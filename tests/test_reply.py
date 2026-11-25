#!/usr/bin/env python3
"""Checks the reply parser: generated text -> reasoning, text and tool-call blocks.

Each reply is fed to the parser in pieces of several sizes (down to single bytes, so markers and
multi-byte characters get split everywhere) and must always produce the same blocks, a stream of
events that adds up to those blocks, and - parsed and written back - the text it started as.

    tests/test_reply.py <gfxi> <model with <function=...> tool calls> [<model with JSON tool calls>]
"""
import json, os, subprocess, sys, tempfile

TOOLS = [
    {'name': 'Read', 'description': 'Read a file', 'input_schema': {'type': 'object', 'properties': {'file_path': {'type': 'string'}, 'limit': {'type': 'integer'}}}},
    {'name': 'Edit', 'description': 'Edit a file', 'input_schema': {'type': 'object', 'properties': {
        'file_path': {'type': 'string'}, 'edits': {'type': 'array'}, 'replace_all': {'type': 'boolean'}, 'new_string': {'type': 'string'}}}},
]
CALL = '<tool_call>\n<function=Read>\n<parameter=file_path>\n/src/main.cpp\n</parameter>\n<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>'
CODE = 'int main() {\n    puts("<|im_end|> 123");\n    return 0;\n}\n'
XML = [
    # name, reply text, flags, expected blocks (type, value), text the reply is written back as (None: same)
    ('reasoning, text, call', 'Let me think.\n</think>\n\nI will read it.\n\n' + CALL, ['--reasoning'],
     [('thinking', 'Let me think.'), ('text', 'I will read it.'), ('tool_use', ('Read', {'file_path': '/src/main.cpp', 'limit': 40}))], '<think>\n{}'),
    ('answer only', '\n\nPlain answer — with “quotes” and 你好.\n', [], [('text', 'Plain answer — with “quotes” and 你好.')], None),
    ('two calls, typed arguments', 'Fixing.\n\n' + CALL + '\n<tool_call>\n<function=Edit>\n<parameter=file_path>\n/a b.c\n</parameter>\n<parameter=edits>\n[{"old": "x", "new": "y"}]\n</parameter>\n'
     '<parameter=replace_all>\ntrue\n</parameter>\n<parameter=new_string>\n' + CODE + '\n</parameter>\n</function>\n</tool_call>', [],
     [('text', 'Fixing.'), ('tool_use', ('Read', {'file_path': '/src/main.cpp', 'limit': 40})),
      ('tool_use', ('Edit', {'file_path': '/a b.c', 'edits': [{'old': 'x', 'new': 'y'}], 'replace_all': True, 'new_string': CODE}))], None),
    ('string argument that looks like JSON', '<tool_call>\n<function=Edit>\n<parameter=new_string>\n{"a": 1}\n</parameter>\n</function>\n</tool_call>', [],
     [('tool_use', ('Edit', {'new_string': '{"a": 1}'}))], None),
    ('call without its wrapper', 'Reading.\n<function=Read>\n<parameter=file_path>\n/x\n</parameter>\n</function>', [],
     [('text', 'Reading.'), ('tool_use', ('Read', {'file_path': '/x'}))], False),
    ('the notation mentioned in text', 'Calls look like <function=Read> here.\n<function=nothing_real>\nand so on', [],
     [('text', 'Calls look like <function=Read> here.\n<function=nothing_real>\nand so on')], None),
    ('reasoning never closed', 'Still thinking', ['--reasoning'], [('thinking', 'Still thinking')], False),
    ('call during reasoning', 'I need the file.\n' + CALL, ['--reasoning'], [('thinking', 'I need the file.'), ('tool_use', ('Read', {'file_path': '/src/main.cpp', 'limit': 40}))], False),
    ('cut off inside an argument', 'Go.\n\n<tool_call>\n<function=Edit>\n<parameter=file_path>\n/a\n</parameter>\n<parameter=new_string>\nint ma', ['--cut'],
     [('text', 'Go.'), ('tool_use', ('Edit', {'file_path': '/a'}))], False),
    ('turn ended inside an argument', '<tool_call>\n<function=Edit>\n<parameter=new_string>\nint x;', [], [('tool_use', ('Edit', {'new_string': 'int x;'}))], False),
]
JSON_STYLE = [
    ('reasoning, text, call', '<think>\nhmm\n</think>\n\nOK.\n<tool_call>\n{"name": "Read", "arguments": {"file_path": "/x", "limit": 3}}\n</tool_call>', [],
     [('thinking', 'hmm'), ('text', 'OK.'), ('tool_use', ('Read', {'file_path': '/x', 'limit': 3}))], None),
    ('not a call after all', 'See <tool_call>this</tool_call> tag', [], [('text', 'See <tool_call>this</tool_call> tag')], None),
]


def run(gfxi, model, req, text, flags, chunk):
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f: json.dump(req, f)
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8', newline='') as t: t.write(text)
    out = subprocess.run([gfxi, 'parse-reply', model, f.name, t.name, '--chunk', str(chunk)] + flags, capture_output=True)
    os.unlink(f.name); os.unlink(t.name)
    if out.returncode != 0: raise RuntimeError(out.stderr.decode())
    return json.loads(out.stdout.decode('utf-8'))


def check(gfxi, model, cases):
    failed = 0
    for name, text, flags, want, back in cases:
        for chunk in (1, 2, 3, 7, 64, 100000):
            r = run(gfxi, model, {'tools': TOOLS}, text, flags, chunk)
            got = [(b['type'], b.get('thinking') if b['type'] == 'thinking' else b.get('text') if b['type'] == 'text' else (b['name'], b['input'])) for b in r['blocks']]
            problems = []
            if got != want: problems.append(f'blocks {got!r}, expected {want!r}')
            # The events must add up to the blocks.
            built, cur = [], None
            for e in r['events']:
                kind = e['event']
                if kind.endswith('_start'): cur = [kind[:-6], e['text'] if kind == 'tool_start' else '', '']
                elif kind == 'stop': built.append(cur); cur = None
                elif kind == 'tool_delta': cur[2] += e['text']
                else: cur[1] += e['text']
            if cur is not None: problems.append('a block was never closed')
            for (kind, a, js), (wk, wv) in zip(built, want):
                if kind == 'tool':
                    if '--cut' in flags and wk == 'tool_use' and (kind, a) == ('tool', wv[0]):
                        try: json.loads(js); problems.append('input of a cut-off call is complete JSON')
                        except ValueError: pass
                    elif (a, json.loads(js)) != wv: problems.append(f'streamed call {a} {js!r}')
                elif a != wv: problems.append(f'streamed {kind} {a!r}')
            if len(built) != len(want): problems.append(f'{len(built)} streamed blocks')
            if back is not False:
                expect = (back or '{}').format(text.strip())
                if r['body'] != expect and r['body'] != '<think>\n\n</think>\n\n' + expect: problems.append(f'written back as {r["body"]!r}')
            if problems:
                failed += 1
                print(f'FAIL {name} (pieces of {chunk}): ' + '; '.join(problems))
                break
    # Stop sequences end the text.
    r = run(gfxi, model, {'tools': TOOLS, 'stop_sequences': ['###', 'END']}, 'one two ##three END four', [], 2)
    if not r['stopped'] or r['blocks'] != [{'type': 'text', 'text': 'one two ##three'}]: failed += 1; print('FAIL stop sequence:', r['blocks'])
    return failed, len(cases) + 1


def main():
    gfxi = sys.argv[1]
    failed, total = check(gfxi, sys.argv[2], XML)
    if len(sys.argv) > 3:
        f, t = check(gfxi, sys.argv[3], JSON_STYLE); failed += f; total += t
    print(f'{total - failed}/{total} reply cases passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
