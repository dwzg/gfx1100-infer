#!/usr/bin/env python3
"""Checks the server's prompt formatting against the chat template stored in each model file.

The server does not interpret the Jinja template; it recognises the format and writes the prompt
itself. Here the template is run with jinja2 on a set of conversations (system prompt, tools, tool
calls and results, reasoning) and the result is compared with `gfxi render`, byte for byte.

    tests/test_chat.py <gfxi> <model.gguf> [more.gguf ...]      (needs: pip install jinja2)

A model file may be truncated after its header.
"""
import json, struct, subprocess, sys, tempfile, os

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


def chat_template(path):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, 'rb') as f:
        d = f.read(64 << 20)
    p = 4
    def rd(fmt):
        nonlocal p
        v = struct.unpack_from('<' + fmt, d, p)[0]; p += struct.calcsize('<' + fmt); return v
    def rstr():
        nonlocal p
        n = rd('Q'); s = d[p:p + n]; p += n; return s
    def skip(t):
        nonlocal p
        if t == 8: return rstr()
        if t == 9:
            et = rd('I'); n = rd('Q')
            if et == 8:
                for _ in range(n): rstr()
            else: p += n * sizes[et]
            return None
        p += sizes[t]; return None
    rd('I'); rd('Q'); nk = rd('Q')
    for _ in range(nk):
        k = rstr().decode(); t = rd('I'); v = skip(t)
        if k == 'tokenizer.chat_template': return v.decode()
    raise SystemExit(f'{path}: no chat template')


def raise_exception(msg): raise jinja2.exceptions.TemplateError(msg)


def run_template(text, messages, tools, **kw):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters['tojson'] = lambda x, ensure_ascii=False, indent=None: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent)
    env.globals['raise_exception'] = raise_exception
    return env.from_string(text).render(messages=messages, tools=tools or None, add_generation_prompt=True, **kw)


def to_openai(req):
    """The request as the list of messages a chat template expects."""
    out = []
    if req.get('system'): out.append({'role': 'system', 'content': req['system']})
    for m in req['messages']:
        c = m['content']
        if isinstance(c, str):
            out.append({'role': m['role'], 'content': c}); continue
        if m['role'] == 'assistant':
            msg = {'role': 'assistant', 'content': ''.join(b['text'] for b in c if b['type'] == 'text')}
            think = [b['thinking'] for b in c if b['type'] == 'thinking']
            if think: msg['reasoning_content'] = think[0]
            calls = [{'type': 'function', 'function': {'name': b['name'], 'arguments': b['input']}} for b in c if b['type'] == 'tool_use']
            if calls: msg['tool_calls'] = calls
            out.append(msg)
        else:
            text = [b['text'] for b in c if b['type'] == 'text']
            for b in c:
                if b['type'] == 'tool_result': out.append({'role': 'tool', 'content': b['content']})
            if text: out.append({'role': 'user', 'content': '\n'.join(text)})
    tools = [{'type': 'function', 'function': {'name': t['name'], 'description': t['description'], 'parameters': t['input_schema']}} for t in req.get('tools', [])]
    return out, tools


TOOLS = [
    {'name': 'Read', 'description': 'Reads a file from the local filesystem.\nUse "absolute" paths — never relative ones.',
     'input_schema': {'type': 'object', 'properties': {'file_path': {'type': 'string', 'description': 'The absolute path'}, 'limit': {'type': 'integer'}}, 'required': ['file_path']}},
    {'name': 'Edit', 'description': 'Performs exact string replacements in files.',
     'input_schema': {'type': 'object', 'properties': {'file_path': {'type': 'string'}, 'edits': {'type': 'array', 'items': {'type': 'object'}}}, 'required': ['file_path', 'edits']}},
]
CALL1 = {'type': 'tool_use', 'id': 'toolu_1', 'name': 'Read', 'input': {'file_path': '/src/main.cpp', 'limit': 40}}
CALL2 = {'type': 'tool_use', 'id': 'toolu_2', 'name': 'Edit', 'input': {'file_path': '/src/main.cpp', 'edits': [{'old': 'int  x;', 'new': 'int x = 0;\n'}]}}
RES1 = {'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': '     1\t#include <cstdio>\n     2\tint main() { puts("<|im_end|> is just text here"); }'}
RES2 = {'type': 'tool_result', 'tool_use_id': 'toolu_2', 'content': 'The file was updated.'}

CASES = {
    'plain question': {'messages': [{'role': 'user', 'content': 'What is a wave size?'}]},
    'system prompt': {'system': 'You are a terse assistant.\n\nAnswer in one line.', 'messages': [{'role': 'user', 'content': '  Hello there.  '}]},
    'several turns': {'system': 'Be brief.', 'messages': [
        {'role': 'user', 'content': 'One'}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Two'}]}, {'role': 'user', 'content': 'Three'}]},
    'tools, first request': {'system': 'You are a coding agent.', 'tools': TOOLS, 'messages': [{'role': 'user', 'content': 'Fix the bug in main.cpp'}]},
    'tools without system prompt': {'tools': TOOLS, 'messages': [{'role': 'user', 'content': 'Fix the bug in main.cpp'}]},
    'tool call and result': {'system': 'You are a coding agent.', 'tools': TOOLS, 'messages': [
        {'role': 'user', 'content': 'Fix the bug in main.cpp'},
        {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'I should read the file first.', 'signature': 'x'}, {'type': 'text', 'text': 'Let me look.'}, CALL1]},
        {'role': 'user', 'content': [RES1]}]},
    'call without text, two results, then a new question': {'system': 'You are a coding agent.', 'tools': TOOLS, 'messages': [
        {'role': 'user', 'content': 'Fix the bug in main.cpp'},
        {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'Read, then edit.', 'signature': 'x'}, CALL1, CALL2]},
        {'role': 'user', 'content': [RES1, RES2]},
        {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'Done now.', 'signature': 'x'}, {'type': 'text', 'text': 'Fixed: x is initialised.'}]},
        {'role': 'user', 'content': 'Thanks. Anything else wrong in there?'}]},
    'result followed by a note': {'system': 'You are a coding agent.', 'tools': TOOLS, 'messages': [
        {'role': 'user', 'content': 'Fix the bug in main.cpp'},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Reading.'}, CALL1]},
        {'role': 'user', 'content': [RES1, {'type': 'text', 'text': '<system-reminder>The user opened main.cpp.</system-reminder>'}]}]},
}


def main():
    gfxi, models = sys.argv[1], sys.argv[2:]
    failed = total = 0
    for model in models:
        text = chat_template(model)
        if '<|im_start|>' not in text:
            print(f'{os.path.basename(model)}: not a ChatML template, skipped'); continue
        # What the template itself does with reasoning from earlier turns, and whether it has effort levels.
        keeps = 'preserve_thinking' in text
        modes = [(False, None), (True, None)]
        if 'reasoning_effort' in text: modes += [(True, 'low'), (True, 'medium')]
        for name, req in CASES.items():
            if 'tools' in req and '<tool_call>' not in text: continue
            if 'messages[0].content +' in text and any(not isinstance(m['content'], str) and m['role'] == 'user' and len(m['content']) > 1 for m in req['messages']): pass
            for think, effort in modes:
                messages, tools = to_openai(req)
                kw = {'enable_thinking': think}
                if effort: kw['reasoning_effort'] = effort
                try: want = run_template(text, messages, tools, **kw)
                except jinja2.exceptions.TemplateError as e:
                    print(f'  (template rejects "{name}": {e})'); continue
                with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
                    json.dump(req, f, ensure_ascii=False)
                cmd = [gfxi, 'render', model, f.name] + (['--think'] if think else []) + (['--effort', effort] if effort else []) + ([] if keeps else ['--drop-thinking'])
                got = subprocess.run(cmd, capture_output=True)
                os.unlink(f.name)
                total += 1
                if got.returncode != 0 or got.stdout.decode() != want:
                    failed += 1
                    print(f'FAIL {os.path.basename(model)}: {name} (thinking {think}, effort {effort})')
                    g = got.stdout.decode() if got.returncode == 0 else got.stderr.decode()
                    i = next((k for k in range(min(len(g), len(want))) if g[k] != want[k]), min(len(g), len(want)))
                    print(f'  first difference at byte {i}:\n    template: {want[max(0, i - 60):i + 80]!r}\n    gfxi:     {g[max(0, i - 60):i + 80]!r}')
        print(f'{os.path.basename(model)}: checked')
    print(f'{total - failed}/{total} prompts identical to the templates')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
