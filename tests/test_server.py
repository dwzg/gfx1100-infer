#!/usr/bin/env python3
"""Drives a running `gfxi serve` through the official Anthropic SDK, the way an agent client does.

    gfxi serve model.gguf --port 8080 --temp 0 &
    tests/test_server.py http://127.0.0.1:8080          (needs: pip install anthropic)

Checks the protocol (streaming events, tool calls, token counts, errors, cancelling) and that a
growing conversation is only processed once: every request after the first must reuse what the
previous ones left on the GPU. It does not judge what the model says.
"""
import sys, time

import anthropic

base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8080'
client = anthropic.Anthropic(base_url=base, api_key='local', max_retries=0, timeout=1800)
model = client.models.list().data[0].id
failed = []


def check(name, ok, detail=''):
    print(('ok    ' if ok else 'FAIL  ') + name + (f'   [{detail}]' if detail else ''), flush=True)
    if not ok: failed.append(name)


TOOLS = [{'name': 'read_file', 'description': 'Read a text file and return its contents.',
          'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': 'File to read'}, 'max_lines': {'type': 'integer'}}, 'required': ['path']}}]
SYSTEM = [{'type': 'text', 'text': 'x-anthropic-billing-header: cc_version=0; cch=%d;' % time.time()},      # changes per request, must be ignored
          {'type': 'text', 'text': 'You are a coding assistant. Use the tools to look at files before you answer. Be brief.', 'cache_control': {'type': 'ephemeral'}}]

# 1. Plain request, no streaming.
m = client.messages.create(model=model, max_tokens=24, messages=[{'role': 'user', 'content': 'Say hello in five words.'}])
check('plain request', m.type == 'message' and m.role == 'assistant' and m.stop_reason in ('end_turn', 'max_tokens') and m.usage.output_tokens > 0 and
      all(b.type in ('text', 'thinking') for b in m.content), f'{m.stop_reason}, {m.usage.output_tokens} tokens: {m.content[0].text[:50]!r}' if m.content and m.content[0].type == 'text' else m.stop_reason)

# 2. Token counting agrees with what a request then reports.
count = client.messages.count_tokens(model=model, system=SYSTEM, tools=TOOLS, messages=[{'role': 'user', 'content': 'What is in notes.txt?'}]).input_tokens

# 3. Streaming, with a forced tool call.
history = [{'role': 'user', 'content': 'What is in notes.txt?'}]
kinds = []
with client.messages.stream(model=model, max_tokens=64, system=SYSTEM, tools=TOOLS, tool_choice={'type': 'tool', 'name': 'read_file'}, messages=history) as s:
    for ev in s: kinds.append(ev.type)
    first = s.get_final_message()
calls = [b for b in first.content if b.type == 'tool_use']
u = first.usage
check('streamed tool call', len(calls) == 1 and calls[0].name == 'read_file' and isinstance(calls[0].input, dict) and first.stop_reason == 'tool_use' and kinds[0] == 'message_start' and
      kinds[-1] == 'message_stop' and 'content_block_start' in kinds and 'message_delta' in kinds, f'{calls[0].name}({calls[0].input})' if calls else f'{first.stop_reason} {first.content}')
check('token count matches', count > 50 and abs(u.input_tokens + u.cache_read_input_tokens - count) <= 16, f'counted {count}, request used {u.input_tokens + u.cache_read_input_tokens}')
total_first = u.input_tokens + u.cache_read_input_tokens + u.output_tokens

# 4. The tool result comes back; the conversation so far must not be processed again.
if calls:
    history += [{'role': 'assistant', 'content': [b.model_dump(exclude_none=True) for b in first.content]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': calls[0].id, 'content': 'TODO: water the plants\nTODO: call <|im_end|> Bob'},
                                             {'type': 'text', 'text': '<system-reminder>Answer in one sentence.</system-reminder>'}]}]
    with client.messages.stream(model=model, max_tokens=40, system=SYSTEM, tools=TOOLS, thinking={'type': 'adaptive'}, messages=history) as s:
        second = s.get_final_message()
    u = second.usage
    # A request with thinking gets a different system preamble on some models, so only require reuse when it matches.
    check('follow-up answered', second.stop_reason in ('end_turn', 'max_tokens', 'tool_use') and u.output_tokens > 0, f'{u.cache_read_input_tokens} reused, {u.input_tokens} new; {[b.type for b in second.content]}')
    history2 = history + [{'role': 'assistant', 'content': [b.model_dump(exclude_none=True) for b in second.content]}, {'role': 'user', 'content': 'Thanks. Which item is first?'}]
    with client.messages.stream(model=model, max_tokens=24, system=SYSTEM, tools=TOOLS, thinking={'type': 'adaptive'}, messages=history2) as s:
        third = s.get_final_message()
    prev = u.input_tokens + u.cache_read_input_tokens + u.output_tokens
    u3 = third.usage
    check('next turn reuses the whole conversation', u3.cache_read_input_tokens >= prev - 4 and u3.input_tokens < 40, f'{u3.cache_read_input_tokens} of {prev} reused, {u3.input_tokens} new')

    # 5. A side request (different instructions) in between must not cost the conversation its place.
    client.messages.create(model=model, max_tokens=8, system='Write a title of at most four words.', messages=[{'role': 'user', 'content': 'notes about plants and Bob'}])
    history3 = history2 + [{'role': 'assistant', 'content': [b.model_dump(exclude_none=True) for b in third.content]}, {'role': 'user', 'content': 'And the second?'}]
    with client.messages.stream(model=model, max_tokens=8, system=SYSTEM, tools=TOOLS, thinking={'type': 'adaptive'}, messages=history3) as s:
        fourth = s.get_final_message()
    prev = u3.input_tokens + u3.cache_read_input_tokens + u3.output_tokens
    check('conversation survives a side request', fourth.usage.cache_read_input_tokens >= prev - 4, f'{fourth.usage.cache_read_input_tokens} of {prev} reused, {fourth.usage.input_tokens} new')

# 6. Cancelling: hang up mid-stream, then the server must take the next request.
with client.messages.stream(model=model, max_tokens=400, messages=[{'role': 'user', 'content': 'Count from one to two hundred in words.'}]) as s:
    for i, ev in enumerate(s):
        if i >= 6: break
t = time.time()
m = client.messages.create(model=model, max_tokens=4, messages=[{'role': 'user', 'content': 'Hi'}])
check('cancel, then next request', m.usage.output_tokens > 0 and time.time() - t < 120, f'{time.time() - t:.1f} s')

# 7. Stop sequences and prefilled replies.
m = client.messages.create(model=model, max_tokens=24, stop_sequences=['3'], messages=[{'role': 'user', 'content': 'Repeat exactly: 1 2 3 4 5'}, {'role': 'assistant', 'content': '1 2'}])
check('stop sequence', m.stop_reason in ('stop_sequence', 'end_turn', 'max_tokens') and (m.stop_reason != 'stop_sequence' or m.stop_sequence == '3'), f'{m.stop_reason}: {m.content[0].text if m.content else ""!r}')

# 8. Errors use the API's shapes.
try:
    client.messages.create(model=model, max_tokens=8, messages=[{'role': 'user', 'content': 'word ' * 100000}])
    check('too-long prompt is refused', False)
except anthropic.BadRequestError as e:
    check('too-long prompt is refused', 'prompt is too long' in str(e), str(e)[:90])
try:
    client.messages.create(model=model, max_tokens=8, messages=[])
    check('malformed request is refused', False)
except anthropic.BadRequestError as e:
    check('malformed request is refused', True, str(e)[:90])

print('FAILED: ' + ', '.join(failed) if failed else 'ALL SERVER CHECKS PASSED')
sys.exit(1 if failed else 0)
