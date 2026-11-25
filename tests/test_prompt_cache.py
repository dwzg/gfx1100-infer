#!/usr/bin/env python3
"""Checks the server's on-disk store of processed prompts: a request is sent, the server is
restarted, and the same request must start from the stored prefix and give the same answer.

    tests/test_prompt_cache.py <gfxi> <model.gguf> [port]       (needs: pip install anthropic)
"""
import os, subprocess, sys, tempfile, time, urllib.request

import anthropic

gfxi, model_path = sys.argv[1], sys.argv[2]
port = int(sys.argv[3]) if len(sys.argv) > 3 else 18182
base = f'http://127.0.0.1:{port}'
os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1'
TOOLS = [{'name': 'read_file', 'description': 'Read a text file and return its contents. ' * 6,
          'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}]
SYSTEM = 'You are a coding assistant. Be brief. ' * 12


def ask(question):
    client = anthropic.Anthropic(base_url=base, api_key='local', max_retries=0, timeout=1800)
    m = client.messages.create(model='local', max_tokens=12, system=SYSTEM, tools=TOOLS, messages=[{'role': 'user', 'content': question}])
    return m.usage.cache_read_input_tokens, m.usage.input_tokens, ''.join(b.text for b in m.content if b.type == 'text')


def serve(store):
    p = subprocess.Popen([gfxi, 'serve', model_path, '--port', str(port), '--ctx', '4096', '--temp', '0', '--prompt-cache', store, '--prompt-cache-min', '64'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(300):
        try:
            opener.open(base + '/health', timeout=2); return p
        except Exception:
            if p.poll() is not None: raise SystemExit('the server did not start')
            time.sleep(1)
    p.kill(); raise SystemExit('the server did not start')


with tempfile.TemporaryDirectory() as store:
    p = serve(store)
    try: reused1, new1, text1 = ask('Say hello.')
    finally: p.terminate(); p.wait()
    files = [f for f in os.listdir(store) if f.endswith('.ctx')]
    p = serve(store)
    try:
        reused2, new2, text2 = ask('Say hello.')
        reused3, new3, text3 = ask('What is two plus two?')
    finally: p.terminate(); p.wait()
ok = reused1 == 0 and files and reused2 > 200 and new2 < new1 / 4 and text2 == text1 and reused3 >= reused2
print(f'first start: {new1} tokens processed; {len(files)} prefix file(s) stored')
print(f'after a restart: {reused2} tokens from the store, {new2} processed, same answer: {text2 == text1}; another question: {reused3} reused')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
