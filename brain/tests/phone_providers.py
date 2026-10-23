"""On the phone: which OpenRouter providers serve the mind's model, and how fast (first token)."""
import http.client, json, ssl, sys, time
sys.path.insert(0, '/opt/robotbrain/bratan/app')
from bratan.common.config import load, secret
c = load(None)
key = secret(c['llm_key'])
prompt = 'Опиши в двух предложениях тихий вечер в квартире робота. ' * 40      # a long-ish prompt
def call(provider=None):
    body = {'model': c['llm_fast'], 'max_tokens': 60, 'stream': True, 'reasoning': {'enabled': False},
            'messages': [{'role': 'user', 'content': prompt}]}
    if provider:
        body['provider'] = {'order': [provider], 'allow_fallbacks': False}
    conn = http.client.HTTPSConnection('openrouter.ai', 443, timeout=30, context=ssl.create_default_context())
    t = time.monotonic()
    conn.request('POST', '/api/v1/chat/completions', json.dumps(body).encode(),
                 {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    r = conn.getresponse()
    first, who, buf = None, None, b''
    while True:
        chunk = r.read1(4096)
        if not chunk:
            break
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            if line.startswith(b'data:') and b'[DONE]' not in line:
                try:
                    d = json.loads(line[5:])
                except ValueError:
                    continue
                who = who or d.get('provider')
                if first is None and ((d.get('choices') or [{}])[0].get('delta') or {}).get('content'):
                    first = time.monotonic() - t
    return r.status, who, first, time.monotonic() - t
for i in range(6):
    print('auto', call())
    time.sleep(1)
