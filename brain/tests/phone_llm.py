"""On the phone: the mind's LLM client — latency over many calls, JSON mode, streaming, retries."""
import sys, time
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '/opt/robotbrain/bratan/app')
from bratan.common.config import load, secret
from bratan.mind.llm import LLM, parse_json
c = load(None)
llm = LLM(c['llm_base_url'], secret(c['llm_key']), c['llm_fast'], pool=3, fallback=c.get('llm_smart'))
times, fails = [], 0
for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 12):
    t = time.monotonic()
    try:
        raw = llm.chat('Ответь одним JSON.', 'Верни {"thought": "короткая мысль робота о вечере", "do": null}',
                       max_tokens=120, json_mode=True, timeout=15.)
        ok = bool(parse_json(raw).get('thought'))
        times.append(time.monotonic() - t)
        print(f'{i:2d} {times[-1]:.2f}s json_ok={ok}', flush=True)
    except Exception as e:
        fails += 1
        print(f'{i:2d} FAIL {type(e).__name__} {e}', flush=True)
    time.sleep(2)
times.sort()
print('median', round(times[len(times)//2], 2), 'max', round(times[-1], 2), 'fails', fails, llm.stats)
