"""On the phone: a real GPT-Live session fed with synthesized speech in real time; latency to first answer audio."""
import json, sys, time, threading, urllib.request
sys.path.insert(0, '/opt/robotbrain/bratan/app')
import numpy as np
from bratan.face import client as face
from bratan.voice.live import LiveSession
from bratan.common.persona import CHARACTER, LIVE_RULES

key = open('/opt/robotbrain/secrets/openai-live-key').read().strip()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': face.proxy_url(19408)}))


def tts(text):
    body = json.dumps({'model': 'gpt-4o-mini-tts', 'voice': 'ash', 'input': text, 'response_format': 'pcm'}).encode()
    raw = opener.open(urllib.request.Request('https://api.openai.com/v1/audio/speech', data=body, headers={
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}), timeout=25).read()
    x = np.frombuffer(raw, dtype='<i2').astype(np.float32)
    return np.interp(np.arange(0, len(x), 1.5), np.arange(len(x)), x).astype('<i2').tobytes()


class Sink:
    speaking = False
    def __init__(self): self.first = None; self.bytes = 0
    def play(self, pcm):
        if self.first is None: self.first = time.monotonic()
        self.bytes += len(pcm)


events = []
for phrase in sys.argv[1:] or ['Братан, привет! Как у тебя дела?']:
    speech = tts(phrase)
    sink = Sink()
    s = LiveSession(key, sink, instructions=lambda: CHARACTER + '\n' + LIVE_RULES, voice='coral',
                    on_event=lambda k, **d: events.append((round(time.monotonic(), 2), k, d)), proxy_url=face.proxy_url)
    s.ask = lambda i, text, said: events.append(('ask', text)) or s.answer(i, 'Мозг: на кухне я был 10 минут назад, там тихо.')
    t0 = time.monotonic()
    s.connect()
    th = threading.Thread(target=s.begin, args=([],), daemon=True); th.start()
    # stream the phrase in real time, 20 ms frames, then 4 s of silence
    pcm = speech + b'\x00' * 32000 * 4
    for i in range(0, len(pcm), 640):
        s.feed(pcm[i:i + 640])
        if i == len(speech) - len(speech) % 640:
            speech_end = time.monotonic()
        if i == 6400:
            s.release()               # name confirmed shortly after it was said
        time.sleep(.02)
    time.sleep(3)
    print(json.dumps({'phrase': phrase, 'session_live_after_s': round((s.started_at or 0) - t0, 2),
                      'first_audio_after_speech_end_s': round(sink.first - speech_end, 2) if sink.first else None,
                      'answer_audio_s': round(sink.bytes / 32000, 1), 'said': s.output_text[:200],
                      'heard': ''.join(s.input_text)[:200]}, ensure_ascii=False), flush=True)
    s.close()
print([e for e in events if e[1] not in ('said', 'heard')][:12])
