"""On the phone: synthesize phrases (OpenAI TTS) and run them through Listener+Confirmer."""
import json, sys, time, urllib.request
sys.path.insert(0, '/opt/robotbrain/bratan/app')
import numpy as np
from vosk import Model, SetLogLevel
from bratan.face import client as face
from bratan.voice.wake import Listener, Confirmer
SetLogLevel(-1)
key = open('/opt/robotbrain/secrets/openai-live-key').read().strip()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': face.proxy_url(19408)}))
model = Model('/opt/robotbrain/models/vosk-ru')
phrases = ['Братан, привет!', 'Эй, братан, как дела?', 'Братан, поехали на кухню.', 'Привет, как дела?',
           'Как дела, братан?', 'Слушай, а что ты сегодня делал, братан?', 'Братан, стоп!', 'Стой!']
for text in phrases:
    body = json.dumps({'model': 'gpt-4o-mini-tts', 'voice': 'ash', 'input': text, 'response_format': 'pcm'}).encode()
    raw = opener.open(urllib.request.Request('https://api.openai.com/v1/audio/speech', data=body, headers={
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}), timeout=25).read()
    x = np.frombuffer(raw, dtype='<i2').astype(np.float32)
    y = np.interp(np.arange(0, len(x), 1.5), np.arange(len(x)), x) * .25          # quieter, like a voice across a room
    noise = np.random.default_rng(1).normal(0, 30, 16000).astype(np.float32)
    pcm = np.concatenate([noise, y, noise]).astype('<i2').tobytes()
    listener = Listener(model)
    confirmer = None
    t = 0.
    result = {'text': text}
    started = time.perf_counter()
    for i in range(0, len(pcm) - 639, 640):
        frame = pcm[i:i + 640]
        t = i / 32000
        heard = listener.feed(t, frame, listen_for_name=confirmer is None)
        if heard == 'stop':
            result['stop_at'] = round(t - 1, 2)
        if heard == 'wake' and confirmer is None:
            result['wake_at'] = round(t - 1, 2)
            confirmer = Confirmer(model, listener.phrase)
            continue
        if confirmer is not None:
            confirmer.feed(frame)
            if confirmer.ended and 'confirmed' not in result:
                ok, said = confirmer.result()
                result.update(confirmed=ok, heard=said, decided_at=round(t - 1, 2))
    result['speech_s'] = round(len(y) / 16000, 2)
    result['cpu_rtf'] = round((time.perf_counter() - started) / (len(pcm) / 32000), 3)
    print(json.dumps(result, ensure_ascii=False), flush=True)
