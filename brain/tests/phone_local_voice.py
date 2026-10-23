"""On the phone: offline voice round trip — Piper speaks, Vosk hears it back, request() strips the name."""
import os, sys, time
sys.path.insert(0, '/opt/robotbrain/bratan/app')
os.sched_setaffinity(0, {4, 5})
from vosk import Model, SetLogLevel
from bratan.common.config import MODELS
from bratan.voice.local import LocalSpeech, Phrase, request


class Sink:
    def __init__(self): self.pcm = bytearray()
    def play(self, pcm): self.pcm += pcm


sink = Sink()
t = time.monotonic()
speech = LocalSpeech(sink, str(MODELS / 'piper' / 'ru_RU-irina-medium.onnx'))
speech.ready.wait(30)
print('ready_s', round(time.monotonic() - t, 2))
for text in ['Да?', 'Слышу тебя, но облачный голос сейчас не работает.', 'Братан, как дела у тебя?']:
    sink.pcm = bytearray()
    first = speech.say(text)
    print(f'first_sound {first:.3f}s audio {len(sink.pcm) / 32000:.2f}s  {text}')
SetLogLevel(-1)
model = Model(str(MODELS / 'vosk-ru'))
pcm = bytes(sink.pcm) + b'\x00' * 32000
phrase = Phrase(model)
for i in range(0, len(pcm) - 639, 640):
    phrase.feed(pcm[i:i + 640])
    if phrase.ended:
        break
heard = phrase.text()
print('heard:', heard, '| request:', request(heard), '| ended', phrase.ended)
