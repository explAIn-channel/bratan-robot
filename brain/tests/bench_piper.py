"""On the phone: how fast Piper speaks Russian (load time, time to the first audio chunk, real-time factor)."""
import os, sys, time, wave
os.sched_setaffinity(0, {4, 5, 6, 7})
from piper import PiperVoice
from piper.config import SynthesisConfig

name = sys.argv[1] if len(sys.argv) > 1 else 'ru_RU-irina-medium'
t0 = time.monotonic()
voice = PiperVoice.load(f'/opt/robotbrain/models/piper/{name}.onnx')
print('load_s', round(time.monotonic() - t0, 2), 'rate', voice.config.sample_rate)
phrases = ['Слышу тебя.', 'Слышу тебя, но голос сейчас не работает.',
           'Я на базе, заряжаюсь. Таффу сегодня видел в коридоре, он спокойно лежал у двери.']
for _ in range(2):
    for text in phrases:
        t = time.monotonic()
        first, samples = None, 0
        for chunk in voice.synthesize(text, SynthesisConfig(length_scale=1.0)):
            if first is None:
                first = time.monotonic() - t
            samples += len(chunk.audio_int16_bytes) // 2
        total = time.monotonic() - t
        audio = samples / voice.config.sample_rate
        print(f'{len(text):3d} chars first {first:.3f}s total {total:.3f}s audio {audio:.2f}s rtf {total / audio:.3f}')
with wave.open('/tmp/piper_test.wav', 'wb') as w:
    voice.synthesize_wav(phrases[2], w)
