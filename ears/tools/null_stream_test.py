"""The streaming RobotNull (bratan/voice/nullvoice.py) as the robot will run it: the pattern first learned on TRAIN
recordings (stored), then the TEST recording block by block — learning while the robot speaks — written as
<test>-stream.wav on the A/B time line for Microsoft.

  python tools/null_stream_test.py TEST_STEM TRAIN_STEM [TRAIN_STEM ...]"""
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bratan'))
from bratan.voice.nullvoice import RobotNull  # noqa: E402

RATE = 16000


def load(stem):
    d = np.load(stem + '-multi.npz')
    with wave.open(stem + '-played.wav') as w:
        played = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
    return d['x'], float(d['t0']), played


def run(null, stem, keep=False):
    x, t0, played = load(stem)
    out = []
    for i in range(0, len(x) - 511, 512):
        t = t0 + i / RATE
        a = int(t * RATE)
        speaking = 0 <= a < len(played) - 512 and 10 * np.log10((played[a:a + 512] ** 2).mean() + 1) > 40
        out.append(null.process(x[i:i + 512], learn=speaking))
    if keep:
        y = np.concatenate(out)
        lead = int(round(-t0 * RATE)) + 256                       # the filter is 256 samples behind
        track = np.zeros(len(played))
        track[max(0, lead):max(0, lead) + len(y)] = y[:len(played) - max(0, lead)]
        with wave.open(stem + '-stream.wav', 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(np.clip(track, -32768, 32767).astype(np.int16).tobytes())


null = RobotNull()
for train in sys.argv[2:]:
    run(null, train)
print('learned windows after training:', null.learned, 'ready', null.ready)
run(null, sys.argv[1], keep=True)
print('learned windows after the test:', null.learned)
