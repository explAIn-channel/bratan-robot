"""A live A/B with the ears' null (C52): what the robot heard while it spoke (the array track: the null's output) against
the three microphones as they were (multi, the best one per 0.5 s) and the phone — levels per half second on the A/B
line, where it spoke (played).

  python tools/null_live_check.py STEM"""
import sys
import wave

import numpy as np

RATE = 16000


def wav(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)


def db(x):
    return 10 * np.log10((x ** 2).mean() + 1.)


stem = sys.argv[1]
array, phone, played = wav(stem + '-array.wav'), wav(stem + '-phone.wav'), wav(stem + '-played.wav')
d = np.load(stem + '-multi.npz')
x, t0 = d['x'].astype(np.float64), float(d['t0'])
step = RATE // 2
rows, gains = [], []
for a in range(0, len(played) - step, step):
    i = int(round((a / RATE - t0) * RATE))
    mics = [db(x[i:i + step, c]) for c in range(3)] if 0 <= i < len(x) - step else [0., 0., 0.]
    rob, ears, ph = db(played[a:a + step]), db(array[a:a + step]), db(phone[a:a + step])
    rows.append(f'{a / RATE:5.1f} s  robot {rob:4.0f}  mics {max(mics):4.0f}  heard {ears:4.0f}  phone {ph:4.0f}')
    if rob > 50 and max(mics) > 0:
        gains.append(ears - max(mics))
print('\n'.join(rows))
if gains:
    print(f'while it spoke ({len(gains)} half-seconds): heard − loudest microphone = median {np.median(gains):+.1f} dB, '
          f'worst {max(gains):+.1f} dB')
