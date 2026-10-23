"""The robot's own voice out of the ears by space, not by its playback (C52 barge-in, 18.10 evening).

The speaker sits in the body at one place: its voice reaches the three microphones with a fixed pattern of delays and
levels in every frequency. Learned from the robot's own speech (its covariance R_robot per frequency) and with the
person's pattern from their call (the principal direction of R_person), the MVDR filter keeps the person undistorted
and takes as much of the robot as three microphones allow. The board's one clock serves all three: no playback
reference and no clock drift (what kept a playback echo canceller on the ears at ~2 dB, aec_try.py).

  python tools/null_robot.py recordings/ab/test3 PERSON_FROM PERSON_TO [OUT_FROM OUT_TO]
Writes <stem>-null.wav (and <stem>-ref.wav, the best single microphone) on the A/B time line.
"""
import sys
import wave

import numpy as np
from scipy.signal import istft, stft

RATE, N, HOP = 16000, 512, 256


def load_wav(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)


def save(path, x):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.clip(x, -32768, 32767).astype(np.int16).tobytes())


def main(stem, p_from, p_to):
    d = np.load(stem + '-multi.npz')
    x, t0 = d['x'].astype(np.float64), float(d['t0'])
    played = load_wav(stem + '-played.wav')
    _, times, Z = stft(x.T, RATE, nperseg=N, noverlap=N - HOP, boundary=None, padded=False)   # Z: ch × F × T
    frame_t = t0 + times                                     # each STFT frame on the A/B time line
    level = lambda a, b: 10 * np.log10((played[max(0, a):max(1, b)] ** 2).mean() + 1)
    robot = np.array([level(int((t - .016) * RATE), int((t + .016) * RATE)) > 40 for t in frame_t])
    person = (frame_t >= p_from) & (frame_t <= p_to) & ~robot
    energy = (np.abs(Z) ** 2).sum(axis=0).sum(axis=0)
    person &= energy > np.percentile(energy[person], 40)     # the voiced part of the call
    print(f'frames: robot {robot.sum()}, person {person.sum()}')
    F = Z.shape[1]
    out = np.zeros(Z.shape[1:], complex)
    gain_person, gain_robot = [], []
    for f in range(F):
        Xf = Z[:, f, :]                                       # ch × T
        Rr = Xf[:, robot] @ Xf[:, robot].conj().T / max(1, robot.sum())
        Rp = Xf[:, person] @ Xf[:, person].conj().T / max(1, person.sum())
        Rr += np.eye(3) * (np.trace(Rr).real / 3 * 1e-3 + 1e-6)
        vals, vecs = np.linalg.eigh(Rp)
        steer = vecs[:, -1] / (vecs[0, -1] if abs(vecs[0, -1]) > 1e-9 else 1)   # the person, mic 1 as reference
        inv = np.linalg.solve(Rr, steer)
        w = inv / (steer.conj() @ inv)
        out[f] = w.conj() @ Xf
        gain_person.append(np.real(w.conj() @ Rp @ w) / max(Rp[0, 0].real, 1e-9))
        gain_robot.append(np.real(w.conj() @ Rr @ w) / max(Rr[0, 0].real, 1e-9))
    band = slice(int(300 / RATE * N), int(3400 / RATE * N))
    db = lambda g: 10 * np.log10(np.mean(g[band]) + 1e-12)
    print(f'speech band, against microphone 1: person {db(np.array(gain_person)):+.1f} dB, '
          f'robot {db(np.array(gain_robot)):+.1f} dB')
    _, y = istft(out, RATE, nperseg=N, noverlap=N - HOP, boundary=False)
    lead = int(round(-t0 * RATE))                            # back onto the A/B time line
    track = np.zeros(len(played))
    a = max(0, -lead)
    track[max(0, lead):max(0, lead) + len(y) - a] = y[a:a + len(track) - max(0, lead)]
    save(stem + '-null.wav', track)
    ref = np.zeros(len(played))
    ref[max(0, lead):max(0, lead) + len(x) - a] = x[a:a + len(ref) - max(0, lead), 0]
    save(stem + '-ref.wav', ref)


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
