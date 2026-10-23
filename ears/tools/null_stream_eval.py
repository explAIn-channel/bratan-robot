"""The streaming RobotNull (bratan/voice/nullvoice.py) on the A/B recordings, as the robot runs it: the pattern learned
first on TRAIN recordings (the stored calibration), then the TEST recording block by block — learning while the robot
speaks, my voice taken out while its sound is in the room, the best microphone as it is otherwise (ArrayLink._pick).

Prints how deep the robot's voice is taken out (out vs. the picked microphone, robot-only windows outside the person's
interruption) and how much of the person is kept: the filters at moments (before / after the interruption) applied to
the person's own pattern (his words with the robot silent, PERSON_FROM..PERSON_TO on the A/B line). Writes
<test>-stream.wav on the A/B line for Microsoft.

  python tools/null_stream_eval.py TEST PERSON_FROM PERSON_TO CUT_FROM CUT_TO TRAIN [TRAIN ...] [--nofit] [--voiced X]
"""
import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bratan'))
from bratan.voice import nullvoice  # noqa: E402
from bratan.voice.array import ArrayLink  # noqa: E402

RATE = 16000


def load(stem):
    d = np.load(stem + '-multi.npz')
    with wave.open(stem + '-played.wav') as w:
        played = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
    return d['x'], float(d['t0']), played


def robot_at(played, t, n=512):
    a = int(t * RATE)
    return 0 <= a < len(played) - n and 10 * np.log10((played[a:a + n] ** 2).mean() + 1) > 40


def run(null, stem, keep=False, marks=(), tail_s=1.):
    x, t0, played = load(stem)
    link = ArrayLink(None)
    out, snaps, last_robot = [], {}, -1e9
    stats = []                                             # (time, robot now, out power, picked power)
    for i in range(0, len(x) - 511, 512):
        t = t0 + i / RATE
        speaking = robot_at(played, t)
        if speaking:
            last_robot = t
        block = x[i:i + 512]
        k = link._pick(block)
        at, y = null.feed(i, block, learn=speaking, clean=t - last_robot < tail_s, pick=k)
        out.append(y)
        if len(y):
            ref = x[at:at + len(y), k].astype(np.float64)
            stats.append((t, speaking, float((y.astype(np.float64) ** 2).mean()), float((ref ** 2).mean())))
        for m in marks:
            if m not in snaps and t >= m:
                snaps[m] = null.W.copy()
    if keep:
        y = np.concatenate(out).astype(np.float64)        # y[0] is x[0]: the A/B time t0
        lead = int(round(t0 * RATE))
        if lead < 0:
            y, lead = y[-lead:], 0
        track = np.zeros(len(played))
        y = y[:len(played) - lead]
        track[lead:lead + len(y)] = y
        with wave.open(stem + '-stream.wav', 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(np.clip(track, -32768, 32767).astype(np.int16).tobytes())
    return stats, snaps


def person_cov(stem, p_from, p_to):
    from scipy.signal import stft
    x, t0, played = load(stem)
    _, times, Z = stft(x.T.astype(np.float64), RATE, nperseg=512, noverlap=256, boundary=None, padded=False)
    t = t0 + times
    robot = np.array([robot_at(played, s - .016) for s in t])
    energy = (np.abs(Z) ** 2).sum(axis=(0, 1))
    person = (t >= p_from) & (t <= p_to) & ~robot
    person &= energy > np.percentile(energy[person], 40)
    return np.einsum('cft,dft->fcd', Z[:, :, person], Z[:, :, person].conj()) / person.sum()


def gain_db(W, R):
    g = np.real(np.einsum('fc,fcd,fd->f', W.conj(), R, W)) / np.maximum(np.einsum('fcc->f', R).real / 3, 1e-9)
    return 10 * np.log10(g[nullvoice.BAND].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('test')
    ap.add_argument('p_from', type=float)
    ap.add_argument('p_to', type=float)
    ap.add_argument('cut_from', type=float)
    ap.add_argument('cut_to', type=float)
    ap.add_argument('train', nargs='+')
    ap.add_argument('--nofit', action='store_true')
    ap.add_argument('--voiced', type=float)
    ap.add_argument('--save')
    a = ap.parse_args()
    if a.nofit:
        nullvoice.FIT_DB = 1e9
    if a.voiced:
        nullvoice.VOICED = a.voiced
    null = nullvoice.RobotNull()
    for stem in a.train:
        run(null, stem)
    print('after training:', null.summary())
    if a.save:
        null.path = Path(a.save)
        null.dirty = True
        null.save()
    marks = (a.cut_from, a.cut_to + .5)
    stats, snaps = run(null, a.test, keep=True, marks=marks)
    print('after the test:', null.summary())
    rob = [(o, p) for t, r, o, p in stats if r and not (a.cut_from - 1 <= t <= a.cut_to + 1) and p > 1e5]
    o, p = np.array(rob).T
    print(f'robot-only windows {len(rob)}: out/picked {10 * np.log10(o.sum() / p.sum()):+.1f} dB '
          f'(median per window {np.median(10 * np.log10(o / p)):+.1f} dB)')
    Rp = person_cov(a.test, a.p_from, a.p_to)
    for m in marks:
        if m in snaps:
            print(f'filters at {m:5.1f} s: person kept {gain_db(snaps[m], Rp):+.1f} dB')


if __name__ == '__main__':
    main()
