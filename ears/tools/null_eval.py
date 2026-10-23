"""How the robot-voice null holds up in real use: the robot's pattern learned on ANOTHER recording (a stored
calibration), the person's from their call or not known at all (min-energy filter). Writes one WAV per variant for
Microsoft and prints the robot/person attenuation in the speech band.

  python tools/null_eval.py TEST_STEM PERSON_FROM PERSON_TO TRAIN_STEM [TRAIN_STEM ...]
"""
import sys
import wave

import numpy as np
from scipy.signal import istft, stft

RATE, N, HOP = 16000, 512, 256


def load_wav(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)


def spectra(stem):
    """STFT of the three microphones (ch × F × T), the frame times on the A/B line, robot-speaking frames."""
    d = np.load(stem + '-multi.npz')
    x, t0 = d['x'].astype(np.float64), float(d['t0'])
    played = load_wav(stem + '-played.wav')
    _, times, Z = stft(x.T, RATE, nperseg=N, noverlap=N - HOP, boundary=None, padded=False)
    t = t0 + times
    idx = np.clip(((t - .016) * RATE).astype(int), 0, len(played) - 512)
    robot = np.array([10 * np.log10((played[i:i + 512] ** 2).mean() + 1) > 40 for i in idx])
    return Z, t, robot, x, t0, len(played)


def cov(Z, mask):
    R = np.einsum('cft,dft->fcd', Z[:, :, mask], Z[:, :, mask].conj()) / max(1, mask.sum())
    load = np.einsum('fcc->f', R).real / 3 * 1e-3 + 1e-6
    return R + load[:, None, None] * np.eye(3)[None]


def filters(Rr, Rp=None):
    W = np.zeros((Rr.shape[0], 3), complex)
    for f in range(Rr.shape[0]):
        if Rp is not None:
            vals, vecs = np.linalg.eigh(Rp[f])
            steer = vecs[:, -1] / (vecs[0, -1] if abs(vecs[0, -1]) > 1e-9 else 1)
            inv = np.linalg.solve(Rr[f], steer)
            W[f] = inv / (steer.conj() @ inv)
        else:                                             # nothing known of the person: least robot, unit norm,
            vals, vecs = np.linalg.eigh(Rr[f])            # scaled to keep the room's level on microphone 1
            w = vecs[:, 0]
            W[f] = w / (abs(w[0]) + 1e-9) * np.exp(-1j * np.angle(w[0]))
    return W


def apply(W, Z):
    return np.einsum('fc,cft->ft', W.conj(), Z)


def gain(W, R):
    return np.real(np.einsum('fc,fcd,fd->f', W.conj(), R, W)) / np.maximum(R[:, 0, 0].real, 1e-9)


def main(test, p_from, p_to, trains):
    Z, t, robot, x, t0, n = spectra(test)
    energy = (np.abs(Z) ** 2).sum(axis=(0, 1))
    person = (t >= p_from) & (t <= p_to) & ~robot
    person &= energy > np.percentile(energy[person], 40)
    Rr_test, Rp = cov(Z, robot), cov(Z, person)
    Rr_train = sum(cov(*spectra(s)[0:3:2]) for s in trains) / len(trains)
    band = slice(int(300 / RATE * N), int(3400 / RATE * N))
    variants = {'in-sample robot + person': filters(Rr_test, Rp), 'stored robot + person': filters(Rr_train, Rp),
                'in-sample robot, no person': filters(Rr_test), 'stored robot, no person': filters(Rr_train)}
    for name, W in variants.items():
        rob, per = gain(W, Rr_test)[band].mean(), gain(W, Rp)[band].mean()
        print(f'{name:28s} robot {10 * np.log10(rob):+6.1f} dB, person {10 * np.log10(per):+6.1f} dB, '
              f'gain {10 * np.log10(per / rob):+5.1f} dB')
        _, y = istft(apply(W, Z), RATE, nperseg=N, noverlap=N - HOP, boundary=False)
        lead = int(round(-t0 * RATE))
        track = np.zeros(n)
        track[max(0, lead):max(0, lead) + len(y)] = y[:n - max(0, lead)]
        with wave.open(f'{test}-null-{name.split()[0]}-{"p" if "+ person" in name else "n"}.wav', 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(np.clip(track, -32768, 32767).astype(np.int16).tobytes())


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:])
