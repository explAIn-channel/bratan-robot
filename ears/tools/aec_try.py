"""Перебивание (C52): убрать из ушей собственный голос робота по записи A/B (…-played.wav — что робот отправил в
динамик и когда, …-array.wav — уши). Офлайн-опыт: задержка и её дрейф, затем адаптивный фильтр по частотам с
подавлением остатка; пишет …-aec.wav и печатает, на сколько дБ ушло эхо.

  python tools/aec_try.py recordings/ab/barge1
"""
import sys
import wave

import numpy as np

RATE = 16000


def load(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)


def lag(ref, mic, max_lag):
    """Задержка mic относительно ref (GCC-PHAT), отсчётов, и острота пика."""
    n = 1
    while n < len(ref) + max_lag:
        n *= 2
    R, M = np.fft.rfft(ref, n), np.fft.rfft(mic, n)
    c = M * np.conj(R)
    c /= np.abs(c) + 1e-9
    cc = np.fft.irfft(c, n)[:max_lag]
    k = int(np.argmax(cc))
    return k, cc[k] / (np.abs(cc).mean() + 1e-12)


class Echo:
    """Блочный адаптивный фильтр по частотам (PBFDAF, overlap-save) + подавление остатка по оценке эха."""

    def __init__(self, block=256, parts=12, mu=.4, floor_db=-30.):
        self.B, self.P, self.mu = block, parts, mu
        self.N = 2 * block
        self.F = block + 1
        self.W = np.zeros((parts, self.F), complex)
        self.X = np.zeros((parts, self.F), complex)
        self.xbuf = np.zeros(self.N)
        self.power = np.full(self.F, 1e6)
        self.floor = 10 ** (floor_db / 20)
        self.Syy = np.full(self.F, 1e3)
        self.See = np.full(self.F, 1e3)
        self.Sxe = np.zeros(self.F, complex)

    def process(self, x, d):
        """x — опора (что сказали), d — уши; оба по `block` отсчётов. Возвращает очищенный блок."""
        B = self.B
        self.xbuf = np.concatenate([self.xbuf[B:], x])
        Xf = np.fft.rfft(self.xbuf)
        self.X = np.roll(self.X, 1, axis=0)
        self.X[0] = Xf
        Y = (self.W * self.X).sum(axis=0)
        y = np.fft.irfft(Y)[B:]
        e = d - y
        Ef = np.fft.rfft(np.concatenate([np.zeros(B), e]))
        self.power = .9 * self.power + .1 * (np.abs(Xf) ** 2)
        # шаг меньше, когда в ушах явно больше, чем объясняет эхо (говорит человек — не сбивать фильтр)
        Df = np.fft.rfft(np.concatenate([np.zeros(B), d]))
        talk = np.sum(np.abs(Ef) ** 2) / (np.sum(np.abs(Df) ** 2) + 1e-9)
        mu = self.mu * (1. if talk < 1.5 else .1)
        G = mu * np.conj(self.X) * Ef / (self.P * self.power + 1e3)
        for p in range(self.P):
            g = np.fft.irfft(G[p])
            g[B:] = 0
            self.W[p] += np.fft.rfft(g)
        # остаток: сколько ещё похоже на эхо — по когерентности ошибки с оценкой эха
        Yf = np.fft.rfft(np.concatenate([np.zeros(B), y]))
        self.Syy = .8 * self.Syy + .2 * np.abs(Yf) ** 2
        self.See = .8 * self.See + .2 * np.abs(Ef) ** 2
        self.Sxe = .8 * self.Sxe + .2 * Yf * np.conj(Ef)
        coh = np.abs(self.Sxe) ** 2 / (self.Syy * self.See + 1e-9)
        gain = np.clip(1 - coh, self.floor, 1.)
        out = np.fft.irfft(Ef * gain)[B:]
        return e, out


def main(stem):
    p, a = load(stem + '-played.wav'), load(stem + '-array.wav')
    n = min(len(p), len(a))
    p, a = p[:n], a[:n]
    print(f'{stem}: {n / RATE:.1f} с')
    win = 4 * RATE
    lags = []
    for s in range(0, n - win, 2 * RATE):
        if np.sqrt((p[s:s + win] ** 2).mean()) < 300:
            continue
        k, sharp = lag(p[s:s + win], a[s:s + win], RATE // 2)
        lags.append((s / RATE, k, sharp))
    print('задержка по окнам 4 с (с, мс, острота):', ' '.join(f'{t:.0f}:{k / 16:.0f}мс/{q:.0f}' for t, k, q in lags))
    good = [k for _, k, q in lags if q > 8]
    delay = int(np.median(good)) if good else 0
    print('медиана задержки', delay / 16, 'мс')
    shift = max(0, delay - 512)                 # фильтр сам ловит ±32 мс вокруг и хвост комнаты
    ref = np.concatenate([np.zeros(shift), p])[:n]
    echo = Echo()
    B = echo.B
    lin, out = np.zeros(n), np.zeros(n)
    for s in range(0, n - B, B):
        e, o = echo.process(ref[s:s + B], a[s:s + B])
        lin[s:s + B], out[s:s + B] = e, o
    speaking = np.array([np.sqrt((p[max(0, s - delay):s - delay + B] ** 2).mean()) > 300 if s >= delay else False
                         for s in range(0, n - B, B)])
    def db(x):
        return 10 * np.log10(np.mean(x ** 2) + 1)
    blocks = lambda x: x[:len(speaking) * B].reshape(-1, B)
    for name, x in (('уши', a), ('после фильтра', lin), ('после подавления', out)):
        xb = blocks(x)
        print(f'{name:18s} во время речи робота {db(xb[speaking]):5.1f} дБ, в тишине {db(xb[~speaking]):5.1f} дБ')
    with wave.open(stem + '-aec.wav', 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.clip(out, -32768, 32767).astype(np.int16).tobytes())


if __name__ == '__main__':
    main(sys.argv[1])
