"""«Уши Братана», стенд на ПК: живы ли 4 микрофона и идут ли два I2S в такт.

    python tools/ushi_bench.py levels          уровни 4 каналов вживую (постучи ногтем по каждому микрофону)
    python tools/ushi_bench.py capture         запись 0,75 с → ushi-<время>.wav и задержки между всеми парами
    python tools/ushi_bench.py sync 10         10 раз: перезапуск I2S и запись — задержка М1↔М3 должна быть одной и той же

Проверка синхронности: сложи все 4 микрофона дырками в одну точку (рядом, на столе), запусти sync и щёлкни пальцами
или постучи по столу во время каждой записи. Все задержки должны быть 0 ± 1 отсчёт; главное — М1↔М3 и М2↔М4 (разные
линии SD, разные I2S) одинаковы от записи к записи.
Нужны pyserial и numpy. Порт находится сам (CP210x / CH340); иначе --port COM5.
"""
import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

BAUD = 921600
NAMES = ('М1 D33 лев', 'М2 D33 прав', 'М3 D32 лев', 'М4 D32 прав')   # слоты, не места в корпусе


def find_port():
    for p in list_ports.comports():
        text = f'{p.description} {p.manufacturer or ""}'.upper()
        if any(k in text for k in ('CP210', 'CH340', 'CH910', 'USB-SERIAL', 'UART')):
            return p.device
    ports = [p.device for p in list_ports.comports()]
    sys.exit('Плата не найдена. Подключи USB; порты: ' + (', '.join(ports) or 'нет'))


def open_port(name):
    # Без DTR/RTS: иначе открытие порта жмёт кнопку EN и плата перезагружается (17.10, аудит; так же в board.py).
    from board import open_board
    port = open_board(name or find_port(), BAUD, timeout=2)
    time.sleep(.3)
    port.reset_input_buffer()
    return port


def levels(port, seconds):
    port.write(b'l')
    end = time.time() + seconds
    while time.time() < end:
        line = port.readline().decode('utf-8', 'replace').strip()
        if line.startswith('L '):
            parts = line.split()[1:]
            out = []
            for i in range(4):
                db, state = float(parts[i * 4 + 1]), parts[i * 4 + 3]
                bar = '#' * max(0, min(30, int((db + 90) / 2)))
                out.append(f'{NAMES[i][:2]} {db:6.1f} дБ {"НЕТ" if state != "ok" else "  "} {bar:<30}')
            print(' | '.join(out), flush=True)
        elif line:
            print(line, flush=True)


def capture(port):
    port.reset_input_buffer()
    port.write(b'c')
    deadline = time.time() + 5
    while time.time() < deadline:
        line = port.readline().decode('utf-8', 'replace').strip()
        if line.startswith('E '):
            sys.exit(line)
        if line.startswith('CAP '):
            frames, channels, rate = map(int, line.split()[1:4])
            raw = port.read(frames * channels * 2)
            if len(raw) != frames * channels * 2:
                sys.exit(f'оборвалась передача: {len(raw)} из {frames * channels * 2} байт')
            return np.frombuffer(raw, '<i2').reshape(frames, channels).astype(np.float64), rate
    sys.exit('плата не ответила на запись (прошивка ushi-bench залита?)')


def gcc_phat(a, b, max_lag):
    """Задержка b относительно a в отсчётах (дробная) и чёткость пика."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    spec = np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n))
    spec /= np.abs(spec) + 1e-12
    cc = np.fft.irfft(spec, n)
    cc = np.concatenate((cc[-max_lag:], cc[:max_lag + 1]))
    k = int(np.argmax(cc))
    shift = 0.
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        shift = .5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return k - max_lag + shift, float(cc[k] / (np.mean(np.abs(cc)) + 1e-12))


def loudest(x, rate, window=.12):
    """Самый громкий кусок (щелчок): на нём задержки чётче, чем на всей записи."""
    energy = np.convolve((x ** 2).sum(axis=1), np.ones(int(rate * .01)), 'same')
    centre = int(np.argmax(energy))
    half = int(rate * window / 2)
    return x[max(0, centre - half):centre + half]


def report(x, rate):
    x = x - x.mean(axis=0)
    rms = 20 * np.log10(np.sqrt((x ** 2).mean(axis=0)) / 32768 + 1e-12)
    print('уровни:', '  '.join(f'{NAMES[i]} {rms[i]:.1f} дБ' for i in range(4)))
    dead = [NAMES[i] for i in range(4) if np.ptp(x[:, i]) < 2]
    if dead:
        print('НЕТ СИГНАЛА:', ', '.join(dead), '— проверь SD, VDD/GND и L/R этого микрофона')
    part = loudest(x, rate)
    max_lag = int(rate * .0015)             # 1,5 мс ≈ 50 см: больше база решётки не бывает
    lags = {}
    for i in range(4):
        for j in range(i + 1, 4):
            lag, sharp = gcc_phat(part[:, i], part[:, j], max_lag)
            lags[(i, j)] = lag
            cross = '  (разные I2S)' if (i < 2) != (j < 2) else ''
            print(f'  {NAMES[i]} → {NAMES[j]}: {lag:+6.2f} отсч. ({lag / rate * 343 * 100:+5.1f} см), чёткость {sharp:4.1f}{cross}')
    return lags


def save_wav(x, rate):
    path = Path(f'ushi-{time.strftime("%Y%m%d-%H%M%S")}.wav')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(4)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.clip(x, -32768, 32767).astype('<i2').tobytes())
    print('записано:', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('levels', 'capture', 'sync'))
    ap.add_argument('count', nargs='?', type=int, default=10)
    ap.add_argument('--port')
    ap.add_argument('--seconds', type=float, default=60.)
    args = ap.parse_args()
    port = open_port(args.port)
    if args.what == 'levels':
        levels(port, args.seconds)
    elif args.what == 'capture':
        x, rate = capture(port)
        save_wav(x, rate)
        report(x, rate)
    else:
        seen = []
        for n in range(args.count):
            port.write(b'r')
            time.sleep(.3)
            print(f'\nзапись {n + 1}/{args.count} — щёлкни пальцами сейчас')
            time.sleep(.3)
            x, rate = capture(port)
            lags = report(x, rate)
            seen.append((lags[(0, 2)], lags[(1, 3)]))
        a = np.array(seen)
        print(f'\nМ1↔М3 по записям: {np.round(a[:, 0], 2).tolist()}')
        print(f'М2↔М4 по записям: {np.round(a[:, 1], 2).tolist()}')
        spread = float(np.ptp(a, axis=0).max())
        print('ИТОГ:', 'два I2S в такт (разброс ≤ 1 отсчёт)' if spread <= 1. else
              f'разброс {spread:.1f} отсчёта между перезапусками — выравнивание не держится')


if __name__ == '__main__':
    main()
