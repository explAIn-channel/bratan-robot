"""Повторяемость второго I2S: N раз перезапустить оба I2S (команда r) и записать 0,75 с — в каком слоте звук и чистый ли.
python tools/restart_check.py [N]"""
import sys
import time

import numpy as np
import serial

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from board import open_board  # noqa: E402
from serial.tools import list_ports

dev = next((p.device for p in list_ports.comports() if 'CP210' in (p.description or '')), None)
if not dev:
    sys.exit('плата не видна по USB')
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
p = open_board(dev)
time.sleep(.3)
p.write(b's')
time.sleep(.3)


def capture():
    p.reset_input_buffer()
    p.write(b'c')
    end = time.time() + 6
    while time.time() < end:
        line = p.readline().decode('utf-8', 'replace').strip()
        if line.startswith('CAP '):
            frames, channels, _ = map(int, line.split()[1:4])
            p.timeout = 4
            raw = p.read(frames * channels * 2)
            p.timeout = .3
            return np.frombuffer(raw, '<i2').reshape(frames, channels).astype(float)
    return None


for k in range(n):
    if k:
        p.write(b'r')
        time.sleep(.6)
    x = capture()
    if x is None:
        print(k, 'нет записи')
        continue
    cells = []
    for ch in range(4):
        col = x[:, ch]
        u = len(np.unique(col))
        zero = float((col == 0).mean())
        cells.append(f'М{ch + 1}: {u:5d} знач., нулей {zero * 100:3.0f}%')
    print(f'{"старт" if k == 0 else "перезапуск " + str(k)}: ' + ' | '.join(cells), flush=True)
p.write(b'l')
