"""Кто где: стучи ногтем по микрофонам по очереди — скрипт пишет, какой канал подпрыгнул и когда.
python tools/tap_watch.py [секунд]"""
import sys
import time
from collections import deque

import serial

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from board import open_board  # noqa: E402
from serial.tools import list_ports

NAMES = ('М1 (D33, левый)', 'М2 (D33, правый)', 'М3 (D32, левый)', 'М4 (D32, правый)')
dev = next((p.device for p in list_ports.comports() if 'CP210' in (p.description or '')), None)
if not dev:
    sys.exit('плата не видна по USB')
seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 90.
p = open_board(dev)
time.sleep(.3)
p.reset_input_buffer()
p.write(b'l')
history = [deque(maxlen=25) for _ in range(4)]
start = time.time()
last_hit = [-9.] * 4
print('слушаю: стучи по микрофонам по очереди', flush=True)
while time.time() - start < seconds:
    line = p.readline().decode('utf-8', 'replace').strip()
    if not line.startswith('L '):
        continue
    parts = line.split()[1:]
    try:
        peaks = [float(parts[i * 4 + 2]) for i in range(4)]
    except (ValueError, IndexError):
        continue
    t = time.time() - start
    for ch in range(4):
        h = history[ch]
        base = sorted(h)[len(h) // 2] if len(h) >= 5 else None
        if base is not None and peaks[ch] > -200 and peaks[ch] - base > 12 and t - last_hit[ch] > 1.:
            last_hit[ch] = t
            print(f'{t:5.1f} с: {NAMES[ch]} подпрыгнул на {peaks[ch] - base:4.1f} дБ (пик {peaks[ch]:.1f})', flush=True)
        h.append(peaks[ch])
print('конец', flush=True)
