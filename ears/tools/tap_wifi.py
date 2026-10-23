"""Кто где, по Wi-Fi: щёлкай возле микрофонов по очереди — скрипт пишет, какой канал подпрыгнул и когда.
python tools/tap_wifi.py [секунд] [--ip 192.168.0.N]"""
import socket
import struct
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
seconds = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace('.', '').isdigit() else 40.
ip = sys.argv[sys.argv.index('--ip') + 1] if '--ip' in sys.argv else (ROOT / '.ushi-ip').read_text().strip()
NAMES = ('М1 (D33, левый слот)', 'М2 (D33, правый слот)', 'М3 (D32, левый слот)', 'М4 (D32, правый слот)')
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('', 0))
s.settimeout(.2)
start = time.time()
last_sub = 0.
buf = []
history = [deque(maxlen=30) for _ in range(4)]
last_hit = [-9.] * 4
print('слушаю 40 с: щёлкай по очереди', flush=True)
while time.time() - start < seconds:
    if time.time() - last_sub > 1.:
        s.sendto(b'sub', (ip, 5771))
        last_sub = time.time()
    try:
        data, _ = s.recvfrom(4096)
    except socket.timeout:
        continue
    if data[:4] != b'BRE1':
        continue
    _, _, n, ch = struct.unpack('<IIHH', data[4:16])
    buf.append(np.frombuffer(data[16:16 + n * ch * 2], '<i2').reshape(n, ch).astype(float))
    if sum(len(b) for b in buf) < 1600:          # окна по 0,1 с
        continue
    x = np.concatenate(buf)
    buf = []
    t = time.time() - start
    peaks = 20 * np.log10(np.abs(x - x.mean(axis=0)).max(axis=0) / 32768 + 1e-9)
    for c in range(4):
        h = history[c]
        base = sorted(h)[len(h) // 2] if len(h) >= 8 else None
        if base is not None and peaks[c] > -150 and peaks[c] - base > 12 and t - last_hit[c] > .8:
            last_hit[c] = t
            print(f'{t:5.1f} с: {NAMES[c]} подпрыгнул на {peaks[c] - base:4.1f} дБ', flush=True)
        h.append(peaks[c])
s.sendto(b'unsub', (ip, 5771))
print('конец', flush=True)
