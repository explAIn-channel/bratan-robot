"""Шевелим провода по очереди, по Wi-Fi: отмечает, когда каждый канал оживает и замолкает (окна по 0,1 с).
python tools/wiggle_wifi.py [секунд] [--ip 192.168.0.N]"""
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
seconds = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace('.', '').isdigit() else 40.
ip = sys.argv[sys.argv.index('--ip') + 1] if '--ip' in sys.argv else (ROOT / '.ushi-ip').read_text().strip()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('', 0))
s.settimeout(.2)
start = time.time()
last_sub = 0.
buf = []
state = [None] * 4
alive_time = [0.] * 4
print(f'слежу {seconds:.0f} с: шевели провода по очереди', flush=True)
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
    buf.append(np.frombuffer(data[16:16 + n * ch * 2], '<i2').reshape(n, ch))
    if sum(len(b) for b in buf) < 1600:
        continue
    x = np.concatenate(buf)
    buf = []
    t = time.time() - start
    for c in range(4):
        alive = len(np.unique(x[:, c])) > 20            # живой микрофон даёт сотни разных значений; ноль/обрывки — нет
        if alive:
            alive_time[c] += .1
        if alive != state[c]:
            if state[c] is not None or alive:
                print(f'{t:5.1f} с: М{c + 1} {"ОЖИЛ" if alive else "замолчал"}', flush=True)
            state[c] = alive
s.sendto(b'unsub', (ip, 5771))
print('итого живы, с: ' + ', '.join(f'М{i + 1} {alive_time[i]:.0f}' for i in range(4)), flush=True)
