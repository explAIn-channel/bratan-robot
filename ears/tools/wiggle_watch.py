"""Шевелим провода по очереди — скрипт отмечает, когда каждый канал оживает и замолкает.
python tools/wiggle_watch.py [секунд]"""
import sys
import time

import serial

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from board import open_board  # noqa: E402
from serial.tools import list_ports

dev = next((p.device for p in list_ports.comports() if 'CP210' in (p.description or '')), None)
if not dev:
    sys.exit('плата не видна по USB')
seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 120.
p = open_board(dev)
time.sleep(.3)
p.reset_input_buffer()
p.write(b'l')
state = [None] * 4
start = time.time()
alive_time = [0.] * 4
last = start
print('слежу: шевели провода по очереди', flush=True)
while time.time() - start < seconds:
    line = p.readline().decode('utf-8', 'replace').strip()
    if not line.startswith('L '):
        continue
    parts = line.split()[1:]
    try:
        alive = [parts[i * 4 + 3] == 'ok' for i in range(4)]
    except IndexError:
        continue
    now = time.time()
    t = now - start
    for ch in range(4):
        if alive[ch]:
            alive_time[ch] += now - last
        if alive[ch] != state[ch]:
            if state[ch] is not None or alive[ch]:
                print(f'{t:5.1f} с: М{ch + 1} {"ОЖИЛ" if alive[ch] else "замолчал"}', flush=True)
            state[ch] = alive[ch]
    last = now
print('итого живы, с: ' + ', '.join(f'М{i + 1} {alive_time[i]:.0f}' for i in range(4)), flush=True)
