"""Быстрая проверка проводки: уровни 4 каналов за N секунд и запись 0,75 с — кто жив, кто молчит, кто «шумит
мусором» (оборванная линия данных). python tools/check_mics.py [секунд]"""
import sys
import time

import numpy as np
import serial

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from board import open_board  # noqa: E402
from serial.tools import list_ports

NAMES = ('М1', 'М2', 'М3', 'М4')
ports = [(p.device, p.description or '') for p in list_ports.comports()]
dev = next((d for d, desc in ports if 'CP210' in desc or 'CH340' in desc), None)
if not dev:
    sys.exit(f'плата не видна по USB; порты: {ports}')
seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 6.
p = open_board(dev)
time.sleep(.3)
p.reset_input_buffer()
p.write(b'l')
rows, messages = [], []
end = time.time() + seconds
while time.time() < end:
    line = p.readline().decode('utf-8', 'replace').strip()
    if line.startswith('L '):
        parts = line.split()[1:]
        try:
            rows.append([(float(parts[i * 4 + 1]), float(parts[i * 4 + 2]), parts[i * 4 + 3]) for i in range(4)])
        except (ValueError, IndexError):
            pass
    elif line.startswith(('I ', 'E ')):
        messages.append(line)
print('сообщения платы:', messages[-4:] or 'нет')
print(f'строк уровней: {len(rows)} за {seconds:.0f} с')
for ch in range(4):
    if not rows:
        break
    rms = [r[ch][0] for r in rows]
    peak = [r[ch][1] for r in rows]
    dead = sum(r[ch][2] != 'ok' for r in rows)
    print(f'{NAMES[ch]}: тишина {np.median(rms):7.1f} дБ (от {min(rms):.1f} до {max(rms):.1f}), пик {max(peak):7.1f} дБ, '
          f'не «ok» в {dead}/{len(rows)} строках')

# Запись: смещение (DC), шум и похожесть каналов (одинаковые данные = закоротило или один микрофон на двух слотах).
p.write(b's')
time.sleep(.3)
p.reset_input_buffer()
p.write(b'c')
deadline = time.time() + 6
x = None
while time.time() < deadline:
    line = p.readline().decode('utf-8', 'replace').strip()
    if line.startswith('CAP '):
        frames, channels, rate = map(int, line.split()[1:4])
        p.timeout = 4                                  # 96 КБ по 921600 бод — около секунды
        raw = p.read(frames * channels * 2)
        x = np.frombuffer(raw, '<i2').reshape(frames, channels).astype(float)
        break
p.write(b'l')
if x is None:
    sys.exit('запись не пришла')
dc = x.mean(axis=0)
ac = x - dc
std = ac.std(axis=0)
print('\nзапись 0,75 с:')
for ch in range(4):
    col = x[:, ch]
    print(f'{NAMES[ch]}: смещение {dc[ch]:8.1f}, шум {std[ch]:7.1f}, мин/макс {col.min():6.0f}/{col.max():6.0f}, '
          f'разных значений {len(np.unique(col))}')
for i in range(4):
    for j in range(i + 1, 4):
        if std[i] > 0 and std[j] > 0:
            r = float(np.corrcoef(ac[:, i], ac[:, j])[0, 1])
            print(f'  похожесть {NAMES[i]}–{NAMES[j]}: {r:+.2f}')
np.save('check_last.npy', x)
