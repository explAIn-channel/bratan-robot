"""Что плата пишет сама по себе: 5 с сырого вывода порта (перезагрузки, ошибки, просадки питания)."""
import sys
import time

import serial

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from board import open_board  # noqa: E402
from serial.tools import list_ports

ports = [(p.device, p.description or '') for p in list_ports.comports()]
print('порты:', ports)
dev = next((d for d, desc in ports if 'CP210' in desc), None)
if not dev:
    sys.exit('плата не видна по USB')
p = open_board(dev, 115200 if '--boot' in sys.argv else 921600)
end = time.time() + 5
data = b''
while time.time() < end:
    data += p.read(4096)
text = data.decode('utf-8', 'replace')
lines = [l for l in text.splitlines() if l.strip()]
print(f'байт: {len(data)}, строк: {len(lines)}')
interesting = [l for l in lines if not l.startswith('L ')]
for l in interesting[-15:]:
    print('  ', l[:160])
print('строк уровней:', sum(l.startswith('L ') for l in lines))
