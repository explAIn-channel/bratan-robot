"""Состояние платы по USB (команда i): версия, Wi-Fi, адрес, сигнал. Ждёт до N секунд, пока плата выйдет в сеть.
python tools/status_usb.py [секунд]"""
import json
import sys
import time

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from board import open_board  # noqa: E402

wait = float(sys.argv[1]) if len(sys.argv) > 1 else 20.
p = open_board()
time.sleep(.3)
p.write(b's')
end = time.time() + wait
last = None
while time.time() < end:
    p.reset_input_buffer()
    p.write(b'i')
    t = time.time() + 1.5
    while time.time() < t:
        line = p.readline().decode('utf-8', 'replace').strip()
        if line.startswith('I {'):
            last = json.loads(line[2:])
            break
    if last and last.get('online'):
        break
    time.sleep(1.)
p.write(b'l')
print(json.dumps(last, ensure_ascii=False) if last else 'плата не ответила')
