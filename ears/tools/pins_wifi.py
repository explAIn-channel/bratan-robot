"""Самопроверка ножек по Wi-Fi (UDP «pins» на 5771): переключения такта и линий данных, сбои I2S.
python tools/pins_wifi.py [--ip 192.168.0.N]"""
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ip = sys.argv[sys.argv.index('--ip') + 1] if '--ip' in sys.argv else (ROOT / '.ushi-ip').read_text().strip()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(2.)
for _ in range(3):
    s.sendto(b'pins', (ip, 5771))
    try:
        print(s.recvfrom(512)[0].decode('utf-8', 'replace'))
    except socket.timeout:
        print('нет ответа')
