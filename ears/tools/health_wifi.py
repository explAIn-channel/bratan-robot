"""The board's own word on its microphones over Wi-Fi: «health» (what is in each of the 4 slots) and «ping» (firmware,
uptime, signal, Bluetooth).  python tools/health_wifi.py [--ip 192.168.0.N]"""
import json
import socket
import sys

ip = sys.argv[sys.argv.index('--ip') + 1] if '--ip' in sys.argv else '192.0.2.85'
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.settimeout(1.5)
    for request in (b'ping', b'health'):
        for _ in range(4):
            try:
                s.sendto(request, (ip, 5771))
                data, _ = s.recvfrom(4096)
                print(request.decode(), json.dumps(json.loads(data), ensure_ascii=False))
                break
            except (OSError, ValueError):
                continue
        else:
            print(request.decode(), 'no answer')
