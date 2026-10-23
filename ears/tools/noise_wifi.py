"""Шум микрофонов в тишине по замерам самой платы (health: rms, frms по слотам) — пока потоки идут и пока не идут.
Вопрос (18.10): шум ушей −45 дБ (на 30+ дБ выше собственного шума INMP441) — наводка от всплесков Wi-Fi по питанию?
Сначала замер «как есть», потом попросить остановить всех подписчиков и замерить снова.

  python tools/noise_wifi.py 192.0.2.85 [секунд]
"""
import json
import socket
import sys
import time

HOST = sys.argv[1] if len(sys.argv) > 1 else '192.0.2.85'
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 10.
SLOTS = (0, 2, 3)


def health(sock):
    sock.sendto(b'health', (HOST, 5771))
    deadline = time.time() + 1.
    while time.time() < deadline:
        try:
            data, _ = sock.recvfrom(8192)
        except socket.timeout:
            break
        if data[:2] == b'H ':
            return json.loads(data[2:].decode('utf-8', 'replace'))
    return None


def measure(label):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(.5)
    rows = []
    end = time.time() + SECONDS
    while time.time() < end:
        h = health(sock)
        if h:
            rows.append(h)
        time.sleep(.5)
    sock.close()
    if not rows:
        print(label, 'плата не ответила')
        return
    print(f'{label}: замеров {len(rows)}')
    for slot in SLOTS:
        vals = {k: sorted(r[k][slot] for r in rows) for k in ('rms', 'frms', 'r1')}
        print('  слот', slot, '  '.join(f'{k} медиана {v[len(v) // 2]:.2f} (мин {v[0]:.2f})' for k, v in vals.items()))


if __name__ == '__main__':
    measure(sys.argv[3] if len(sys.argv) > 3 else 'замер')
