"""Приём звука «ушей» по Wi-Fi: подписаться (UDP «sub» на 5771 раз в секунду) и принимать пакеты BRE1.
Показывает потери, уровни каналов и пишет ushi-wifi-<время>.wav (4 канала) и -mono.wav (сумма живых каналов).
python tools/stream_rx.py [секунд] [--ip 192.168.0.N]"""
import socket
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
seconds = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace('.', '').isdigit() else 10.
ip = sys.argv[sys.argv.index('--ip') + 1] if '--ip' in sys.argv else (ROOT / '.ushi-ip').read_text().strip() \
    if (ROOT / '.ushi-ip').exists() else None
if not ip:
    sys.exit('адрес платы неизвестен: --ip 192.168.0.N')
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('', 0))
s.settimeout(.2)
chunks, seqs, firsts = [], [], []
start = last_sub = 0.
start = time.time()
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
    seq, first, n, ch = struct.unpack('<IIHH', data[4:16])
    seqs.append(seq)
    firsts.append(first)
    chunks.append(np.frombuffer(data[16:16 + n * ch * 2], '<i2').reshape(n, ch))
s.sendto(b'unsub', (ip, 5771))
if not chunks:
    sys.exit('звук не пришёл: плата не в сети, не тот адрес или брандмауэр Windows режет UDP')
seqs = np.array(seqs)
lost = int(seqs[-1] - seqs[0] + 1 - len(seqs))
x = np.concatenate(chunks).astype(np.float64)
dur = len(x) / 16000
print(f'пакетов {len(seqs)}, потеряно {lost} ({lost / (seqs[-1] - seqs[0] + 1) * 100:.1f} %), '
      f'звука {dur:.1f} с за {seconds:.0f} с')
names = ('М1', 'М2', 'М3', 'М4')
alive = []
for c in range(x.shape[1]):
    col = x[:, c] - x[:, c].mean()
    level = 20 * np.log10(np.sqrt((col ** 2).mean()) / 32768 + 1e-12)
    # Живой — звук почти всё время (17.10: «копия» соседа с болтающимся L/R — это половина нулей, в сумму не берём).
    ok = np.ptp(x[:, c]) > 0 and (x[:, c] != 0).mean() > .9
    alive.append(ok)
    print(f'  {names[c]}: {"живой" if ok else "молчит"}, уровень {level:6.1f} дБ')
stamp = time.strftime('%Y%m%d-%H%M%S')
OUT = ROOT / 'recordings'
OUT.mkdir(exist_ok=True)
with wave.open(str(OUT / f'ushi-wifi-{stamp}.wav'), 'wb') as w:
    w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(np.clip(x, -32768, 32767).astype('<i2').tobytes())
live = x[:, [c for c in range(x.shape[1]) if alive[c]]]
if live.size:
    mono = live - live.mean(axis=0)
    mono = mono.sum(axis=1) / max(1, live.shape[1])
    with wave.open(str(OUT / f'ushi-wifi-{stamp}-mono.wav'), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(np.clip(mono, -32768, 32767).astype('<i2').tobytes())
print('записано:', f'recordings/ushi-wifi-{stamp}.wav', 'и -mono.wav')
