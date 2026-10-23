"""Сторож «ушей»: опрашивает плату раз в секунду и завершается с описанием, как только что-то не так —
микрофон (М1, М2, М3) дольше 5 с не «ok», такт сбоит, выросли переполнения или сбои такта, свободной памяти
меньше 60 КБ, плата молчит дольше 3 мин. Короткие провалы Wi-Fi (17.10 21:20 и 21:54: эфир забит 1–2 мин,
подключение не рвалось — помеха) только пишутся в watch.log. Запускается в фоне; его завершение — сигнал
«посмотри» (17.10, владелец: «просто периодически чекай»).
    python tools/alarm.py [--ip 192.168.0.N] [--max-hours 12]"""
import json
import re
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ip = sys.argv[sys.argv.index('--ip') + 1] if '--ip' in sys.argv else (ROOT / '.ushi-ip').read_text().strip()
max_hours = float(sys.argv[sys.argv.index('--max-hours') + 1]) if '--max-hours' in sys.argv else 12.
SLOTS = {0: 'М1 спереди', 2: 'М2 справа', 3: 'М3 слева'}
PINS = re.compile(r'WS25 (\d+) (\d+)%')

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('', 0))
s.settimeout(.8)
bad_since, last_answer, base, started = {}, time.time(), None, time.time()
pins_at = 0.


def stop(text):
    print(time.strftime('%H:%M:%S'), text, flush=True)
    note(text)
    sys.exit(1)


def note(text):
    with open(ROOT / 'watch.log', 'a', encoding='utf-8') as handle:
        handle.write(f'{time.strftime("%H:%M:%S")} сторож: {text}\n')


silent_noted = False
heap_at = 0.
while time.time() - started < max_hours * 3600:
    time.sleep(1.)
    try:
        s.sendto(b'health', (ip, 5771))
        data = s.recvfrom(4096)[0]
        h = json.loads(data[2:]) if data[:2] == b'H ' else None
    except (socket.timeout, ValueError, OSError):
        h = None
    now = time.time()
    if h is None:
        if now - last_answer > 15 and not silent_noted:
            silent_noted = True
            note('плата не отвечает больше 15 с (провал Wi-Fi?)')
        if now - last_answer > 180:
            stop(f'плата не отвечает {now - last_answer:.0f} с — дольше обычного провала Wi-Fi (питание? плата?)')
        continue
    if silent_noted:
        silent_noted = False
        note(f'плата снова отвечает (молчала {now - last_answer:.0f} с)')
    last_answer = now
    if now - heap_at > 30:
        heap_at = now
        try:
            s.sendto(b'ping', (ip, 5771))
            status = json.loads(s.recvfrom(1024)[0])
            if status.get('heap', 10 ** 6) < 60000:
                stop(f'у платы мало свободной памяти: {status["heap"]} байт (утечка?)')
            if status.get('reconnects') or status.get('drops'):
                note(f'Wi-Fi платы: обрывов {status.get("drops")}, переподключений {status.get("reconnects")}')
        except (socket.timeout, ValueError, OSError):
            pass
    counters = {k: h.get(k, 0) for k in ('ovf', 'fail')}
    if base is None:
        base = counters
    grown = {k: counters[k] - base[k] for k in counters if counters[k] > base[k]}
    if grown:
        stop(f'приём перезапускался: {grown} (переполнение — что-то держало приём; сбой такта — провод SCK/WS)')
    for slot, name in SLOTS.items():
        v = h['v'][slot]
        if v in ('ok', 'wait'):
            bad_since.pop(slot, None)
        else:
            bad_since.setdefault(slot, now)
            if now - bad_since[slot] >= 5:
                stop(f'{name}: «{v}» уже {now - bad_since[slot]:.0f} с (пусто {h["zero"][slot]:.0f} %, '
                     f'наводка {h["tail"][slot]:.0f} %, копия {h["copy"][slot]:.0f} %)')
    if now - pins_at > 30:
        pins_at = now
        try:
            s.sendto(b'pins', (ip, 5771))
            line = s.recvfrom(512)[0].decode('utf-8', 'replace')
            m = PINS.search(line)
            if m and (abs(int(m.group(1)) - 16000) > 480 or not 35 <= int(m.group(2)) <= 65):
                stop(f'такт сбоит: {line}')
        except socket.timeout:
            pass
print(time.strftime('%H:%M:%S'), f'за {max_hours:.0f} ч всё было в порядке', flush=True)
