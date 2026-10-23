"""Живая панель «ушей»: что в каждом из 4 слотов, вспышка на стук, такты платы, связь.
Вывод о слоте делает сама плата (прошивка ushi-7+, запрос health): она видит все 32 бита слова — пустую линию,
наводку, копию соседа, битые данные. Панель только показывает.
Подписи и подсказки — в столбиках мини-макетки, как владелец втыкает провода (17.10: «я по названиям путаюсь»):
2/3 — 3V3, 5/6 — GND, 8 — SCK, 10 — WS, 12 — SD линии D33 (М1, М2), 14 — SD линии D32 (М3, М4).

Ниже — компас (17.10, C51): откуда робот слышит звук, тем же модулем, что и робот (bratan/voice/array.py), и
калибровка: встать с одной из 8 сторон, нажать её кнопку, сказать «Братан» — отпечаток записан; «На робота» —
таблица уходит на телефон (голос робота перечитывает её сам).

    python tools/live.py                 по Wi-Fi (UDP 5771), открыть http://localhost:8777
    python tools/live.py --usb           по кабелю (порт занят панелью, пока она работает)
    python tools/live.py --once [--usb]  таблица в консоль и выход
    python tools/live.py --lan           панель видна и с телефона в домашней сети (http://<адрес ПК>:8777)
    [--ip 192.168.0.N] [--port 8777] [--no-ears]
    [--layout М1:0,М2:2,М3:3]   имена владельца → слоты платы (0 D33 левый, 1 D33 правый, 2 D32 левый,
                                3 D32 правый); слоты вне списка не показываются, но если в них что-то появится — предупреждение
"""
import json
import re
import socket
import statistics
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
PINS = re.compile(r'SCK26 (\d+) (\d+)% \| WS25 (\d+) (\d+)% \| SD33 (\d+) (\d+)% \| SD32 (\d+) (\d+)% '
                  r'\| i2s_fail (\d+) \| ovf (\d+)')


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


# Слоты платы: 0 — D33 левый, 1 — D33 правый, 2 — D32 левый, 3 — D32 правый. Провода: столбик SD, столбик L/R.
WIRES = (('12', '5 или 6', 'GND'), ('12', '2 или 3', '3V3'), ('14', '5 или 6', 'GND'), ('14', '2 или 3', '3V3'))
CHECK = 'VDD → 2/3, GND → 5/6, SCK → 8, WS → 10'
# Имена владельца (17.10: «м1 — м1, м3 — м2, м4 — м3, м2 убери: у нас всего 3 микрофона»): имя → слот платы.
LAYOUT = [(name, int(slot)) for name, slot in (p.split(':') for p in arg('--layout', 'М1:0,М2:2,М3:3').split(','))]
NAMES = {slot: name for name, slot in LAYOUT}
# Где стоит в роботе (17.10, щелчки по корпусу: справа громче всех М2, слева — М3, спереди — М1, на 10–30 дБ).
PLACE = {0: 'спереди', 2: 'справа', 3: 'слева'}


def named(slot):
    return NAMES.get(slot, 'лишний слот')


CALIBRATION = ROOT / 'ears-calibration.json'
PHONE_CALIBRATION = '/opt/robotbrain/bratan/state/ears-calibration.json'
SSH_CONFIG = ROOT.parents[1] / 'access' / 'ssh_config'
STAND = (('спереди', 0), ('слева-спереди', 45), ('слева', 90), ('слева-сзади', 135), ('сзади', 180),
         ('справа-сзади', -135), ('справа', -90), ('справа-спереди', -45))


class Ears:
    """Компас и калибровка: подписка на три микрофона (без голоса), направление раз в 0,3 с тем же модулем, что у
    робота. Калибровка: нажата сторона → ждём звук до 10 с, после него ещё 1 с (слово целиком) → отпечаток."""

    def __init__(self, ip):
        sys.path.insert(0, str(ROOT.parent / 'bratan'))
        from bratan.voice import array
        self.array = array
        self.link = array.ArrayLink(ip, voice=None).start()
        self.direction = array.Direction(self.link.ring, array.Calibration(CALIBRATION))
        self.lock = threading.Lock()
        self.now = None                     # направление сейчас (последняя секунда)
        self.last = None                    # последний уверенный звук (держится 6 с)
        self.armed = None                   # {'deg', 'name', 'since', 'heard'} — ждём звук для калибровки
        self.message = ''
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            time.sleep(.3)
            try:
                found = self.direction.estimate(1.)
            except Exception as error:      # noqa: BLE001 — панель не должна падать от одного кадра
                found = None
                self.message = f'ошибка расчёта: {type(error).__name__}: {error}'[:160]
            with self.lock:
                # Неуверенная оценка не рисуется как «слышу сейчас» (18.10: на хвосте звука стрелка прыгала).
                self.now = found if found and found['confidence'] >= .3 else None
                if self.now:
                    self.last = {**found, 't': time.time()}
            self.calibrate_tick()

    def start_round(self):
        """По кругу: все 8 сторон по очереди; после каждой записи 3 с — перейти на следующую сторону."""
        self.round = [d for _, d in STAND]
        self.round_next_at = time.time()
        self.message = 'по кругу: начинаем спереди'

    def round_tick(self):
        rnd = getattr(self, 'round', None)
        if not rnd or self.armed or time.time() < self.round_next_at:
            return
        self.arm(rnd.pop(0), patience=40.)
        if not rnd:
            self.round = None

    def calibrate_tick(self):
        self.round_tick()
        armed = self.armed
        if not armed:
            return
        now = time.time()
        feat = self.direction.features(1.)
        if armed.get('heard') is None:
            if feat is not None and feat['onsets'] >= 3:                      # слово (несколько начал слогов), не шаг
                armed['heard'] = now
            elif now - armed['since'] > armed.get('patience', 10.):
                self.armed = None
                if getattr(self, 'round', None) is not None:
                    self.round = None
                    self.message = f'{armed["name"]}: «Братан» не было — круг остановлен, начни заново'
                else:
                    self.message = f'{armed["name"]}: звука не было — нажми ещё раз и скажи громче'
            return
        if now - armed['heard'] < 1.:
            return
        feat = self.direction.features(2.5)
        self.armed = None
        if feat is None:
            self.message = f'{armed["name"]}: звук потерялся — ещё раз'
            return
        fit, taus = self.direction.fingerprint(feat)
        # Сам звук фразы (три микрофона, 3 с) — для настройки расчёта под корпус по записям с известной стороны.
        got = self.link.ring.latest(3.)
        if got is not None:
            out = ROOT / 'recordings' / f'calib-{armed["deg"]:+04d}-{time.strftime("%Y%m%d-%H%M%S")}.npz'
            out.parent.mkdir(exist_ok=True)
            import numpy as np
            np.savez_compressed(out, x=got[0].astype(np.int16), valid=got[1], deg=armed['deg'])
        cal = self.direction.calibration
        cal.entries = [e for e in cal.entries if round(e['deg']) != armed['deg'] and 'fit' in e]  # сторону можно переписать
        cal.add(armed['deg'], fit, taus)
        cal.save()
        self.message = (f'{armed["name"]}: записано (модель услышала {fit:+.0f}°, задержки '
                        f'{[round(float(v), 1) for v in taus]}). Сторон: {len(cal.entries)} из 8')
        if getattr(self, 'round', None) is not None:
            self.round_next_at = time.time() + 3.
            following = next((n for n, d in STAND if d == self.round[0]), '')
            self.message += f' → переходи: {following}' if following else ''
        elif getattr(self, 'round', None) is None and len(cal.entries) >= 8:
            self.message += ' → все 8 записаны, жми «На робота»'

    def arm(self, deg, patience=10.):
        name = next((n for n, d in STAND if d == deg), f'{deg}°')
        self.armed = {'deg': deg, 'name': name, 'since': time.time(), 'heard': None, 'patience': patience}
        self.message = f'{name}: скажи «Братан» (жду {patience:.0f} с)'

    def clear(self):
        self.direction.calibration.entries = []
        self.direction.calibration.save()
        self.message = 'калибровка очищена: робот снова судит по самому громкому микрофону'

    def push(self):
        """Таблица → телефон (голос робота перечитает её сам при следующем звуке)."""
        import subprocess
        if not CALIBRATION.exists():
            self.message = 'нечего отправлять: калибровки нет'
            return
        r = subprocess.run(['scp', '-F', str(SSH_CONFIG), str(CALIBRATION), f'phone:{PHONE_CALIBRATION}'],
                           capture_output=True, timeout=60)
        self.message = ('на робота отправлено' if r.returncode == 0 else
                        f'не отправилось: {r.stderr.decode("utf-8", "replace")[-160:]}')

    def snapshot(self):
        with self.lock:
            now, last = self.now, self.last
        cal = self.direction.calibration
        done = sorted({round(e['deg']) for e in cal.entries})
        summary = self.link.summary()
        fresh_last = last if last and time.time() - last['t'] < 6 else None
        pick = lambda f: None if not f else {k: f[k] for k in ('deg', 'side', 'confidence', 'loudest', 'excess',
                                                                'tau23', 'method')}
        return {'now': pick(now), 'last': pick(fresh_last), 'ago': round(time.time() - last['t'], 1) if fresh_last else None,
                'stream': {'loss': summary['multi_loss'], 'age': summary['multi_age']},
                'calibrated': done, 'ready': cal.ready(), 'armed': (self.armed or {}).get('name'),
                'message': self.message, 'stand': [{'name': n, 'deg': d} for n, d in STAND]}


class Board:
    """Опрашивает плату: health 5 раз в секунду, pins раз в секунду, ping раз в 3 с. Хранит последние ответы."""

    def __init__(self, usb):
        self.usb = usb
        self.lock = threading.Lock()
        self.health = self.pins = self.status = None
        self.health_t = self.pins_t = self.status_t = 0.
        self.asked = deque(maxlen=200)
        self.answered = deque(maxlen=200)
        self.peaks = [deque(maxlen=15) for _ in range(4)]
        self.flash_until = [0.] * 4
        self.taps = [0] * 4
        self.first_counters = None
        if usb:
            from board import open_board
            self.port = open_board()
            time.sleep(.3)
            self.port.write(b's')
            self.where = 'USB'
        else:
            self.ip = arg('--ip') or (ROOT / '.ushi-ip').read_text().strip()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('', 0))
            self.sock.settimeout(.05)
            self.where = f'Wi-Fi {self.ip}'
        threading.Thread(target=self.run, daemon=True).start()

    def ask(self, what):
        if self.usb:
            self.port.write({'health': b'h', 'pins': b'd', 'ping': b'i'}[what])
        else:
            self.sock.sendto(what.encode(), (self.ip, 5771))
        if what == 'health':
            self.asked.append(time.time())

    def receive(self):
        if self.usb:
            line = self.port.readline().decode('utf-8', 'replace').strip()
            if line.startswith('I {'):
                line = line[2:]
            return line
        try:
            data, _ = self.sock.recvfrom(4096)
        except socket.timeout:
            return ''
        return data.decode('utf-8', 'replace').strip()

    def run(self):
        last = {'health': 0., 'pins': 0., 'ping': 0.}
        every = {'health': .2, 'pins': 1., 'ping': 3.}
        while True:
            now = time.time()
            for what in last:
                if now - last[what] >= every[what]:
                    last[what] = now
                    try:
                        self.ask(what)
                    except OSError:
                        pass
            try:
                line = self.receive()
            except OSError:
                time.sleep(.2)
                continue
            if line.startswith('H {'):
                try:
                    self.on_health(json.loads(line[2:]))
                except ValueError:
                    pass
            elif line.startswith('D '):
                m = PINS.search(line)
                if m:
                    with self.lock:
                        self.pins, self.pins_t = [int(g) for g in m.groups()], time.time()
            elif line.startswith('{'):
                try:
                    status = json.loads(line)
                    with self.lock:
                        self.status, self.status_t = status, time.time()
                except ValueError:
                    pass

    def on_health(self, h):
        now = time.time()
        with self.lock:
            self.answered.append(now)
            if self.first_counters is None:
                self.first_counters = {k: h.get(k, 0) for k in ('ovf', 'fail', 'epoch', 'qdrop')}
            for c in range(4):
                p = self.peaks[c]
                if len(p) >= 8 and h['v'][c] not in ('none', 'wait'):
                    base = statistics.median(p)
                    if h['fpeak'][c] - base > 12 and now > self.flash_until[c]:
                        self.flash_until[c] = now + .8
                        self.taps[c] += 1
                p.append(h['fpeak'][c])
            self.health, self.health_t = h, now
        self.journal(h)

    def journal(self, h):
        """watch.log: каждая смена вывода по слоту (держится ≥1 с), сбои такта, переполнения — чтобы между проверками
        ничего не пропало (17.10, владелец: «просто периодически чекай»)."""
        now = time.time()
        state = tuple(h['v'])
        pending = getattr(self, 'pending_state', None)
        if pending is None or pending[0] != state:
            self.pending_state = (state, now)
            return
        logged = getattr(self, 'logged_state', None)
        counters = (h.get('ovf'), h.get('fail'), h.get('epoch'))
        if (state != logged and now - pending[1] >= 1.) or counters != getattr(self, 'logged_counters', counters):
            self.logged_state, self.logged_counters = state, counters
            text = ' | '.join(f'{named(c)} {state[c]}' for c in range(4) if c in NAMES)
            stray = [f'слот {c}: {state[c]}' for c in range(4) if c not in NAMES and state[c] not in ('none', 'wait')]
            line = (f'{time.strftime("%H:%M:%S")} {text}' + (' | лишний ' + ', '.join(stray) if stray else '')
                    + f' | ovf {counters[0]} fail {counters[1]} epoch {counters[2]}\n')
            with open(ROOT / 'watch.log', 'a', encoding='utf-8') as handle:
                handle.write(line)
        self.logged_counters = counters

    def snapshot(self):
        now = time.time()
        with self.lock:
            h, pins, status = self.health, self.pins, self.status
            fresh = h is not None and now - self.health_t < 2.
            recent = [t for t in self.asked if now - 5 < t < now - .5]
            got = [t for t in self.answered if now - 5 < t]
            link = round(min(100, 100 * len(got) / len(recent))) if recent else None
            flash = [now < f for f in self.flash_until]
            taps = list(self.taps)
            first = self.first_counters
        chans, stray = [], []
        for c in range(4):
            name = named(c)
            sd, lr_col, lr = WIRES[c]
            pair, left, right = named(c ^ 1), named(c & 2), named(c | 1)
            place = PLACE.get(c, '')
            wires = f'SD → столбик {sd} · L/R → столбик {lr_col} ({lr})'
            if c not in NAMES:                     # лишний слот: не показываем, но если в нём что-то есть — сказать
                if fresh and h['v'][c] not in ('none', 'wait'):
                    stray.append(f'В лишнем слоте (SD → {sd}, L/R → {lr_col}) появился звук — какой-то микрофон '
                                 f'воткнут не туда: проверь SD и L/R по подсказкам на карточках')
                continue
            if not fresh:
                chans.append(dict(name=name, place=place, wires=wires, state='—', cls='dim',
                                  why='плата не отвечает', level=None, flash=False, taps=taps[c], extra=''))
                continue
            v = h['v'][c]
            zero, tail, copy = h['zero'][c], h['tail'][c], h['copy'][c]
            state, cls, why = {
                'ok': ('ПАШЕТ', 'ok', ''),
                'spill': ('ПАШЕТ*', 'warn', f'живой, но лезет и в {pair}: дожми его L/R в столбик {lr_col}'),
                'ghost': ('ЧУЖОЙ ЗВУК', 'copy', f'своего микрофона здесь нет: сюда перетекает {pair}, у которого '
                                                f'болтается L/R. Проверь у {name}: SD → {sd}, L/R → {lr_col}'),
                'none': ('НЕТ', 'bad', f'тишина. Проверь у этого микрофона: SD → {sd}, L/R → {lr_col}, {CHECK}; '
                                       f'не надет ли разъём зеркально'),
                'float': ('НАВОДКА', 'bad', f'в столбике {sd} мусор: провод SD висит без микрофона '
                                            f'или в столбик {sd} попал чужой провод'),
                'hop': ('ПРЫГАЕТ', 'bad', f'микрофон скачет между {left} и {right}: его L/R ни к чему '
                                          f'не подключён. L/R в 5/6 — будет {left}, в 2/3 — {right}'),
                'gaps': ('РВЁТСЯ', 'warn', f'звук то есть, то нет ({zero:.0f} % пусто): дожми провода этого микрофона'),
                'noise': ('МУСОР', 'bad', f'битые данные: в столбике {sd} два микрофона с одинаковым L/R '
                                          f'или чужой провод'),
                'wait': ('…', 'dim', 'плата собирает данные'),
            }.get(v, (v, 'dim', ''))
            level = h['frms'][c] if h['frms'][c] > -139 else None
            chans.append(dict(name=name, place=place, wires=wires, state=state, cls=cls, why=why,
                              level=level, flash=flash[c], taps=taps[c],
                              extra=f'пусто {zero:.0f} % · наводка {tail:.0f} % · копия {copy:.0f} %'))
        chans.sort(key=lambda ch: ch['name'])
        legend = ' · '.join(f'{name}' + (f' ({PLACE[c]})' if c in PLACE else '') + f': SD → {WIRES[c][0]}, '
                            f'L/R → {WIRES[c][1]}' for name, c in sorted(LAYOUT))
        clock = None
        if pins and now - self.pins_t < 5:
            sck, sck_d, ws, ws_d, d33, d33_d, d32, d32_d, fail, ovf = pins
            good = abs(ws - 16000) < 480 and abs(sck - 64 * ws) < 64 * ws * .03 and 35 <= ws_d <= 65 and 35 <= sck_d <= 65
            clock = dict(ok=good, ws=ws, sck=sck, ws_d=ws_d, sck_d=sck_d, d33=d33, d32=d32)
        counters = {}
        if fresh and first:
            counters = {k: h.get(k, 0) - first[k] for k in first}
        label = lambda c: f'{c["name"]} ({c["place"]})' if c['place'] else c['name']
        ok = [label(c) for c in chans if c['cls'] in ('ok', 'warn') and c['state'] != '—']
        bad = [label(c) for c in chans if c['cls'] in ('bad', 'copy')]
        return dict(where=self.where, fresh=fresh, link=link, chans=chans, clock=clock, counters=counters,
                    fw=(h or {}).get('fw'), rssi=(status or {}).get('rssi') if status and now - self.status_t < 10 else None,
                    ok=ok, bad=bad, stray=stray, legend=legend)


PAGE = r'''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Уши Братана</title>
<style>
:root{--bg:#111418;--card:#1b2026;--text:#e8edf2;--dim:#8c96a3;--ok:#2ecc71;--bad:#e74c3c;--warn:#f39c12;--copy:#b07cd8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.4 system-ui,sans-serif;padding:16px}
h1{margin:0 0 2px;font-size:22px}.sub{color:var(--dim);margin-bottom:12px;font-size:14px}
.sum{font-size:28px;font-weight:800;margin:4px 0 14px}
.top{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.pill{background:var(--card);border-radius:10px;padding:8px 12px;font-size:14px}.pill b{font-size:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}
.card{background:var(--card);border-radius:16px;padding:16px;border:4px solid transparent;transition:border-color .1s,background .1s}
.card.flash{border-color:#fff;background:#2c3440}
.head{display:flex;justify-content:space-between;align-items:baseline}
.name{font-size:30px;font-weight:700}.place{font-size:24px;font-weight:600;color:#7fb3ff}.tap{font-size:18px;font-weight:800;color:#fff;visibility:hidden}.card.flash .tap{visibility:visible}
.wires{color:var(--dim);font-size:13px;margin-bottom:6px}
.state{font-size:42px;font-weight:800;letter-spacing:.5px;line-height:1.1}
.why{min-height:40px;color:var(--text);opacity:.85;font-size:14px;margin:6px 0 8px}
.bar{height:16px;background:#0b0d10;border-radius:8px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok);width:0;transition:width .15s}
.meta{display:flex;justify-content:space-between;color:var(--dim);font-size:12px;margin-top:6px;gap:8px;flex-wrap:wrap}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}.copy{color:var(--copy)}.dim{color:var(--dim)}
h2{font-size:22px;margin:22px 0 8px}
.ears{display:flex;gap:18px;flex-wrap:wrap;align-items:center}
#compass{width:260px;height:260px;max-width:100%;background:var(--card);border-radius:50%}
#earsinfo{font-size:22px;line-height:1.35;max-width:560px}#earsinfo b{font-size:30px}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.btns button{background:var(--card);color:var(--text);border:2px solid #3a4250;border-radius:10px;padding:10px 14px;font-size:16px;cursor:pointer}
.btns button.done{border-color:var(--ok)}.btns button.armed{border-color:#ffd166;background:#3a3420}
.btns button.act{border-color:#7fb3ff}
#calmsg{color:var(--dim);font-size:15px;min-height:22px}
</style></head><body>
<h1>Уши Братана — живая проверка</h1>
<div class="sub">Плата сама проверяет каждый микрофон 2 раза в секунду. Стучи по микрофону — его карточка вспыхнет.<br>
Столбики макетки: <b>2/3</b> — 3V3 · <b>5/6</b> — GND · <b>8</b> — SCK · <b>10</b> — WS. <span id="legend"></span></div>
<div class="sum" id="sum">…</div><div class="top" id="top"></div><div class="grid" id="grid"></div>
<div id="earsbox"><h2>Откуда слышит</h2>
<div class="ears"><svg id="compass" viewBox="0 0 240 240"></svg><div id="earsinfo">…</div></div>
<h2>Калибровка</h2>
<div class="sub">Встань в 1–2 м от робота с одной из сторон, нажми её кнопку и скажи «Братан». Зелёные — записаны.
Хватает 4 сторон, лучше все 8. Потом «На робота».</div>
<div class="btns" id="calib"></div><div id="calmsg"></div></div>
<script>
const COMPASS_BASE = `<circle cx="120" cy="120" r="112" fill="none" stroke="#2c3440" stroke-width="2"/>
<text x="120" y="20" fill="#8c96a3" font-size="13" text-anchor="middle">перёд</text>
<text x="226" y="124" fill="#8c96a3" font-size="13" text-anchor="end">справа</text>
<text x="14" y="124" fill="#8c96a3" font-size="13">слева</text>
<text x="120" y="232" fill="#8c96a3" font-size="13" text-anchor="middle">сзади</text>
<rect x="88" y="78" width="64" height="94" rx="12" fill="#2c3440" stroke="#4a5568"/>
<circle cx="120" cy="80" r="6" fill="#7fb3ff"/><text x="120" y="98" fill="#7fb3ff" font-size="12" text-anchor="middle">М1</text>
<circle cx="152" cy="140" r="6" fill="#7fb3ff"/><text x="146" y="160" fill="#7fb3ff" font-size="12" text-anchor="end">М2</text>
<circle cx="88" cy="140" r="6" fill="#7fb3ff"/><text x="94" y="160" fill="#7fb3ff" font-size="12">М3</text>`;
const xy = (d, r) => [120 - r * Math.sin(d * Math.PI / 180), 120 - r * Math.cos(d * Math.PI / 180)];
const sign = d => (d > 0 ? '+' : '') + Math.round(d);
async function earsTick(){
  try{
    const e = await (await fetch('/ears')).json();
    if (e.off){ document.getElementById('earsbox').style.display = 'none'; return; }
    const f = e.now || e.last;
    let arrow = '';
    if (f){
      const [x, y] = xy(f.deg, 92), col = e.now ? '#ffd166' : '#5f7fa8';
      arrow = `<line x1="120" y1="125" x2="${x}" y2="${y}" stroke="${col}" stroke-width="8" stroke-linecap="round"/><circle cx="${x}" cy="${y}" r="10" fill="${col}"/>`;
    }
    const dots = e.calibrated.map(d => { const [x, y] = xy(d, 104); return `<circle cx="${x}" cy="${y}" r="5" fill="#2ecc71"/>`; }).join('');
    document.getElementById('compass').innerHTML = COMPASS_BASE + dots + arrow;
    const how = f ? (f.method === 'calibrated' ? 'по калибровке' : 'по громкости микрофонов') : '';
    document.getElementById('earsinfo').innerHTML = e.now
      ? `Слышу: <b>${esc(f.side)}</b> (${sign(f.deg)}°)<br>уверенность ${f.confidence.toFixed(2)} · громче всех ${esc(f.loudest)} · ${how}`
      : (e.last ? `<span class="dim">Последний звук ${e.ago} с назад: <b>${esc(f.side)}</b> (${sign(f.deg)}°)</span>`
                : '<span class="dim">Тихо — скажи что-нибудь</span>')
      + `<br><span class="dim" style="font-size:14px">поток трёх микрофонов: потери ${(e.stream.loss * 100).toFixed(1)} %</span>`;
    document.getElementById('calib').innerHTML = e.stand.map(s =>
      `<button class="${e.calibrated.includes(s.deg) ? 'done' : ''} ${e.armed === s.name ? 'armed' : ''}" onclick="fetch('/ears/arm?deg=${s.deg}')">${esc(s.name)}</button>`).join('')
      + `<button class="act" onclick="fetch('/ears/round')">Пройти все 8 по кругу</button><button class="act" onclick="fetch('/ears/push')">На робота</button><button onclick="if(confirm('Стереть калибровку?')) fetch('/ears/clear')">Стереть</button>`;
    document.getElementById('calmsg').textContent = e.message || (e.ready ? 'калибровка готова — робот определяет направление по ней' : '');
  }catch(err){}
  setTimeout(earsTick, 300);
}
earsTick();
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function tick(){
  try{
    const d = await (await fetch('/state')).json();
    let sum;
    if(!d.fresh) sum = '<span class="bad">ПЛАТА НЕ ОТВЕЧАЕТ</span>';
    else sum = (d.ok.length ? `<span class="ok">Пашут: ${d.ok.join(', ')}</span>` : '<span class="bad">Ни один не пашет</span>')
             + (d.bad.length ? ` · <span class="bad">не пашут: ${d.bad.join(', ')}</span>` : '');
    document.getElementById('sum').innerHTML = sum;
    const c = d.clock;
    const clk = !c ? '<b class="dim">такт: жду замер</b>' : c.ok
      ? `<b class="ok">такт исправен</b> · WS ${c.ws} Гц · SCK ${(c.sck/1000).toFixed(1)} кГц`
      : `<b class="bad">ТАКТ СБОИТ</b> · WS ${c.ws} Гц (${c.ws_d}%) · SCK ${(c.sck/1000).toFixed(1)} кГц (${c.sck_d}%) — чужой провод на WS/SCK?`;
    const lines = c ? `линия D33: ${c.d33 ? '<b class="ok">идут данные</b>' : 'пусто'} · линия D32: ${c.d32 ? '<b class="ok">идут данные</b>' : 'пусто'}` : '';
    const k = d.counters || {};
    const trouble = (k.fail || k.ovf) ? `<div class="pill"><b class="warn">за сеанс: сбоев такта ${k.fail||0}, переполнений ${k.ovf||0}</b></div>` : '';
    const stray = (d.stray || []).map(s => `<div class="pill"><b class="warn">${esc(s)}</b></div>`).join('');
    document.getElementById('legend').textContent = d.legend || '';
    document.getElementById('top').innerHTML =
      `<div class="pill">${d.fresh ? '<b class="ok">связь есть</b>' : '<b class="bad">нет ответа</b>'} · ${esc(d.where)}${d.link!=null ? ' · ответов ' + d.link + ' %' : ''}${d.rssi!=null ? ' · сигнал ' + d.rssi + ' дБм' : ''}${d.fw ? ' · ' + esc(d.fw) : ''}</div>
       <div class="pill">${clk}</div>${lines ? `<div class="pill">${lines}</div>` : ''}${trouble}${stray}`;
    document.getElementById('grid').innerHTML = d.chans.map(ch => {
      const w = ch.level == null ? 0 : Math.max(2, Math.min(100, (ch.level + 90) / 75 * 100));
      return `<div class="card ${ch.flash ? 'flash' : ''}"><div class="head"><span class="name">${ch.name}${ch.place ? ' <span class="place">' + esc(ch.place) + '</span>' : ''}</span><span class="tap">● СТУК</span></div>
        <div class="wires">${esc(ch.wires)}</div><div class="state ${ch.cls}">${esc(ch.state)}</div>
        <div class="why">${esc(ch.why)}</div><div class="bar"><i style="width:${w}%"></i></div>
        <div class="meta"><span>${ch.level == null ? 'уровень —' : 'уровень ' + ch.level.toFixed(0) + ' дБ'}</span><span>стуков ${ch.taps}</span></div>
        <div class="meta"><span>${esc(ch.extra)}</span></div></div>`;
    }).join('');
  }catch(e){ document.getElementById('sum').innerHTML = '<span class="bad">панель не отвечает — запусти tools/live.py</span>'; }
  setTimeout(tick, 200);
}
tick();
</script></body></html>'''


def serve(board, port, ears=None, lan=False):
    from urllib.parse import parse_qs, urlparse

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            as_json = lambda v: (json.dumps(v, ensure_ascii=False).encode(), 'application/json; charset=utf-8')
            if url.path == '/state':
                body, ctype = as_json(board.snapshot())
            elif url.path == '/ears':
                body, ctype = as_json(ears.snapshot() if ears else {'off': True})
            elif url.path.startswith('/ears/') and ears:
                if url.path == '/ears/arm':
                    ears.arm(int(parse_qs(url.query).get('deg', ['0'])[0]))
                elif url.path == '/ears/clear':
                    ears.clear()
                elif url.path == '/ears/round':
                    ears.start_round()
                elif url.path == '/ears/push':
                    threading.Thread(target=ears.push, daemon=True).start()
                body, ctype = as_json({'ok': True})
            else:
                body, ctype = PAGE.encode(), 'text/html; charset=utf-8'
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f'панель: http://localhost:{port}  (плата: {board.where})' + (' — видна в домашней сети' if lan else ''),
          flush=True)
    ThreadingHTTPServer(('0.0.0.0' if lan else '127.0.0.1', port), Handler).serve_forever()


def once(board):
    end = time.time() + 6
    while time.time() < end:
        s = board.snapshot()
        if s['fresh'] and s['clock'] and len(board.answered) >= 3:
            break
        time.sleep(.1)
    s = board.snapshot()
    if not s['fresh']:
        sys.exit(f'плата не отвечает ({s["where"]})')
    c = s['clock']
    clock = 'такт: нет замера' if not c else (
        f'такт исправен: WS {c["ws"]} Гц, SCK {c["sck"] / 1000:.1f} кГц' if c['ok'] else
        f'ТАКТ СБОИТ: WS {c["ws"]} Гц ({c["ws_d"]} %), SCK {c["sck"] / 1000:.1f} кГц ({c["sck_d"]} %)')
    lines = f'D33 {"данные" if c and c["d33"] else "пусто"}, D32 {"данные" if c and c["d32"] else "пусто"}' if c else ''
    print(f'{s["fw"]} · {s["where"]}' + (f' · сигнал {s["rssi"]} дБм' if s['rssi'] is not None else '')
          + f' · {clock} · {lines}')
    for ch in s['chans']:
        level = f'{ch["level"]:6.1f} дБ' if ch['level'] is not None else '     — '
        print(f'  {ch["name"]} {ch["place"]:<8} {ch["state"]:<10} {level}  {ch["extra"]}  {ch["why"]}')
    for line in s['stray']:
        print('  !', line)
    k = s['counters']
    if k.get('fail') or k.get('ovf'):
        print(f'  за время проверки: сбоев такта {k["fail"]}, переполнений {k["ovf"]}')


if __name__ == '__main__':
    board = Board('--usb' in sys.argv)
    if '--once' in sys.argv:
        once(board)
        if board.usb:
            board.port.write(b'l')
    else:
        ears = None
        if '--no-ears' not in sys.argv:
            try:
                ears = Ears(arg('--ip') or (ROOT / '.ushi-ip').read_text().strip())
            except Exception as error:      # noqa: BLE001 — без numpy/модуля робота панель работает без компаса
                print('компас недоступен:', type(error).__name__, error, flush=True)
        serve(board, int(arg('--port', 8777)), ears, lan='--lan' in sys.argv)
