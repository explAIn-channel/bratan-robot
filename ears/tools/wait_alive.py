"""Ждать, пока нужные микрофоны оживут и продержатся (по USB, уровни платы). Выход — сразу.
python tools/wait_alive.py [секунд] [--need 1,2] [--hold 2]"""
import sys
import time

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from board import open_board  # noqa: E402

limit = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace('.', '').isdigit() else 540.
need = [int(c) - 1 for c in sys.argv[sys.argv.index('--need') + 1].split(',')] if '--need' in sys.argv else None
hold = float(sys.argv[sys.argv.index('--hold') + 1]) if '--hold' in sys.argv else .6
p = open_board()
time.sleep(.3)
p.reset_input_buffer()
p.write(b'l')
since = [None] * 4
seen = set()
start = time.time()
while time.time() - start < limit:
    line = p.readline().decode('utf-8', 'replace').strip()
    if not line.startswith('L '):
        continue
    parts = line.split()[1:]
    try:
        alive = [parts[i * 4 + 3] == 'ok' for i in range(4)]
    except IndexError:
        continue
    now = time.time()
    for c in range(4):
        since[c] = (since[c] or now) if alive[c] else None
        if alive[c] and c not in seen:
            seen.add(c)
            print(f'{now - start:.0f} с: М{c + 1} подаёт признаки жизни', flush=True)
    targets = need if need is not None else [c for c in range(4) if since[c]]
    if targets and all(since[c] and now - since[c] >= hold for c in targets):
        print(f'{now - start:.0f} с: ЖИВЫ И ДЕРЖАТСЯ ' + ', '.join(f'М{c + 1}' for c in range(4) if alive[c]), flush=True)
        sys.exit(0)
print('за отведённое время не дождался')
sys.exit(1)
