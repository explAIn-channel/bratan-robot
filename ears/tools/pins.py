"""Самопроверка ножек платы (команда d): переключения такта и линий данных — «прошивка или провода»."""
import sys
import time

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from board import open_board  # noqa: E402

p = open_board()
time.sleep(.3)
p.write(b's')
time.sleep(.3)
for _ in range(3):
    p.reset_input_buffer()
    p.write(b'd')
    end = time.time() + 5
    while time.time() < end:
        line = p.readline().decode('utf-8', 'replace').strip()
        if line.startswith('D '):
            print(line, flush=True)
            break
    time.sleep(.3)
p.write(b'l')
