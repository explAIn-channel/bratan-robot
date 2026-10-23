"""Подключить «уши» к Wi-Fi по кнопке WPS роутера (без ввода пароля): python tools/wps.py
Плата ждёт 2 минуты; за это время нажать кнопку WPS на роутере. Потом показывает адрес и силу сигнала."""
import json
import sys
import time

import serial

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from board import open_board  # noqa: E402
from serial.tools import list_ports

port = next((p.device for p in list_ports.comports() if 'CP210' in (p.description or '') or 'CH340' in (p.description or '')), None)
if not port:
    sys.exit('плата не подключена по USB')
s = open_board(port)
time.sleep(.3)
s.reset_input_buffer()
s.write(b's')                                  # без потока уровней: только сообщения
time.sleep(.2)
s.write(b'p')
print('WPS запущен: нажми кнопку WPS на роутере (2 минуты)', flush=True)
end = time.time() + 150
connected = False
asked = 0.
while time.time() < end:
    try:
        line = s.readline().decode('utf-8', 'replace').strip()
    except serial.SerialException:
        sys.exit('плата пропала с USB (выдернули кабель или порт отключил питание)')
    if line.startswith('E '):
        print(line, flush=True)
    if 'WPS не вышел' in line:
        sys.exit(1)
    if time.time() - asked >= 3:                 # спросить состояние раз в 3 секунды
        asked = time.time()
        s.write(b'i')
    if line.startswith('I {'):
        status = json.loads(line[2:])
        if status.get('online'):
            connected = True
            print(f'ПОДКЛЮЧЕНО: адрес {status["ip"]}, сигнал {status["rssi"]} дБм', flush=True)
            break
s.write(b'l')
sys.exit(0 if connected else 1)
