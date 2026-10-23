"""Открыть порт платы, не перезагружая её: по умолчанию открытие порта дёргает DTR/RTS, а на DevKit это кнопка EN
(15.10: каждая проверка начиналась с перезагрузки — микрофоны только просыпались, первые записи с нулями)."""
import serial
from serial.tools import list_ports


def find():
    return next((p.device for p in list_ports.comports() if 'CP210' in (p.description or '') or
                 'CH340' in (p.description or '')), None)


def open_board(dev=None, baud=921600, timeout=.3):
    s = serial.Serial()
    s.port = dev or find()
    s.baudrate = baud
    s.timeout = timeout
    s.dtr = False
    s.rts = False
    s.open()
    return s
