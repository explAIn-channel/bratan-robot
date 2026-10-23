"""Прошить «уши»: первый раз по USB, дальше по Wi-Fi — и убедиться, что новая прошивка жива.

    python tools/flash.py usb       плата на проводе (порт находится сам)
    python tools/flash.py ota       по домашнему Wi-Fi; после приёма ждёт, пока плата вернётся в сеть и подтвердит
                                    прошивку (60 с устойчивой связи). Не подтвердила — загрузчик сам вернул прежнюю.
    python tools/flash.py status    найти плату в сети: версия, сигнал (RSSI), обрывы, переподключения
    python tools/flash.py build     только собрать

Плата ищется так: имя bratan-ushi.local → последний известный адрес (.ushi-ip) → широковещательный «ping» по UDP 5771.
Пароль прошивки по воздуху создаётся один раз в .ota-pass и вшивается при сборке; на экран не выводится.
PlatformIO берётся из E:/pio-core-ushi: обычный ~/.platformio — ярлык в несуществующий архив (не трогать).
"""
import argparse
import json
import os
import secrets
import socket
import string
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASS_FILE = ROOT / '.ota-pass'
IP_FILE = ROOT / '.ushi-ip'
PING_PORT = 5771


NEW_PASS_FILE = ROOT / '.ota-pass.new'


def new_password():
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(24))


def ota_pass():
    if not PASS_FILE.exists():
        PASS_FILE.write_text(new_password(), encoding='ascii')
        print('создан новый пароль прошивки по воздуху:', PASS_FILE.name)
    return PASS_FILE.read_text(encoding='ascii').strip()


TOKEN_FILE = ROOT / '.ble-token'


def ble_token():
    """The token the board's Bluetooth stream opens to (the face app writes it, C52 18.10); made once, never shown."""
    if not TOKEN_FILE.exists():
        TOKEN_FILE.write_text(new_password(), encoding='ascii')
    return TOKEN_FILE.read_text(encoding='ascii').strip()


def run_quiet(command, env, secrets_):
    """pio без пароля в выводе: espota в отладке печатает свои параметры вместе с паролем (утёк так 18.10)."""
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace')
    for line in proc.stdout:
        if any(s and s in line for s in secrets_) or 'auth' in line.lower() or 'Options' in line:
            continue
        print(line, end='', flush=True)
    return proc.wait()


def ping(address, timeout=1.5, tries=4):
    """Ответ платы на «ping». Несколько попыток: при слабом Wi-Fi (18.10, −85 дБм) один запрос из трёх теряется."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.settimeout(timeout)
        for _ in range(tries):
            try:
                s.sendto(b'ping', (address, PING_PORT))
                data, sender = s.recvfrom(2048)
                return json.loads(data.decode('utf-8')), sender[0]
            except (OSError, ValueError):
                continue
        return None, None


def find_board():
    candidates = []
    try:
        candidates.append(socket.gethostbyname('bratan-ushi.local'))
    except OSError:
        pass
    if IP_FILE.exists():
        candidates.append(IP_FILE.read_text().strip())
    candidates.append('255.255.255.255')
    for address in candidates:
        status, ip = ping(address)
        if status:
            IP_FILE.write_text(ip)
            return status, ip
    return None, None


def wait_confirmed(old_up, version_before):
    """После приёма: плата перезагрузилась, вернулась в сеть и через 60 с подтвердила прошивку."""
    deadline = time.time() + 240
    rebooted = False
    while time.time() < deadline:
        status, ip = find_board()
        if status:
            if status['up_s'] < (old_up or 10 ** 9) or rebooted:
                rebooted = True
                if not status['trial']:
                    same = status['fw'] == version_before
                    print(f'плата в сети ({ip}), прошивка {status["fw"]} подтверждена, сигнал {status["rssi"]} дБм'
                          + (' — та же версия, что была (если ждали новую — загрузчик откатил её)' if same else ''))
                    return 0
                print(f'  новая прошивка на проверке: {status["up_s"]} с работы, сигнал {status["rssi"]} дБм', flush=True)
        else:
            print('  плата пока не в сети…', flush=True)
        time.sleep(5)
    print('плата не подтвердила прошивку за 4 минуты: загрузчик вернёт прежнюю; если и её нет в сети — нужен USB')
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('how', choices=('usb', 'ota', 'status', 'build'))
    ap.add_argument('--ip', help='адрес платы, если сама не находится')
    ap.add_argument('--rotate', action='store_true',
                    help='ota: вшить новый пароль (вход — по старому); после подтверждения он заменит .ota-pass')
    args = ap.parse_args()
    if args.how == 'status':
        status, ip = ping(args.ip) if args.ip else find_board()
        if not status:
            sys.exit('плата в сети не найдена (выключена, нет Wi-Fi или открыта точка Bratan-Ushi)')
        print(ip, json.dumps(status, ensure_ascii=False))
        rssi = status.get('rssi', 0)
        print('сигнал:', 'хороший' if rssi > -60 else 'достаточный' if rssi > -70 else
              'СЛАБЫЙ — переставить плату дальше от металла/аккумулятора или внешняя антенна')
        return
    current = ota_pass()
    built = current
    if args.rotate and args.how == 'ota':
        if not NEW_PASS_FILE.exists():
            NEW_PASS_FILE.write_text(new_password(), encoding='ascii')
        built = NEW_PASS_FILE.read_text(encoding='ascii').strip()
    token = ble_token()
    env = dict(os.environ, USHI_OTA_PASS=built, USHI_OTA_AUTH=current, USHI_BLE_TOKEN=token,
               PLATFORMIO_CORE_DIR=os.environ.get('PLATFORMIO_CORE_DIR', 'E:/pio-core-ushi'))
    command = ['pio', 'run', '-e', 'ota' if args.how == 'ota' else 'usb']
    before = None
    if args.how == 'ota':
        before, ip = (ping(args.ip) if args.ip else find_board())
        if not before:
            sys.exit('плата в сети не найдена — прошить по воздуху нельзя (status покажет, в чём дело)')
        command += ['-t', 'upload', '--upload-port', ip]
    elif args.how == 'usb':
        command += ['-t', 'upload']
    code = run_quiet(command, env, (current, built, token))
    if code or args.how != 'ota':
        sys.exit(code)
    result = wait_confirmed(before['up_s'], before['fw'])
    if result == 0 and built != current:
        status, _ = find_board()
        if status and status['fw'] != before['fw']:            # новая прошивка живёт — у неё новый пароль
            PASS_FILE.with_name('.ota-pass.old').write_text(current, encoding='ascii')
            NEW_PASS_FILE.replace(PASS_FILE)
            print('пароль прошивки по воздуху сменён (.ota-pass), прежний — в .ota-pass.old')
        else:
            print('прошивка откатилась: пароль прежний, новый остался в .ota-pass.new')
    sys.exit(result)


if __name__ == '__main__':
    main()
