"""Мост Bluetooth глазами платы (C52): два «ping» с паузой — сколько кусков в секунду ушло и сколько брошено.
Нужно ~62 куска/с (31 пакет по 2 куска).

  python tools/ble_rate.py [секунд] [--ip адрес]
"""
import argparse
import time

from flash import find_board, ping

ap = argparse.ArgumentParser()
ap.add_argument('seconds', nargs='?', type=float, default=15.)
ap.add_argument('--ip')
args = ap.parse_args()
first, ip = ping(args.ip) if args.ip else find_board()
if not first:
    raise SystemExit('плата не найдена')
t0 = time.time()
time.sleep(args.seconds)
last, _ = ping(ip)
took = time.time() - t0
a, b = first['ble'], last['ble']
rate = lambda key: (b[key] - a[key]) / took
print(f'{ip} {last["fw"]}, работает {last["up_s"]} с (сброс {last.get("reset")}), Wi-Fi {last["rssi"]} дБм, '
      f'подписчиков по Wi-Fi {last["subs"]}')
print(f'Bluetooth: подписан {b["sub"]}, MTU {b["mtu"]}, интервал {b.get("itvl_ms")} мс, DLE {b.get("dle")}; '
      f'кусков {rate("chunks"):.1f}/с (надо ~62), брошено кусков {rate("failed"):.1f}/с, '
      f'пакетов не влезло в очередь {rate("qdrops"):.1f}/с')
