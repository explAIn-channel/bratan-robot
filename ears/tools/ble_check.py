"""Мост «уши → Bluetooth» с ПК (C52): найти «Bratan-ushi», подписаться на звук, собрать пакеты BRM1 из кусков, как
приложение лица, и посчитать: пакетов в секунду (надо ~31), потерянных, байт в секунду (~25 КБ/с).

  python tools/ble_check.py [секунд]
"""
import asyncio
import struct
import sys
import time

from bleak import BleakClient, BleakScanner

SERVICE = '6b9c0001-3f6e-4f4b-9a54-6272617461e1'
AUDIO = '6b9c0002-3f6e-4f4b-9a54-6272617461e1'
UNLOCK = '6b9c0003-3f6e-4f4b-9a54-6272617461e1'
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 15.
sys.path.insert(0, __import__('os').path.dirname(__file__))


async def main():
    device = await BleakScanner.find_device_by_filter(
        lambda d, adv: SERVICE in [u.lower() for u in adv.service_uuids] or (d.name or '').startswith('Bratan'),
        timeout=15.)
    if device is None:
        sys.exit('плата по Bluetooth не найдена')
    print('нашёл', device.address, device.name, flush=True)
    state = {'expected': -1, 'buf': bytearray(), 'last': -1, 'packets': 0, 'dropped': 0, 'bytes': 0, 'chunks': 0,
             'firsts': []}

    def on_chunk(_, data: bytearray):
        state['chunks'] += 1
        state['bytes'] += len(data)
        head = data[0]
        counter, start = head & 0x7F, bool(head & 0x80)
        if start:
            if state['expected'] >= 0:
                state['dropped'] += 1
            state['expected'] = data[1] | (data[2] << 8)
            state['buf'] = bytearray(data[3:])
        elif state['expected'] < 0 or counter != ((state['last'] + 1) & 0x7F):
            if state['expected'] >= 0:
                state['dropped'] += 1
            state['expected'] = -1
            state['last'] = counter
            return
        else:
            state['buf'] += data[1:]
        state['last'] = counter
        if state['expected'] >= 0 and len(state['buf']) >= state['expected']:
            packet = bytes(state['buf'][:state['expected']])
            if packet[:4] == b'BRM1':
                state['firsts'].append(struct.unpack_from('<I', packet, 8)[0])
            state['packets'] += 1
            state['expected'] = -1

    lost = {}

    def on_disconnect(_):
        lost.setdefault('at', time.time())

    async with BleakClient(device, disconnected_callback=on_disconnect) as client:
        print('MTU', client.mtu_size, flush=True)
        await client.start_notify(AUDIO, on_chunk)
        if '--token' in sys.argv or '--wrong-token' in sys.argv:          # ushi-19: the stream opens to the token
            from flash import ble_token
            value = ble_token() if '--token' in sys.argv else 'wrong'
            try:
                await client.write_gatt_char(UNLOCK, value.encode('ascii'), response=True)
                print('token written' if '--token' in sys.argv else 'wrong token written', flush=True)
            except Exception as error:
                print('token write failed:', type(error).__name__, flush=True)
        t0 = time.time()
        while time.time() - t0 < SECONDS and 'at' not in lost:
            await asyncio.sleep(.2)
        if 'at' in lost:
            print(f'связь оборвалась через {lost["at"] - t0:.1f} с', flush=True)
        else:
            await client.stop_notify(AUDIO)
        took = lost.get('at', time.time()) - t0
    firsts = state['firsts']
    gaps = sum(1 for a, b in zip(firsts, firsts[1:]) if b - a != 512)
    print(f'за {took:.1f} с: пакетов {state["packets"]} ({state["packets"] / took:.1f}/с, надо ~31), кусков '
          f'{state["chunks"]}, {state["bytes"] / took / 1000:.1f} КБ/с, недособранных {state["dropped"]}, '
          f'разрывов по кадрам {gaps}', flush=True)


asyncio.run(main())
