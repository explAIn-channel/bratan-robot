"""Does the ears board answer over Bluetooth at all (C52, 18.10 22:17: the phone connected, but service discovery was
never answered)? From the PC: find «Bratan-ushi», connect, list its services — only reading, no stream, no token.

  python tools/ble_gatt_probe.py"""
import asyncio
import sys
import time

from bleak import BleakClient, BleakScanner

SERVICE = '6b9c0001-3f6e-4f4b-9a54-6272617461e1'


async def main():
    device = await BleakScanner.find_device_by_filter(
        lambda d, adv: SERVICE in [u.lower() for u in adv.service_uuids] or (d.name or '').startswith('Bratan'),
        timeout=15.)
    if device is None:
        sys.exit('not found by Bluetooth')
    print('found', device.address, device.name, flush=True)
    started = time.monotonic()
    try:
        async with BleakClient(device, timeout=20.) as client:
            print(f'connected in {time.monotonic() - started:.1f} s, mtu {client.mtu_size}', flush=True)
            for service in client.services:
                print('service', service.uuid, [c.uuid[:8] for c in service.characteristics])
    except Exception as error:
        print('failed after', round(time.monotonic() - started, 1), 's:', type(error).__name__, error)


asyncio.run(main())
