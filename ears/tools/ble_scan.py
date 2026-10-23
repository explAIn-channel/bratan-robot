"""From the PC: every Bluetooth LE device heard in N seconds (name, address, signal) — read-only."""
import asyncio
import sys

from bleak import BleakScanner


async def main(seconds):
    found = await BleakScanner.discover(timeout=seconds, return_adv=True)
    for address, (device, adv) in sorted(found.items(), key=lambda kv: -kv[1][1].rssi):
        print(f'{adv.rssi:4d} dBm  {address}  {device.name or adv.local_name or ""}  {list(adv.manufacturer_data)}')


asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 12.))
