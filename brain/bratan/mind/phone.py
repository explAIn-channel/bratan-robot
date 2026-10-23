"""Phone battery care: hold charging at 80 % and resume below 75 % (Qualcomm restricted charging).

The original controls are saved before the first change and restored on release,
so a crash never leaves the phone unable to charge for long (released at start).
"""
from __future__ import annotations

import json
from pathlib import Path

SYS = Path('/sys/class')


class ChargeLimit:
    def __init__(self, marker: Path, limit=80, resume=75, sysfs=SYS):
        self.marker = Path(marker)
        self.limit, self.resume = limit, resume
        self.enable = sysfs / 'qcom-battery/restricted_charging'
        self.current = sysfs / 'qcom-battery/restricted_current'
        self.capacity = sysfs / 'power_supply/battery/capacity'
        self.usb = sysfs / 'power_supply/usb/present'
        self.state = {}
        self.release()

    @property
    def supported(self):
        return self.enable.exists() and self.current.exists()

    def _read(self, path):
        return int(path.read_text().strip())

    def _write(self, path, value):
        if self._read(path) != value:
            path.write_text(f'{value}\n')

    def hold(self):
        if self.marker.exists():
            return
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        self.marker.write_text(json.dumps({'enable': self._read(self.enable), 'current': self._read(self.current)}))
        self._write(self.current, 0)
        self._write(self.enable, 1)

    def release(self):
        if not self.marker.exists() or not self.supported:
            return
        saved = json.loads(self.marker.read_text())
        self._write(self.current, int(saved.get('current', 0)))
        self._write(self.enable, int(saved.get('enable', 0)))
        self.marker.unlink()

    def poll(self):
        if not self.supported:
            self.state = {'supported': False}
            return self.state
        try:
            percent = self._read(self.capacity)
            usb = bool(self._read(self.usb))
            if not usb or percent <= self.resume:
                self.release()
            elif percent >= self.limit:
                self.hold()
            self.state = {'percent': percent, 'usb': usb, 'held': self.marker.exists()}
        except (OSError, ValueError) as error:
            try:
                self.release()
            except (OSError, ValueError):
                pass
            self.state = {'error': f'{type(error).__name__}:{error}'}
        return self.state
