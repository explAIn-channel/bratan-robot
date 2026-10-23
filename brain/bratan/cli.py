"""Operator tool on the phone.

python -m bratan.cli pub body.do '{"skill":"rotate","args":{"angle":1.0}}'
python -m bratan.cli tail body.event voice. eyes.seen        (prints messages, Ctrl-C to stop)
python -m bratan.cli get body.state                          (one retained value)
python -m bratan.cli autonomy on|off
"""
from __future__ import annotations

import json
import os
import sys
import time

from .common import bus


def main(argv):
    command = argv[0]
    if command == 'pub':
        client = bus.Client('cli').start()
        client.wait_connected(3)
        client.publish(argv[1], json.loads(argv[2]) if len(argv) > 2 else {})
        time.sleep(.2)
    elif command == 'autonomy':
        client = bus.Client('cli').start()
        client.wait_connected(3)
        client.publish('mind.autonomy', {'enabled': argv[1] == 'on'})
        time.sleep(.2)
    elif command in ('tail', 'get'):
        done = []

        def show(topic, data):
            text = json.dumps(data, ensure_ascii=False)
            try:
                print(time.strftime('%H:%M:%S'), topic, text[:600] if command == 'tail' else text, flush=True)
            except (BrokenPipeError, OSError):
                os._exit(0)                   # nobody reads any more
            done.append(1)
        client = bus.Client('cli', argv[1:], show).start()
        deadline = time.time() + (3 if command == 'get' else 1e9)
        beat = time.time()
        try:
            while time.time() < deadline and not (command == 'get' and done):
                time.sleep(.05)
                if command == 'tail' and time.time() - beat > 60:
                    # A reader that went away (the PC agent's SSH dropped) is only noticed on a write:
                    # without this an idle `tail` lived for days on the robot (07.10: two orphans).
                    beat = time.time()
                    try:
                        print('', flush=True)
                    except (BrokenPipeError, OSError):
                        os._exit(0)
        except KeyboardInterrupt:
            pass
    else:
        print(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:] or ['help'])
