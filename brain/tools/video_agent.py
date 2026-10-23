"""PC agent: pull trip videos as soon as the robot docks after a trip; catch up at start.

Started at logon by the Windows task BratanVideos (pythonw, no window). Keeps one SSH
stream to the robot's bus open and reacts to `docked`; reconnects quietly when the
robot or the network is away.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pull_videos  # noqa: E402

LOG = pull_videos.TARGET / 'agent.log'


def log(*parts):
    pull_videos.TARGET.mkdir(parents=True, exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as handle:
        handle.write(time.strftime('%Y-%m-%d %H:%M:%S ') + ' '.join(str(p) for p in parts) + '\n')


def pull(reason):
    try:
        pull_videos.main()
        log('pulled after', reason)
    except Exception as error:
        log('pull failed', reason, type(error).__name__, error)


def soul_loop():
    """Once a night a copy of the robot's character, memory and faces comes to the PC (09.10 audit)."""
    import soul_backup
    while True:
        try:
            if soul_backup.due():
                log('soul copied', soul_backup.backup())
        except Exception as error:
            log('soul copy failed', type(error).__name__, error)
        time.sleep(900)


def main():
    import threading
    threading.Thread(target=soul_loop, name='soul', daemon=True).start()
    pull('agent start')
    while True:
        command = pull_videos.SSH + ['cd /opt/robotbrain/bratan/app && PYTHONPATH=. '
                                     '/opt/robotbrain/venvs/perception/bin/python -u -m bratan.cli tail body.event']
        try:
            stream = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                      encoding='utf-8', errors='replace',
                                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            log('connected')
            for line in stream.stdout:
                if '"kind": "docked"' in line:
                    time.sleep(15)          # the recorder closes its segment ~8 s after the wheels stop
                    pull('docked')
            stream.wait()
        except Exception as error:
            log('stream error', type(error).__name__, error)
        time.sleep(30)
        pull('reconnect')


if __name__ == '__main__':
    main()
