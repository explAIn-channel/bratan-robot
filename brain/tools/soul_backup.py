"""PC side: a nightly copy of Братан's «soul» — character, memory, faces, album, doors and map pose — on the PC.

09.10 audit: a broken phone would erase the personality. The cloud model is a replaceable cortex; the character,
the memory and the faces live at home. The video agent (tools/video_agent.py) runs this once a night; by hand:
python tools/soul_backup.py

Copies are never deleted (about 3 MB a night). The memory is copied with SQLite's backup API (consistent while
the robot writes); mind.json and the rest are written atomically by the robot, so a plain copy is whole.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pull_videos  # noqa: E402

NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
TARGET = Path(os.environ.get('BRATAN_SOUL_TARGET', 'E:/RobotBrain-soul' if Path('E:/').exists()
                             else Path.home() / 'Documents' / 'RobotBrain-soul'))
DUE_S = 20 * 3600          # one copy a night
LATE_S = 30 * 3600         # the PC was off at night: copy whenever it can
SNAPSHOT = r'''set -e
rm -rf /tmp/bratan-soul /tmp/bratan-soul.tgz
mkdir -p /tmp/bratan-soul
cd /opt/robotbrain/bratan
/opt/robotbrain/venvs/perception/bin/python - <<'PY'
import sqlite3
source = sqlite3.connect('file:state/memory.sqlite?mode=ro', uri=True)
copy = sqlite3.connect('/tmp/bratan-soul/memory.sqlite')
source.backup(copy)
copy.close()
source.close()
PY
for f in state/mind.json state/doors.json state/location.json state/dock-reference.json config.json \
         /opt/robotbrain/state/faces.json; do
    if [ -f "$f" ]; then cp "$f" /tmp/bratan-soul/; fi
done
if [ -d state/album ]; then cp -r state/album /tmp/bratan-soul/album; fi
tar czf /tmp/bratan-soul.tgz -C /tmp/bratan-soul .
rm -rf /tmp/bratan-soul
echo ready
'''


def last_copy():
    copies = sorted(TARGET.glob('bratan-soul-*.tgz'), key=lambda p: p.stat().st_mtime)
    return copies[-1] if copies else None


def due(now=None):
    """Time for tonight's copy: none for 20 h and it is night (2-9), or none for 30 h at all."""
    now = time.time() if now is None else now
    last = last_copy()
    age = now - last.stat().st_mtime if last else 1e9
    return age > LATE_S or (age > DUE_S and 2 <= time.localtime(now).tm_hour < 9)


def backup():
    """Make the copy; returns its path (or raises)."""
    TARGET.mkdir(parents=True, exist_ok=True)
    # Bytes, not text: on Windows a text pipe turns the script's \n into \r\n and sh fails on «set -e\r».
    made = subprocess.run(pull_videos.SSH + ['sh -s'], input=SNAPSHOT.encode(), capture_output=True, timeout=300,
                          creationflags=NO_WINDOW)
    if made.returncode != 0 or b'ready' not in made.stdout:
        raise RuntimeError(f'snapshot failed: {made.returncode} {made.stderr.decode(errors="replace")[-300:]}')
    name = time.strftime('bratan-soul-%Y%m%d-%H%M.tgz')
    partial = TARGET / (name + '.part')
    copied = subprocess.run(pull_videos.SCP + ['phone:/tmp/bratan-soul.tgz', str(partial)], capture_output=True,
                            text=True, timeout=600, creationflags=NO_WINDOW)
    if copied.returncode != 0 or not partial.exists() or partial.stat().st_size < 1000:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f'copy failed: {copied.returncode} {copied.stderr[-300:]}')
    final = TARGET / name
    partial.replace(final)
    subprocess.run(pull_videos.SSH + ['rm -f /tmp/bratan-soul.tgz'], timeout=60, creationflags=NO_WINDOW)
    return final


if __name__ == '__main__':
    print(backup())
