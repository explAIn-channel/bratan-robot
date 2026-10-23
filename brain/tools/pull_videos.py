"""PC side: pull finished trip videos from the phone, convert to MP4, free the phone.

python tools/pull_videos.py   (run by a Windows scheduled task every 30 minutes)
"""
import os
import shutil
import subprocess
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SSH = ['ssh', '-F', str(HERE.parents[1] / 'access' / 'ssh_config'), 'phone']
SCP = ['scp', '-F', str(HERE.parents[1] / 'access' / 'ssh_config')]
TARGET = Path(os.environ.get('BRATAN_VIDEO_TARGET', Path.home() / 'Videos' / 'Bratan'))
CAP_BYTES = 10 * 1024 ** 3
FFMPEG = shutil.which('ffmpeg') or str(Path.home() / 'AppData/Local/Microsoft/WinGet/Packages/'
                                       'Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0-full_build/bin/ffmpeg.exe')


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(SSH + ['ls -1 /opt/robotbrain/bratan/videos/ 2>/dev/null'], capture_output=True,
                             text=True, timeout=60, creationflags=NO_WINDOW).stdout.split()
    done = [name for name in listing if name.endswith('.avi')]
    for name in sorted(done):
        local = TARGET / name
        mp4 = local.with_suffix('.mp4')
        if not mp4.exists():
            if subprocess.run(SCP + [f'phone:/opt/robotbrain/bratan/videos/{name}', str(local)],
                              timeout=900, creationflags=NO_WINDOW).returncode != 0:
                continue
            result = subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-i', str(local), '-c:v', 'libx264',
                                     '-preset', 'veryfast', '-crf', '26', '-pix_fmt', 'yuv420p', str(mp4)],
                                    timeout=1800, creationflags=NO_WINDOW)
            if result.returncode != 0 or not mp4.exists() or mp4.stat().st_size < 1000:
                continue
            local.unlink(missing_ok=True)
        subprocess.run(SSH + [f'rm -f /opt/robotbrain/bratan/videos/{name}'], timeout=60, creationflags=NO_WINDOW)
        print('pulled', mp4.name, flush=True)
    # 720p trips with sound from the face app: copied as they are (already H.264 + AAC).
    android = '/proc/1/root/data/media/0/Android/data/ai.robotbrain.face/files/trips'
    listing = subprocess.run(SSH + [f'ls -1 {android}/ 2>/dev/null'], capture_output=True, text=True, timeout=60,
                             creationflags=NO_WINDOW).stdout.split()
    for name in sorted(n for n in listing if n.endswith('.mp4')):
        local = TARGET / name
        if not local.exists():
            partial = local.with_suffix('.part')
            if subprocess.run(SCP + [f'phone:{android}/{name}', str(partial)], timeout=1800,
                              creationflags=NO_WINDOW).returncode != 0:
                continue
            partial.replace(local)
        subprocess.run(SSH + [f'rm -f {android}/{name}'], timeout=60, creationflags=NO_WINDOW)
        print('pulled', name, flush=True)
    files = sorted(TARGET.glob('trip-*.mp4'), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    while files and total > CAP_BYTES:
        oldest = files.pop(0)
        total -= oldest.stat().st_size
        oldest.unlink()


if __name__ == '__main__':
    sys.exit(main())
