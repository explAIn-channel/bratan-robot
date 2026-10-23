"""Проверка «ушей» как у робота, но на ПК: тот же модуль bratan/voice/array.py — подписка на плату, голос кадрами по
20 мс, три микрофона для направления. Пишет здоровье потоков и направление звука раз в секунду.
    python tools/ears_check.py [секунд] [--ip 192.168.0.N] [--wav]    (--wav: сохранить голос в recordings/)"""
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'bratan'))
from bratan.voice import array  # noqa: E402

seconds = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace('.', '').isdigit() else 15.
ip = sys.argv[sys.argv.index('--ip') + 1] if '--ip' in sys.argv else (ROOT / '.ushi-ip').read_text().strip()
frames, gaps = [], [0]


def on_voice(mono, frame):
    if frame is None:
        gaps[0] += 1
    else:
        frames.append(frame)


link = array.ArrayLink(ip, on_voice=on_voice).start()
direction = array.Direction(link.ring)
start = time.time()
while time.time() - start < seconds:
    time.sleep(1.)
    s = link.summary()
    found = direction.estimate(1.)
    where = (f"{found['deg']:+6.1f}° {found['side']:<14} увер. {found['confidence']:.2f} громче {found['loudest']} "
             f"над фоном {found['excess']} τ23 {found['tau23']}") if found else 'тихо'
    print(f"{time.time() - start:4.0f} с | голос: потери {s['voice_loss'] * 100:4.1f} %, слот {s['voice_slot']}, "
          f"кадров {len(frames)}, разрывов {gaps[0]} | 3 мик.: потери {s['multi_loss'] * 100:4.1f} % | {where}",
          flush=True)
link.stop()
if '--wav' in sys.argv and frames:
    out = ROOT / 'recordings' / f'ears-voice-{time.strftime("%Y%m%d-%H%M%S")}.wav'
    out.parent.mkdir(exist_ok=True)
    with wave.open(str(out), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(array.RATE)
        w.writeframes(b''.join(frames))
    print('голос записан:', out.relative_to(ROOT))
