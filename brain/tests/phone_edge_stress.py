"""On the phone: the backup Microsoft voice under real use — phrases at random pauses, first-sound times, hedges, failures."""
import random, sys, time
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '/opt/robotbrain/bratan/app')
from bratan.voice.edge import EdgeSpeech


class Sink:
    def __init__(self): self.bytes = 0
    def play(self, pcm): self.bytes += len(pcm)


phrases = ['Слышу тебя.', 'Я на базе, заряжаюсь.', 'Таффу сегодня видел в коридоре, он лежал у двери.',
           'Сейчас посмотрю, что там на кухне.', 'Хорошо, еду на базу.', 'Не понял, повтори, пожалуйста.',
           'Вижу тебя, ты стоишь у окна.', 'Да вроде нормально, никуда не спешу.']
sink = Sink()
edge = EdgeSpeech(sink)
time.sleep(4)                                  # channels open, ready lines cached
firsts = []
count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
for i in range(count):
    pause = random.choice([0, 1, 3, 8, 15, 25, 40])
    time.sleep(pause)
    text = random.choice(phrases)
    before = sink.bytes
    first = edge.say(text)
    firsts.append(first)
    print(f'{i:2d} after {pause:2d}s first {"FAIL " if first is None else f"{first:.3f}s"} '
          f'audio {(sink.bytes - before) / 32000:.2f}s  {text[:28]}', flush=True)
ok = sorted(f for f in firsts if f is not None)
print("stats", edge.stats)
for t in edge.trace: print("  ", t)
if ok:
    print(f'first sound: median {ok[len(ok) // 2]:.3f}s  worst {ok[-1]:.3f}s  failures {firsts.count(None)}/{len(firsts)}')
