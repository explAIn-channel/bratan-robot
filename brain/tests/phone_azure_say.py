"""On the phone: AzureSpeech.say — first sound with the first clause fetched in parallel."""
import sys, time
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '/opt/robotbrain/bratan/app')
from bratan.voice.azure import AzureSpeech

class Sink:
    def __init__(self): self.bytes = 0
    def play(self, pcm): self.bytes += len(pcm)

key = open('/opt/robotbrain/secrets/azure-speech-key').read().strip()
region = open('/opt/robotbrain/secrets/azure-speech-region').read().strip()
sink = Sink()
azure = AzureSpeech(sink, key, region)
time.sleep(3)
for text in ['Да?', 'Слышу тебя.', 'Не понял, повтори, пожалуйста.', 'Таффу сегодня видел в коридоре, он лежал у двери.',
             'Да вроде нормально. Стою на кухне, оглядываюсь. Скучаю по Таффе.', 'Хорошо, еду на базу.'] * 2:
    before = sink.bytes
    first = azure.say(text)
    print(f'first {first if first is None else round(first, 3)}s audio {(sink.bytes - before) / 32000:.2f}s  {text}', flush=True)
    time.sleep(2)
print(azure.stats)
