"""On the phone, A/B of the ears (C51, 17.10): the same speech through the phone's microphone and the body's array,
both sent to Microsoft's recognition phrase by phrase. Prints what each heard and, against the reference phrases, how
many words each got right — the owner decides by numbers which microphone hears words (config ears.asr).

  cd /opt/robotbrain/bratan/app
  PYTHONPATH=. /opt/robotbrain/venvs/perception/bin/python -m bratan.voice.ab 45 1m        # record 45 s, then compare
  PYTHONPATH=. /opt/robotbrain/venvs/perception/bin/python -m bratan.voice.ab --files 1m   # compare a recording

Read the phrases (printed at the start) with a pause between them; one run per distance (1, 2, 4 m) and side.
The robot's voice must be running (it records both microphones: voice.ab).
"""
import json
import sys
import time
import wave

import numpy as np

from ..common import bus
from ..common.config import STATE, load, secret
from .hearing import Hearing, normalize

# Без имени и без команд ехать (18.10, владелец: «запрети ему ехать, иначе он тупо поедет»): робот не просыпается
# и не отвечает — сравниваются только микрофоны. Первый прогон 1 м (ab1m2) был со старым списком.
PHRASES = ['Привет, как у тебя дела?', 'Какая сегодня погода на улице?', 'Расскажи что-нибудь интересное про космос.',
           'Сколько сейчас времени?', 'Поставь будильник на семь утра.', 'Что ты сегодня видел интересного?',
           'Включи спокойную музыку.', 'Таффа сегодня гуляла два раза.', 'Завтра будет холодно и пойдёт дождь.',
           'Спасибо, ты молодец.']
OLD_PHRASES = ['Братан, привет, как дела?', 'Поехали на кухню.', 'Найди Таффу и посмотри, что она делает.',
               'Братан, стоп!', 'Братан, иди ко мне.']
FRAME = 320


def read(path):
    with wave.open(str(path), 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), '<i2')


def segments(tracks, pad_s=.25, gap_s=.35, min_s=.4):
    """Phrases: 20 ms frames 12 dB above either track's own floor, gaps under 0.35 s merged."""
    n = min(len(t) for t in tracks) // FRAME
    active = np.zeros(n, bool)
    for t in tracks:
        x = t[:n * FRAME].astype(np.float32).reshape(n, FRAME)
        db = 10 * np.log10((x * x).mean(axis=1) + 1.)
        active |= db > np.percentile(db, 20) + 12.
    spans, start, last = [], None, None
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            elif (i - last) * .02 > gap_s:
                spans.append((start, last))
                start = i
            last = i
    if start is not None:
        spans.append((start, last))
    pad = int(pad_s / .02)
    return [(max(0, s - pad), min(n, e + 1 + pad)) for s, e in spans if (e - s + 1) * .02 >= min_s]


def heard(hearing, track, span, tries=3):
    """Microsoft's text for one phrase. 18.10: after ~13 quick requests in a row the service refused (429) and the
    rest came back empty — so a pause between requests and, on a refusal, a wait and another try."""
    s, e = span
    frames = [track[i * FRAME:(i + 1) * FRAME].tobytes() for i in range(s, e)]
    for attempt in range(tries):
        time.sleep(.8)
        hearing.down_until = 0.                        # a refusal here is a pace problem, not a dead key
        utterance = hearing.start(frames)
        text = utterance.result(wait=15.) if utterance is not None else None
        if text is not None:
            return text
        time.sleep(5. * (attempt + 1))
    return None


def words(text):
    return normalize(text or '').split()


def accuracy(said, reference):
    """Share of the reference's words heard right (1 − WER, floored at 0)."""
    ref, hyp = words(reference), words(said)
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d = d, np.zeros(len(hyp) + 1, int)
        d[0] = i
        for j, h in enumerate(hyp, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r != h))
    return max(0., 1. - d[-1] / max(1, len(ref)))


def best_reference(*texts):
    def overlap(phrase):
        ref = set(words(phrase))
        return max(len(ref & set(words(t))) / max(1, len(ref)) for t in texts)
    phrase = max(PHRASES + OLD_PHRASES, key=overlap)
    return phrase if overlap(phrase) >= .34 else None


def record(seconds, name):
    print('Читай фразы с паузой между ними:', flush=True)
    for i, p in enumerate(PHRASES, 1):
        print(f'  {i:2d}. {p}', flush=True)
    events = []
    client = bus.Client('ab', ('voice.event',), lambda topic, data: events.append(data)).start()
    client.wait_connected(3)
    client.publish('voice.ab', {'seconds': seconds, 'name': name})
    print(f'пишу {seconds:.0f} с…', flush=True)
    deadline = time.time() + seconds + 15
    while time.time() < deadline:
        saved = [e for e in events if e.get('kind') == 'ab_saved' and e.get('name') == name]
        if saved:
            return saved[0]
        time.sleep(.2)
    sys.exit('голос робота не прислал запись (он запущен? плата ушей в сети?)')


def live_sum(x):
    """The whole recording through the robot's own live sum (voice/beam.py Beam), the reference microphone chosen as
    the robot chooses it (ArrayLink._pick); put back on the recording's time line (the sum lags by Beam.LAG)."""
    from .array import ArrayLink
    from .beam import Beam
    link, beam = ArrayLink(None, voice='multi'), Beam()
    out = [beam.feed(x[i:i + 512], link._pick(x[i:i + 512])) for i in range(0, len(x) - 511, 512)]
    track = np.concatenate(out + [np.zeros(len(x) - 512 * len(out), np.int16)])
    return np.concatenate([track[Beam.LAG:], np.zeros(Beam.LAG, np.int16)])


def beam_compare(name, hearing):
    """C52: the same phrases through the ears as the robot hears them now (the board's own pick, PCM), the best single
    microphone of the three, their delay-and-sum, and each of those cleaned of the hum — which one Microsoft hears
    best. Needs <name>-multi.npz (A/B records it since 18.10)."""
    from .beam import delay_and_sum, delays
    from .denoise import denoise_track
    folder = STATE / 'ab'
    voice = read(folder / f'{name}-array.wav')
    multi = np.load(folder / f'{name}-multi.npz')
    x, t0 = multi['x'], float(multi['t0'])
    spans = segments([voice])
    print(f'фраз найдено: {len(spans)}; три микрофона: {len(x) / 16000:.1f} с, сдвиг {t0:+.2f} с', flush=True)
    variants = ('доска', 'лучший', 'сумма', 'сумма живая', 'доска+чистка', 'сумма+чистка')
    live = live_sum(x)
    score = {v: [] for v in variants}
    clean_voice = denoise_track(voice)
    for k, (s, e) in enumerate(spans, 1):
        a, b = int((s * .02 - t0) * 16000), int((e * .02 - t0) * 16000)
        if a < 0 or b > len(x) or b - a < 8000:
            continue
        part = x[a:b]
        d, ref = delays(part)
        noise = np.array([np.percentile(np.sqrt((part[:len(part) // 512 * 512, ch].astype(float).reshape(-1, 512) ** 2)
                                                .mean(axis=1)), 20) for ch in range(3)]) + 1e-6
        snr = part.astype(float).std(axis=0) / noise
        best = part[:, int(np.argmax(snr))].copy()
        summed = delay_and_sum(part, d)
        tracks = {'доска': voice[s * FRAME:e * FRAME], 'лучший': best, 'сумма': summed, 'сумма живая': live[a:b],
                  'доска+чистка': clean_voice[s * FRAME:e * FRAME], 'сумма+чистка': denoise_track(summed)}
        texts = {}
        for variant, track in tracks.items():
            frames_n = len(track) // FRAME
            texts[variant] = heard(hearing, track[:frames_n * FRAME], (0, frames_n))
        reference = best_reference(*texts.values())
        line = f'{k:2d}. {s * .02:5.1f} с, задержки {np.round(d, 1)} (опора М{ref + 1}) | ' + \
               ' | '.join(f'{v}: {texts[v]!r}' for v in variants)
        if reference:
            for v in variants:
                score[v].append(accuracy(texts[v], reference))
            line += f' | эталон {reference!r}'
        print(line, flush=True)
    if score['доска']:
        print(f'\nИТОГ по {len(score["доска"])} фразам с эталоном (верно слов / фраз целиком):', flush=True)
        for v in variants:
            print(f'  {v:14s} {np.mean(score[v]):.0%} / {sum(s == 1 for s in score[v])}', flush=True)


def main(argv):
    if '--beam' in argv:
        config = load()
        hearing = Hearing(secret(config['azure_speech_key']), secret(config['azure_speech_region']))
        beam_compare(argv[argv.index('--beam') + 1], hearing)
        print('распознавание:', json.dumps(hearing.stats, ensure_ascii=False), flush=True)
        return
    if '--files' in argv:
        name = argv[argv.index('--files') + 1]
    else:
        seconds = float(argv[0]) if argv else 45.
        name = argv[1] if len(argv) > 1 else time.strftime('%H%M%S')
        saved = record(seconds, name)
        print('записано кадров:', json.dumps(saved.get('frames'), ensure_ascii=False), flush=True)
    folder = STATE / 'ab'
    phone, array = read(folder / f'{name}-phone.wav'), read(folder / f'{name}-array.wav')
    config = load()
    hearing = Hearing(secret(config['azure_speech_key']), secret(config['azure_speech_region']))
    spans = segments([phone, array])
    print(f'фраз найдено: {len(spans)}', flush=True)
    score = {'phone': [], 'array': []}
    for k, span in enumerate(spans, 1):
        texts = {source: heard(hearing, track, span) for source, track in (('phone', phone), ('array', array))}
        reference = best_reference(*texts.values())
        line = f'{k:2d}. {span[0] * .02:5.1f}–{span[1] * .02:5.1f} с | телефон: {texts["phone"]!r} | уши: {texts["array"]!r}'
        if reference:
            for source in score:
                score[source].append(accuracy(texts[source], reference))
            line += f' | эталон: {reference!r} → {score["phone"][-1]:.0%} / {score["array"][-1]:.0%}'
        print(line, flush=True)
    if score['phone']:
        print(f'\nИТОГ по {len(score["phone"])} фразам с эталоном: верно слов — телефон {np.mean(score["phone"]):.0%}, '
              f'уши {np.mean(score["array"]):.0%}; фраз целиком — телефон '
              f'{sum(s == 1 for s in score["phone"])}, уши {sum(s == 1 for s in score["array"])}', flush=True)
    print('распознавание:', json.dumps(hearing.stats, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
