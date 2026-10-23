"""Hearing when OpenAI is down: Microsoft's speech recognition (Azure, the same key as the voice).

The offline Vosk model misheard a lot (07.10: «бросаем привет», «никуда больше не девять»); the owner
asked to work on Microsoft. The audio is streamed while the person speaks (chunked upload), so the
text is ready ~0.1 s after the phrase ends (measured 0.06 s). Vosk still decides the name and the end
of the phrase, and stays the answer whenever the cloud is slow or refuses.
"""
from __future__ import annotations

import http.client
import json
import queue
import re
import ssl
import struct
import threading
import time

PATH = '/speech/recognition/conversation/cognitiveservices/v1?language=ru-RU&format=simple&profanity=raw'
RESULT_WAIT_S = 1.5           # after the phrase ended; longer and Vosk's text is used instead
DOWN_S = 120.                 # after a refusal (key, quota): do not try again for a while
MAX_FRAMES = 50 * 15          # a phrase is never longer; the service takes up to 60 s
_PUNCT = re.compile(r'[^\w\s-]+')


def _wav_header():
    """A WAV header of unknown length: the audio follows as it is heard."""
    return (b'RIFF' + struct.pack('<I', 0xFFFFFFFF - 8) + b'WAVEfmt ' +
            struct.pack('<IHHIIHH', 16, 1, 1, 16000, 32000, 2, 16) +
            b'data' + struct.pack('<I', 0xFFFFFFFF - 44))


def normalize(text):
    """«Братан, ты меня слышишь?» -> «братан ты меня слышишь»: the form the name checks expect."""
    return ' '.join(_PUNCT.sub(' ', text.lower().replace('ё', 'е')).split())


# A name Microsoft does not know comes back as what it sounds like (18.10 A/B: «Таффа сегодня гуляла» → «дафа сегодня
# гулял», «тафа»): a word that sounds the same (voiced = unvoiced, a double letter = one, о = а) with a noun's ending
# is the name in that case — «дафу» → «таффу». Only these endings: «давай» must stay «давай».
_SOUND = str.maketrans({'д': 'т', 'б': 'п', 'г': 'к', 'з': 'с', 'в': 'ф', 'ж': 'ш', 'о': 'а'})
_ENDINGS = ('', 'а', 'у', 'е', 'ы', 'и', 'ой', 'ою', 'ей', 'ю', 'я', 'ом', 'ем')
_STEM = re.compile(r'^(.*?)([аеиоуыэюяй]*)$')


def _sounds(word):
    return re.sub(r'(.)\1+', r'\1', word.translate(_SOUND))


def fix_names(text, names):
    """Words that sound like one of `names` (in any case ending) written as that name."""
    if not text or not names:
        return text
    out = []
    for word in text.split():
        fixed = word
        for name in names:
            name = name.casefold().replace('ё', 'е')
            stem, _ = _STEM.match(name).groups()
            w_stem, ending = _STEM.match(word).groups()
            if len(stem) >= 3 and word != name and ending in _ENDINGS and _sounds(w_stem) == _sounds(stem):
                fixed = stem + ending
                break
        out.append(fixed)
    return ' '.join(out)


class Hearing:
    def __init__(self, key, region):
        self.key = key
        self.host = f'{region}.stt.speech.microsoft.com'
        self.context = ssl.create_default_context()
        self.down_until = 0.
        self.stats = {'phrases': 0, 'failed': 0, 'late': 0, 'again': 0, 'last_error': None, 'last_after_end_s': None}
        self.names = []                  # names the service mishears, written right (fix_names): the dog's

    def start(self, frames=()):
        """Begin streaming one phrase; None while the service is refusing."""
        if time.monotonic() < self.down_until:
            return None
        return Utterance(self, frames)


class Utterance:
    def __init__(self, hearing, frames=()):
        self.hearing = hearing
        self.frames: queue.Queue = queue.Queue()
        self.kept = []                  # what was sent: heard again if the answer was «nothing» (see `again`)
        self.count = 0
        self.closed = False
        self.ended_at = None
        self.text = None
        self.done = threading.Event()
        for frame in frames:
            self.feed(frame)
        threading.Thread(target=self._run, name='hearing', daemon=True).start()

    def feed(self, frame):
        if not self.closed and self.count < MAX_FRAMES:
            self.frames.put(frame)
            self.kept.append(frame)
            self.count += 1

    def again(self):
        """The same audio once more, all at once (18.10: «nothing» came back for phrases Vosk heard clearly —
        «какая погода на улице», «расскажи мне пожалуйста»)."""
        self.hearing.stats['again'] += 1
        utterance = self.hearing.start(self.kept)
        if utterance is not None:
            utterance.finish()
        return utterance

    def finish(self):
        """The phrase ended: send what is left and let the service answer."""
        if not self.closed:
            self.closed = True
            self.ended_at = time.monotonic()
            self.frames.put(None)

    cancel = finish                 # an empty phrase costs nothing more than its seconds

    def result(self, wait=RESULT_WAIT_S):
        """The recognized text (normalized), '' for nothing heard, None if not in time or failed."""
        self.finish()
        if not self.done.wait(wait):
            self.hearing.stats['late'] += 1
            return None
        return self.text

    def _body(self):
        yield _wav_header()
        while True:
            try:
                frame = self.frames.get(timeout=20.)
            except queue.Empty:
                return                      # never finished (dropped on the way): close the request
            if frame is None:
                return
            yield frame

    def _run(self):
        hearing = self.hearing
        try:
            conn = http.client.HTTPSConnection(hearing.host, 443, timeout=20, context=hearing.context)
            conn.request('POST', PATH, self._body(), {
                'Ocp-Apim-Subscription-Key': hearing.key, 'Accept': 'application/json',
                'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000'}, encode_chunked=True)
            response = conn.getresponse()
            body = response.read()
            conn.close()
            if response.status != 200:
                hearing.stats['failed'] += 1
                hearing.stats['last_error'] = f'http {response.status}'
                if response.status in (401, 403, 429):
                    hearing.down_until = time.monotonic() + DOWN_S
                return
            data = json.loads(body.decode('utf-8'))
            status = data.get('RecognitionStatus')
            if status == 'Success':
                self.text = fix_names(normalize(data.get('DisplayText') or ''), hearing.names)
            elif status in ('NoMatch', 'InitialSilenceTimeout', 'BabbleTimeout'):
                self.text = ''
            else:
                hearing.stats['last_error'] = status
            hearing.stats['phrases'] += 1
            if self.ended_at is not None:
                hearing.stats['last_after_end_s'] = round(time.monotonic() - self.ended_at, 3)
        except Exception as error:                   # network: Vosk's text is used
            hearing.stats['failed'] += 1
            hearing.stats['last_error'] = f'{type(error).__name__}: {error}'[:120]
            self.closed = True
        finally:
            self.done.set()
