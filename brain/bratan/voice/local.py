"""Offline voice for when the cloud voice is down: Piper speech and open-vocabulary Vosk hearing.

Piper (Russian, ONNX) runs on the voice's big cores: a short phrase is ready in
~0.2 s and it synthesizes about five times faster than it speaks. Text is cut at
punctuation and every piece goes to the speaker as soon as it is ready, so the
first sound comes after the first clause, not the whole answer.
"""
from __future__ import annotations

import json
import re
import threading
import time

import numpy as np

from .audio import RATE, as_heard, rms
from .vad import Talk
from .wake import FILLERS, GATE_FLOOR, is_name

READY_LINES = ('Да?', 'Слушаю.', 'Секунду.')     # synthesized at start: zero delay
PIECES = re.compile(r'[^,.!?…;:—]+[,.!?…;:—]*')


class LocalSpeech:
    def __init__(self, audio, model_path, threads=2):
        self.audio = audio
        self.model_path = model_path
        self.threads = threads
        self.voice = None
        self.cache = {}
        self.ready = threading.Event()
        threading.Thread(target=self._load, name='piper-load', daemon=True).start()

    def _load(self):
        try:
            import onnxruntime
            from piper import PiperVoice
            voice = PiperVoice.load(self.model_path)
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            voice.session = onnxruntime.InferenceSession(self.model_path, sess_options=options,
                                                         providers=['CPUExecutionProvider'])
            for _ in voice.synthesize('Привет.'):       # first run allocates: do it now, not on the first line
                pass
            self.voice = voice
            for line in READY_LINES:
                self.cache[line] = b''.join(self._pieces(line))
            self.ready.set()
        except Exception as error:                     # no offline voice: the robot stays silent instead
            print('piper_unavailable', type(error).__name__, error, flush=True)

    def say(self, text, is_current=lambda: True):
        """Speak; returns seconds until the first sound or None."""
        if not self.ready.wait(5.):
            return None
        started = time.monotonic()
        text = text.strip()
        if text in self.cache:
            self.audio.play(self.cache[text])
            return time.monotonic() - started
        first = None
        for pcm in self._pieces(text, is_current):
            if first is None:
                first = time.monotonic() - started
            self.audio.play(pcm)
        return first

    def _pieces(self, text, is_current=lambda: True):
        """16 kHz PCM, clause by clause, as soon as each is synthesized."""
        from scipy.signal import resample_poly
        rate = self.voice.config.sample_rate
        for piece in PIECES.findall(text) or [text]:
            piece = piece.strip()
            if not piece:
                continue
            for chunk in self.voice.synthesize(piece):
                if not is_current():
                    return
                x = chunk.audio_float_array.astype(np.float64)
                y = resample_poly(x, RATE, rate) if rate != RATE else x
                yield (np.clip(y, -1, 1) * 32767).astype('<i2').tobytes()


class Phrase:
    """One spoken phrase, open vocabulary; `ended` after a 1.2 s pause (or `max_s`).

    Speech/silence: by the speech detector's probabilities when given (voice/vad.py, 18.10: the body's ears hear the
    room only 12-20 dB below a voice), else by loudness against the room's own noise (a kitchen hood is not speech).
    """

    def __init__(self, model, frames=(), noise=.002, max_s=7., probs=None):
        from vosk import KaldiRecognizer
        self.recognizer = KaldiRecognizer(model, RATE)
        self.max_frames = int(50 * max_s)
        # The same floor as the name's gate: a voice from across the room is quieter than 1.5x of it and its
        # syllables counted as silence («устал немного на работе» -> «устном немного на», 08.10 20:24).
        self.threshold = max(GATE_FLOOR, noise * 2.5)
        self.words = []
        self.frames = 0
        self.silence = 0
        self.speech = False
        self.speech_frame = None          # the frame the speech began at
        self.voiced = 0                   # frames of speech
        self.partial_words = 0
        self.talk = Talk()
        probs = list(probs) if probs is not None else []
        for i, frame in enumerate(frames):
            self.feed(frame, probs[i] if i < len(probs) else None)

    def feed(self, frame, speech=None):
        heard = as_heard(frame)
        if self.recognizer.AcceptWaveform(heard):
            self.words += (json.loads(self.recognizer.Result()).get('text') or '').split()
        self.frames += 1
        if (self.talk.feed(speech) if speech is not None else rms(heard) > self.threshold):
            if not self.speech:
                self.speech_frame = self.frames - 1
            self.speech, self.silence = True, 0
            self.voiced += 1
        else:
            self.silence += 1
            if self.silence == 30:                 # a pause: how much has been said so far?
                partial = json.loads(self.recognizer.PartialResult()).get('partial', '')
                self.partial_words = len(self.words) + len(partial.split())

    @property
    def ended(self):
        # 1.2 s of silence ends a turn: between two sentences people pause 0.6-1 s («…в загадки. — Загадай мне
        # что-нибудь», «…в города. — Москва» lost the second, 08.10 21:11); after a short opener («Молодец.»,
        # «Слушай.», «Так.») 1.5 s — the rest usually follows.
        pause = 75 if self.partial_words <= 2 else 60
        return (self.speech and self.silence >= pause) or self.frames >= self.max_frames

    def text(self):
        self.words += (json.loads(self.recognizer.FinalResult()).get('text') or '').split()
        return ' '.join(self.words)


def request(text):
    """The words after the name: «эй братан как дела» -> «как дела». A greeting is kept: «братан привет»
    is answered «Привет!», not «Да?» (08.10)."""
    words = text.split()
    while words and ((words[0] in FILLERS and words[0] not in ('привет', 'хэй')) or is_name(words[0])):
        words.pop(0)
    return ' '.join(words)
