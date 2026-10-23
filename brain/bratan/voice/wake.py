"""Hearing the name and the urgent "стоп", locally with Vosk (Russian).

Stage 1 (`Listener`): an energy gate feeds a tiny grammar decoder; a partial hit
on «братан» fires immediately, mid-sentence, so the live session can open while
the person is still speaking. The same gate feeds a STOP grammar.
Stage 2 (`Confirmer`): an open-vocabulary decoder runs over the same speech from
the gate's pre-roll; at the end of the phrase it confirms the name is an
address (within the first two words), not a word inside other speech.
"""
from __future__ import annotations

import json
from collections import deque

from .audio import FRAME, RATE, as_heard, rms
from .vad import START, Talk

NAME = 'братан'
VARIANTS = frozenset({'братан', 'братана', 'братаном', 'братаны'})
# How the name sounds to the small Vosk model from across a room (18.10, 3-4 m from the corridor: «врата сколько
# сейчас времени»): these wake the ears too, and the check that follows decides — Vosk's own, then Microsoft's (a
# «false wake» is heard by Microsoft before it is dropped). They never count as the name in that check.
SOUNDS_LIKE = frozenset({'брата', 'врата', 'братом', 'братану', 'батан'})
FILLERS = frozenset({'эй', 'слушай', 'ну', 'а', 'о', 'так', 'привет', 'хэй'})
STOP_PHRASES = frozenset({'стоп', 'стой', 'остановись', 'братан стоп', 'братан стой', 'братан остановись'})
GATE_FLOOR = .0018
GATE_FACTOR = 2.
HANGOVER_S = 1.2
PREROLL = 60                       # frames (1.2 s) kept before the gate opens; 18.10: with 0.6 s the gate, opened
                                   # late by the speech detector, cut «Бра-» off and the name was missed (12 calls
                                   # recorded: 10 → 12 heard, no false wake; work/checks/wake_eval.py)
OPEN_AT = START                    # the speech detector's probability that opens the name's gate
STOP_MIN_PEAK = .004
STOP_QUIET_S = .3                  # «стоп» counts when no more voice follows it this long:
STOP_GOES_ON = 6                   # 6 frames (0.12 s) of voice within it are more words (the detector's own tail is
                                   # shorter)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def is_name(word):
    return word in VARIANTS or (len(word) >= 5 and edit_distance(word, NAME) <= 1)


def wakes(word):
    """The name or what it sounds like from afar: enough to listen closer (not to answer)."""
    return is_name(word) or word in SOUNDS_LIKE


def addressed(words):
    """Name among the first two words after fillers («эй, братан», «братан, привет»)."""
    words = [w for w in words if w]
    head = [w for w in words[:4] if w not in FILLERS][:2]
    return any(is_name(w) for w in head) or any(is_name(w) for w in words[:2])


class Listener:
    """One small grammar decoder for both the name and STOP (one acoustic pass per frame)."""

    def __init__(self, model, preroll=PREROLL, floor=GATE_FLOOR, open_at=OPEN_AT):
        """`preroll` — frames kept before the gate opens; `floor` — the loudness gate's lowest threshold (no speech
        detector: the phone's second listener — its stream is filtered almost to silence, a voice from across the room
        is 34-40 dB there); `open_at` — the speech detector's probability that opens the gate."""
        from vosk import KaldiRecognizer
        grammar = sorted(VARIANTS) + ['эй ' + v for v in sorted(VARIANTS)] +             ['слушай ' + v for v in sorted(VARIANTS)] + sorted(STOP_PHRASES) + sorted(SOUNDS_LIKE) + ['[unk]']
        self.model = model
        self.grammar = json.dumps(sorted(set(grammar)), ensure_ascii=False)
        self.decoder = KaldiRecognizer(model, RATE, self.grammar)
        self.decoder.SetWords(True)
        self.noise = .002
        self.open_until = 0.
        self.floor, self.open_at = floor, open_at
        self.preroll = deque(maxlen=preroll)
        self.phrase = []                 # frames since the gate opened (incl. pre-roll)
        self.peak = 0.
        self.speech_peak = 0.            # the speech detector's highest probability in this phrase
        self.cooldown = 0.
        self.stop_cooldown = 0.
        self.named = False
        self.frames_open = 0
        self.frames_total = 0
        self.finals = deque(maxlen=6)    # (mono, text) of the last grammar results: what the ears made of it
        self.name_finals = 0             # results with the name in them (heard, whether or not acted on)

    def renew(self):
        """A fresh grammar decoder; the noise floor and the counters stay (the voice's watchdog, in quiet)."""
        from vosk import KaldiRecognizer
        decoder = KaldiRecognizer(self.model, RATE, self.grammar)
        decoder.SetWords(True)
        self.decoder = decoder
        self.phrase = []
        self.peak = 0.
        self.speech_peak = 0.
        self.named = False
        self.renewed = getattr(self, 'renewed', 0) + 1

    def reset(self):
        self.decoder.Reset()
        self.phrase = []
        self.peak = 0.
        self.speech_peak = 0.
        self.open_until = 0.
        self.named = False

    def feed(self, mono, frame, *, listen_for_name=True, speech=None, strict=False):
        """Returns 'wake', 'stop' or None (the word that woke — in `woke_by`). `strict` — only «братан» itself wakes:
        not what it sounds like from afar, nor «братана» (said about me, not to me) — while we speak our own words
        come back through the microphone (18.10, the barge A/B: «это понятно» over our story became «братана»; the
        owner's real «Братан!» came as «братан»). `self.phrase` holds the audio so far. `speech` — the speech detector's
        probability for this frame (voice/vad.py): with it the gate opens on a voice, not on loudness (18.10: the
        body's ears hear the room 12-20 dB below a voice, a loudness gate was open most of the time)."""
        heard = as_heard(frame)
        level = rms(heard)
        self.frames_total += 1
        pending = getattr(self, 'stop_pending', None)
        if pending is not None:
            if speech is not None and speech >= START:
                self.stop_voiced = getattr(self, 'stop_voiced', 0) + 1
            if getattr(self, 'stop_voiced', 0) >= STOP_GOES_ON:
                self.stop_pending, self.stop_voiced = None, 0     # the words go on: part of something else
                self.stop_dropped = getattr(self, 'stop_dropped', 0) + 1
            elif mono - pending >= STOP_QUIET_S:
                self.stop_pending, self.stop_voiced = None, 0
                return 'stop'
        if mono >= self.open_until and (speech is None or speech < .2):
            # Learn the room quickly when quieter, slowly when louder.
            clipped = min(level, self.noise * 1.35)
            alpha = .06 if clipped < self.noise else .003
            self.noise = min(.04, max(.0005, self.noise * (1 - alpha) + clipped * alpha))
        self.preroll.append(frame)
        threshold = max(self.floor, self.noise * GATE_FACTOR)
        was_open = mono < self.open_until
        if (speech >= self.open_at) if speech is not None else (level >= threshold):
            self.open_until = mono + HANGOVER_S
        if mono >= self.open_until:
            if was_open:
                self.reset()
            return None
        self.frames_open += 1
        if not was_open:
            self.phrase = list(self.preroll)
            for earlier in self.phrase[:-1]:
                self.decoder.AcceptWaveform(as_heard(earlier))
        else:
            self.phrase.append(frame)
        self.peak = max(self.peak, level)
        if speech is not None:
            self.speech_peak = max(getattr(self, 'speech_peak', 0.), speech)
        if len(self.phrase) > 50 * 8:            # 8 s of one phrase: start over
            # Eight seconds without a pause is the room (music, a TV), not a person: lift the noise floor to
            # it, so the gate opens on a voice above it again (08.10: open 95% of the time, the name lost).
            levels = sorted(rms(as_heard(f)) for f in self.phrase)
            self.noise = min(.04, max(self.noise, levels[len(levels) // 2] * .8))
            self.reset()
            return None
        if self.decoder.AcceptWaveform(heard):
            payload = json.loads(self.decoder.Result())
            words = payload.get('result') or []
            text = ' '.join(w.get('word', '') for w in words)
            self.last_text = text
            if text:
                self.finals.append((round(mono, 1), text))
                if any(is_name(w) for w in text.split()):
                    self.name_finals += 1
            # Loud enough to be meant: 5x the room by loudness alone; a clear voice by the detector needs less (the
            # ears' room is loud: «стоп» from 4 m is ~4x of it).
            loud = self.peak >= max(STOP_MIN_PEAK, 5 * self.noise) or (
                getattr(self, 'speech_peak', 0.) >= .8 and self.peak >= max(STOP_MIN_PEAK, 2.5 * self.noise))
            if (text in STOP_PHRASES and all(w.get('conf', 0) >= .85 for w in words)
                    and loud and mono >= self.stop_cooldown):
                self.stop_cooldown = mono + 2.
                if speech is None:
                    # No speech detector to tell more words from quiet: at once, as before it. With the name too the
                    # quiet after it is waited for (18.10 18:33: «Братан, сколько сейчас времени» came as «братан
                    # стой» twice and stopped the body); over my own voice OverMe (Microsoft) hears «стоп» instead.
                    return 'stop'
                # A real «стоп!» is followed by quiet; «сколько сейчас времени» made into «стой» goes on (18.10 A/B
                # at 2 m): stop once STOP_QUIET_S passes with no more voice.
                self.stop_pending, self.stop_voiced = mono, 0
                return None
            woke = [w for w in text.split() if (w == NAME if strict else wakes(w))]
            if listen_for_name and not self.named and mono >= self.cooldown and woke:
                self.cooldown = mono + 1.5
                self.woke_by = woke[0]
                return 'wake'
            self.named = False
            return None
        # Partial hypotheses are the expensive part: read them every 4th frame (80 ms).
        if not listen_for_name or self.named or mono < self.cooldown or self.frames_open % 4:
            return None
        partial = json.loads(self.decoder.PartialResult()).get('partial', '')
        words = partial.split()
        woke = [w for w in words if (w == NAME if strict else wakes(w))]
        if woke and not any(w in ('стоп', 'стой', 'остановись') for w in words):
            self.named = True
            self.cooldown = mono + 1.5
            self.woke_by = woke[0]
            return 'wake'
        return None


class Confirmer:
    """Open-vocabulary decode of one phrase, fed frame by frame.

    The end of the phrase: 1.2 s of silence by the speech detector when there is one, else by loudness against the
    room's noise as the name's gate learned it. (Until 18.10 the noise here was a fixed .002: below the body's ears'
    own noise, so the phrase never ended before its 8 s limit — the robot answered «Братан, привет» 7 s late, with the
    next phrase merged in.)"""

    def __init__(self, model, frames, noise=.002):
        from vosk import KaldiRecognizer
        self.recognizer = KaldiRecognizer(model, RATE)
        self.recognizer.SetWords(True)
        self.name_check = KaldiRecognizer(model, RATE, json.dumps(sorted(VARIANTS) + ['[unk]'], ensure_ascii=False))
        self.name_check.SetWords(True)
        self.frames = 0
        self.final_words = []
        self.silence = 0
        self.speech = False
        self.noise = noise
        self.talk = Talk(speaking=True)              # the name was just heard: someone is speaking
        for frame in frames:
            self.feed(frame)

    def feed(self, frame, speech=None):
        heard = as_heard(frame)
        if self.recognizer.AcceptWaveform(heard):
            self.final_words += [w.get('word', '') for w in json.loads(self.recognizer.Result()).get('result') or []]
        if self.frames < 125:                       # constrained check on the first 2.5 s
            self.name_check.AcceptWaveform(heard)
        self.frames += 1
        if speech is not None:
            self.talk.feed(speech)
            self.speech = True                       # the name was spoken: now only the pause after it counts
            self.silence = self.talk.silence
            return
        level = rms(heard)
        if level > max(GATE_FLOOR, self.noise * 2.5):
            self.speech, self.silence = True, 0
        else:
            self.silence += 1

    def early(self):
        """The name is already heard at the start of the phrase: confirm without waiting."""
        if self.frames % 5:
            return False
        words = self.final_words + json.loads(self.recognizer.PartialResult()).get('partial', '').split()
        return len(words) >= 2 and addressed(words)

    @property
    def ended(self):
        # 1.2 s pause or 8 s: «Братан, привет! Как дела?» and «Братан, давай в загадки. Загадай мне что-нибудь»
        # are one phrase (0.6-0.8 s cut them after the first sentence).
        return (self.speech and self.silence >= 60) or self.frames >= 50 * 8

    def result(self):
        words = json.loads(self.recognizer.FinalResult()).get('result') or []
        heard = self.final_words + [w.get('word', '') for w in words]
        text = ' '.join(heard)
        if addressed(heard):
            return True, text
        named = json.loads(self.name_check.FinalResult()).get('result') or []
        for hit in named:
            if hit.get('word') in VARIANTS and hit.get('conf', 0) >= .5:
                before = sum(1 for w in words if w.get('end', 0) <= hit.get('start', 0) + .05
                             and w.get('word') not in FILLERS)
                return before <= 1, (NAME + ' ' + text).strip()
        return False, text
