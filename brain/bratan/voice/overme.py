"""Someone talking over me (C52, 18.10; the owner: «Братан, стоп не сработало»).

While I speak the small name grammar hears my own voice in the microphone and almost never the person's word under
it: in the barge A/B it caught one «Братан» of several, and in the owner's live test none. Microsoft does hear the
person through my voice — over two-second windows of the phone's microphone during my story it wrote «…вода сама держит
человека ПОДОЖДИ», «…рыбы и креветки БРАТАН киты», and in the recordings with my voice alone not one word to stop on.
So while I speak, the microphone goes to Microsoft window by window; a word to stop on that I am not saying myself
stops me: «стоп», «хватит», «замолчи»… — silence; «братан», «подожди», «погоди» — silence and I listen.

The windows cost Microsoft's per-second price only while I speak to someone (an answer in a talk, my own line); one
request at a time, a refusal makes it rest without touching the conversation's own recognizer.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from difflib import SequenceMatcher

from .hearing import Hearing, normalize

WINDOW_S = 2.0                # each request: this much of the microphone…
STEP_S = 1.2                  # …every this often (a word cut by one window's edge is whole in the next); 18.10: with
                              # 2.6 s / 2 s my short answers (3 s) were over before the owner's «Братан» came back
FRAME_S = .02
LATE_S = 4.                   # an answer this long after its window ended is not acted on (I may be quiet by then)
STOP = re.compile(r'^(стоп\w*|стой|хватит|замолчи|замолкни|тихо|помолчи|заткнись)$')
# «брат»: «Братан» cut by my voice. Not «слушай», «секунду»: Microsoft made «слушай» out of my own «Осьминоги…» at a
# window's start and I cut my own story off (18.10 21:11).
LISTEN = re.compile(r'^(брат|братан\w{0,2}|подожди|погоди)$')
SIMILAR = .6                  # a word this close to one of mine by letters is mine, misheard…
SIMILAR_SHORT = .75           # …a short one closer: «стоп» is 67 % «место»
FOREIGN_MIN = 2               # this many words of someone else's in one window: they are talking to me
CUE_BACK_S = .9               # cut in by a word: heard from this long before the window's end (the word itself)


def cues(text, own_words):
    """The words to stop on in `text` that are none of mine (by sound too: my «стопка» cut to «стоп»). A word at the
    window's very start does not count unless it is alone there: the window cuts my previous word, and what is left
    of it comes back as some other word."""
    words, own = normalize(text or '').split(), set(own_words)
    # …or someone else's words follow it (19.10, the owner's test: «подожди а сколько сейчас» began a window and was
    # thrown away, the question lost)
    led = len(words) == 1 or any(len(w) >= 3 and not w.isdigit() and not _mine(w, own) for w in words[1:])
    return [w for i, w in enumerate(words) if (STOP.match(w) or LISTEN.match(w)) and not _my_cue(w, own)
            and (i > 0 or led)]


def _my_cue(word, own):
    """A stop word is mine only if I said it, or a word it is cut from («стопки» → «стоп»), or one as long and close by
    sound — not a shorter word of mine inside it (19.10: I was saying «дожди» and the owner's «подожди» was taken for
    it, 83 % alike by letters: no barge-in)."""
    for o in own:
        if o == word or (len(o) > len(word) and o.startswith(word)):
            return True
        if abs(len(o) - len(word)) <= 1 and SequenceMatcher(None, word, o).ratio() >= SIMILAR_SHORT:
            return True
    return False


def _mine(word, own):
    """Is `word` one of mine, as Microsoft writes my voice: the same, a cut piece of it, or close by sound."""
    if word in own:
        return True
    for o in own:
        if len(o) >= 3 and (o.startswith(word) or word.startswith(o[:max(3, len(o) - 2)])):
            return True
        if SequenceMatcher(None, word, o).ratio() >= (SIMILAR if len(word) >= 6 else SIMILAR_SHORT):
            return True
    return False


def strip_mine(text, own_words):
    """The person's words without mine in front: heard from the window where they cut in, the phrase starts with the
    end of my own line («…солнечную систему а сколько сейчас времени» → «а сколько сейчас времени»)."""
    words, own = normalize(text or '').split(), set(own_words)
    while words and (_mine(words[0], own) or words[0].isdigit()):
        words.pop(0)
    return ' '.join(words)


def foreign(text, own_words):
    """Someone else's words in a window over my voice (18.10, the owner: «хочу, чтобы его можно было перебивать
    любой фразой»): the words that are none of mine. The window's first and last words are left out — my words cut by
    its edges come back as other words («после», «город») — and so are numbers (I say «три», it writes «3») and
    one- or two-letter words."""
    words = normalize(text or '').split()
    own = set(own_words)
    return [w for w in words[1:-1] if len(w) >= 3 and not w.isdigit() and not _mine(w, own)]


class OverMe:
    def __init__(self, key, region, own_words, on_cue, event):
        self.hearing = Hearing(key, region)      # its own: a refusal here never pauses the talk's recognizer
        self.own_words = own_words               # () -> the words I said lately
        self.on_cue = on_cue                     # (cues, text) — called from a worker thread
        self.event = event
        self.frames = deque(maxlen=int(WINDOW_S / FRAME_S))
        self.last_sent = 0.
        self.busy = False
        self.generation = 0
        self.stats = {'windows': 0, 'cues': 0}

    def stop(self):
        """I stopped speaking: windows still on their way are not acted on."""
        if self.frames:
            self.frames.clear()
            self.generation += 1
            self.prev = (0., 0., ())

    def check_now(self, mono):
        """The name grammar thinks it heard «Братан» or «Братан, стоп» over my voice: it also makes up «братан стой»
        out of my own words, so Microsoft checks the window right now instead of me stopping on its word."""
        self.urgent = True
        self.feed(mono, None)

    def feed(self, mono, frame):
        if frame is not None:
            self.frames.append(frame)
        urgent = getattr(self, 'urgent', False)
        if (self.busy or (mono - self.last_sent < STEP_S and not urgent) or len(self.frames) < self.frames.maxlen // 2
                or time.monotonic() < self.hearing.down_until):
            return
        self.urgent = False
        self.last_sent = mono
        self.busy = True
        threading.Thread(target=self._check, args=(list(self.frames), self.generation, time.monotonic(),
                                                   mono - len(self.frames) * FRAME_S),
                         name='over-me', daemon=True).start()

    def _check(self, frames, generation, sent_at, start):
        try:
            utterance = self.hearing.start(frames)
            text = utterance.result(wait=LATE_S) if utterance is not None else None
            self.stats['windows'] += 1
            self.stats['last'] = text
            self.event('over_me_heard', text=text, took=round(time.monotonic() - sent_at, 2))   # to tune it by
            if not text or generation != self.generation or time.monotonic() - sent_at > LATE_S:
                return
            own = self.own_words()
            found = cues(text, own)
            other = foreign(text, own)
            # Someone else's words of the window before count too (19.10: «подожди а сколько сейчас», «сколько сейчас
            # времени» — one such word in each, the phrase was let go)
            prev_at, prev_start, prev_other = getattr(self, 'prev', (0., start, ()))
            joined = list(dict.fromkeys(other + [w for w in prev_other if sent_at - prev_at < 3 * STEP_S]))
            self.prev = (sent_at, start, other)
            if found or len(joined) >= FOREIGN_MIN:
                self.stats['cues'] += 1
                self.event('over_me', text=text, cues=found, foreign=joined)
                # Cut in by a last word («…подожди», «братан»): what they want follows it — heard from just before the
                # window's end (18.10 20:07: from its start the phrase began with my own «…всё плавится», ended in
                # the pause before the question, and «а сколько сейчас времени» was lost). By a phrase, or a word with
                # more after it: from the window's start (the one before, if it held some of it), my words taken off.
                end = start + len(frames) * FRAME_S
                words = normalize(text).split()
                if found and words and words[-1] in found:
                    at = max(start, end - CUE_BACK_S)
                elif not found and len(other) < FOREIGN_MIN:
                    at = prev_start
                else:
                    at = start
                self.prev = (0., start, ())
                self.on_cue(found or joined, text, at)
        except Exception as error:                   # never in the way of the voice
            self.event('over_me_failed', error=f'{type(error).__name__}: {error}'[:160])
        finally:
            self.busy = False
