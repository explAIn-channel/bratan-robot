"""How Братан feels right now: a living mood, not a label (core affect: pleasure and arousal).

Russell's circumplex (valence x arousal) as used in Kismet, aibo and later social robots: what happens
pushes the mood (a pat, a «молодец», Taffa running in, a bump, being ignored), and between events it drifts
back to his own temperament — calm and kind by nature, sleepier at night. Kept across restarts.

Praise and scolding are heard in what people say to him (aibo reacts to «good boy»); who praises and pets
him most becomes dearer (bonds count them).
"""
from __future__ import annotations

import math
import re
import time

VALENCE_HALF_LIFE_S = 20 * 60
AROUSAL_HALF_LIFE_S = 8 * 60

# event kind -> (valence push, arousal push)
IMPULSES = {
    'praise': (.35, .2), 'scold': (-.3, .15), 'touch': (.2, .05), 'dog_appeared': (.15, .25),
    'person_appeared': (.08, .15), 'conversation_start': (.1, .1), 'action_done': (.03, .02),
    'action_failed': (-.06, .05), 'refused': (-.02, 0.), 'bump': (-.1, .2), 'startle': (-.05, .4),
    'carried': (-.05, .3), 'door_opened': (.05, .15), 'game_won': (.3, .3), 'game_played': (.25, .25),
    'dreamed': (.1, 0.), 'lost': (-.1, .2), 'found': (.08, 0.), 'boop': (.15, .2), 'peekaboo': (.15, .15),
    'dog_photo': (.05, .05), 'waved': (.15, .15), 'cheered': (.2, .25), 'smiled': (.08, .03),
    'looks_sad': (-.06, 0.), 'handled': (.02, .2),
}
SOUND_IMPULSES = {'laughter': (.12, .15), 'music': (.1, .15), 'dog': (.05, .15), 'doorbell': (.05, .2),
                  'bang': (-.05, .25), 'crying': (-.1, .15), 'clapping': (.1, .15)}

# Only what is said about him: «да нормально, всё хорошо» is an answer, not praise (08.10 19:27), «мне плохо»
# is not a scolding. Words that mean praise or scolding on their own, or «ты ...»/«... робот» around a quality.
QUALITY_GOOD = r'(хорош\w*|класс\w*|крут\w*|мил\w*|лучш\w*|умн\w*|супер\w*|славн\w*|чудесн\w*|замечательн\w*)'
QUALITY_BAD = r'(плох\w*|глуп\w*|туп\w*|противн\w*|вредн\w*|негодн\w*|ужасн\w*)'
PRAISE = re.compile(r'\b(молод\w*|умниц\w*|умничк\w*|красав\w*|золотце|котик|зайка|солнышко|хвалю|спасибо|'
                    r'люблю тебя|обожаю тебя|ты\s+(у меня\s+)?(такой\s+|самый\s+|очень\s+)?' + QUALITY_GOOD + r'|'
                    + QUALITY_GOOD + r'\s+(мальчик|робот|парень|братан))\b', re.I)
SCOLD = re.compile(r'\b(фу+|дурак\w*|бестолоч\w*|отстань|уйди|заткнись|надоел\w*|'
                   r'ты\s+(такой\s+|очень\s+)?' + QUALITY_BAD + r'|' + QUALITY_BAD + r'\s+(мальчик|робот|братан))\b', re.I)


def heard_tone(text):
    """'praise', 'scold' or None for what someone said to the robot."""
    text = str(text or '').casefold()
    if not text:
        return None
    praise, scold = bool(PRAISE.search(text)), bool(SCOLD.search(text))
    if praise and not scold and not re.search(r'\bне\s+(молод|умни|хорош)', text):
        return 'praise'
    if scold and not praise:
        return 'scold'
    return None


class Affect:
    def __init__(self, state, config=None):
        self.state = state
        self.config = config or {}
        saved = state.get('affect') or {}
        self.valence = float(saved.get('v', self.base_valence()))
        self.arousal = float(saved.get('a', .4))
        self.at = float(saved.get('t', time.time()))

    def base_valence(self):
        # Temperament: «спокойный, тёплый, в целом оптимистичный» (persona) — a little above neutral.
        return .25

    def base_arousal(self, now):
        hour = time.localtime(now).tm_hour
        return .15 if hour < 7 or hour >= 23 else .4

    def _decay(self, now):
        dt = max(0., now - self.at)
        self.at = now
        if not dt:
            return
        kv = .5 ** (dt / VALENCE_HALF_LIFE_S)
        ka = .5 ** (dt / AROUSAL_HALF_LIFE_S)
        self.valence = self.base_valence() + (self.valence - self.base_valence()) * kv
        self.arousal = self.base_arousal(now) + (self.arousal - self.base_arousal(now)) * ka

    def push(self, dv, da, now=None):
        now = time.time() if now is None else now
        self._decay(now)
        # Diminishing returns near the ends: joy upon joy adds less (habituation).
        self.valence = max(-1., min(1., self.valence + dv * (1 - abs(self.valence) * .5)))
        self.arousal = max(0., min(1., self.arousal + da * (1 - self.arousal * .5)))
        self.save()

    def on_event(self, kind, data=None):
        data = data or {}
        if kind in ('sound', 'sound_note'):
            impulse = SOUND_IMPULSES.get(data.get('sound'))
        else:
            impulse = IMPULSES.get(kind)
        if kind == 'person_appeared' and data.get('who') not in (None, 'человек'):
            impulse = (.15, .2)                  # one of ours, not a stranger
        if impulse:
            self.push(*impulse)

    def save(self):
        self.state.set('affect', {'v': round(self.valence, 3), 'a': round(self.arousal, 3), 't': round(self.at, 1)})

    def now(self, now=None):
        now = time.time() if now is None else now
        self._decay(now)
        return self.valence, self.arousal

    def words(self, now=None):
        v, a = self.now(now)
        if a < .18:
            return 'сонный' + (', но довольный' if v > .2 else '')
        if v >= .5:
            return 'весёлый, игривый' if a >= .5 else 'довольный, на душе тепло'
        if v >= .15:
            return 'бодрый, любопытный' if a >= .55 else 'спокойный, в хорошем настроении'
        if v > -.15:
            return 'немного взбудоражен' if a >= .6 else 'задумчивый'
        if v > -.45:
            return 'расстроен, взвинчен' if a >= .55 else 'немного грустно'
        return 'обидно и тревожно' if a >= .5 else 'грустно'

    def face(self, now=None):
        """An idle face that matches the mood (when nothing else is shown)."""
        v, a = self.now(now)
        if a < .18:
            return 'sleepy'
        if v >= .5:
            return 'happy'
        if v <= -.3:
            return 'concerned'
        return 'curious' if a >= .5 else 'idle'

    def text(self, now=None):
        v, a = self.now(now)
        return f'{self.words(now)} (радость {v:+.1f}, бодрость {a:.1f})'
