"""Emotional sounds without words, R2-D2 style (owner, 08.10): chirps, trills, whistles and warbles.

Synthesised on the spot (no recordings): exponential pitch glides, vibrato, a little third harmonic for a
robotic timbre, soft attack and release (no clicks). Every call varies pitch, timing and the number of notes
a little, so the same feeling never sounds exactly the same — like a voice, not a ringtone.
"""
from __future__ import annotations

import random

import numpy as np

RATE = 16000
LEVEL = .032                   # owner 08.10: quiet — .32 -> .16 -> .064 -> half again (12:25); speech plays at .8


def _tone(f0, f1, seconds, *, vibrato=(0., 0.), harmonic=.25, attack=.012, release=.03, level=1.):
    n = max(1, int(seconds * RATE))
    t = np.arange(n) / RATE
    # Exponential glide f0 -> f1, then vibrato on top.
    freq = f0 * (f1 / f0) ** (t / max(seconds, 1e-6))
    rate, depth = vibrato
    if rate and depth:
        freq = freq * (1 + depth * np.sin(2 * np.pi * rate * t))
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    wave = np.sin(phase) + harmonic * np.sin(3 * phase) * .6 + harmonic * .4 * np.sin(2 * phase)
    env = np.ones(n)
    a, r = min(n, int(attack * RATE)), min(n, int(release * RATE))
    if a:
        env[:a] = np.linspace(0., 1., a)
    if r:
        env[-r:] *= np.linspace(1., 0., r)
    return wave * env * level


def _gap(seconds):
    return np.zeros(int(seconds * RATE))


def _purr(seconds, base=180.):
    """A soft warble: a low tone pulsing ~22 times a second (content, being petted)."""
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    pulse = .55 + .45 * np.sin(2 * np.pi * random.uniform(19, 25) * t)
    wave = np.sin(2 * np.pi * base * t + 2 * np.sin(2 * np.pi * 3 * t)) * pulse
    env = np.minimum(1., np.minimum(t / .15, (seconds - t) / .25))
    return wave * np.clip(env, 0., 1.) * .8


def _u(a, b):
    return random.uniform(a, b)


def _chatter(count=(6, 11), low=1100, high=3000, level=1.):
    """R2-D2's «talking»: a quick run of short random notes with tiny glides (Ben Burtt's ARP 2600 chatter)."""
    notes = []
    for _ in range(random.randint(*count)):
        f = _u(low, high)
        notes += [_tone(f, f * _u(.85, 1.2), _u(.025, .06), harmonic=.15, attack=.004, release=.012, level=level),
                  _gap(_u(.005, .03))]
    return notes


def _warble(f0, f1, seconds, rate=(18, 30), depth=.18, level=1.):
    """A fast frequency-modulated whistle: the bubbly R2 trill."""
    return _tone(f0, f1, seconds, vibrato=(_u(*rate), depth), harmonic=.1, level=level)


def _happy():
    return _chatter((5, 9)) + [_warble(_u(1400, 1800), _u(2200, 2800), _u(.18, .28))]


def _happy_plain():
    notes, f = [], _u(900, 1200)
    for _ in range(random.randint(3, 5)):
        g = f * _u(1.15, 1.35)
        notes += [_tone(f, g, _u(.05, .09)), _gap(_u(.02, .05))]
        f = g * _u(.9, 1.05)
    notes.append(_tone(f, f * _u(1.1, 1.3), _u(.14, .22), vibrato=(_u(14, 22), .04)))
    return notes


def _excited():
    return _chatter((8, 13), 1300, 3200) + [_warble(_u(1200, 1500), _u(2600, 3200), .3, rate=(28, 36), depth=.22)]


def _excited_plain():
    """Bubbly, uneven bursts of short chirps going up (not a steady sweep: that reads as a siren)."""
    notes, f = [], _u(800, 1000)
    for burst in range(random.randint(2, 3)):
        for _ in range(random.randint(2, 4)):
            g = f * _u(1.2, 1.5)
            notes += [_tone(f, g, _u(.03, .05), harmonic=.3), _gap(_u(.01, .03))]
            f = g * _u(.75, .95)
        notes.append(_gap(_u(.06, .1)))
        f *= _u(1.05, 1.2)
    notes.append(_tone(f, f * _u(1.3, 1.6), .06))
    return notes


def _curious():
    return _chatter((3, 5), 900, 1800) + [_gap(.04), _warble(_u(800, 1000), _u(1900, 2400), _u(.25, .35),
                                                             rate=(7, 10), depth=.05)]


def _curious_plain():
    f = _u(700, 900)
    return [_tone(f, f * _u(.95, 1.05), _u(.08, .12)), _gap(.06),
            _tone(f * .9, f * _u(1.7, 2.1), _u(.18, .26))]


def _surprised():
    f = _u(600, 800)
    return [_tone(f, f * 4.5, _u(.09, .12), harmonic=.15), _tone(f * 4.5, f * 2.2, .06),
            _tone(f * 2.2, f * 4.8, .07)]


def _surprised_plain():
    f = _u(500, 700)
    return [_tone(f, f * _u(4., 5.), _u(.10, .14), harmonic=.15), _tone(f * 4.4, f * 3.2, .07)]


def _sad():
    """R2's «aww»: one long falling, wobbling whistle, a little falling chirp (no steady tail: that reads as a phone)."""
    f = _u(1200, 1400)
    return [_warble(f, f * .4, _u(.65, .8), rate=(5, 7), depth=.07), _gap(.06),
            _tone(f * .5, f * .28, .16, harmonic=.1, level=.6)]


def _sad_plain():
    f = _u(850, 950)
    return [_tone(f, f * .62, _u(.35, .45), vibrato=(5, .02)), _gap(.08),
            _tone(f * .6, f * .42, _u(.4, .55), vibrato=(4, .03), level=.8)]


def _concerned():
    f = _u(600, 750)
    return [_tone(f, f * 1.05, .28, vibrato=(_u(7, 9), .07)), _gap(.05), _tone(f * 1.02, f * .8, .22)]


def _laughing():
    notes, f = [], _u(1500, 1800)
    for _ in range(random.randint(4, 6)):
        notes += [_tone(f, f * .82, _u(.045, .06), harmonic=.35), _gap(_u(.035, .05))]
        f *= _u(.9, .96)
    return notes


def _sleepy():
    """Powering down for a nap: three soft beeps, each lower and quieter (a glide reads as a siren)."""
    f = _u(650, 750)
    notes = []
    for k, level in enumerate((.8, .55, .35)):
        notes += [_tone(f, f * _u(.72, .8), _u(.2, .26), vibrato=(_u(5, 7), .035), harmonic=.05, level=level,
                        release=.1), _gap(_u(.1, .2))]
        f *= _u(.8, .88)
    return notes


def _hello():
    f = _u(1300, 1600)
    return [_tone(f, f * 1.5, .09), _gap(.03), _tone(f * 1.5, f * .9, .11)] + _chatter((4, 6)) + \
        [_warble(f * 1.2, f * 1.8, .16)]


def _hello_plain():
    f = _u(900, 1100)
    return [_tone(f, f, .09), _gap(.05), _tone(f * 1.5, f * 1.5, .12), _gap(.03),
            _tone(f * 1.5, f * 2.1, .09)]


def _yes():
    f = _u(900, 1100)
    return [_tone(f, f * 1.1, .08), _gap(.05), _tone(f * 1.3, f * 1.45, .11)]


def _no():
    f = _u(1000, 1200)
    return [_tone(f, f * .95, .1), _gap(.06), _tone(f * .75, f * .7, .14)]


def _ouch():
    f = _u(1600, 2000)
    return [_tone(f, f * .35, _u(.12, .18), harmonic=.4)]


def _thinking():
    notes = []
    for _ in range(random.randint(4, 7)):
        f = _u(900, 2200)
        notes += [_tone(f, f * _u(.9, 1.1), _u(.03, .05), level=.6), _gap(_u(.03, .08))]
    return notes


def _found():
    f = _u(800, 1000)
    return [_tone(f, f, .07), _gap(.02), _tone(f * 1.26, f * 1.26, .07), _gap(.02),
            _tone(f * 1.5, f * 1.5, .07), _gap(.02), _tone(f * 2., f * 2.2, .2, vibrato=(18, .03))]


def _affectionate():
    return [_purr(_u(.7, 1.1), base=_u(160, 210))]


RECIPES = {
    'happy': _happy, 'excited': _excited, 'curious': _curious, 'surprised': _surprised, 'sad': _sad,
    'concerned': _concerned, 'laughing': _laughing, 'sleepy': _sleepy, 'hello': _hello, 'yes': _yes,
    'no': _no, 'ouch': _ouch, 'thinking': _thinking, 'found': _found, 'affectionate': _affectionate,
}
WORDS = {
    'happy': 'радостная трель', 'excited': 'восторженный каскад', 'curious': 'вопросительное «бип?»',
    'surprised': 'удивлённый свист', 'sad': 'грустный спуск', 'concerned': 'тревожное дрожание',
    'laughing': 'смешок чириками', 'sleepy': 'зевок', 'hello': 'приветственное «би-ду»', 'yes': '«да» бипами',
    'no': '«нет» бипами', 'ouch': '«ой!» писком', 'thinking': 'задумчивые блипы', 'found': 'победные ноты',
    'affectionate': 'мурчание',
}


OWN_SOUNDS = None          # set by the voice: a folder with sounds/<emotion>/*.wav the owner put there


def own_sound(emotion, level):
    """One of the owner's own WAV files for this feeling (any rate, mono or stereo), as 16 kHz PCM."""
    if OWN_SOUNDS is None:
        return b''
    import wave
    from pathlib import Path
    files = sorted((Path(OWN_SOUNDS) / emotion).glob('*.wav'))
    if not files:
        return b''
    try:
        with wave.open(str(random.choice(files)), 'rb') as handle:
            rate, channels, width = handle.getframerate(), handle.getnchannels(), handle.getsampwidth()
            frames = handle.readframes(min(handle.getnframes(), rate * 6))
        if width != 2:
            return b''
        x = np.frombuffer(frames, dtype='<i2').astype(np.float64)
        if channels > 1:
            x = x.reshape(-1, channels).mean(axis=1)
        if rate != RATE:
            from scipy.signal import resample_poly
            x = resample_poly(x, RATE, rate)
        peak = float(np.max(np.abs(x))) or 1.
        return (x / peak * level * 32767).astype('<i2').tobytes()
    except Exception:
        return b''


def synth(emotion, level=LEVEL):
    """16 kHz s16le PCM of one emotional sound (the owner's own file if there is one), b'' if unknown."""
    own = own_sound(emotion, level)
    if own:
        return own
    recipe = RECIPES.get(emotion)
    if recipe is None:
        return b''
    wave = np.concatenate(recipe() + [_gap(.03)])
    peak = float(np.max(np.abs(wave))) or 1.
    return (wave / peak * level * 32767).astype('<i2').tobytes()
