"""Hearing the house (voice/sounds.py): categories, habituation, startle, never our own voice."""
import time
from pathlib import Path

import numpy as np
import pytest

from bratan.voice import sounds as soundsmod

MODEL = Path(__file__).resolve().parents[2] / 'models' / 'yamnet'


def pcm(x):
    return (np.clip(x, -1, 1) * 32767).astype('<i2').tobytes()


def feed(sounds, x, start, own=False, talking=False):
    data = pcm(x)
    for i in range(0, len(data) - 640 + 1, 640):
        sounds.feed(start + i / 32000, data[i:i + 640], noise=.002, own=own, talking=talking)
        if i % (640 * 25) == 0:
            time.sleep(.002)                     # let the worker take the windows as they come
    time.sleep(.3)


def tone(seconds=3., level=.2):
    sr = 16000
    t = np.arange(int(sr * .4)) / sr
    notes = [261.6, 329.6, 392., 523.3, 392., 329.6] * int(seconds / 2.4 + 1)
    return np.concatenate([sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 5)) * np.exp(-3 * t) * level
                           for f in notes])


needs_model = pytest.mark.skipif(not (MODEL / 'yamnet.onnx').exists(), reason='YAMNet model not on this machine')


@needs_model
def test_music_is_heard_once_then_habituated_and_our_own_sound_never():
    heard = []
    sounds = soundsmod.Sounds(MODEL, heard.append)
    feed(sounds, tone(), 0.)
    assert any(h['kind'] == 'music' for h in heard)
    count = len(heard)
    feed(sounds, tone(), 10.)                     # the same music again within its gap: no news
    assert len(heard) == count
    sounds.last.clear()
    feed(sounds, tone(), 400., own=True)          # our own speaker playing: never classified
    assert len(heard) == count


@needs_model
def test_a_sudden_bang_after_quiet_is_a_startle_and_silence_is_nothing():
    heard = []
    sounds = soundsmod.Sounds(MODEL, heard.append)
    feed(sounds, np.zeros(32000), 0.)
    assert heard == []
    bang = np.concatenate([np.zeros(16000), np.random.default_rng(1).normal(0, .5, 1600) * np.exp(-np.arange(1600) / 300),
                           np.zeros(16000)])
    feed(sounds, bang, 10.)
    assert heard and heard[0]['kind'] == 'startle' and heard[0]['urgent']



def test_our_own_words_coming_back_are_not_a_stop():
    """08.10 18:53: our «Стою на кухне» came back through the microphone as «стой» and closed the talk."""
    import time
    import types
    from collections import deque
    from bratan.voice.main import DIALOG_STOP, Voice
    voice = types.SimpleNamespace(spoken_log=deque([(time.monotonic(), 'Стою на кухне, осматриваюсь. Ты как?')]))
    assert Voice.own_words_said(voice, 'стой')
    assert Voice.own_words_said(voice, 'стоп')    # right after our «стою» a bare «стоп» is ambiguous; «Братан, стоп» works
    voice.spoken_log = deque([(time.monotonic(), 'Привет. Хорошо, что ты здесь.')])
    assert not Voice.own_words_said(voice, 'стоп')
    assert DIALOG_STOP.match('хватит') and DIALOG_STOP.match('стоп пожалуйста') and not DIALOG_STOP.match('стоп-кран')
