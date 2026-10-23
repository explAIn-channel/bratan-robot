"""Body language (eyes/gesture.py): a wave, hands up, a smile held for a moment — and how Братан answers."""
import types
from pathlib import Path

import numpy as np
import pytest

from bratan.eyes import gesture as gesturemod

MODELS = Path(__file__).resolve().parents[2] / 'models'


def body(wrist_l=(200, 420), wrist_r=(280, 420), track=1):
    kp = np.zeros((17, 3))
    kp[:, 2] = .9
    kp[gesturemod.NOSE] = (240, 250, .9)
    kp[gesturemod.L_SHOULDER] = (200, 300, .9)
    kp[gesturemod.R_SHOULDER] = (280, 300, .9)
    kp[gesturemod.L_WRIST] = (*wrist_l, .9)
    kp[gesturemod.R_WRIST] = (*wrist_r, .9)
    return {'track': track, 'kp': kp, 'box': [150, 150, 350, 600]}


def run(gestures, frames, start=0., step=.12):
    found = []
    for index, person in enumerate(frames):
        found += [kind for kind, _ in gestures.update([person], now=start + index * step)]
    return found


def test_a_hand_waving_above_the_shoulder_is_a_wave_once():
    g = gesturemod.Gestures()
    waving = [body(wrist_r=(300 + 60 * (1 if k % 4 < 2 else -1), 200)) for k in range(20)]
    assert run(g, waving) == ['wave']
    # Still waving right after: not a second hello for the same wave.
    assert run(g, waving, start=3.) == []
    assert run(g, waving, start=10.) == ['wave']


def test_a_raised_still_hand_or_talking_hands_are_not_a_wave():
    g = gesturemod.Gestures()
    still = [body(wrist_r=(300, 200)) for _ in range(20)]
    assert run(g, still) == []
    talking = [body(wrist_r=(300 + 60 * (1 if k % 4 < 2 else -1), 380)) for k in range(20)]   # below the shoulders
    assert run(g, talking, start=5.) == []


def test_both_hands_above_the_head_is_hands_up():
    g = gesturemod.Gestures()
    cheer = [body(wrist_l=(190, 180), wrist_r=(290, 180)) for _ in range(3)]
    assert run(g, cheer) == ['hands_up']


def test_a_smile_held_for_a_moment_is_news_and_not_every_frame():
    expressions = gesturemod.Expressions.__new__(gesturemod.Expressions)
    expressions.recent, expressions.said = {}, {}
    smile = dict.fromkeys(gesturemod.FER_LABELS, .02) | {'happiness': .86}
    sad = dict.fromkeys(gesturemod.FER_LABELS, .04) | {'sadness': .72}
    assert [expressions.update('Сергей', smile, now=t) for t in (0., .5, 1.)] == [None, None, 'smile']
    assert expressions.update('Сергей', smile, now=1.5) is None
    # A flicker of sadness is nothing; held, it is noticed once.
    assert expressions.update('Сергей', sad, now=2.) is None
    assert [expressions.update('Сергей', sad, now=t) for t in (2.5, 3.)] == [None, 'sad']
    assert expressions.update('Сергей', sad, now=4.) is None


@pytest.mark.skipif(not (MODELS / 'pose' / 'yolo26n-pose.onnx').exists(), reason='pose model not on this machine')
def test_pose_model_runs_on_an_empty_room():
    import cv2
    pose = gesturemod.Pose(MODELS / 'pose' / 'yolo26n-pose.onnx')
    assert pose(cv2, np.full((640, 480, 3), 180, np.uint8)) == []


@pytest.mark.skipif(not (MODELS / 'fer' / 'emotion-ferplus-8.onnx').exists(), reason='FER+ model not on this machine')
def test_expression_model_gives_eight_feelings():
    import cv2
    expressions = gesturemod.Expressions(MODELS / 'fer' / 'emotion-ferplus-8.onnx')
    probs = expressions(cv2, np.full((640, 480, 3), 128, np.uint8), (100, 100, 300, 300))
    assert set(probs) == set(gesturemod.FER_LABELS) and abs(sum(probs.values()) - 1) < .02
    assert expressions(cv2, np.zeros((640, 480, 3), np.uint8), (5, 5, 12, 12)) is None    # too small to read


def test_a_wave_is_answered_a_smile_mirrored_silently():
    from bratan.mind.life import Life
    calls = []
    world = types.SimpleNamespace(event=lambda kind, text, **data: calls.append(('event', kind, data.get('who'))))
    life = types.SimpleNamespace(
        world=world, play=types.SimpleNamespace(active=False),
        react=lambda state: calls.append(('react', state)), sfx=lambda emotion, **_: calls.append(('sfx', emotion)),
        wiggle=lambda kind='wiggle', sound=None: calls.append(('wiggle', kind, sound)) or True)
    Life.on_gesture(life, {'kind': 'wave', 'who': 'Сергей'})
    # One sound: the wiggle carries the hello (two in a row was too much).
    assert calls == [('event', 'waved', 'Сергей'), ('react', 'happy'), ('wiggle', 'wiggle', 'hello')]
    calls.clear()
    life.wiggle = lambda kind='wiggle', sound=None: False          # just wiggled a moment ago: the hello alone
    Life.on_gesture(life, {'kind': 'wave', 'who': 'Сергей'})
    assert calls == [('event', 'waved', 'Сергей'), ('react', 'happy'), ('sfx', 'hello')]
    calls.clear()
    Life.on_gesture(life, {'kind': 'smile', 'who': None})
    assert calls == [('event', 'smiled', 'человек'), ('react', 'happy')]        # no sound for a smile
    calls.clear()
    life.play.active = True                      # a game reads its own gestures: only noted
    Life.on_gesture(life, {'kind': 'wave', 'who': 'Сергей'})
    assert calls == [('event', 'waved', 'Сергей')]


def test_no_sounds_on_the_dock_unless_asked_or_playing(monkeypatch):
    from bratan.mind import life as lifemod
    sent = []
    world = types.SimpleNamespace(body={'docked': True}, visible=lambda: ['Сергей'], available=lambda: ['Сергей'],
                                  live=False)
    life = types.SimpleNamespace(world=world, play=types.SimpleNamespace(active=False), quiet=lambda: False,
                                 state=types.SimpleNamespace(get=lambda key, default=None: default),
                                 publish=lambda topic, data: sent.append(data['emotion']), SFX_GAP_S=30.)
    life.with_people = lambda: lifemod.Life.with_people(life)
    lifemod.Life.sfx(life, 'hello')                 # sees him from the base: silent (owner 08.10 12:35)
    lifemod.Life.sfx(life, 'surprised', serious=True)
    assert sent == []
    lifemod.Life.sfx(life, 'yes', asked=True)       # asked for it: heard
    life.play.active = True
    lifemod.Life.sfx(life, 'laughing', force=True)  # a game started on the dock
    world.body = {'docked': False}
    life.play.active = False
    lifemod.Life.sfx(life, 'hello', force=True)     # out and about with people: a hello
    assert sent == ['yes', 'laughing', 'hello']
