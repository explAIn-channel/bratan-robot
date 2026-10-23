"""The eyes keep a name through a turned head and learn new angles of a known face (11.10, the owner: «по ходу дела
людей распознавал и корректировал»): python -m pytest tests/test_faces_learn.py"""
import json

import numpy as np

from bratan.eyes import detect


def test_a_turned_head_keeps_the_name_on_its_track():
    tracker = detect.Tracker()
    box = [100, 50, 300, 600]
    first = tracker.update([{'label': 'person', 'box': box, 'name': 'Сергей'}], 0.)
    turned = tracker.update([{'label': 'person', 'box': box, 'face_unknown': True}], .2)
    back = tracker.update([{'label': 'person', 'box': box}], .4)
    assert first[0]['track'] == turned[0]['track'] == back[0]['track']
    assert turned[0]['name'] == back[0]['name'] == 'Сергей' and turned[0]['name_kept']


def test_another_name_on_the_track_wins():
    tracker = detect.Tracker()
    box = [100, 50, 300, 600]
    tracker.update([{'label': 'person', 'box': box, 'name': 'Сергей'}], 0.)
    assert tracker.update([{'label': 'person', 'box': box, 'name': 'Марина'}], .2)[0]['name'] == 'Марина'
    assert tracker.update([{'label': 'person', 'box': box}], .4)[0]['name'] == 'Марина'


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def faces_with(tmp_path, people):
    db = tmp_path / 'faces.json'
    db.write_text(json.dumps({'schema': 1, 'people': {n: {'name': n, 'samples': [list(map(float, s)) for s in ss]}
                                                      for n, ss in people.items()}}), encoding='utf-8')
    faces = object.__new__(detect.Faces)
    faces.db, faces.mtime, faces.people, faces.learned_at = db, None, {}, {}
    return faces, db


def test_a_new_angle_is_learned_carefully(tmp_path):
    faces, db = faces_with(tmp_path, {'Сергей': [unit([1, 0, 0, 0])] * 3, 'Марина': [unit([0, 1, 0, 0])] * 3})
    angle = unit([.35, .05, .9, 0])                   # closer to Сергей than to Марина, below the naming bar
    face = {'name': None, 'best': 'Сергей', 'similarity': .35, 'margin': .3, 'embedding': angle}
    assert faces.may_learn('Сергей', face, 0.)
    assert faces.learn('Сергей', angle, 0.)
    data = json.loads(db.read_text(encoding='utf-8'))['people']['Сергей']
    assert len(data['learned']) == 1 and len(data['samples']) == 3          # the enrolled ones are kept
    assert not faces.may_learn('Сергей', face, 10.)                          # not again at once
    assert faces.may_learn('Сергей', face, 31.)
    faces._load()
    assert len(faces.people['Сергей']) == 4                                  # used for matching from now on


def test_doubtful_faces_are_not_learned(tmp_path):
    faces, _ = faces_with(tmp_path, {'Сергей': [unit([1, 0, 0, 0])]})
    base = {'name': None, 'best': 'Сергей', 'similarity': .35, 'margin': .3}
    assert not faces.may_learn('Сергей', dict(base, best='Марина'), 0.)    # closest to someone else
    assert not faces.may_learn('Сергей', dict(base, similarity=.2), 0.)      # too unlike
    assert not faces.may_learn('Сергей', dict(base, margin=.01), 0.)         # as close to another
    assert not faces.may_learn('Сергей', dict(base, name='Сергей'), 0.)      # already recognised: nothing new
    assert not faces.learn('Пётр', unit([1, 0, 0, 0]), 0.)                    # never makes up a person


def test_learned_looks_are_capped(tmp_path):
    faces, db = faces_with(tmp_path, {'Сергей': [unit([1, 0, 0, 0])]})
    for i in range(30):
        faces.learn('Сергей', unit([1, .01 * i, 0, 0]), 100. * i)
    data = json.loads(db.read_text(encoding='utf-8'))['people']['Сергей']
    assert len(data['learned']) == detect.LEARNED_KEPT and len(data['samples']) == 1
