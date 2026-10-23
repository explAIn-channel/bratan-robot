"""A pivot that spins up slowly is not stuck, offline: python -m pytest tests/test_pivot.py

10.10 evening: 0.42 rad/s asked, 0.03-0.09 measured after 1.2 s while the heading did turn — taken for «rubbing»,
a 10 cm back-off three times in a row at the same spot, then a run-up: the detours on the way to the hall.
"""
import types

from bratan.body import skills


def nav():
    skill = skills.Navigate({'xy': [6., -2.4]})
    skill.info.update(pivot_stuck=0., pivot_yaw=0.)
    return skill


def test_a_slowly_turning_pivot_is_not_stuck():
    skill = nav()
    assert not skill._pivot_stuck(types.SimpleNamespace(now=1.3, odom=(0., 0., .04)))     # the old 1.2 s verdict
    assert not skill._pivot_stuck(types.SimpleNamespace(now=2.2, odom=(0., 0., .12)))     # turning: not stuck


def test_a_pivot_that_does_not_turn_is_stuck():
    skill = nav()
    assert skill._pivot_stuck(types.SimpleNamespace(now=2.2, odom=(0., 0., .01)))
