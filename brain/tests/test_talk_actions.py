"""What the owner says, as the talking model turns it into actions (10.10: 145 phrases through the real model on the
phone): python -m pytest tests/test_talk_actions.py"""
import math

from tests.test_mind import clock, make_mind, world  # noqa: F401  (fixtures)


def body(acts):
    return [d for k, d in acts if k == 'body_do']


def test_back_is_back_and_a_turn_is_a_turn(world, clock):
    mind, acts = make_mind(world)
    mind.do(mind.as_asked('сдай назад', {'tool': 'move', 'args': {'direction': 'back'}}), owner=True)
    assert body(acts)[-1]['skill'] == 'drive' and body(acts)[-1]['args']['distance'] < 0
    mind.do(mind.as_asked('развернись', {'tool': 'move', 'args': {'action': 'turn', 'angle': 180}}), owner=True)
    assert body(acts)[-1]['skill'] == 'rotate' and abs(body(acts)[-1]['args']['angle'] - math.pi) < 1e-6
    mind.do({'tool': 'move', 'args': {'meters': .5, 'direction': 'назад'}}, owner=True)
    assert body(acts)[-1]['args']['distance'] == -.5


def test_to_me_is_me(world, clock):
    mind, acts = make_mind(world)
    assert mind.as_asked('подъедь ко мне', {'tool': 'approach', 'args': {'who': 'собака'}})['args']['who'] == 'меня'
    assert mind.as_asked('поезжай за мной', {'tool': 'follow', 'args': {'who': 'человек'}})['args']['who'] == 'меня'
    assert mind.as_asked('подъедь к Таффе', {'tool': 'approach', 'args': {'who': 'собака'}})['args']['who'] == 'собака'


def test_follow_me_before_i_am_seen(world, clock):
    mind, acts = make_mind(world)
    ok, _ = mind.do({'tool': 'follow', 'args': {'who': 'человек'}}, owner=True)
    assert ok and body(acts)[-1]['skill'] == 'track' and body(acts)[-1]['args']['mode'] == 'follow'


def test_emotions_in_other_words(world, clock):
    mind, acts = make_mind(world)
    for args, kind in (({'kind': 'нет'}, 'shake'), ({'emotion': 'happy'}, 'wiggle'), ({'kind': 'поклон'}, 'bow'),
                       ({'kind': 'кивок'}, 'nod'), ({'name': 'dog'}, 'wiggle')):
        mind.do({'tool': 'emote', 'args': args}, owner=True)
        assert [d for k, d in acts if k == 'express'][-1]['emote'] == kind


def test_a_place_does_not_hide_a_room(world, clock):
    mind, acts = make_mind(world)
    ok, why = mind.do({'tool': 'remember_place', 'args': {'name': 'кухня'}}, owner=True)
    assert not ok and 'комната' in why
    world.state.set('places', [{'name': 'кухня', 'xy': [1.5, .4]}])       # one made before the fix
    mind.do({'tool': 'go', 'args': {'to': 'Кухня'}}, owner=True)
    assert body(acts)[-1]['args']['xy'] != [1.5, .4]
    assert mind.do({'tool': 'remember_place', 'args': {'name': 'у окна'}}, owner=True)[0]


def test_the_photo_is_shown_to_the_one_talking(world, clock):
    world.state.set('album', [{'id': 'p1', 't': clock.now}])
    world.voice = {'live': True}
    mind, acts = make_mind(world)
    assert mind.do({'tool': 'show_photo', 'args': {}}, owner=True) == (True, 'показал снимок')
