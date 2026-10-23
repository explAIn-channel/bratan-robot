"""Hundreds of «the world changed on the way» scenarios through the real world model, mind and executive (no model,
no robot): python -m pytest tests/test_scenarios.py -q

Owner 10.10: «едет к точке, на ходу увидел Таффу, а хотел к Таффе — пусть сразу меняет траекторию; придумай сотни
сценариев и прогони». Each case: a plan the mind made (as the model writes it, with «если»), a change in the world,
a circumstance. Expected: the reflex the plan asks for within a second — or nothing, where nothing is right.
"""
import types

import pytest

from bratan.mind import executive as execmod
from tests.test_mind import clock, make_mind, world  # noqa: F401  (fixtures)

DOG = {'label': 'dog', 'score': .8, 'box': [0, 300, 200, 480], 'bearing': .1, 'distance': 1.5, 'frames': 4}


def person(name=None):
    return {'label': 'person', 'score': .9, 'box': [100, 50, 300, 600], 'name': name, 'face': bool(name),
            'facing': bool(name), 'bearing': 0., 'distance': 1.8, 'frames': 5}


GO_HALL = {'tool': 'go', 'args': {'to': 'Зал'}}
GOALS = {
    # name: (thought, do, «если»)
    'look_for_taffa': ('Соскучился — поеду поищу Таффу в зале.', GO_HALL, []),
    'hall_dog_photo': ('Поеду в зал.', GO_HALL,
                       [{'когда': 'собака', 'do': {'tool': 'approach', 'args': {'who': 'собака', 'intent': 'сфотографировать'}}}]),
    'kitchen_person': ('Поеду на кухню.', {'tool': 'go', 'args': {'to': 'Кухня'}},
                       [{'когда': 'человек', 'do': {'tool': 'approach', 'args': {'intent': 'поздороваться'}}}]),
    'hall_sergey': ('Поеду в зал, там может быть Сергей.', GO_HALL,
                    [{'когда': 'увижу Сергея', 'do': {'tool': 'approach', 'args': {}}}]),
    'hall_door_kitchen': ('Поеду в зал.', GO_HALL,
                          [{'когда': 'дверь закрыта', 'do': {'tool': 'go', 'args': {'to': 'Кухня'}}}]),
    'hall_arrived_look': ('Поеду в зал.', GO_HALL, [{'когда': 'доехал', 'do': {'tool': 'look_around', 'args': {}}}]),
    'hall_failed_rest': ('Поеду в зал.', GO_HALL, [{'когда': 'не вышло', 'do': {'tool': 'rest', 'args': {}}}]),
    'explore_dark_rest': ('Осмотрю спальню.', {'tool': 'explore_room', 'args': {}},
                          [{'когда': 'темно', 'do': {'tool': 'rest', 'args': {}}}]),
    'explore_bark_search': ('Осмотрю спальню.', {'tool': 'explore_room', 'args': {}},
                            [{'когда': 'звук: лай', 'do': {'tool': 'search', 'args': {'who': 'собака'}}}]),
    'plain_trip': ('Поеду в зал.', GO_HALL, []),
    'all_three': ('Поеду в зал.', GO_HALL,
                  [{'когда': 'собака', 'do': {'tool': 'approach', 'args': {'who': 'собака'}}},
                   {'когда': 'дверь закрыта', 'do': {'tool': 'go', 'args': {'to': 'Кухня'}}},
                   {'когда': 'доехал', 'do': {'tool': 'look_around', 'args': {}}}]),
    'thing_then_dog': ('Подъеду к комоду.', {'tool': 'go', 'args': {'to': 'Коридор'}},
                       [{'когда': 'Таффа', 'do': {'tool': 'approach', 'args': {}}}]),
}
CHANGES = ('nothing', 'dog', 'dog_flicker', 'person', 'sergey', 'marina', 'door_closed', 'arrived', 'failed',
           'dark', 'bark', 'music', 'sergey_flicker')
CIRCUMSTANCES = ('normal', 'talk', 'owner_errand', 'autonomy_off', 'game')

# what each kind of condition answers to
# a face recognised even in one frame is someone real (the world trusts a name at once); a lone dog frame is not
MATCH = {'dog': {'dog'}, 'person': {'person', 'sergey', 'marina', 'sergey_flicker'},
         'name:Сергей': {'sergey', 'sergey_flicker'},
         # «не нашёл дороги — проход закрыт» is a closed door to a plan («дверь закрыта → на кухню»)
         'door_closed': {'door_closed', 'failed'}, 'arrived': {'arrived'}, 'failed': {'failed'}, 'dark': {'dark'},
         'sound:лай': {'bark'}, 'seek:собака': {'dog'}}
BODY = {'approach': {'track', 'approach'}, 'go': {'go'}, 'look_around': {'look_around'}, 'rest': {'dock'},
        'search': {'track', 'go'}}


def expected(goal, change, circumstance):
    """The oracle: the first condition of the plan the change satisfies -> the tool, else None."""
    if circumstance != 'normal':
        return None                                    # a talk, the owner's errand, a game, autonomy off: no reflex
    thought, do, conditions = GOALS[goal]
    keys = []
    if goal == 'look_for_taffa':
        keys.append(('seek:собака', 'approach'))
    from bratan.mind.brain import when_of
    for c in conditions:
        when, arg = when_of(c['когда'], ['Сергей', 'Марина'], 'Таффа')
        keys.append((f'{when}:{arg}' if when in ('name', 'sound') else when, c['do']['tool']))
    for key, tool in keys:
        if change in MATCH.get(key, ()):
            return tool
    return None


CASES = [(g, c, k) for g in GOALS for c in CHANGES for k in CIRCUMSTANCES]


def setup(world, circumstance, mind):
    life = types.SimpleNamespace(
        world=world, mind=mind, config={'autonomy': circumstance != 'autonomy_off', 'pets': {'dog': 'Таффа'}},
        play=types.SimpleNamespace(active=circumstance == 'game'), archive=world.archive, state=world.state,
        publish=lambda topic, data: None, pool=None,
        loop=types.SimpleNamespace(run_in_executor=lambda pool, fn, *args: fn(*args)))
    return execmod.Executive(life)


@pytest.mark.parametrize('goal,change,circumstance', CASES, ids=['-'.join(c) for c in CASES])
def test_scenario(world, clock, goal, change, circumstance):
    mind, acts = make_mind(world, config={'people': ['Сергей', 'Марина']})
    ex = setup(world, circumstance, mind)
    thought, do, conditions = GOALS[goal]
    mind.enact({'thought': thought, 'say': None, 'do': dict(do), 'if': conditions, 'face': None}, 'heartbeat', False)
    plan_id = mind.sent_id
    assert plan_id, 'the plan started no body command'
    kind = {'go': 'go', 'explore_room': 'explore'}[do['tool']]
    world.on_body({**world.body, 'skill': {'kind': kind, 'origin': 'mind', 'id': plan_id}})
    if circumstance == 'talk':
        world.voice = {'live': True}
    if circumstance == 'owner_errand':
        world.on_body({**world.body, 'skill': {'kind': 'go', 'origin': 'owner', 'id': 'owner-1'}})
    before = len([a for a in acts if a[0] == 'body_do'])
    started = clock.now
    reacted_at = None
    for step in range(40):                                   # 4 s at the executive's 10 Hz
        t = clock.now - started
        if change in ('dog', 'person', 'sergey', 'marina') and t <= 2.:
            item = DOG if change == 'dog' else person({'sergey': 'Сергей', 'marina': 'Марина'}.get(change))
            world.on_eyes({'items': [item], 'stats': {'light': 120}})
        if change in ('dog_flicker', 'sergey_flicker') and step == 0:
            world.on_eyes({'items': [DOG if change == 'dog_flicker' else person('Сергей')], 'stats': {'light': 120}})
        if change == 'door_closed' and step == 2:
            world.event('door_closed', 'дверь между «Коридор» и «Зал» закрыли', rooms=['Коридор', 'Зал'])
        if change in ('arrived', 'failed') and step == 2:
            world.on_body({**world.body, 'skill': None})
            world.on_body_event({'kind': 'skill', 'id': plan_id, 'skill': kind, 'origin': 'mind',
                                 'status': 'done' if change == 'arrived' else 'failed',
                                 'reason': '' if change == 'arrived' else 'no_route'})
        if change == 'dark':
            world.on_eyes({'items': [], 'stats': {'light': 3}})
        if change in ('bark', 'music') and step == 2:
            world.event('sound', 'слышу: лает собака' if change == 'bark' else 'слышу: музыка')
        ex.tick(clock.now)
        if reacted_at is None and len([a for a in acts if a[0] == 'body_do']) > before:
            reacted_at = clock.now - started
        clock(.1)
    reflexes = [d for k, d in acts if k == 'body_do'][before:]
    want = expected(goal, change, circumstance)
    if want is None:
        assert not reflexes, f'no reflex expected, got {[r["skill"] for r in reflexes]}'
    else:
        assert reflexes, f'expected {want}, the body did nothing'
        assert reflexes[0]['skill'] in BODY[want], f'expected {want}, got {reflexes[0]["skill"]}'
        assert len(reflexes) == 1, f'one reflex, not {len(reflexes)}'
        assert reacted_at <= 1.0, f'reflex after {reacted_at:.1f} s'
        if want == 'approach' and change == 'dog':
            assert reflexes[0]['args'].get('who') == 'собака'
