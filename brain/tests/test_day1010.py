"""The day of 10.10, offline: python -m pytest tests/test_day1010.py"""
from bratan.mind import brain as brainmod
from tests.test_mind import clock, make_mind, world  # noqa: F401  (fixtures)


def kinds(world):
    return [e['kind'] for e in world.events]


def test_lost_the_mind_lets_the_body_find_itself(world, clock):
    # 11:13-13:13: 96 «dock: потерялся на карте» — the mind kept the body busy while it could not search for itself.
    world.on_body({**world.body, 'localized': False})
    mind, acts = make_mind(world)
    ok, why = mind.do({'tool': 'go', 'args': {'to': 'Кухня'}})
    assert not ok and 'потерялся' in why and not [a for a in acts if a[0] == 'body_do']
    assert mind.do({'tool': 'rest', 'args': {}}, owner=True)[0]              # the owner's «на базу» still goes


def test_driving_on_while_lost_is_not_being_carried(world, clock):
    # 17:47: «меня перенесли на руках» after driving home lost.
    world.on_body_event({'kind': 'lost'})
    world.on_body({**world.body, 'localized': False, 'pose': [1.5, .4, 0.]})
    clock(4)
    world.tick()
    world.on_body({**world.body, 'localized': False, 'pose': [3.2, 1.0, 0.]})   # the wheels drove 1.8 m
    world.on_body_event({'kind': 'found', 'pose': [3.3, 1.05, 0.]})
    assert 'carried' not in kinds(world)


def test_a_photo_shown_is_the_last_one_and_said_so(world, clock):
    # 20:55: show_photo with no id — nothing shown and «Не получается: показал снимок».
    world.state.set('album', [{'id': 'p1', 't': clock.now}, {'id': 'p2', 't': clock.now}])
    world.on_eyes({'items': [{'label': 'person', 'score': .8, 'box': [100, 50, 300, 600], 'name': 'Сергей',
                              'face': True, 'facing': True, 'bearing': 0., 'frames': 3}], 'stats': {'light': 120}})
    mind, acts = make_mind(world)
    for _ in range(3):
        world.on_eyes({'items': [{'label': 'person', 'score': .8, 'box': [100, 50, 300, 600], 'name': 'Сергей',
                                  'face': True, 'facing': True, 'bearing': 0., 'frames': 3}], 'stats': {'light': 120}})
    assert mind.do({'tool': 'show_photo', 'args': {}}, owner=True) == (True, 'показал снимок')
    assert ('show_photo', {'id': 'p2'}) in acts


def test_words_the_model_reaches_for(world, clock):
    # 18:22: «попробуй ещё раз встать на базу» -> «dock» («не умею»); «отъедь вперёд» -> go {"to": "вперёд"}.
    mind, acts = make_mind(world)
    assert mind.do({'tool': 'dock', 'args': {}}, owner=True)[0]
    assert acts[-1][0] == 'body_do' and acts[-1][1]['skill'] == 'dock'
    ok, text = mind.do({'tool': 'go', 'args': {'to': 'вперёд'}}, owner=True)
    assert ok and acts[-1][1]['skill'] == 'drive' and acts[-1][1]['args']['distance'] == .5
    mind.do({'tool': 'move', 'args': {'meters': -3}}, owner=True)
    assert acts[-1][1]['args']['distance'] == -1.                          # at most a metre


def test_the_charge_is_known(world, clock):
    # 15:45 «какой процент зарядки?» -> «точных данных о батарее у меня нет».
    for i in range(8):
        world.on_body({**world.body, 'docked': True, 'voltage': 15.6 + .05 * i})
        clock(30)
    text = world.situation({'board_reserve_voltage': 15., 'departure_voltage': 16.3, 'pets': {}})
    assert 'Батарея: 15.9 В, выезжаю с 16.3 В — выеду примерно через' in text


def test_hearing_talk_is_not_an_invented_sense():
    # 17:46: «слышу, как рядом разговаривают» (true: the ears hear speech) -> «поправка: я не слышу звуков комнаты».
    assert not brainmod.INVENTED_SENSE.search('слышу, как рядом разговаривают')
    assert brainmod.INVENTED_SENSE.search('слышу, как Таффа сопит')


def test_gone_is_said_only_after_seen(world, clock):
    # 78 «Таффа ушёл из виду» and 13 «вижу собаку» in a day: every short return ended in a «left» thought.
    dog = {'items': [{'label': 'dog', 'score': .8, 'box': [0, 300, 200, 480], 'bearing': .1, 'frames': 4}],
           'stats': {'light': 120}}
    for _ in range(2):
        world.on_eyes(dog)
        clock(.5)
    clock(20)
    world.tick()
    assert kinds(world) == ['dog_appeared', 'dog_left']
    for _ in range(3):                                   # back a minute later, gone again: no news either way
        clock(30)
        for _ in range(2):
            world.on_eyes(dog)
            clock(.5)
        clock(20)
        world.tick()
    assert kinds(world) == ['dog_appeared', 'dog_left']
