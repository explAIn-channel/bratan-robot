"""One being, not two (11.10 11:30: «вижу человека», a minute later «вижу: Сергей», then «человек ушёл» and «Сергей
ушёл»): python -m pytest tests/test_recognize.py"""
from tests.test_mind import clock, kinds, person, world  # noqa: F401  (fixtures)


def seen(world, clock, *items, seconds=.5, times=1):
    for _ in range(times):
        world.on_eyes({'items': [dict(i) for i in items], 'stats': {'light': 120}})
        clock(seconds)


def tracked(name=None, track=7, xy=(2., 1.), face=False):
    item = person(name, face=face)
    item.update(track=track, map_xy=list(xy))
    return item


def appeared(world):
    return [e for e in world.events if e['kind'] == 'person_appeared']


def test_named_within_the_grace_is_one_hello(world, clock):
    seen(world, clock, tracked(), times=3)                     # 1.5 s, no face yet
    seen(world, clock, tracked('Сергей', face=True), times=2)
    seen(world, clock, tracked(), times=4)                     # turned away: same track keeps the name
    texts = [e['text'] for e in appeared(world)]
    assert len(texts) == 1 and texts[0].startswith('вижу: Сергей')
    clock(16)
    world.tick()
    assert [e['text'] for e in world.events if e['kind'] == 'person_left'] == ['Сергей ушёл из виду']


def test_recognised_late_rewrites_the_record_and_tells_the_mind(world, clock):
    seen(world, clock, tracked(), times=8)                     # 4 s unnamed: «вижу человека» said
    assert [e['who'] for e in appeared(world)] == ['человек']
    world.state.heard(clock.now, 'человек', 'ты что, меня не узнаёшь?')
    seen(world, clock, tracked('Сергей', face=True), times=2)
    events = appeared(world)
    assert events[0]['who'] == 'Сергей' and 'сначала не узнал' in events[0]['text']
    assert 'это и есть Сергей' in events[-1]['text']
    assert world.state.get('dialogue')[-1]['who'] == 'Сергей'
    assert 'человек' not in world.presence
    clock(16)
    world.tick()
    assert [e['who'] for e in world.events if e['kind'] == 'person_left'] == ['Сергей']     # no «человек ушёл»
    assert world.archive.events(0, kinds=['recognized_late'])


def test_the_name_survives_a_broken_track_close_by(world, clock):
    seen(world, clock, tracked('Сергей', face=True, track=3), times=2)
    seen(world, clock, tracked(track=9, xy=(2.4, 1.2)), times=8)     # the robot turned: a new track, no face
    assert 'человек' not in world.presence and len(appeared(world)) == 1


def test_someone_else_far_away_is_not_given_the_name(world, clock):
    seen(world, clock, tracked('Сергей', face=True, track=3), times=2)
    seen(world, clock, tracked(track=9, xy=(6., -2.)), times=8)
    assert [e['who'] for e in appeared(world)] == ['Сергей', 'человек']


def test_two_people_in_view_are_not_merged(world, clock):
    seen(world, clock, tracked(track=1, xy=(2., 1.)), times=8)
    seen(world, clock, tracked(track=1, xy=(2., 1.)), tracked('Марина', face=True, track=2, xy=(3., 1.)), times=2)
    assert 'человек' in world.presence
    assert [e['who'] for e in appeared(world)] == ['человек', 'Марина']


def test_greeted_earlier_then_a_stranger_who_is_them(world, clock):
    seen(world, clock, tracked('Сергей', face=True, track=1), times=2)
    clock(60)
    seen(world, clock, tracked(track=5, xy=(5., -1.)), times=8)        # far from before: unknown at first
    seen(world, clock, tracked('Сергей', face=True, track=5, xy=(5., -1.)), times=1)
    assert world.events[-1]['kind'] == 'recognized' and 'Сергей' in world.events[-1]['text']


def test_equal_quiet_hours_mean_asleep_all_day(world, clock):
    from tests.test_mind import make_mind
    mind, _ = make_mind(world, config={'quiet_start_hour': 11, 'quiet_end_hour': 11})
    assert all(mind.in_quiet(clock.now + 3600 * h) for h in range(24))
    mind, _ = make_mind(world, config={'quiet_start_hour': 22, 'quiet_end_hour': 11})
    assert not all(mind.in_quiet(clock.now + 3600 * h) for h in range(24))
