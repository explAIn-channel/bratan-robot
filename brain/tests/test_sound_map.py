"""Where the house's sounds come from (C52): a direction from the body's ears is a ray on the apartment map — which
room it points at, through which door, and where two rays of one sound cross."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bratan.mind import hearing
from bratan.mind import world as worldmod

DATA = Path(__file__).parent / 'data'
ROOMS = json.loads((DATA / 'rooms.json').read_text(encoding='utf-8'))
DOORS = [{'rooms': ['Коридор', 'Спальня и кабинет'], 'state': 'open', 'seen': 0., 'center': [3.19, .95]},
         {'rooms': ['Коридор', 'Кухня'], 'state': 'open', 'seen': 0., 'center': [8.62, .27]},
         {'rooms': ['Коридор', 'Зал'], 'state': 'open', 'seen': 0., 'center': [6.28, .35]}]
DOCK = (-.24, -.13, .08)


@pytest.fixture(scope='module')
def room_map():
    from bratan.body.grid import Grid
    return Grid(DATA / 'apartment-final-20261006-dock.yaml'), np.load(DATA / 'rooms-labels.npy')


def make_world(room_map, pose, room, twist=(0., 0.)):
    world = worldmod.World(ROOMS, 'Таффа', None, room_map)
    world.on_body({'pose': list(pose), 'localized': True, 'room': room, 'docked': False, 'twist': list(twist),
                   'doors': [dict(d) for d in DOORS]})
    return world


def bearing_to(pose, xy):
    return math.atan2(math.sin(math.atan2(xy[1] - pose[1], xy[0] - pose[0]) - pose[2]),
                      math.cos(math.atan2(xy[1] - pose[1], xy[0] - pose[0]) - pose[2]))


def test_from_the_dock_a_sound_through_the_bedroom_door_comes_from_the_corridor(room_map):
    world = make_world(room_map, DOCK, 'Спальня и кабинет')
    sounds = hearing.SoundMap(world)
    ray = sounds.add('dog', 'лает собака', bearing_to(DOCK, (3.19, .95)) + .15, .8)
    assert ray['toward'] == 'Коридор' and ray['via'] == 'door'
    assert sounds.words(ray) == 'со стороны «Коридор»'


def test_a_sound_away_from_every_door_is_in_this_room_and_named_by_its_side_only(room_map):
    world = make_world(room_map, (1.5, .4, 0.), 'Спальня и кабинет')
    sounds = hearing.SoundMap(world)
    ray = sounds.add('speech', 'рядом разговаривают', math.radians(-100), .9)       # south: the bedroom's own wall
    assert ray['toward'] == 'Спальня и кабинет' and ray['via'] == 'here' and ray['free'] < 1.5
    assert sounds.words(ray) == ''                          # no room said: by the wall it may be a reflection


def test_in_the_corridor_the_kitchen_door_means_the_kitchen(room_map):
    pose = (7.4, 1.3, math.pi / 2)
    world = make_world(room_map, pose, 'Коридор')
    sounds = hearing.SoundMap(world)
    ray = sounds.add('kitchen', 'на кухне что-то готовят', bearing_to(pose, (8.62, .27)), .7)
    assert ray['toward'] == 'Кухня' and ray['via'] == 'door'
    hall = sounds.add('music', 'музыка', bearing_to(pose, (6.28, .35)), .7)
    assert hall['toward'] == 'Зал'


def test_unsure_turning_or_lost_gives_no_ray(room_map):
    world = make_world(room_map, DOCK, 'Спальня и кабинет')
    sounds = hearing.SoundMap(world)
    assert sounds.add('dog', 'лает собака', .5, .2) is None
    assert sounds.add('dog', 'лает собака', None, .9) is None
    turning = make_world(room_map, DOCK, 'Спальня и кабинет', twist=(0., .5))
    assert hearing.SoundMap(turning).add('dog', 'лает собака', .5, .9) is None
    lost = make_world(room_map, DOCK, 'Спальня и кабинет')
    lost.body['localized'] = False
    assert hearing.SoundMap(lost).add('dog', 'лает собака', .5, .9) is None
    assert not sounds.rays


def test_two_rays_of_one_sound_from_two_spots_cross_where_it_is(room_map):
    source = (6.0, -2.5)                                    # in the hall
    world = make_world(room_map, (5.0, 1.2, 0.), 'Коридор')
    sounds = hearing.SoundMap(world)
    first = (5.0, 1.2, 0.)
    sounds.add('music', 'музыка', bearing_to(first, source), .8, t=100.)
    second = (7.2, 1.0, math.pi)
    world.body['pose'] = list(second)
    sounds.add('music', 'музыка', bearing_to(second, source), .8, t=110.)
    found = sounds.cross(('music',), now=120.)
    assert found is not None and math.dist(found['xy'], source) < .3 and found['room'] == 'Зал'
    assert sounds.cross(('dog',), now=120.) is None
    assert sounds.cross(('music',), now=1000.) is None       # old rays place nothing


def test_rays_too_close_or_parallel_do_not_cross():
    a = {'x': 0., 'y': 0., 'heading': 0.}
    assert hearing.intersect(a, {'x': .3, 'y': 0., 'heading': 1.}) is None             # heard from the same spot
    assert hearing.intersect(a, {'x': 0., 'y': 2., 'heading': .1}) is None             # nearly parallel
    assert hearing.intersect(a, {'x': 0., 'y': 2., 'heading': math.pi / 2}) is None    # it would be behind
    point = hearing.intersect(a, {'x': 2., 'y': 2., 'heading': -math.pi / 2})
    assert point is not None and math.dist(point, (2., 0.)) < 1e-6


def test_the_mind_is_told_the_side_and_the_room(room_map):
    world = make_world(room_map, DOCK, 'Спальня и кабинет')
    world.on_sound({'sound': 'dog', 'words': 'лает собака', 'urgent': True, 'side': 'слева-спереди',
                    'bearing': bearing_to(DOCK, (3.19, .95)), 'bearing_conf': .7})
    event = world.events[-1]
    assert event['text'] == 'слышу слева-спереди, со стороны «Коридор»: лает собака' and event['toward'] == 'Коридор'
    assert 'со стороны «Коридор»' in world.hearing_text()
    world.on_sound({'sound': 'music', 'words': 'музыка', 'urgent': True, 'side': 'справа', 'bearing': -1.5,
                    'bearing_conf': .7})
    assert world.events[-1]['text'] == 'слышу справа: музыка'          # by the wall: the side, no guessed room


# ---------------------------------------------------------------- going to a sound (the mind's to_sound)
def mind_on_map(room_map, tmp_path, pose, room, docked=False, voltage=16.4):
    from bratan.common.config import DEFAULTS
    from bratan.mind import brain as brainmod
    from bratan.mind.memory import Archive, MindState
    archive = Archive(tmp_path / 'memory.sqlite')
    world = worldmod.World(ROOMS, 'Таффа', archive, room_map, state=MindState(tmp_path / 'mind.json'))
    world.on_body({'pose': list(pose), 'localized': True, 'room': room, 'docked': docked, 'twist': [0., 0.],
                   'voltage': voltage, 'doors': [dict(d) for d in DOORS]})
    acts = []

    def act(kind, **data):
        acts.append((kind, data))
        if kind == 'quiet':
            return False
        if kind == 'body_do':
            return f'm{len(acts)}'
        return True
    config = dict(DEFAULTS, people=['Сергей', 'Марина'])
    return brainmod.Mind(world.state, world, archive, None, config, act), world, acts


def commands(acts):
    return [(d['skill'], d.get('args'), d.get('origin')) for k, d in acts if k == 'body_do']


def test_a_bark_from_the_corridor_takes_me_there_to_find_taffa(room_map, tmp_path):
    pose = (1.5, .4, 0.)
    mind, world, acts = mind_on_map(room_map, tmp_path, pose, 'Спальня и кабинет')
    world.on_sound({'sound': 'dog', 'words': 'лает собака', 'urgent': True, 'side': 'спереди',
                    'bearing': bearing_to(pose, (3.19, .95)), 'bearing_conf': .8})
    ok, text = mind.do({'tool': 'to_sound', 'args': {'sound': 'лай'}})
    assert ok and 'со стороны «Коридор»' in text and 'послушаю в проёме' in text
    skill, args, origin = commands(acts)[0]
    assert skill == 'go' and args['xy'] == [3.19, .95] and origin == 'mind'       # first to the doorway
    assert mind.skill_run['name'] == 'поиск: собака'                    # the executive comes up to him when seen
    assert mind.skill_run['steps'][-1] == {'tool': 'search', 'args': {'who': 'собака'}}
    done(mind, acts)                                                    # in the doorway: listen
    assert commands(acts)[-1][0] == 'wait'
    done(mind, acts)                                                    # quiet there: where it was heard from
    corridor = next(r for r in ROOMS['rooms'] if r['name'] == 'Коридор')
    skill, args, _ = commands(acts)[-1]
    assert skill == 'go' and args['xy'] == corridor['visit']


def done(mind, acts):
    """The body finished the last command it was given."""
    mind.next_skill_step({'id': f'm{len(acts)}', 'status': 'done'})


def test_in_the_doorway_heard_again_it_says_the_kitchen(room_map, tmp_path):
    """C52, the owner: stop in the doorway, listen again. From the bedroom the bark came in through its door (it only
    says «the corridor's side»); heard again from that doorway it comes through the kitchen's door."""
    pose = (1.5, .4, 0.)
    mind, world, acts = mind_on_map(room_map, tmp_path, pose, 'Спальня и кабинет')
    world.on_sound({'sound': 'dog', 'words': 'лает собака', 'urgent': True, 'side': 'спереди',
                    'bearing': bearing_to(pose, (3.19, .95)), 'bearing_conf': .8})
    world.sounds.latest(('dog',))['t'] -= 5.
    mind.do({'tool': 'to_sound', 'args': {'sound': 'лай'}})
    done(mind, acts)
    door = (3.19, .95, 0.)
    world.on_body({**world.body, 'pose': list(door), 'room': 'Коридор'})
    world.sounds.add('dog', 'лает собака', bearing_to(door, (8.62, .27)), .8)     # the kitchen's door
    done(mind, acts)
    skill, args, _ = commands(acts)[-1]
    kitchen = next(r for r in ROOMS['rooms'] if r['name'] == 'Кухня')
    assert skill == 'go' and args['xy'] == kitchen['visit']


def test_music_in_this_room_takes_me_closer_then_a_look(room_map, tmp_path):
    pose = (1.0, -.3, 0.)
    mind, world, acts = mind_on_map(room_map, tmp_path, pose, 'Спальня и кабинет')
    world.sounds.add('music', 'музыка', math.radians(-90), .8)          # south, the room's far side
    ok, text = mind.do({'tool': 'to_sound', 'args': {'sound': 'музыка'}})
    assert ok
    skill, args, _ = commands(acts)[0]
    assert skill == 'go' and abs(args['xy'][0] - 1.0) < .05 and args['xy'][1] < -.5       # straight towards it
    free = world.sounds.rays[-1]['free']
    assert math.dist(args['xy'], pose[:2]) <= min(1.5, free - .7) + .01
    assert mind.skill_run['steps'][-1]['tool'] == 'look_around'


def test_heard_from_two_spots_goes_where_the_rays_cross(room_map, tmp_path):
    mind, world, acts = mind_on_map(room_map, tmp_path, (5.0, 1.2, 0.), 'Коридор')
    source = (6.0, -2.5)
    world.sounds.add('music', 'музыка', bearing_to((5.0, 1.2, 0.), source), .8)
    world.body['pose'] = [7.2, 1.0, math.pi]
    world.sounds.add('music', 'музыка', bearing_to((7.2, 1.0, math.pi), source), .8)
    ok, text = mind.do({'tool': 'to_sound', 'args': {'sound': 'музыка'}})
    hall = next(r for r in ROOMS['rooms'] if r['name'] == 'Зал')
    assert ok and 'услышал с двух мест' in text and commands(acts)[0][1]['xy'] == hall['visit']


def test_without_a_placed_sound_it_says_so(room_map, tmp_path):
    mind, world, acts = mind_on_map(room_map, tmp_path, (1.5, .4, 0.), 'Спальня и кабинет')
    ok, text = mind.do({'tool': 'to_sound', 'args': {'sound': 'лай'}})
    assert not ok and 'не знаю, откуда был звук' in text and not commands(acts)


def test_from_the_dock_house_chatter_is_no_reason_but_a_call_or_the_owner_is(room_map, tmp_path):
    mind, world, acts = mind_on_map(room_map, tmp_path, DOCK, 'Спальня и кабинет', docked=True)
    world.sounds.add('music', 'музыка', bearing_to(DOCK, (3.19, .95)), .8)
    ok, text = mind.do({'tool': 'to_sound', 'args': {}})
    assert not ok and 'не повод съезжать с базы' in text and not commands(acts)
    ok, text = mind.do({'tool': 'to_sound', 'args': {}}, owner=True)
    assert ok and commands(acts)[0][0] == 'go' and commands(acts)[0][2] == 'owner'
    world.sounds.add('alarm', 'звенит будильник или телефон', bearing_to(DOCK, (3.19, .95)), .8)
    mind.skill_run = None
    ok, text = mind.do({'tool': 'to_sound', 'args': {'sound': 'звон'}})
    assert ok


def test_come_to_me_said_from_the_corridor_goes_there_to_find_me(room_map, tmp_path):
    """«Братан, иди ко мне» out of view: the ears placed the voice behind the bedroom door — the corridor first, then
    find the speaker and come up (until 18.10: a look round on the spot, «the voice came from near»)."""
    pose = (1.5, .4, 0.)
    mind, world, acts = mind_on_map(room_map, tmp_path, pose, 'Спальня и кабинет')
    world.sounds.add('wake', 'меня позвали', bearing_to(pose, (3.19, .95)), .7)
    ok, text = mind.do({'tool': 'approach', 'args': {'who': 'меня'}}, owner=True)
    assert ok and 'голос был со стороны «Коридор»' in text
    corridor = next(r for r in ROOMS['rooms'] if r['name'] == 'Коридор')
    assert commands(acts)[0][0] == 'go' and commands(acts)[0][1]['xy'] == corridor['visit']
    assert mind.skill_run['steps'][-1]['tool'] == 'approach'


def test_come_to_me_in_this_room_looks_toward_the_voice_first(room_map, tmp_path):
    pose = (1.0, -.3, 0.)
    mind, world, acts = mind_on_map(room_map, tmp_path, pose, 'Спальня и кабинет')
    world.sounds.add('wake', 'меня позвали', math.radians(-120), .7)
    ok, text = mind.do({'tool': 'approach', 'args': {'who': 'меня'}}, owner=True)
    skill, args, _ = commands(acts)[0]
    assert ok and skill == 'track' and args['mode'] == 'seek' and args['bearing'] == pytest.approx(math.radians(-120), abs=.01)


def test_a_bark_placed_in_the_corridor_tells_the_need_where_taffa_is(room_map, tmp_path):
    from bratan.mind.drives import Drives
    pose = (1.5, .4, 0.)
    mind, world, acts = mind_on_map(room_map, tmp_path, pose, 'Спальня и кабинет')
    world.sounds.add('dog', 'лает собака', bearing_to(pose, (3.19, .95)), .8)
    needs = Drives(world.archive, world, world.state, {'timezone': 'Europe/Moscow'}).needs(time_now())
    how = next(n[4] for n in needs if n[0] == 'скучаю по Таффе')
    assert 'лаял со стороны «Коридор»' in how and 'to_sound' in how


def test_with_two_in_view_the_voice_tells_who_is_speaking(room_map):
    import time
    world = make_world(room_map, (1.0, 0., 0.), 'Спальня и кабинет')
    now = time.time()
    world.presence = {'Сергей': {'seen': now, 'face': now, 'present': True, 'xy': [2.5, 1.0]},     # ahead-left
                      'Марина': {'seen': now, 'face': now, 'present': True, 'xy': [2.5, -1.0]}}  # ahead-right
    assert world.speaker_by_voice() is None                                   # nothing heard yet
    world.sounds.add('voice', 'со мной говорят', bearing_to((1.0, 0., 0.), (2.5, -1.0)), .7)
    assert world.speaker_by_voice() == 'Марина'
    world.sounds.add('voice', 'со мной говорят', bearing_to((1.0, 0., 0.), (2.5, 1.0)), .7)
    assert world.speaker_by_voice() == 'Сергей'
    world.sounds.add('voice', 'со мной говорят', math.pi, .7)                 # from behind: neither of them
    assert world.speaker_by_voice() is None


def test_the_diary_says_where_a_sound_came_from(tmp_path):
    import time
    from bratan.mind.memory import Archive
    archive = Archive(tmp_path / 'memory.sqlite')
    t = time.time() - 600
    archive.event('heard_sound', {'kind': 'sound', 'sound': 'dog', 'words': 'лает собака', 'toward': 'Коридор'}, t=t)
    archive.event('heard_sound', {'kind': 'sound', 'sound': 'doorbell', 'words': 'звонок в дверь'}, t=t + 60)
    diary = archive.diary(t - 10)
    clock = time.strftime('%H:%M', time.localtime(t))
    assert f'лаял Таффа ({clock} со стороны «Коридор»)' in diary and 'звонили в дверь (' in diary


def test_the_sounds_of_the_evenings_become_a_habit_and_say_where_people_are(room_map, tmp_path):
    """C52, the owner: the TV in the hall in the evening means people there. Heard on enough different evenings, from
    the hall's side, it is a habit: the mind reads it, and missing my people with nobody's sightings known, I go there."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from bratan.mind.drives import Drives
    pose = (1.5, .4, 0.)
    mind, world, acts = mind_on_map(room_map, tmp_path, pose, 'Спальня и кабинет')
    tz = ZoneInfo('Europe/Moscow')
    now = datetime.now(tz).replace(hour=20, minute=30, second=0, microsecond=0).timestamp()
    for day in range(5):
        evening = (datetime.fromtimestamp(now, tz) - timedelta(days=day + 1)).replace(hour=20).timestamp()
        for k in range(3):
            world.archive.event('heard_sound', {'kind': 'sound', 'sound': 'tv', 'words': 'телевизор',
                                                'toward': 'Зал', 'here': 'Спальня и кабинет'}, t=evening + 600 * k)
    world.archive.event('heard_sound', {'kind': 'sound', 'sound': 'dog', 'words': 'лает'}, t=now - 3600)   # one day
    drives = Drives(world.archive, world, world.state, {'timezone': 'Europe/Moscow', 'people': ['Сергей']})
    habits = drives.sound_habits(now)
    assert set(habits) == {'tv'} and habits['tv']['parts'] == ['вечером'] and habits['tv']['room'] == 'Зал'
    assert drives.sound_habits_text(now) == 'телевизор — почти каждый день, чаще вечером, обычно со стороны «Зал»'
    how = next(n[4] for n in drives.needs(now) if n[0] == 'скучаю по своим')
    assert 'обычно слышно «телевизор» со стороны «Зал»' in how and 'поехать в «Зал»' in how
    morning = datetime.fromtimestamp(now, tz).replace(hour=9).timestamp()
    drives._sounds = (0., {})
    assert drives.heard_usually(morning) is None                    # in the morning it is not the TV's time


def time_now():
    import time
    return time.time()


def test_latest_by_kind_and_freshness(room_map):
    world = make_world(room_map, DOCK, 'Спальня и кабинет')
    sounds = hearing.SoundMap(world)
    sounds.add('dog', 'лает собака', .3, .8, t=1000.)
    sounds.add('music', 'музыка', .4, .8, t=1010.)
    assert sounds.latest(('dog',), now=1020.)['kind'] == 'dog'
    assert sounds.latest(now=1020.)['kind'] == 'music'
    assert sounds.latest(('dog',), now=1200.) is None
