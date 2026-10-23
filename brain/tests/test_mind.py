"""The mind's honesty and liveliness rules, offline (no model, no robot): python -m pytest tests/test_mind.py"""
import json
import time
import types
from pathlib import Path

import pytest

from bratan.common.config import DEFAULTS
from bratan.mind import brain as brainmod
from bratan.mind import drives as drivesmod
from bratan.mind import world as worldmod
from bratan.mind.memory import Archive, MindState

DATA = Path(__file__).parent / 'data'
ROOMS = json.loads((DATA / 'rooms.json').read_text(encoding='utf-8'))


class Clock:
    def __init__(self, start=1_800_000_000.):
        self.now = start

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def __call__(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    fake = types.SimpleNamespace(time=clock.time, monotonic=clock.monotonic, localtime=time.localtime)
    for module in (worldmod, brainmod, drivesmod):
        monkeypatch.setattr(module, 'time', fake)
    return clock


@pytest.fixture
def world(tmp_path, clock):
    archive = Archive(tmp_path / 'memory.sqlite')
    state = MindState(tmp_path / 'mind.json')
    w = worldmod.World(ROOMS, 'Таффа', archive, None, state=state)
    w.on_body({'pose': [1.5, .4, 0.], 'localized': True, 'room': 'Спальня и кабинет', 'docked': False,
               'voltage': 15.8, 'twist': [0., 0.]})
    return w


def person(name=None, face=False, bearing=0.):
    return {'label': 'person', 'score': .8, 'box': [100, 50, 300, 600], 'name': name, 'face': face,
            'facing': face, 'bearing': bearing, 'frames': 3}


def kinds(world):
    return [e['kind'] for e in world.events]


# ---------------------------------------------------------------- world
def test_flicker_is_not_a_visit(world, clock):
    world.on_eyes({'items': [person()], 'stats': {'light': 120}})
    clock(20)
    world.tick()
    assert 'person_left' not in kinds(world) and 'person_appeared' not in kinds(world)


def test_real_visit_comes_and_goes(world, clock):
    for _ in range(3):
        world.on_eyes({'items': [person()], 'stats': {'light': 120}})
        clock(.5)
    assert 'person_appeared' not in kinds(world)        # 11.10: a moment for the face to be named first
    for _ in range(6):
        world.on_eyes({'items': [person()], 'stats': {'light': 120}})
        clock(.5)
    assert 'person_appeared' in kinds(world) and world.visible()
    clock(16)
    world.tick()
    assert kinds(world).count('person_left') == 1


def test_short_loss_of_place_is_not_news(world, clock):
    world.on_body_event({'kind': 'lost'})
    clock(1)
    world.on_body_event({'kind': 'found', 'pose': [1.5, .4, 0.]})
    clock(5)
    world.tick()
    assert 'lost' not in kinds(world) and 'found' not in kinds(world)


def test_long_loss_and_being_carried(world, clock):
    world.on_body_event({'kind': 'lost'})
    world.on_body({**world.body, 'localized': False})
    clock(3.5)
    world.tick()
    assert kinds(world).count('lost') == 1
    world.on_body({**world.body, 'localized': True, 'room': 'Зал', 'pose': [6.5, -1.3, 1.]})
    world.on_body_event({'kind': 'found', 'pose': [6.5, -1.3, 1.]})
    assert kinds(world)[-1] == 'carried'


def test_dock_chatter_is_one_event(world, clock):
    for docked in (True, False, True):
        world.on_body({**world.body, 'docked': docked})
        world.tick()
        clock(.5)
    clock(2.5)
    world.tick()
    assert kinds(world).count('docked') == 1 and 'undocked' not in kinds(world)


def test_voltage_without_driving_sag(world, clock):
    for v in (15.9, 15.2, 15.1, 15.3):
        world.on_body({**world.body, 'voltage': v})
        clock(5)
    assert world.voltage == 15.9
    clock(60)
    world.on_body({**world.body, 'voltage': 15.4})
    assert world.voltage == 15.4


def test_room_visits_and_light_survive_restart(world, clock, tmp_path):
    world.on_eyes({'items': [], 'stats': {'light': 10}})
    clock(40)
    world.on_body({**world.body, 'room': 'Коридор'})
    world.state.save(force=True)
    again = worldmod.World(ROOMS, 'Таффа', world.archive, None, state=MindState(tmp_path / 'mind.json'))
    assert again.room_visits.get('Спальня и кабинет') == pytest.approx(clock.now)
    assert again.room_dark('Спальня и кабинет', clock.now)


def look_without_seeing(world, clock, seconds=20):
    for _ in range(seconds):
        world.on_eyes({'items': [], 'stats': {'light': 120}})
        clock(1.1)


def test_a_thing_taken_away_is_hidden_after_two_looks_and_comes_back(world, clock):
    world.archive.saw('chair', (3.0, .4), 'Спальня и кабинет', clock.now, count=4)
    oid = world.archive.objects(min_count=1)[0]['id']
    look_without_seeing(world, clock)
    assert world.archive.object(oid) is not None          # once missing: maybe just hidden behind someone
    clock(1500)
    look_without_seeing(world, clock)
    assert world.archive.object(oid) is None and 'object_gone' in kinds(world)
    world.archive.saw('chair', (3.05, .4), 'Спальня и кабинет', clock.now)     # it is back: remembered again
    assert world.archive.object(oid) is not None


def test_furniture_is_never_erased_by_not_seeing_it(world, clock):
    # 08.10: the bed (510 sightings) and the sofa were «not there» up close and erased for good.
    world.archive.saw('bed', (3.0, .4), 'Спальня и кабинет', clock.now, count=500)
    oid = world.archive.objects(min_count=1)[0]['id']
    for _ in range(3):
        look_without_seeing(world, clock)
        clock(1500)
    assert world.archive.object(oid) is not None


def test_a_thing_behind_a_wall_is_not_missing(tmp_path, clock):
    import numpy as np
    from bratan.body.grid import Grid
    room_map = (Grid(DATA / 'apartment-final-20261006-dock.yaml'), np.load(DATA / 'rooms-labels.npy'))
    archive = Archive(tmp_path / 'memory.sqlite')
    w = worldmod.World(ROOMS, 'Таффа', archive, room_map, state=MindState(tmp_path / 'mind.json'))
    # In the bedroom, facing the corridor wall: a chair remembered on the other side of it.
    w.on_body({'pose': [1.2, -.2, .04], 'localized': True, 'room': 'Спальня и кабинет', 'docked': False,
               'voltage': 15.8, 'twist': [0., 0.]})
    archive.saw('chair', (3.5, -.1), 'Коридор', clock.now, count=6)
    oid = archive.objects(min_count=1)[0]['id']
    assert not w.line_of_sight((1.2, -.2), (3.5, -.1)) and w.line_of_sight((1.2, -.2), (2.2, .3))
    for _ in range(3):
        look_without_seeing(w, clock)
        clock(1500)
    assert archive.object(oid) is not None


def test_a_thing_seen_now_and_then_stays(world, clock):
    world.archive.saw('chair', (3.0, .4), 'Спальня и кабинет', clock.now, count=4)
    oid = world.archive.objects(min_count=1)[0]['id']
    chair = {'label': 'chair', 'score': .6, 'box': [0, 0, 50, 50], 'bearing': 0.}
    for i in range(60):
        world.on_eyes({'items': [chair] if i % 8 == 0 else [], 'stats': {'light': 120}})
        clock(1.1)
    assert world.archive.object(oid) is not None


def test_failed_way_with_someone_near_suggests_asking(world, clock):
    for _ in range(3):
        world.on_eyes({'items': [person()], 'stats': {'light': 120}})
        clock(.3)
    world.on_body_event({'kind': 'skill', 'status': 'failed', 'skill': 'go', 'reason': 'blocked', 'origin': 'mind'})
    assert 'попросить' in world.events[-1]['text']


# ---------------------------------------------------------------- drives
def test_habits_send_the_search_where_taffa_usually_is(world, clock):
    for day in range(4):
        world.archive.sighting('собака', 'Коридор', clock.now - day * 86400 - 3600)
    drives = drivesmod.Drives(world.archive, world, world.state, dict(DEFAULTS))
    habits = drives.habits(clock.now)
    assert habits['собака']['room'] == 'Коридор'
    needs = {n[0]: n for n in drives.needs(clock.now)}
    assert '«Коридор»' in needs['скучаю по Таффе'][4]
    assert 'Таффа' in drives.habits_text(clock.now)


def test_dark_room_does_not_pull(world, clock):
    drives = drivesmod.Drives(world.archive, world, world.state, dict(DEFAULTS))
    world.room_light['Зал'] = (clock.now - 600, 8)
    home = [n for n in drives.needs(clock.now) if n[0] == 'знать дом'][0]
    assert 'Зал' not in home[3] or 'темно' in home[3]


# ---------------------------------------------------------------- mind
class FakeLLM:
    def __init__(self, reply=''):
        self.reply = reply

    def chat(self, *args, **kw):
        return self.reply


def make_mind(world, config=None, reply=''):
    acts = []

    def act(kind, **data):
        acts.append((kind, data))
        if kind in ('quiet', 'playing'):
            return False
        if kind == 'may_leave_dock':
            return True
        if kind == 'body_do':
            return f'm{len(acts)}'               # the id life gives the body command
        return True
    mind = brainmod.Mind(world.state, world, world.archive, FakeLLM(reply), dict(DEFAULTS, **(config or {})), act)
    return mind, acts


def test_the_internet_answer_comes_later_and_is_said_in_the_talk(world, clock, monkeypatch):
    """18.10, the owner: «подруби к нему интернет, чтоб он гуглить мог». web: the search runs in the background, the
    answer (no links, no markdown) is said as part of the talk and kept in the dialogue; a daily cap guards the money."""
    calls = []

    class Web(FakeLLM):
        def chat(self, system, user, **kw):
            calls.append(kw)
            return 'Курс доллара сегодня около **82 рублей** [ЦБ](https://cbr.ru).'
    mind, acts = make_mind(world)
    mind.llm = Web()
    monkeypatch.setattr(brainmod.threading, 'Thread',
                        lambda target, args, **kw: types.SimpleNamespace(start=lambda: target(*args)))
    ok, text = mind.do({'tool': 'web', 'args': {'query': 'курс доллара сегодня'}}, owner=True)
    assert ok and 'курс доллара' in text, text
    assert calls[0]['plugins'] == [{'id': 'web', 'max_results': 3}] and calls[0]['model'].endswith(':online')
    said = [d for k, d in acts if k == 'say']
    assert said == [{'text': 'Курс доллара сегодня около 82 рублей .', 'answer': True}]
    from datetime import datetime
    world.state.set('web_today', {'day': datetime.now(mind.tz).strftime('%Y-%m-%d'), 'n': brainmod.WEB_PER_DAY})
    ok, text = mind.do({'tool': 'web', 'args': {'query': 'ещё'}}, owner=True)
    assert not ok and 'кончились' in text


def test_old_technical_lessons_and_stale_plans_are_tidied(world, clock):
    world.state.set('character', {'traits': {}, 'lessons': [
        "Попытки ... ошибкой 'dock_retry_reference_expired' — системная проблема",
        'Таффа вечером обычно в коридоре']})
    world.state.set('plans', [{'what': 'старый план', 'when': 'завтра днём', 't': clock.now - 40 * 3600},
                              {'what': 'свежий', 'at_ts': clock.now + 600, 't': clock.now}])
    world.state.set('wants', [{'text': 'Хочу поехать в зал', 't': clock.now},
                              {'text': 'Хочу поехать в зал!', 't': clock.now}])
    make_mind(world)
    assert world.state.get('character')['lessons'] == ['Таффа вечером обычно в коридоре']
    assert [p['what'] for p in world.state.get('plans')] == ['свежий']
    assert len(world.state.get('wants')) == 1


def test_self_talk_never_narrates_actions(world, clock):
    mind, _ = make_mind(world)
    assert 'действия' in mind.check_say({'to': 'себе', 'text': 'Поеду в зал, посмотрю.'}, False)
    assert mind.check_say({'to': 'себе', 'text': 'Тихий вечер.'}, False) is None


def test_one_hello_to_taffa_then_just_be_glad(world, clock):
    world.dog['seen'] = clock.now
    mind, _ = make_mind(world)
    assert mind.check_say({'to': 'Таффа', 'text': 'Привет, Таффа!'}, False) is None
    mind.speak({'to': 'собака', 'text': 'Привет, Таффа!'})
    clock(60)
    world.dog['seen'] = clock.now
    reason = mind.check_say({'to': 'собака', 'text': 'Таффа, рад тебя видеть'}, False)
    assert reason and 'молча' in reason


def test_hush_keeps_initiative_quiet(world, clock):
    mind, _ = make_mind(world)
    ok, text = mind.inner('hush', {'minutes': 30})
    assert ok
    assert 'помолчать' in mind.check_say({'to': 'все', 'text': 'Есть кто дома?'}, False)
    mind.inner('hush', {'minutes': 0})
    assert world.state.get('hush_until') == 0


def test_approach_someone_in_view_not_the_chair(world, clock):
    # 08.10 00:41: «подъеду к нему поближе» went to the chair (object 150) the owner sat on.
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'approach', 'args': {'who': 'Сергей'}})
    assert not ok and 'не вижу' in text
    for _ in range(3):
        world.on_eyes({'items': [person()], 'stats': {'light': 120}})
        clock(.5)
    ok, text = mind.do({'tool': 'approach', 'args': {'who': 'Сергей'}})
    body = [d for k, d in acts if k == 'body_do']
    assert ok and body[-1]['skill'] == 'track'
    assert body[-1]['args']['mode'] == 'seek' and body[-1]['args']['distance'] == .8    # up to them, then the intent


def test_rest_on_the_dock_is_just_resting(world, clock):
    world.on_body({'pose': [0., 0., 0.], 'localized': True, 'room': 'Спальня и кабинет', 'docked': True,
                   'voltage': 16.6, 'twist': [0., 0.]})
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'rest', 'args': {}})
    assert ok and not [d for k, d in acts if k == 'body_do']


def test_dozes_off_when_nothing_happens(world, clock):
    world.on_body({'pose': [0., 0., 0.], 'localized': True, 'room': 'Спальня и кабинет', 'docked': True,
                   'voltage': 16.6, 'twist': [0., 0.]})
    mind, acts = make_mind(world, reply='{"thought": "Ночь, тихо. Подожду утра.", "do": {"tool": "rest", "args": {}}}')
    gaps = []
    for _ in range(4):
        mind.step('heartbeat')
        gaps.append(mind.next_heartbeat - clock.now)
        clock(10)
    assert gaps[0] < 1200 and gaps[-1] > 1250 and gaps[-1] > gaps[0]    # 15 min, then up to 30
    mind.step('person_appeared')                  # something happened: awake again
    assert mind.dozing == 0 and mind.next_heartbeat - clock.now < 800


def test_explore_is_remembered_only_when_done_and_a_shut_room_is_not_tried_again(world, clock):
    mind, acts = make_mind(world)
    ok, _ = mind.do({'tool': 'explore_room', 'args': {'room': 'Зал'}})
    assert ok and mind.last_explore == 'Зал' and 'Зал' not in mind.state.get('explored', {})
    mind.state.set('closed_ways', {'зал': clock.now})          # life, on «no_route»
    ok, text = mind.do({'tool': 'explore_room', 'args': {'room': 'Зал'}})
    assert not ok and 'не было дороги' in text


def test_a_plan_in_the_quiet_hours_waits_for_their_end(world, clock):
    # 08.10: «поехать в зал в 07:00» fired at 07:00 inside the quiet hours (till 10) and was lost to «не езжу».
    mind, acts = make_mind(world, config={'quiet_start_hour': 2, 'quiet_end_hour': 10, 'timezone': 'UTC'})
    clock.now = 1_800_000_000 - 1_800_000_000 % 86400 + 1 * 3600          # 01:00 UTC
    ok, text = mind.do({'tool': 'plan', 'args': {'what': 'поехать в зал', 'at': '07:00'}})
    assert ok and 'перенёс' in text and mind.state.get('plans')[-1]['when'] == 'в 10:00'
    plan = mind.state.get('plans')[-1]
    plan['at_ts'] = clock.now + 60                      # an old plan already due inside the quiet hours
    clock(120)
    assert mind.due(clock.now, True) != 'plan_due' and not plan.get('fired')
    assert mind.due(clock.now, False) == 'plan_due'


DOORS = [{'rooms': ['Коридор', 'Спальня и кабинет'], 'state': 'shut', 'seen': None},
         {'rooms': ['Коридор', 'Кухня'], 'state': 'open', 'seen': None},
         {'rooms': ['Коридор', 'Зал'], 'state': None, 'seen': 0}]


def test_a_shut_bedroom_door_means_one_room_and_says_which_door(world, clock):
    # 08.10: the bedroom door was shut; the mind decided «the hall and the kitchen are closed» and kept trying.
    doors = [dict(d, seen=clock.now - 60 if d['seen'] is None else d['seen']) for d in DOORS]
    world.on_body({'pose': [1.5, .4, 0.], 'localized': True, 'room': 'Спальня и кабинет', 'docked': False,
                   'voltage': 15.8, 'twist': [0., 0.], 'doors': doors})
    reach = world.reachable()
    assert set(k for k in reach if k != '_why') == {'Спальня и кабинет'}
    assert 'Спальня и кабинет' in reach['_why']['Кухня'] and 'Коридор' in reach['_why']['Кухня']
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'go', 'args': {'to': 'Кухня'}})
    assert not ok and 'не попасть' in text and not [d for k, d in acts if k == 'body_do']
    assert 'не попасть' in world.rooms_facts()
    # Shut in: the needs still come out whole (08.10 11:37–12:47 they failed and the mind thought without them).
    needs, goals, mood = mind.drives.text(clock.now, False)
    assert 'отдых' in needs and 'двери в другие комнаты закрыты' in needs and 'не разобрался' not in needs
    world.on_body_event({'kind': 'door', 'rooms': ['Коридор', 'Спальня и кабинет'], 'state': 'open'})
    clock(9)                                            # door news once the change has held (C39: a flicker is none)
    world.tick()
    assert 'door_opened' in kinds(world)
    world.on_body({**world.body, 'doors': [dict(doors[0], state='open', seen=clock.now)] + doors[1:]})
    ok, text = mind.do({'tool': 'go', 'args': {'to': 'Кухня'}})
    assert ok


def test_a_sound_without_words_and_joy_with_the_body(world, clock):
    mind, acts = make_mind(world, reply='{"thought": "Сергей пришёл!", "face": "happy", "sound": "happy", '
                                        '"do": {"tool": "emote", "args": {"kind": "wiggle"}}}')
    mind.step('heartbeat')
    assert not [d for k, d in acts if k == 'sfx']        # alone: no sounds (owner 08.10: only with people)
    for _ in range(3):
        world.on_eyes({'items': [person()], 'stats': {'light': 120}})
        clock(.5)
    mind.step('person_appeared')
    assert ('sfx', {'emotion': 'happy'}) in acts
    assert ('express', {'emote': 'wiggle'}) in acts           # body + lamp + sound, as one feeling (life decides)
    acts.clear()
    ok, text = mind.do({'tool': 'emote', 'args': {'kind': 'bounce'}})
    assert ok and acts[-1] == ('express', {'emote': 'bounce'}) and not [d for k, d in acts if k == 'body_do']
    clock(30)
    mind.step('heartbeat')                                  # a second sound so soon: not again
    assert [d for k, d in acts if k == 'sfx'].count({'emotion': 'happy'}) == 0
    assert mind.parse('{"thought": "x", "sound": "siren"}')['sound'] is None


def test_praise_and_scolding_are_felt_and_the_mood_drifts_back(world, clock, monkeypatch):
    from bratan.mind import affect as affectmod
    monkeypatch.setattr(affectmod, 'time', types.SimpleNamespace(time=clock.time, localtime=time.localtime))
    assert affectmod.heard_tone('Братан, ты молодец!') == 'praise'
    assert affectmod.heard_tone('фу, плохой') == 'scold'
    assert affectmod.heard_tone('ты не молодец') is None and affectmod.heard_tone('поехали на кухню') is None
    # 08.10 19:27: an answer is not praise; a feeling is not a scolding.
    for text in ('да нормально все хорошо', 'хорошо', 'отлично, поехали', 'лучше поехали на кухню', 'мне плохо',
                 'плохо слышно', 'глупый вопрос'):
        assert affectmod.heard_tone(text) is None, text
    for text in ('умница', 'хороший мальчик', 'ты хороший', 'спасибо, братан', 'ты у меня самый лучший'):
        assert affectmod.heard_tone(text) == 'praise', text
    for text in ('плохой робот', 'ты дурак', 'отстань', 'ты тупой'):
        assert affectmod.heard_tone(text) == 'scold', text
    mind, acts = make_mind(world, reply='Спасибо!\n###\n{"thought": "приятно", "face": "affectionate"}')
    before = mind.affect.now()[0]
    mind.converse('молодец, Братан', '')
    assert mind.affect.now()[0] > before + .2 and ('reflex', {'tone': 'praise'}) in acts
    assert 'praise' in kinds(world)
    high = mind.affect.now()[0]
    clock(3 * 3600)                                       # hours later: back near his own temperament
    assert abs(mind.affect.now()[0] - mind.affect.base_valence()) < .05 < high - mind.affect.base_valence()


def test_helper_tools_messages_alarms_timers_places_and_things(world, clock):
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'tell', 'args': {'to': 'Марина', 'text': 'ужин готов'}}, owner=True)
    assert ok and acts[-1] == ('play', {'game': 'deliver', 'who': 'Марина',
                                        'args': {'to': 'Марина', 'text': 'ужин готов', 'from': None}})
    ok, text = mind.do({'tool': 'timer', 'args': {'minutes': 10, 'label': 'чайник'}}, owner=True)
    assert ok and mind.state.get('plans')[-1]['errand'] == 'timer'
    clock(601)
    acts.clear()
    mind.due(clock.now, False)
    plays = [d for k, d in acts if k == 'play']
    assert plays and plays[0]['game'] == 'timer'
    assert mind.state.get('plans')[-1].get('fired')
    ok, text = mind.do({'tool': 'alarm', 'args': {'to': 'Сергей', 'at': '08:00'}}, owner=True)
    assert ok and mind.state.get('plans')[-1]['errand'] == 'wake'
    ok, text = mind.do({'tool': 'remember_place', 'args': {'name': 'у окна'}})
    assert ok and mind.state.get('places')[-1]['name'] == 'у окна'
    world.items = [{'label': 'backpack', 'score': .8, 'box': [0, 0, 9, 9], 'bearing': .05, 'map_xy': [2.0, .3]}]
    world.eyes_t = clock.now
    ok, text = mind.do({'tool': 'name_thing', 'args': {'name': 'мой рюкзак'}})
    assert ok
    assert 'мой рюкзак' in mind.things_in('Братан, где мой рюкзак?', clock.now)


def test_home_diary_and_weather_line(world, clock):
    from bratan.mind import weather as weathermod
    world.archive.event('heard_sound', {'kind': 'sound', 'sound': 'doorbell', 'words': 'звонок в дверь'}, t=clock.now)
    world.archive.event('door_opened', {'text': 'дверь между «Коридор» и «Кухня» открыли'}, t=clock.now)
    text = world.archive.diary(clock.now - 3600, clock.now + 10)
    assert 'звонили в дверь' in text and 'двери:' in text
    line = weathermod.describe({'current': {'temperature_2m': 7.2, 'apparent_temperature': 3.1, 'weather_code': 61,
                                            'wind_speed_10m': 5.4},
                                'daily': {'temperature_2m_min': [4, 2], 'temperature_2m_max': [9, 6],
                                          'precipitation_probability_max': [80, 10], 'weather_code': [61, 3]}})
    assert line.startswith('сейчас +7° (ощущается +3°), небольшой дождь') and 'осадки 80%' in line


def test_a_thought_alone_keeps_the_calm_face_a_real_event_shows_a_feeling(world, clock):
    # Owner 08.10: idle = one waiting face; feelings only for something real, so they can be read.
    mind, acts = make_mind(world, reply='{"thought": "Тихо тут.", "face": "sleepy"}')
    mind.step('heartbeat')
    assert not [d for k, d in acts if k == 'face']
    mind, acts = make_mind(world, reply='{"thought": "Сергей!", "face": "happy"}')
    mind.step('person_appeared')
    assert ('face', {'state': 'happy'}) in acts


def test_search_goes_where_he_was_last_seen(world, clock):
    world.archive.sighting('собака', 'Кухня', clock.now - 600)
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'search', 'args': {'who': 'Таффа'}})
    assert ok and 'Кухня' in text
    body = [d for k, d in acts if k == 'body_do']
    assert body and body[0]['skill'] == 'go'
    assert mind.skill_run and mind.skill_run['steps'][1]['tool'] == 'search'


def test_refused_words_do_not_wake_the_mind_again(world, clock):
    mind, _ = make_mind(world)
    mind.enact({'thought': 'x', 'say': {'to': 'Сергей', 'text': 'Привет, Сергей'}, 'do': None, 'face': None},
               'heartbeat', False)
    assert world.events[-1]['kind'] == 'quiet_note'
    mind.last_step = clock.now - 10
    assert mind.due(clock.now, False) is None


def test_dream_plans_get_a_time_and_fire(world, clock):
    reply = json.dumps({'traits': {}, 'lessons': [], 'self': '', 'wants': [], 'mood': 'спокойно', 'dream': 'сон',
                        'tomorrow': [{'what': 'проверить кухню', 'at': '11:15'}, 'поискать Таффу']},
                       ensure_ascii=False)
    mind, _ = make_mind(world, reply=reply)
    mind.dream(clock.now)
    plans = world.state.get('plans')
    assert len(plans) == 2 and all(p.get('at_ts') for p in plans)
    assert plans[0]['when'] == 'в 11:15'
    clock(30 * 3600)
    fired = [mind._plan_ready(p, clock.now) for p in plans]
    assert all(fired)


def test_each_night_dreams_otherwise_and_away_from_the_last_dreams(world, clock):
    """19.10, the owner: «ему снятся однообразные сны» — five nights of one dream (an open door, Таффа on a sofa,
    Сергей, «наконец разглядел»)."""
    mind, _ = make_mind(world)
    world.archive.event('dream', {'dream': 'Еду по бесконечному коридору, а все двери открыты.'}, t=clock.now - 86400)
    world.archive.event('converse', {'heard': 'расскажи что-нибудь интересное про космос', 'reply': '…'},
                        t=clock.now - 5 * 86400)
    world.state.set('dreamed_at', clock.now - 8 * 86400)               # no dream for a week (19.10: since 11.10)
    prompt = mind.dream_prompt(clock.now)
    assert 'ПРОШЛЫЕ СНЫ' in prompt and 'бесконечному коридору' in prompt
    assert 'про космос' in prompt                                       # an old memory to weave in
    assert any(kind in prompt for kind in mind.DREAM_KINDS)
    kinds = {next(k for k in mind.DREAM_KINDS if k in mind.dream_prompt(clock.now + n * 86400)) for n in range(8)}
    assert len(kinds) >= 4                                              # not the same kind of dream every night


def test_photo_keeps_the_minds_caption(tmp_path, monkeypatch):
    from bratan.mind import life as lifemod
    life = lifemod.Life.__new__(lifemod.Life)
    life.state = MindState(tmp_path / 'mind.json')
    life.captions = {'photo-1': 'Таффа на коврике'}
    life.on_photo({'id': 'photo-1', 'path': str(tmp_path / 'p.jpg'), 'items': [{'label': 'dog'}]})
    assert life.state.get('album')[-1]['caption'].startswith('Таффа на коврике')
    assert 'собака' in life.state.get('album')[-1]['caption']


# ---------------------------------------------------------------- voice face activity
def test_voice_shows_listening_thinking_speaking(monkeypatch):
    from bratan.voice import main as voicemod
    voice = voicemod.Voice.__new__(voicemod.Voice)
    voice.audio = types.SimpleNamespace(speaking=False)
    voice.dialog, voice.confirmer, voice.session = None, None, None
    voice.activity, voice.activity_wanted, voice.activity_seq = 'idle', ('idle', 0.), 0
    import queue
    voice.face_queue = queue.Queue()
    published = []
    voice.publish_state = lambda: published.append(voice.activity)   # at once, not with the next second (19.10)
    sent = []

    def tick(t):
        voice.activity_tick(t)
        while not voice.face_queue.empty():
            path = voice.face_queue.get_nowait()
            sent.append('wake' if path.startswith('/display') else path.split('state=')[1].split('&')[0])
    voice.confirmer = object()
    tick(0.)
    voice.confirmer, voice.dialog = None, {'waiting': True}
    tick(.1)
    voice.audio.speaking = True
    tick(.2)
    voice.audio.speaking = False                  # a gap between two sentences
    tick(.5)
    voice.audio.speaking = True
    tick(.7)
    voice.audio.speaking, voice.dialog = False, None
    tick(1.)
    tick(2.)
    tick(3.)
    # 19.10, the owner: on the dock the dark screen and the lamp must wake the moment it hears «Братан»
    assert sent == ['wake', 'listening', 'thinking', 'speaking', 'idle']
    assert published == ['listening', 'thinking', 'speaking', 'idle']


def test_while_the_name_is_checked_at_night_i_show_i_am_listening():
    from bratan.voice import main as voicemod
    voice = voicemod.Voice.__new__(voicemod.Voice)
    voice.audio = types.SimpleNamespace(speaking=False)
    voice.dialog, voice.confirmer, voice.session = None, None, None
    voice.night_checking = True
    assert voice.wanted_activity() == 'listening'
    voice.night_checking = False
    assert voice.wanted_activity() == 'idle'


def test_the_screen_stays_awake_while_the_voice_is_listening(world, clock):
    from tests.test_senses import make_life
    life = make_life(world, docked=True)
    life.quiet = lambda: True                       # the owner's quiet hours: all day
    world.voice = {'live': False, 'speaking': False, 'activity': 'idle'}
    life.display_awake, life.last_activity, life.display_said_at = False, clock.now - 100, clock.now
    sets = []
    life.set_display = lambda awake: sets.append(awake) or setattr(life, 'display_awake', awake)
    life.display_tick(clock.now)
    assert sets == []                               # dark and idle on the dock
    world.voice = {'live': False, 'speaking': False, 'activity': 'listening'}   # heard «Братан»
    life.display_tick(clock.now)
    assert sets == [True]


def test_back_home_after_hours_is_met_with_the_news(world, clock):
    world.archive.sighting('Сергей', 'Спальня и кабинет', clock.now - 5 * 3600)
    world.archive.event('heard_sound', {'kind': 'sound', 'sound': 'doorbell', 'words': 'звонок в дверь'},
                        t=clock.now - 3600)
    world.on_eyes({'items': [person('Сергей', face=True)], 'stats': {'light': 120}})
    back = [e for e in world.events if e['kind'] == 'person_appeared'][-1]
    assert 'не виделись' in back['text'] and 'звонили в дверь' in back['news']
    mind, _ = make_mind(world)
    text = mind.prompt(clock.now, 'person_appeared', False)
    assert 'ВСТРЕЧА: Сергей снова дома' in text
    assert 'ВСТРЕЧА' not in mind.prompt(clock.now, 'heartbeat', False)


def test_what_the_owner_asked_is_not_cut_short_by_a_passing_thought(world, clock):
    # 08.10 12:48: a dance asked for was replaced by the mind's own «поеду на базу отдохну».
    world.on_body({**world.body, 'skill': {'kind': 'dance', 'origin': 'owner', 'args': {'seconds': 20}}})
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'go', 'args': {'to': 'Зал'}})
    assert ok and 'доделаю' in text and not [d for k, d in acts if k in ('body_do', 'body_stop')]
    ok, text = mind.do({'tool': 'stop', 'args': {}})
    assert 'доделаю' in text and not [d for k, d in acts if k == 'body_stop']
    ok, text = mind.do({'tool': 'look_around', 'args': {}}, owner=True)       # the owner changes his mind: fine
    assert [d for k, d in acts if k == 'body_do']


def test_off_the_dock_is_said_plainly(world, clock):
    # 08.10 12:53: standing on the floor the mind twice thought «я на базе, заряжаюсь».
    text = world.situation(dict(DEFAULTS))
    assert 'НЕ на базе' in text and 'стою на полу' in text
    world.on_body({**world.body, 'docked': True})
    text = world.situation(dict(DEFAULTS))
    assert 'Стою на базе' in text and 'НЕ на базе' not in text


def test_wanting_attention_with_people_around_keeps_all_needs(world, clock):
    # 08.10 evening: «хочу внимания» had no words and every thought with people around ran without needs.
    world.archive.event('conversation_start', {}, t=clock.now - 3 * 3600)
    for _ in range(3):
        world.on_eyes({'items': [person('Сергей', face=True)], 'stats': {'light': 120}})
        clock(.5)
    mind, _ = make_mind(world)
    needs, goals, mood = mind.drives.text(clock.now, False)
    assert 'хочу внимания' in needs and 'не разобрался' not in needs



def test_who_i_talk_to_stays_when_the_face_turns_away(world, clock):
    # 08.10 18:51: Марина was recognized, turned away mid-talk, and «ты меня узнаёшь?» got «ты человек».
    for _ in range(2):
        world.on_eyes({'items': [person('Марина', face=True)], 'stats': {'light': 120}})
        clock(.5)
    clock(20)
    world.on_eyes({'items': [], 'stats': {'light': 120}})
    mind, _ = make_mind(world, reply='Конечно, ты Марина!\n###\n{"thought": "узнал", "do": null}')
    prompts = []
    chat = mind.llm.chat
    mind.llm.chat = lambda system, prompt, **kw: prompts.append(prompt) or chat(system, prompt, **kw)
    spoken, _ = mind.converse('ты меня узнаёшь', '')
    assert 'С КЕМ ГОВОРЮ: Марина' in prompts[-1] and '(разговор только начался)' in prompts[-1]
    assert world.state.get('dialogue')[-2]['who'] == 'Марина'
    clock(5)
    mind.converse('а как дела', '')
    assert 'я: Конечно, ты Марина!' in prompts[-1]      # my own answer is part of this talk
    assert brainmod.LOOK_WORDS.search('ты меня узнаешь') is None


def test_hello_to_everyone_goes_to_whoever_is_here(world, clock):
    # 08.10 18:45: «Привет, ребята» to «все» was refused because Сергей was in view.
    for _ in range(2):
        world.on_eyes({'items': [person('Сергей', face=True)], 'stats': {'light': 120}})
        clock(.5)
    mind, _ = make_mind(world)
    say = {'to': 'все', 'text': 'Привет, ребята.'}
    assert mind.check_say(say, False) is None and say['to'] == 'Сергей'



def test_come_to_me_and_refusals_in_words(world, clock):
    # 08.10: «подъедь ко мне» came as {"target": "Сергей"}; the refusal with tool hints was spoken aloud.
    for _ in range(2):
        world.on_eyes({'items': [person('Сергей', face=True)], 'stats': {'light': 120}})
        clock(.5)
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'approach', 'args': {'target': 'Сергей'}}, owner=True)
    assert ok and [d for k, d in acts if k == 'body_do' and d['skill'] == 'track']
    ok, text = mind.do({'tool': 'approach', 'args': {'who': 'ко мне'}}, owner=True)
    assert ok
    assert brainmod.said_reason('такого предмета нет в памяти — approach по id из «что помню вокруг»') == \
        'такого предмета нет в памяти'
    assert brainmod.said_reason('{"who": ...}') == 'не выходит'



def test_to_the_base_means_the_dock_and_known_slips_are_fixed(world, clock):
    # 08.10 20:06: «едь на базу» heard as «есть на базу» went to a place «База» and failed.
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'go', 'args': {'to': 'База'}}, owner=True)
    assert ok and [d for k, d in acts if k == 'body_do' and d['skill'] == 'dock']
    assert brainmod.fix_heard('есть на базу') == 'едь на базу'
    assert brainmod.fix_heard('подожди ваза же тоже находится в спальне') == 'подожди база же тоже находится в спальне'
    assert brainmod.fix_heard('красивая ваза') == 'красивая ваза'
    assert brainmod.fix_heard('поесть пожалуйста в зал побудь там минут 10') == 'поезжай пожалуйста в зал побудь там минут 10'
    assert brainmod.fix_heard('хочу поесть на кухне') == 'хочу поесть на кухне'       # really eating: kept
    from bratan.voice.local import request
    assert request('братан привет') == 'привет' and request('эй братан как дела') == 'как дела'



def test_a_turn_is_short():
    text = ('Сегодня я в основном катался по дому: зал, кухню, коридор, спальню. Рассматривал стол, кровать и '
            'кресло, а потом заряжаюсь. И ещё видел Таффу два раза, он бегал по коридору туда и сюда.')
    short = brainmod.clip_words(text, brainmod.SPOKEN_WORDS)
    assert short.endswith('заряжаюсь.') and 'Таффу' not in short
    assert brainmod.clip_words('Очень длинное предложение ' * 10, 5).startswith('Очень')     # one sentence at least



def test_go_to_a_thing_by_its_name(world, clock):
    # 08.10 21:01: «съезди к кровати» -> «кровать я сейчас не вижу»: the talk did not know the things around.
    world.archive.saw('bed', (1.2, .9), 'Спальня и кабинет', clock.now, count=5)
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'approach', 'args': {'who': 'кровати'}}, owner=True)
    assert ok and [d for k, d in acts if k == 'body_do' and d['skill'] == 'approach']
    assert mind.thing_named('Сергей') is None and mind.thing_named('кровать')['label'] == 'bed'




def test_two_asked_actions_go_one_after_the_other(world, clock):
    # 08.10 bench: «сначала покрутись, а потом вернись на базу» did only the first.
    reply = ('Покручусь и вернусь.\n###\n{"thought": "", "do": {"tool": "dance", "args": {}}, '
             '"then": {"tool": "rest", "args": {}}}')
    mind, acts = make_mind(world, reply=reply)
    mind.converse('сначала покрутись а потом вернись на базу', '')
    assert [d['skill'] for k, d in acts if k == 'body_do'] == ['dance']
    mind.run_then()
    assert [d['skill'] for k, d in acts if k == 'body_do'] == ['dance', 'dock']


def test_a_feeling_it_has_no_move_for_and_errands_with_other_words(world, clock):
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'emote', 'args': {'kind': 'meow'}}, owner=True)
    assert ok
    for _ in range(2):
        world.on_eyes({'items': [person('Сергей', face=True)], 'stats': {'light': 120}})
        clock(.5)
    ok, text = mind.do({'tool': 'alarm', 'args': {'time': '08:00', 'text': 'вставай'}}, owner=True)
    assert ok and 'Сергей' in text



def test_on_the_dock_nobody_is_greeted(world, clock):
    # Owner 08.10 21:30: on the dock, seeing him, it kept trying to talk.
    world.on_body({**world.body, 'docked': True})
    clock(30)
    world.tick()
    clock(5)
    mind, acts = make_mind(world)
    mind.last_step = clock.now
    mind.next_heartbeat = clock.now + 900
    for _ in range(3):
        world.on_eyes({'items': [person('Сергей', face=True)], 'stats': {'light': 120}})
        clock(.5)
    assert 'person_appeared' in kinds(world)
    assert mind.due(clock.now + 5, False) is None                      # no thought just for that
    assert 'не заговариваю' in mind.check_say({'to': 'Сергей', 'text': 'Привет!'}, False)
    ok, text = mind.do({'tool': 'approach', 'args': {'who': 'Сергей'}})
    assert not ok and not [d for k, d in acts if k == 'body_do']
    ok, text = mind.do({'tool': 'approach', 'args': {'who': 'Сергей'}}, owner=True)   # asked: goes
    assert ok



# ---------------------------------------------------------------- C36: stage 0 of the 09.10 liveness audit
def test_own_skill_steps_follow_only_their_own_commands(world, clock):
    # 09.10 audit: the next step went after ANY skill's end (a dock for the battery, the owner's errand), and a
    # failed step did not stop the chain.
    mind, acts = make_mind(world)
    ok, _ = mind.do({'tool': 'new_skill', 'args': {'name': 'поклон', 'steps': [
        {'tool': 'emote', 'args': {'kind': 'nod'}}, {'tool': 'look_around'}, {'tool': 'emote', 'args': {'kind': 'bow'}}]}})
    assert ok
    ok, _ = mind.do({'tool': 'use_skill', 'args': {'name': 'поклон'}})
    assert ok and mind.skill_run['wait'] == 'emote-nod'
    assert mind.next_skill_step({'id': 'energy-dock', 'status': 'done'}) is None      # not my step
    assert not [d for k, d in acts if k == 'body_do']
    mind.next_skill_step({'id': 'emote-nod', 'status': 'done'})
    assert [d['skill'] for k, d in acts if k == 'body_do'] == ['look_around'] and mind.skill_run['wait']
    ok, text = mind.next_skill_step({'id': mind.skill_run['wait'], 'status': 'failed'})
    assert not ok and mind.skill_run is None                                           # a failed step ends it
    assert not [d for k, d in acts if k == 'express' and d.get('emote') == 'bow']


def test_the_owners_errand_drops_my_own_skill(world, clock):
    mind, acts = make_mind(world)
    mind.do({'tool': 'new_skill', 'args': {'name': 'обход', 'steps': [{'tool': 'look_around'},
                                                                        {'tool': 'wait', 'args': {'seconds': 5}}]}})
    mind.do({'tool': 'use_skill', 'args': {'name': 'обход'}})
    first = mind.skill_run['wait']
    world.on_body({**world.body, 'skill': {'kind': 'go', 'origin': 'owner', 'id': 'o1'}})
    ok, text = mind.next_skill_step({'id': first, 'status': 'done'})
    assert not ok and mind.skill_run is None and 'попросили' in text
    assert [d['skill'] for k, d in acts if k == 'body_do'] == ['look_around']


def test_a_pause_in_my_skill_is_waited_for(world, clock):
    mind, acts = make_mind(world)
    mind.do({'tool': 'new_skill', 'args': {'name': 'пауза', 'steps': [{'tool': 'wait', 'args': {'seconds': 5}},
                                                                        {'tool': 'look_around'}]}})
    mind.do({'tool': 'use_skill', 'args': {'name': 'пауза'}})
    assert [d['skill'] for k, d in acts if k == 'body_do'] == ['wait']
    mind.next_skill_step({'id': mind.skill_run['wait'], 'status': 'done'})
    assert [d['skill'] for k, d in acts if k == 'body_do'] == ['wait', 'look_around']


def test_the_same_want_in_other_words_is_one_want(world, clock):
    # 09.10 audit: four of eight wants retold «осмотреть зал»; similarity by letters did not see it.
    world.state.set('wants', [
        {'text': 'Обследовать зал — давно там не был, интересно, что изменилось.', 't': clock.now},
        {'text': 'Хочу поехать в зал и посмотреть, что там изменилось с прошлого раза.', 't': clock.now},
        {'text': 'Хочу завтра с утра поехать в зал и спокойно всё там осмотреть, без спешки.', 't': clock.now},
        {'text': 'Проверить спальню и кабинет утром, когда откроют дверь — найти Таффу или Сергея.', 't': clock.now},
        {'text': 'Хочу поехать на кухню', 't': clock.now}])
    mind, _ = make_mind(world)
    wants = world.state.get('wants')
    assert [w['text'][:12] for w in wants] == ['Обследовать ', 'Проверить сп', 'Хочу поехать']
    assert wants[0]['times'] == 3
    assert brainmod.same_want('Хочу в ближайшие дни спокойно осмотреть зал, пока там никого нет.', wants[0]['text'])
    assert not brainmod.same_want('Хочу поехать в зал', 'Хочу поехать на кухню')
    ok, text = mind.do({'tool': 'want', 'args': {'text': 'Хочу в ближайшие дни спокойно осмотреть зал'}})
    assert 'уже' in text and len(world.state.get('wants')) == 3
    assert '(хочу давно)' in mind.prompt(clock.now, 'heartbeat', False)


def test_character_stays_near_his_temperament_and_waiting_is_not_a_lesson(world, clock):
    # 09.10 audit: «cautious» .86 -> .92 in three nights; lessons taught to wait on the dock.
    world.state.set('character', {'traits': {'cautious': .92, 'curious': .8, 'talkative': .43, 'tidy': .76},
                                  'lessons': ['Попытки вернуться на базу при плохой локализации у док-станции часто '
                                              'проваливаются — лучше не настаивать, а просто ждать в стороне.',
                                              'Когда тело бодрое, а делать нечего — лучше не мельтешить по дому, а '
                                              'дождаться утра на базе, чтобы не копить ошибки.',
                                              'Если Сергей просит не мешать, это значит не просто сидеть на базе, а не '
                                              'инициировать контакт — можно тихо рассматривать предметы рядом.',
                                              'Ночью в зале и на кухне темно — осматривать лучше при дневном свете.']})
    world.state.set('self', 'Я Братан, робот-корги.')
    reply = json.dumps({'traits': {'cautious': 1.0, 'curious': .5},
                        'lessons': ['Лучше подождать на базе, пока не позовут.', 'Таффа вечером спит в коридоре.'],
                        'self': 'Я Братан, люблю кататься по дому и смотреть на своих.', 'wants': [],
                        'mood': 'спокойно', 'dream': 'сон', 'tomorrow': []}, ensure_ascii=False)
    mind, _ = make_mind(world, reply=reply)
    character = world.state.get('character')
    assert character['traits']['cautious'] == .8 and character['traits']['tidy'] == .7       # back in the band
    assert len(character['lessons']) == 2 and len(character['frozen_lessons']) == 2
    assert 'не настаивать' not in mind.prompt(clock.now, 'heartbeat', False)
    mind.dream(clock.now)
    character = world.state.get('character')
    assert character['traits']['cautious'] <= .8                     # a step of .03 at most, then back towards .6
    assert .74 <= character['traits']['curious'] <= .8
    assert 'Таффа вечером спит в коридоре.' in character['lessons']
    assert 'Лучше подождать на базе, пока не позовут.' in character['frozen_lessons']
    assert world.state.get('self').startswith('Я Братан, люблю')
    assert world.state.get('self_history')[-1]['text'] == 'Я Братан, робот-корги.'


def test_reflection_leaves_who_i_am_to_the_night(world, clock):
    world.state.set('self', 'Я Братан, робот-корги.')
    reply = json.dumps({'self': 'Я теперь совсем другой.', 'insight': 'Лучше просто ждать на базе, пока позовут.',
                        'want': 'Хочу сфотографировать Таффу у окна'}, ensure_ascii=False)
    mind, _ = make_mind(world, reply=reply)
    mind.reflect(clock.now)
    assert world.state.get('self') == 'Я Братан, робот-корги.'
    assert not world.state.get('notes')                              # a lesson in waiting is not written down
    assert world.state.get('wants')[-1]['text'] == 'Хочу сфотографировать Таффу у окна'


def test_asleep_on_the_dock_no_thoughts_but_promises_are_kept(world, clock):
    # 09.10 audit: in the quiet hours the pulse woke the model every 30 min («Ночь, все спят…» 36 times a night).
    world.on_body({**world.body, 'docked': True})
    clock(30)
    world.tick()
    clock(5)
    mind, acts = make_mind(world)
    mind.do({'tool': 'timer', 'args': {'minutes': 1, 'label': 'чайник'}}, owner=True)
    world.event('sound', 'громко хлопнула дверь', sound='bang')
    mind.next_heartbeat = clock.now - 1
    clock(61)
    assert mind.due(clock.now, True) is None                         # asleep: neither the sound nor the pulse
    assert [d for k, d in acts if k == 'play' and d.get('game') == 'timer']      # the timer still rang
    assert mind.due(clock.now, False) == 'sound'                     # awake: the night's news wakes a thought


def test_an_empty_answer_is_not_a_thought(world, clock):
    mind, _ = make_mind(world, reply='не json вовсе')
    mind.step('person_appeared')
    assert not world.state.get('journal')


def test_what_would_be_refused_is_said_before_asking(world, clock):
    # 09.10 17:10-18:12: 18 of 98 thoughts were refusals, five of them «look round again» a minute after one.
    mind, _ = make_mind(world)
    not_now = lambda trigger='heartbeat': next((x for x in mind.prompt(clock.now, trigger, False).split('\n')
                                                if x.startswith('СЕЙЧАС НЕ ВЫЙДЕТ')), '')
    assert not_now() == ''
    assert mind.do({'tool': 'look_around', 'args': {}})[0]
    clock(30)
    assert 'look_around и search здесь же — оглядывался 30 с назад' in not_now()
    ok, why = mind.do({'tool': 'look_around', 'args': {}})
    assert not ok and 'только что оглядывался' in why                  # the line tells what do() would say
    assert 'look_around' not in not_now('dog_appeared')                # a newcomer is worth a look round
    clock(200)
    assert not_now() == ''


def test_rest_while_the_dock_settles_is_not_a_new_dock(world, clock):
    # 09.10 22:55: docked, the dock skill still settling — «rest» started a new dock: off the contacts and back.
    world.on_body({**world.body, 'docked': True, 'skill': {'kind': 'dock', 'stage': 'charging', 'origin': 'mind'}})
    mind, acts = make_mind(world)
    assert mind.do({'tool': 'rest', 'args': {}}) == (True, 'отдыхаю на базе')
    world.on_body({**world.body, 'docked': False, 'skill': {'kind': 'dock', 'stage': 'route', 'origin': 'mind'}})
    assert mind.do({'tool': 'rest', 'args': {}})[0] and not [a for a in acts if a[0] == 'body_do']


def test_key_thoughts_are_said_aloud_while_driving(world, clock):
    # owner 10.10: «пусть вслух произносит ключевые мысли, пока ездит — не все»
    world.on_body({**world.body, 'docked': False, 'skill': {'kind': 'go', 'origin': 'mind'}})
    mind, acts = make_mind(world)
    spoken = lambda: [d['text'] for k, d in acts if k == 'say' and d.get('aloud')]
    go = {'tool': 'go', 'args': {'to': 'Зал'}}
    decision = {'thought': 'Зарядился, пора двигаться. Поеду в зал, там давно не был.', 'say': None, 'do': go, 'face': None}
    mind.enact(decision, 'heartbeat', False)
    assert spoken() == ['Зарядился, пора двигаться.']                         # the first sentence only
    mind.enact({**decision, 'thought': 'В зале пусто, никого нет рядом совсем.'}, 'heartbeat', False)
    assert len(spoken()) == 1                                                  # not twice within 45 s
    clock(60)
    mind.enact({**decision, 'thought': 'Зарядился, пора двигаться. Поеду в зал.'}, 'heartbeat', False)
    assert len(spoken()) == 1                                                  # the same line is not said again
    mind.enact({**decision, 'thought': 'Таффа на диване, подъеду и сфотографирую.',
                'do': {'tool': 'approach', 'args': {'who': 'собака'}}}, 'dog_appeared', False)
    assert spoken()[-1] == 'Таффа на диване, подъеду и сфотографирую.'
    clock(60)
    mind.enact({**decision, 'thought': 'Тихо и спокойно вокруг, постою немного.', 'do': None}, 'heartbeat', False)
    assert len(spoken()) == 2                                                  # an idle thought is not a key one


def test_thoughts_stay_silent_on_the_dock_at_night_in_a_talk_or_when_asked_to_hush(world, clock):
    mind, acts = make_mind(world)
    decision = {'thought': 'Зарядился, пора двигаться. Поеду в зал.', 'say': None,
                'do': {'tool': 'go', 'args': {'to': 'Зал'}}, 'face': None}
    aloud = lambda: [1 for k, d in acts if k == 'say' and d.get('aloud')]
    world.on_body({**world.body, 'docked': True})
    mind.enact(decision, 'heartbeat', False)                                   # on the dock
    world.on_body({**world.body, 'docked': False})
    mind.enact(decision, 'heartbeat', True)                                    # quiet hours
    world.voice = {'live': True}
    mind.enact(decision, 'heartbeat', False)                                   # in a talk
    world.voice = {'live': False}
    world.state.set('hush_until', clock.now + 600)
    mind.enact(decision, 'heartbeat', False)                                   # asked to be quiet
    assert not aloud()
    world.state.set('hush_until', 0)
    mind.config['think_aloud'] = False
    mind.enact(decision, 'heartbeat', False)                                   # switched off
    assert not aloud()
    mind.config['think_aloud'] = True
    mind.enact(decision, 'heartbeat', False)
    assert len(aloud()) == 1


def test_one_line_of_mood(world, clock):
    # 09.10 audit: «спокойный (бодрость 0,2); бодрый; …» — two moods in one line contradicted each other.
    mind, _ = make_mind(world)
    line = next(x for x in mind.prompt(clock.now, 'heartbeat', False).split('\n')
                if x.startswith('КАК Я СЕБЯ ЧУВСТВУЮ'))
    assert 'бодрость' in line and line.count(';') <= 1 and 'на душе хорошо' not in line


def test_the_last_half_hour_survives_a_restart_as_context_not_news(world, clock):
    world.archive.event('person_appeared', {'text': 'вижу Сергея', 'who': 'Сергей'}, t=clock.now - 300)
    world.archive.event('door_opened', {'text': 'дверь открыли'}, t=clock.now - 4000)         # too old
    world.archive.event('mind_step', {'thought': 'x', 'text': 'y'}, t=clock.now - 200)        # not a world event
    again = worldmod.World(ROOMS, 'Таффа', world.archive, None, state=world.state)
    assert [(e['kind'], e['text']) for e in again.events] == [('person_appeared', 'вижу Сергея')]
    assert again.recent(0) == []
    again.on_body(world.body)
    mind, _ = make_mind(again)
    mind.next_heartbeat = clock.now + 600
    assert mind.due(clock.now + 5, False) is None
    assert 'вижу Сергея' in mind.prompt(clock.now, 'heartbeat', False)


def test_at_night_only_a_clear_call_starts_a_talk(monkeypatch):
    # 09.10 01:08: dictation at the PC — Vosk made out «братан…», Microsoft heard no name — and he said
    # «Не расслышал, повтори?» in the quiet hours.
    from bratan.voice import main as voicemod
    voice = voicemod.Voice.__new__(voicemod.Voice)
    voice.config = {'quiet_start_hour': 0, 'quiet_end_hour': 24, 'timezone': 'UTC'}
    voice.dialog = voice.session = None
    events, started = [], []
    voice.event = lambda kind, **data: events.append(kind)
    voice.start_dialog = lambda text, utterance=None: started.append(text)
    said = types.SimpleNamespace(result=lambda: 'весьма там все статистику охвата весь мировой опыт')
    voice._night_call(said, 'братан лишь на танк весь мировой опыт')
    assert started == [] and events == ['night_not_addressed']
    said = types.SimpleNamespace(result=lambda: 'братан который час')
    voice._night_call(said, 'братан который час')
    assert started == ['братан который час']
    voice.dialog = {'id': 'local-1', 'said': [], 'heard': ['что']}
    voice.say_queue = __import__('queue').Queue()
    closed = []
    voice.end_dialog = lambda reason: closed.append(reason)
    voice.on_message('voice.answer_part', {'id': 'local-1', 'text': 'Не расслышал, повтори?'})
    voice.on_message('voice.answer', {'id': 'local-1', 'text': 'Не расслышал, повтори?', 'streamed': True})
    assert voice.say_queue.empty() and closed == ['night_unclear']



def test_the_second_asked_action_waits_for_the_first_one_itself(world, clock):
    reply = ('Покручусь и вернусь.\n###\n{"thought": "", "do": {"tool": "dance", "args": {}}, '
             '"then": {"tool": "rest", "args": {}}}')
    mind, acts = make_mind(world, reply=reply)
    mind.converse('сначала покрутись а потом вернись на базу', '')
    dance = mind.then[2]
    assert dance and [d['skill'] for k, d in acts if k == 'body_do'] == ['dance']
    assert mind.run_then({'id': 'energy-dock', 'status': 'done', 'origin': 'owner'}) is None    # not the dance
    assert [d['skill'] for k, d in acts if k == 'body_do'] == ['dance']
    mind.run_then({'id': dance, 'status': 'done', 'origin': 'owner'})
    assert [d['skill'] for k, d in acts if k == 'body_do'] == ['dance', 'dock']


def test_stop_takes_the_second_asked_action_with_it(world, clock):
    reply = ('Покручусь и вернусь.\n###\n{"thought": "", "do": {"tool": "dance", "args": {}}, '
             '"then": {"tool": "rest", "args": {}}}')
    mind, acts = make_mind(world, reply=reply)
    mind.converse('сначала покрутись а потом вернись на базу', '')
    assert mind.run_then({'id': mind.then[2], 'status': 'cancelled', 'origin': 'owner'}) is None
    assert [d['skill'] for k, d in acts if k == 'body_do'] == ['dance'] and mind.then is None



# ---------------------------------------------------------------- C37: asked to come, asked to find
def test_asked_to_find_or_come_he_looks_round_and_comes_up(world, clock):
    # 09.10 10:16: «подъедь к Таффе» came as search (done at once, «found», he stayed), 10:18 «найди меня» only
    # turned; 10:17 approach went to the bed Taffa lay by.
    mind, acts = make_mind(world)
    for request in ({'tool': 'search', 'args': {'who': 'Таффа'}}, {'tool': 'approach', 'args': {'who': 'собака'}},
                    {'tool': 'search', 'args': {'who': 'Сергей'}}, {'tool': 'approach', 'args': {'who': 'меня'}}):
        acts.clear()
        ok, text = mind.do(request, owner=True)
        body = [d for k, d in acts if k == 'body_do']
        assert ok and body and body[0]['skill'] == 'track' and body[0]['args']['mode'] == 'seek', (request, text)
        assert body[0]['args']['seconds'] >= 40
    assert body[0]['args']['who'] == 'человек'                         # nobody known in view: anyone in view
    mind.do({'tool': 'search', 'args': {'who': 'собака'}})               # the mind's own search: turn and look
    assert [d for k, d in acts if k == 'body_do'][-1]['args']['mode'] == 'search'


def test_asked_to_find_me_on_the_dock_he_leaves_it(world, clock):
    world.on_body({**world.body, 'docked': True})
    clock(30)
    world.tick()
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'search', 'args': {'who': 'Сергей'}}, owner=True)
    assert ok and 'с базы' not in text
    assert [d['args']['mode'] for k, d in acts if k == 'body_do'] == ['seek']
    ok, text = mind.do({'tool': 'approach', 'args': {'who': 'собака'}})
    assert not ok and 'пусть подойдёт сам' in text                      # his own idea: no driving at the dog


def test_after_a_talk_a_word_reaches_them_out_of_view(world, clock):
    mind, _ = make_mind(world)
    assert 'не вижу' in mind.check_say({'to': 'Сергей', 'text': 'Не нашёл тебя.'}, False)
    world.state.heard(clock.now, 'Сергей', 'найди меня')
    clock(30)
    assert mind.check_say({'to': 'Сергей', 'text': 'Не нашёл тебя — ты где?'}, False) is None
    clock(120)
    assert 'не вижу' in mind.check_say({'to': 'Сергей', 'text': 'Ты где?'}, False)


def test_a_cat_is_our_dog_seen_badly():
    from bratan.eyes.detect import merge_pet_labels
    items = merge_pet_labels([{'label': 'cat', 'score': .5, 'box': [0, 0, 100, 100]},
                              {'label': 'dog', 'score': .6, 'box': [200, 0, 300, 100]},
                              {'label': 'cat', 'score': .41, 'box': [205, 5, 300, 100]},
                              {'label': 'person', 'score': .9, 'box': [0, 0, 50, 200]}])
    assert [i['label'] for i in items] == ['dog', 'dog', 'person'] and items[0]['seen_as'] == 'cat'


def at(xy, name=None, face=False, track=None):
    return {**person(name, face), 'map_xy': list(xy), 'track': track}


def seen_now(world, *items):
    world.on_eyes({'items': list(items), 'stats': {'light': 120}})
    return {who for who, entry in world.presence.items() if entry['seen'] == clock_now(world)}


def clock_now(world):
    return max(entry['seen'] for entry in world.presence.values())


@pytest.mark.parametrize('after_s, xy, want', [
    (180., (2.3, 1.1), 'Сергей'),        # face covered, the same place 3 min later: him
    (180., (4.5, 1.0), 'человек'),       # across the room: someone
    (360., (2.3, 1.1), 'человек'),       # six minutes on: not assumed
])
def test_someone_at_the_place_a_known_person_was_minutes_ago_is_them(world, clock, after_s, xy, want):
    """19.10, the owner: «стоит мне чуток лицо закрыть — он меня не узнаёт; пусть считает, что это тот же человек, если
    он на том же месте в течение 5 минут»."""
    world.on_eyes({'items': [at((2.0, 1.0), 'Сергей', face=True, track=1)], 'stats': {'light': 120}})
    clock(after_s)
    world.on_eyes({'items': [at(xy, track=7)], 'stats': {'light': 120}})
    assert want in seen_now(world)


def test_two_known_people_at_that_place_leave_it_unknown(world, clock):
    world.on_eyes({'items': [at((2.0, 1.0), 'Сергей', face=True, track=1)], 'stats': {'light': 120}})
    world.on_eyes({'items': [at((2.4, 1.2), 'Марина', face=True, track=2)], 'stats': {'light': 120}})
    clock(60.)
    world.on_eyes({'items': [at((2.2, 1.1), track=9)], 'stats': {'light': 120}})
    assert 'человек' in seen_now(world)
