"""Senses and quick reactions (C37): turning to a rustle, checking a thing on the way, where to look first.

python -m pytest tests/test_senses.py
"""
import asyncio
import math
import types

from bratan.body import skills
from bratan.mind import life as lifemod
from tests.test_mind import clock, make_mind, person, world  # noqa: F401  (fixtures)


# ---------------------------------------------------------------- an approach checks its thing on the way
class SeeCtx:
    def __init__(self, pose=(0., 0., 0.), hit=False, light=120):
        self.pose, self.hit, self.light, self.frame = pose, hit, light, 0

    def sees(self, label, bearing, tolerance=.3):
        self.frame += 1
        return self.frame, self.hit, self.light


def approach(xy, label='cup'):
    skill = skills.Approach({'xy': list(xy), 'label': label, 'distance': .6}, id='a', origin='mind')
    skill.kind = 'approach'
    return skill


def test_a_thing_not_there_any_more_stops_the_approach():
    skill, ctx = approach((2., 0.)), SeeCtx()
    results = [skill.missing(ctx) for _ in range(skills.VERIFY_FRAMES)]
    assert results[-1] and not any(results[:-1])                          # ten looks in good view: gone


def test_a_thing_seen_once_is_confirmed_and_not_checked_again():
    skill, ctx = approach((2., 0.)), SeeCtx(hit=True)
    assert not skill.missing(ctx) and skill.info['confirmed']
    ctx.hit = False
    assert not any(skill.missing(ctx) for _ in range(30))


def test_no_verdict_out_of_view_too_far_too_close_or_in_the_dark():
    for xy, light in (((2., 1.5), 120), ((4., 0.), 120), ((.6, 0.), 120), ((2., 0.), 20)):
        skill, ctx = approach(xy), SeeCtx(light=light)
        assert not any(skill.missing(ctx) for _ in range(30)), xy
    skill, ctx = approach((2., 0.), label=None), SeeCtx()                 # furniture is not checked
    assert not any(skill.missing(ctx) for _ in range(30))


def test_only_things_that_can_be_taken_away_are_checked(world, clock):
    mind, acts = make_mind(world)
    world.archive.saw('cup', (2.5, .4), 'Спальня и кабинет', clock.now)
    world.archive.saw('bed', (3.5, -.5), 'Спальня и кабинет', clock.now)
    ids = {o['label']: o['id'] for o in world.archive.objects(min_count=1)}
    for item_id, label in ((ids['cup'], 'cup'), (ids['bed'], None)):
        acts.clear()
        mind.do({'tool': 'approach', 'args': {'id': item_id}}, owner=True)
        body = [d for k, d in acts if k == 'body_do']
        assert body and body[0]['args'].get('label') == label


# ---------------------------------------------------------------- turning to a rustle
def make_life(world, docked=False, skill=None, live=False):
    life = lifemod.Life.__new__(lifemod.Life)
    life.world, life.config = world, {'autonomy': True, 'pets': {'dog': 'Таффа'}, 'timezone': 'UTC'}
    world.on_body({**world.body, 'docked': docked, 'skill': skill})
    world.voice = {'live': live}
    life.play = types.SimpleNamespace(active=False)
    life.display_awake = True
    life.sent, life.gazes, life.faces = [], [], []
    life.publish = lambda topic, data: life.sent.append((topic, data))
    life.look_at = lambda x, y: life.gazes.append((x, y))
    life.react = lambda state: life.faces.append(state)
    life.quiet = lambda: False
    return life


def turns(life):
    return [d for t, d in life.sent if t == 'body.do' and d['origin'] == 'reflex']


def test_a_rustle_beside_turns_the_eyes_then_the_body(world, clock):
    life = make_life(world)
    life.on_motion({'bearing': -1.6, 'distance': 1.2})                    # on the right
    assert life.gazes and life.gazes[0][0] == -1.                         # eyes first, to that side
    assert [d['args']['angle'] for d in turns(life)] == [-1.6]
    assert 'motion_seen' in [e['kind'] for e in world.events]


def test_not_on_the_dock_not_during_an_errand_or_a_talk(world, clock):
    for kwargs in ({'docked': True}, {'skill': {'kind': 'go', 'origin': 'owner'}}, {'live': True}):
        life = make_life(world, **kwargs)
        life.on_motion({'bearing': 1.7, 'distance': 1.})
        assert not turns(life), kwargs
    life = make_life(world, skill={'kind': 'look_around', 'origin': 'mind'})   # an own look round: yes
    life.on_motion({'bearing': 1.7, 'distance': 1.})
    assert turns(life)


def test_not_again_and_again_to_the_same_place(world, clock):
    life = make_life(world)
    now = [1000.]
    real_time = lifemod.time.time
    lifemod.time.time = lambda: now[0]
    try:
        for at in (0., 1., 3.1, 6.2, 9.):                                  # rustles on the left
            now[0] = 1000. + at
            life.on_motion({'bearing': 1.6, 'distance': 1.})
        assert len(turns(life)) == 2                                       # not closer than 3 s, 2 in 10 s
        now[0] = 1009.5
        life.on_motion({'bearing': -1.6, 'distance': 1.})                  # something new on the right: yes
        assert len(turns(life)) == 3
    finally:
        lifemod.time.time = real_time


def test_after_the_turn_the_mind_hears_what_was_there(world, clock):
    life = make_life(world)
    life.ORIENT_LOOK_S = 0.
    for _ in range(3):
        world.on_eyes({'items': [person('Сергей', face=True)], 'stats': {'light': 120}})
        clock(.5)
    asyncio.run(life.look_after_orienting({'side': 'справа', 'heading': 0., 'interrupted': None}))
    assert world.events[-1]['kind'] == 'oriented' and 'Сергей' in world.events[-1]['text']


# ---------------------------------------------------------------- where to look first
def test_a_search_starts_where_they_were_last_seen(world, clock):
    world.on_body({**world.body, 'pose': [0., 0., 0.]})
    for _ in range(3):
        world.on_eyes({'items': [{**person('Сергей', face=True), 'map_xy': [0., 2.]}], 'stats': {'light': 120}})
        clock(.5)
    clock(20)                                                              # gone from view since
    assert abs(world.where_last('Сергей') - math.pi / 2) < .01             # to the left
    mind, acts = make_mind(world)
    mind.do({'tool': 'approach', 'args': {'who': 'Сергей'}}, owner=True)
    seek = [d for k, d in acts if k == 'body_do'][0]['args']
    assert seek['mode'] == 'seek' and abs(seek['bearing'] - math.pi / 2) < .01


# ---------------------------------------------------------------- the executive: thinking on the go
from bratan.mind import executive as execmod  # noqa: E402


def make_exec(world, clock, skill):
    mind, acts = make_mind(world)
    world.on_body({**world.body, 'skill': skill})
    sent = []
    life = types.SimpleNamespace(
        world=world, mind=mind, config={'autonomy': True, 'pets': {'dog': 'Таффа'}},
        play=types.SimpleNamespace(active=False), archive=world.archive, state=world.state,
        publish=lambda topic, data: sent.append((topic, data)), pool=None,
        loop=types.SimpleNamespace(run_in_executor=lambda pool, fn, *args: fn(*args)))
    return execmod.Executive(life), mind, acts, sent


def test_found_on_the_way_goes_to_them_at_once(world, clock):
    # A search took him to another room; the one he looks for is right here on the way.
    ex, mind, acts, sent = make_exec(world, clock, {'kind': 'go', 'origin': 'mind', 'id': 'm1'})
    mind.skill_run = {'name': 'поиск: собака', 'index': 1, 'wait': 'm1',
                      'steps': [{'tool': 'go', 'args': {'to': 'Кухня'}}, {'tool': 'search', 'args': {'who': 'собака'}}]}
    for _ in range(6):                                                    # ~2 s: not a flicker
        world.on_eyes({'items': [{'label': 'dog', 'score': .8, 'box': [0, 300, 200, 480], 'bearing': .1,
                                  'distance': 1.5, 'frames': 4}], 'stats': {'light': 120}})
        ex.tick(clock.now)
        clock(.4)
    assert mind.skill_run is None
    assert 'found_on_the_way' in [e['kind'] for e in world.events]
    body = [d for k, d in acts if k == 'body_do']
    assert body and body[-1]['skill'] == 'track' and body[-1]['args']['who'] == 'собака'


def test_a_trip_to_look_for_taffa_turns_to_her_when_she_is_seen(world, clock):
    # 10.10 21:23 «поеду поищу его в коридоре» (a plain go); 21:24:30 she was in view, the mind took 3 s and chose
    # a photo, she was gone by 21:24:50. Now the executive knows what the trip is for.
    ex, mind, acts, sent = make_exec(world, clock, None)
    mind.enact({'thought': 'Соскучился по Таффе — поеду поищу его в коридоре и сфотографирую.', 'say': None,
                'do': {'tool': 'go', 'args': {'to': 'Коридор'}}, 'face': None}, 'heartbeat', False)
    assert mind.seeking and mind.seeking['who'] == 'собака' and mind.seeking['intent'] == 'сфотограф'
    world.on_body({**world.body, 'skill': {'kind': 'go', 'origin': 'mind', 'id': mind.seeking['id']}})
    for _ in range(4):                                                    # 1.2 s of her in view
        world.on_eyes({'items': [{'label': 'dog', 'score': .8, 'box': [0, 300, 200, 480], 'bearing': .1,
                                  'distance': 1.5, 'frames': 4}], 'stats': {'light': 120}})
        ex.tick(clock.now)
        clock(.3)
    body = [d for k, d in acts if k == 'body_do']
    assert body[-1]['skill'] == 'track' and body[-1]['args']['who'] == 'собака'
    assert 'found_on_the_way' in [e['kind'] for e in world.events] and mind.seeking is None
    mind.enact({'thought': 'Поеду в зал, там давно не был.', 'say': None,
                'do': {'tool': 'go', 'args': {'to': 'Зал'}}, 'face': None}, 'heartbeat', False)
    assert mind.seeking is None                                         # a plain trip is not a search


def test_someone_who_comes_in_view_is_worth_changing_course_for(world, clock):
    # 10.10: «подъеду к Таффе» on dog_appeared dropped 6 times as «не стал бросать начатое» — the old trip went on.
    mind, acts = make_mind(world)
    mind.enact({'thought': 'Поеду в зал.', 'say': None, 'do': {'tool': 'go', 'args': {'to': 'Зал'}}, 'face': None},
               'heartbeat', False)
    world.on_body({**world.body, 'skill': {'kind': 'go', 'origin': 'mind', 'id': mind.sent_id}})
    for _ in range(3):
        world.on_eyes({'items': [{'label': 'dog', 'score': .8, 'box': [0, 300, 200, 480], 'bearing': .1,
                                  'distance': 1.5, 'frames': 4}], 'stats': {'light': 120}})
        clock(.4)
    effect = mind.enact({'thought': 'Таффа! Подъеду к нему.', 'say': None, 'face': None,
                         'do': {'tool': 'approach', 'args': {'who': 'собака'}}}, 'dog_appeared', False)
    assert 'не стал бросать' not in (effect or '')
    assert [d for k, d in acts if k == 'body_do'][-1]['skill'] == 'track'
    effect = mind.enact({'thought': 'Сниму его.', 'say': None, 'face': None, 'do': {'tool': 'photo', 'args': {}}},
                        'heartbeat', False)
    assert 'не стал бросать' not in (effect or '')                     # a photo does not abandon a trip
    effect = mind.enact({'thought': 'А поеду-ка на кухню.', 'say': None, 'face': None,
                         'do': {'tool': 'go', 'args': {'to': 'Кухня'}}}, 'heartbeat', False)
    assert 'не стал бросать' in effect                                  # a whim still waits for the trip to end


def test_asked_in_a_talk_to_find_taffa_he_goes(world, clock):
    # 10.10 22:20 «иди поищи Таффу»: the errand's first step was dropped «со мной разговаривают — своё потом».
    world.archive.sighting('собака', 'Коридор', clock.now - 300)
    world.voice = {'live': True}
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'approach', 'args': {'who': 'собака'}}, owner=True)
    assert ok and 'прерван' not in text
    assert [d for k, d in acts if k == 'body_do'][-1]['skill'] == 'go'


def test_the_way_shut_on_the_way_stops_at_once(world, clock):
    ex, mind, acts, sent = make_exec(world, clock, {'kind': 'go', 'origin': 'mind', 'id': 'm2'})
    mind.last_go_target = 'зал'
    ex.tick(clock.now)
    assert not sent                                                       # the way is open: drive on
    world.on_body({**world.body, 'doors': [{'rooms': ['Коридор', 'Спальня и кабинет'], 'state': 'shut',
                                            'seen': clock.now}]})
    ex.tick(clock.now)
    ex.tick(clock.now + .1)
    assert [t for t, d in sent] == ['body.stop']                         # once
    assert world.events[-1]['kind'] == 'refused' and 'Зал' in world.events[-1]['text']
    assert 'зал' in world.state.get('closed_ways')


def test_no_looking_round_in_the_dark(world, clock):
    ex, mind, acts, sent = make_exec(world, clock, {'kind': 'look_around', 'origin': 'mind', 'id': 'm3'})
    world.on_eyes({'items': [], 'stats': {'light': 8}})
    ex.tick(clock.now)
    assert not sent
    ex.tick(clock.now + 3.5)
    assert [t for t, d in sent] == ['body.stop'] and 'темно' in world.events[-1]['text']
    ok, text = mind.do({'tool': 'look_around', 'args': {}})              # and not started again in the dark
    assert not ok and 'темно' in text
    assert mind.do({'tool': 'look_around', 'args': {}}, owner=True)[0]   # asked: yes


def test_the_owners_errand_is_left_alone(world, clock):
    ex, mind, acts, sent = make_exec(world, clock, {'kind': 'go', 'origin': 'owner', 'id': 'o1'})
    mind.last_go_target = 'зал'
    world.on_body({**world.body, 'doors': [{'rooms': ['Коридор', 'Спальня и кабинет'], 'state': 'shut',
                                            'seen': clock.now}]})
    world.on_eyes({'items': [], 'stats': {'light': 8}})
    for step in range(5):
        ex.tick(clock.now + step)
    assert not sent


def test_called_and_nobody_in_view_he_turns_where_someone_was(world, clock):
    life = make_life(world)
    life.mind = types.SimpleNamespace(version=0)
    life.archive = world.archive
    life.face = lambda state: None
    world.on_body({**world.body, 'pose': [0., 0., 0.]})
    life.turn_to_voice(None)
    assert not turns(life)                                                # nothing known: no guess
    world.last_motion = (clock.now, -2.0)                                 # a rustle behind on the right
    life.turn_to_voice(None)
    assert [d['args']['angle'] for d in turns(life)] == [-2.0] and life.gazes[-1][0] == -1.
    life.sent.clear()
    world.on_body({**world.body, 'docked': True})
    life.turn_to_voice(None)
    assert not turns(life)                                                # on the dock: the eyes only


def test_find_the_dog_who_was_seen_in_another_room_goes_there(world, clock):
    world.archive.sighting('собака', 'Кухня', clock.now - 900)          # Taffa in the kitchen a quarter ago
    mind, acts = make_mind(world)
    ok, text = mind.do({'tool': 'approach', 'args': {'who': 'Таффа'}}, owner=True)
    body = [d for k, d in acts if k == 'body_do']
    assert ok and 'Кухня' in text and body[0]['skill'] == 'go' and body[0]['origin'] == 'owner'
    assert mind.skill_run['owner'] and mind.skill_run['steps'][1]['tool'] == 'approach'
    mind.next_skill_step({'id': mind.skill_run['wait'], 'status': 'done'})   # there: look round and come up
    world.on_body({**world.body, 'room': 'Кухня'})
    assert [d['args'].get('mode') for k, d in acts if k == 'body_do'][-1] == 'seek'


def test_find_me_looks_round_here_first(world, clock):
    world.archive.sighting('Сергей', 'Кухня', clock.now - 300)
    world.state.heard(clock.now, 'Сергей', 'найди меня')
    mind, acts = make_mind(world)
    mind.do({'tool': 'approach', 'args': {'who': 'Сергей'}}, owner=True)
    assert [d['args'].get('mode') for k, d in acts if k == 'body_do'] == ['seek']   # the voice was near: here


# ---------------------------------------------------------------- no names in every line; asides; intentions
from bratan.mind import brain as brainmod  # noqa: E402


def test_no_calling_by_name_in_every_line():
    names = ['Сергей', 'Марина']
    cases = {  # 09.10 11:19, as said
        'Да я просто тут с тобой болтаю, Сергей.': 'Да я просто тут с тобой болтаю.',
        'Сергей, ты про Казахстан?': 'Ты про Казахстан?',
        'Не совсем понял, что ты имел в виду, Сергей.': 'Не совсем понял, что ты имел в виду.',
        'Ага, понял, Сергей, идём гулять позже.': 'Ага, понял, идём гулять позже.',
        'Конечно, ты Сергей.': 'Конечно, ты Сергей.',               # who it is: kept
        'Это Марина пришла.': 'Это Марина пришла.',
    }
    for said, want in cases.items():
        assert brainmod.no_vocative(said, names) == want, said


def test_an_aside_is_not_asked_to_repeat(world, clock):
    reply = 'Сергей, ты про Казахстан? Не совсем понял, повтори.\n###\n{"thought": "", "do": null}'
    mind, _ = make_mind(world, reply=reply)
    spoken, _ = mind.converse('смотри вот он сейчас нашёл да потом не высмотрит казахстан мне кажется', '', aside=True)
    assert spoken == ''
    spoken, _ = mind.converse('чего по', '')                               # short, to me: asking again is right
    assert spoken == 'Ты про Казахстан? Не совсем понял, повтори.'


def test_after_two_asides_the_talk_is_someone_elses(world, clock):
    # 09.10 19:19: dictation at the PC; four phrases held back, the fifth got «Понял, сейчас посмотрю, что с
    # анимацией», then «это я не тебе».
    mind, _ = make_mind(world, reply='Не расслышал, повтори, пожалуйста.\n###\n{"thought": "", "do": null}')
    for heard in ('это и здесь задавали так можно то есть один час на баске',
                  'другой проверил я напал тридцать два ранее всех там'):
        assert mind.converse(heard, '', aside=True)[0] == ''
    mind.llm.reply = 'Понял, сейчас посмотрю, что с анимацией.\n###\n{"thought": "", "do": null}'
    assert mind.converse('то что она приближается нормально а вы дергаться она не должна', '', aside=True) == ('', {})
    assert mind.converse('а что ты сейчас видишь', '')[0]                  # a short one, to me: answered
    assert mind.asides_in_row == 0
    assert mind.converse('то что она приближается нормально а вы дергаться она не должна', '', aside=True)[0]


def test_told_it_is_not_for_him_he_stops_answering(world, clock):
    # 09.10 19:20 «там я вообще не тебе, меня не слушай и не отвечай» -> «Понял, не мешаю.», and the next phrase
    # about an animation got an answer again.
    assert brainmod.NOT_FOR_ME.search('там я вообще не тебе пока меня не слушаю и не отвечаю')
    assert brainmod.NOT_FOR_ME.search('горе братан это я не тебе')
    assert not brainmod.NOT_FOR_ME.search('а тебе нравится')
    mind, _ = make_mind(world, reply='Понял, не мешаю.\n###\n{"thought": "", "do": null}')
    assert mind.converse('там я вообще не тебе пока меня не слушаю и не отвечаю', '', aside=True)[0] == ''
    assert mind.asides_in_row == brainmod.ASIDES_IN_ROW                  # and the voice is told to stop
    mind.llm.reply = 'Понял, про анимацию с приближением.\n###\n{"thought": "", "do": null}'
    assert mind.converse('кстати анимация с приближающимся кадром отображат много где везде', '', aside=True) == ('', {})


def test_an_announced_silence_is_not_said(world, clock):
    # 09.10 19:29-19:31: «Это не мне адресовано, молчу.» aloud, five times into the owner's phone call.
    mind, _ = make_mind(world, reply='Это не мне адресовано, молчу.\n###\n{"thought": "", "do": null}')
    assert mind.converse('начальником для них стало самостоятельности компания немецкая', '', aside=True)[0] == ''
    assert mind.asides_in_row == 1
    mind.llm.reply = 'Понял.\n###\n{"thought": "", "do": null}'
    assert mind.converse('братан это я не тебе', '')[0] == 'Понял.'          # by name: a short «понял» is fine


def test_asked_to_stay_he_stays(world, clock):
    # 09.10 19:21 «поезжай, пожалуйста, в зал, побудь там минут 10»: four minutes in the hall, then off to the base;
    # 19:30 «в зал, побудь там ещё какое-то время».
    assert brainmod.stay_minutes('поезжай пожалуйста в зал побудь там минут 10') == 10
    assert brainmod.stay_minutes('в зал побудь там еще какойто время') == brainmod.STAY_DEFAULT_MIN
    assert brainmod.stay_minutes('посиди тут пять минут') == 5
    assert brainmod.stay_minutes('подожди я сейчас') is None
    assert brainmod.stay_minutes('подожди на секунду') is None
    assert brainmod.stay_minutes('побудь рядом') is None
    reply = 'Понял, побуду в зале.\n###\n{"thought": "", "do": {"tool": "go", "args": {"to": "Зал"}}}'
    mind, _ = make_mind(world, reply=reply)
    mind.converse('поезжай пожалуйста в зал побудь там минут 10', '')
    assert world.state.get('stay')['room'] == 'Зал'
    ok, why = mind.do({'tool': 'go', 'args': {'to': 'Кухня'}})
    assert not ok and 'побыть в «Зал»' in why
    assert not mind.do({'tool': 'rest', 'args': {}})[0]                     # the battery is fine: stay
    assert mind.do({'tool': 'go', 'args': {'to': 'Зал'}})[0]                 # there is fine
    stay_line = lambda: [x for x in mind.prompt(clock.now, 'heartbeat', False).split('\n')
                         if x.startswith('МЕНЯ ПОПРОСИЛИ ПОБЫТЬ')]
    assert stay_line()
    clock(601)
    assert not stay_line() and 'stay_over' in [e['kind'] for e in world.events]
    assert mind.do({'tool': 'go', 'args': {'to': 'Кухня'}})[0]
    mind.converse('поезжай пожалуйста в зал побудь там минут 10', '')
    mind.do({'tool': 'go', 'args': {'to': 'Кухня'}}, owner=True)            # the owner's next request replaces it
    assert not world.state.get('stay')


def test_the_eyes_look_below_the_ceiling():
    # 10.10: the top quarter of the upright frame is ceiling; the detector sees the floor larger without it.
    import numpy as np
    from bratan.eyes.detect import detect_lower
    seen = []

    def detector(image):
        seen.append(image.shape)
        return [{'label': 'dog', 'score': .8, 'box': [100., 300., 160., 400.]},
                {'label': 'person', 'score': .9, 'box': [200., 0., 300., 470.]}]      # cut by the crop's top
    items = detect_lower(detector, np.zeros((640, 480, 3), np.uint8))
    assert seen == [(480, 480, 3)]
    assert items[0]['box'] == [100., 460., 160., 560.]                          # back in full-frame rows
    assert items[1]['box'] == [200., 0., 300., 630.]                            # the person goes on to the top


def test_going_to_someone_is_going_for_something(world, clock):
    world.state.heard(clock.now, 'Сергей', 'найди Таффу')
    mind, acts = make_mind(world)
    mind.do({'tool': 'approach', 'args': {'who': 'Таффа'}}, owner=True)
    intent = mind.intents[mind.sent_id]
    assert intent['kind'] == 'report' and intent['asker'] == 'Сергей' and intent['who'] == 'собака'
    for _ in range(2):
        world.on_eyes({'items': [{'label': 'dog', 'score': .8, 'box': [0, 300, 200, 480], 'bearing': .1,
                                  'distance': 1.5, 'frames': 4}], 'stats': {'light': 120}})
        clock(.5)
    mind.do({'tool': 'approach', 'args': {'who': 'собака', 'intent': 'поиграть с ним'}})
    assert mind.intents[mind.sent_id]['kind'] == 'play'


def test_on_arrival_he_does_what_he_went_for(world, clock):
    life = make_life(world)
    life.mind, _ = make_mind(world)
    life.state, life.archive = world.state, world.archive
    life.loop = types.SimpleNamespace(run_in_executor=None)
    shown = []
    life.express = lambda kind, origin='mind', sound=None: shown.append((kind, sound))
    for _ in range(2):
        world.on_eyes({'items': [{'label': 'dog', 'score': .8, 'box': [0, 300, 200, 480], 'bearing': .1,
                                  'distance': 1.5, 'frames': 4}], 'stats': {'light': 120}})
        clock(.5)
    intent = {'who': 'собака', 'kind': 'report', 'words': '', 'owner': True, 'asker': 'Сергей', 't': clock.now}
    asyncio.run(life.carry_out(intent, {'reason': 'arrived'}))
    said = [d['text'] for t, d in life.sent if t == 'voice.say']
    assert said == ['Нашёл Таффу!'] and shown[0] == ('wiggle', 'found')
    assert world.events[-1]['kind'] == 'arrived' and 'Таффу' in world.events[-1]['text']
    assert lifemod.accusative('Сергей') == 'Сергея' and lifemod.accusative('Марина') == 'Марину'


# ---------------------------------------------------------------- doors: someone in the doorway is not a shut door
def test_a_door_flicker_is_no_news_and_an_opened_door_clears_no_way(world, clock):
    # 09.10 17:12-17:54: the hall door «closed» and «opened» within 2-5 s six times (someone in the doorway); the
    # trip was stopped and the hall refused five times after the door had opened again.
    rooms = ['Коридор', 'Зал']

    def body_doors(state):
        world.on_body({**world.body, 'doors': [{'rooms': rooms, 'state': state, 'seen': clock.now}]})

    def news():
        return [e['kind'] for e in world.events if e['kind'] in ('door_opened', 'door_closed')]
    body_doors('open')
    world.on_body_event({'kind': 'door', 'rooms': rooms, 'state': 'open'})
    clock(9)
    world.tick()
    assert news() == ['door_opened']
    body_doors('shut')                                    # someone in the doorway for 2 s
    world.on_body_event({'kind': 'door', 'rooms': rooms, 'state': 'shut'})
    clock(1)
    world.tick()
    assert not world.door_shut(world.doors[0])            # not yet: it may be someone passing
    clock(1)
    body_doors('open')
    world.on_body_event({'kind': 'door', 'rooms': rooms, 'state': 'open'})
    clock(9)
    world.tick()
    assert news() == ['door_opened']                      # the flicker told nothing
    body_doors('shut')                                    # really shut now
    world.on_body_event({'kind': 'door', 'rooms': rooms, 'state': 'shut'})
    clock(9)
    world.tick()
    assert world.door_shut(world.doors[0]) and news() == ['door_opened', 'door_closed']
    world.state.set('closed_ways', {'зал': clock.now, 'кухня': clock.now})
    body_doors('open')
    world.on_body_event({'kind': 'door', 'rooms': rooms, 'state': 'open'})
    clock(9)
    world.tick()
    assert news()[-1] == 'door_opened' and list(world.state.get('closed_ways')) == ['кухня']


def test_someone_in_view_gets_a_glance_not_a_turn(world, clock):
    # 09.10 17:25: Taffa padding round the robot — four turns and four alike thoughts.
    life = make_life(world)
    for _ in range(2):
        world.on_eyes({'items': [{'label': 'dog', 'score': .8, 'box': [0, 300, 200, 480], 'bearing': .1,
                                  'distance': 1., 'frames': 4}], 'stats': {'light': 120}})
        clock(.5)
    life.on_motion({'bearing': -1.6, 'distance': 1.})
    assert life.gazes and not turns(life)


def test_a_thing_not_there_is_forgotten(world, clock):
    world.archive.saw('chair', (2.5, .4), 'Спальня и кабинет', clock.now)
    oid = world.archive.objects(min_count=1)[0]['id']
    life = make_life(world)
    life.mind, _ = make_mind(world)
    life.archive, life.state = world.archive, world.state
    life.mind.last_approach = (oid, 'chair')
    life.on_body_event({'kind': 'skill', 'skill': 'approach', 'status': 'failed', 'reason': 'not_there', 'id': 'x',
                        'origin': 'mind'})
    assert world.archive.object(oid) is None
