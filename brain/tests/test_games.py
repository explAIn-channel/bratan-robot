"""Games (mind/games.py) against a pretend body, voice and camera: no robot, no clock waits beyond seconds."""
import asyncio
import time
import types

import pytest

from bratan.mind import games as gamesmod


class FakeWorld:
    def __init__(self):
        self.voice = {}
        self.items = []
        self.eyes_t = time.time()
        self.room = 'Спальня и кабинет'
        self.rooms = {'Спальня и кабинет': {}, 'Коридор': {}}
        self.events = []

    def reachable(self):
        return {'Спальня и кабинет': True, '_why': {'Коридор': 'закрыта дверь'}}

    def room_point(self, name):
        return name, (1., 1.)

    def event(self, kind, text, **data):
        self.events.append((kind, text, data))

    def see(self, box, **extra):
        self.items = [{'label': 'person', 'score': .9, 'box': box, 'frames': 5, **extra}]
        self.eyes_t = time.time()


class FakeLife:
    def __init__(self, loop, finish_after=.05):
        self.loop = loop
        self.world = FakeWorld()
        self.published = []
        self.looks, self.captions = {}, {}
        self.io = types.SimpleNamespace(submit=lambda *a, **k: None)
        self.archive = types.SimpleNamespace(event=lambda *a, **k: None)
        self.play = gamesmod.Play(self)
        self.finish_after = finish_after
        self.faces = []

    def publish(self, topic, data):
        self.published.append((topic, data))
        if topic == 'body.do':
            self.loop.call_later(self.finish_after, self.play.on_skill,
                                 {'kind': 'skill', 'id': data['id'], 'status': 'done'})

    def face_now(self, state):
        self.faces.append(state)

    def sfx(self, emotion, force=False):
        self.published.append(('sfx', emotion))


@pytest.fixture
def fast(monkeypatch):
    """Games wait seconds; tests do not: shrink sleeps."""
    real = asyncio.sleep

    async def quick(seconds, *a, **k):
        await real(min(seconds, .01))
    monkeypatch.setattr(gamesmod.asyncio, 'sleep', quick)

    async def no_wait(self, chars):
        return None
    monkeypatch.setattr(gamesmod.Game, 'spoken', no_wait)

    async def short(self, seconds):
        await real(min(seconds, .01))
        self.check()
    monkeypatch.setattr(gamesmod.Game, 'sleep', short)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_high_five_with_the_button(fast):
    async def scenario():
        life = FakeLife(asyncio.get_running_loop())
        ok, _ = life.play.start('дай пять', 'Сергей')
        assert ok and life.play.active
        await asyncio.sleep(.05)
        life.play.on_touch()
        await life.play.task
        return life
    life = run(scenario())
    kind, text, data = life.world.events[-1]
    assert kind == 'game_played' and 'дали пять' in text and data['who'] == 'Сергей'


def test_red_light_catches_a_move_and_ends_when_the_player_arrives(fast):
    async def scenario():
        life = FakeLife(asyncio.get_running_loop())
        game = gamesmod.TisheEdesh(life.play, 'Сергей')
        life.world.see([200, 100, 260, 300])
        assert await game.watch_freeze(.05) == 'still'

        async def step_aside():
            await asyncio.sleep(.02)
            life.world.see([280, 100, 340, 300])          # moved sideways while «red»
        asyncio.ensure_future(step_aside())
        assert await game.watch_freeze(.3) == 'moved'
        life.world.see([100, 0, 400, 600], distance=.7)    # right in front: arrived
        assert await game.watch_freeze(.05) == 'arrived'
    run(scenario())


def test_hide_and_seek_finds_the_one_who_hid(fast):
    async def scenario():
        life = FakeLife(asyncio.get_running_loop(), finish_after=.3)
        life.play.start('прятки', 'Марина')

        async def appear():
            await asyncio.sleep(.25)
            life.world.see([200, 100, 260, 300], name='Марина')
        asyncio.ensure_future(appear())
        await asyncio.wait_for(life.play.task, 10.)
        return life
    life = run(scenario())
    kind, text, _ = life.world.events[-1]
    assert kind == 'game_played' and 'нашёл Марина' in text
    assert 'laughing' in life.faces


def test_hide_and_seek_calls_cuckoo_and_goes_to_the_answer(fast):
    """Not in view after the look round: «Ку-ку! Где ты?», the ears place the answer behind on the left — the body
    turns there and looks, and finds her (C52: the seeker listens, as people play it)."""
    async def scenario():
        life = FakeLife(asyncio.get_running_loop(), finish_after=.05)
        asked = []
        real_publish = life.publish

        def publish(topic, data):
            real_publish(topic, data)
            if topic == 'voice.locate':
                asked.append(data)
                life.loop.call_later(.01, life.play.on_located, {'kind': 'located', 'id': data['id'],
                                                                 'bearing': 2.4, 'bearing_conf': .7, 'side': 'слева-сзади'})
            if topic == 'body.do' and data['skill'] == 'rotate' and abs(data['args'].get('angle', 0) - 2.4) < 1e-6:
                life.loop.call_later(.02, lambda: life.world.see([200, 100, 260, 300], name='Марина'))
        life.publish = publish
        life.play.start('прятки', 'Марина')
        await asyncio.wait_for(life.play.task, 10.)
        return life, asked
    life, asked = run(scenario())
    said = [d['text'] for t, d in life.published if t == 'voice.say']
    assert 'Ку-ку! Где ты?' in said and 'Слышу! Иду!' in said and asked
    skills = [(d['skill'], d['args']) for t, d in life.published if t == 'body.do']
    turned = [i for i, (s, a) in enumerate(skills) if s == 'rotate' and abs(a.get('angle', 0) - 2.4) < 1e-6]
    assert turned and skills[turned[0] - 1][0] == 'look_around'          # first a look round, then the call
    assert 'нашёл Марина' in life.world.events[-1][1]


def test_hide_and_seek_without_an_answer_goes_on_as_before(fast):
    async def scenario():
        life = FakeLife(asyncio.get_running_loop(), finish_after=.05)
        real_publish = life.publish

        def publish(topic, data):
            real_publish(topic, data)
            if topic == 'voice.locate':
                life.loop.call_later(.01, life.play.on_located, {'kind': 'located', 'id': data['id']})
        life.publish = publish
        life.play.start('прятки', 'Марина')

        async def appear():
            await asyncio.sleep(.4)
            life.world.see([200, 100, 260, 300], name='Марина')
        asyncio.ensure_future(appear())
        await asyncio.wait_for(life.play.task, 10.)
        return life
    life = run(scenario())
    skills = [d['skill'] for t, d in life.published if t == 'body.do']
    assert 'explore' in skills or 'нашёл' in life.world.events[-1][1]


def test_unknown_game_and_one_at_a_time(fast):
    async def scenario():
        life = FakeLife(asyncio.get_running_loop())
        assert not life.play.start('шахматы')[0]
        assert life.play.start('фото')[0]
        assert not life.play.start('прятки')[0]
        life.play.stop()
        await asyncio.wait_for(life.play.task, 5.)
    run(scenario())


def test_a_message_is_delivered_to_the_one_found(fast):
    async def scenario():
        life = FakeLife(asyncio.get_running_loop())
        life.world.see([200, 100, 260, 300], name='Марина')
        ok, _ = life.play.start('deliver', 'Марина', {'to': 'Марина', 'text': 'ужин готов', 'from': 'Сергей'})
        assert ok
        await asyncio.wait_for(life.play.task, 5.)
        return life
    life = run(scenario())
    said = [d['text'] for t, d in life.published if t == 'voice.say']
    assert any('Марина, тебе сообщение от Сергей: ужин готов' in x for x in said)
    assert 'передал' in life.world.events[-1][1]


def test_the_timer_rings(fast):
    async def scenario():
        life = FakeLife(asyncio.get_running_loop())
        life.play.start('timer', None, {'label': 'чайник'})
        await asyncio.wait_for(life.play.task, 5.)
        return life
    life = run(scenario())
    assert any('Время вышло: чайник' in d['text'] for t, d in life.published if t == 'voice.say')
