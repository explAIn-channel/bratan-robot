"""Games Братан plays with his people (owner 08.10: «чтобы он мог играть»).

Each game is a short script on the Life loop: it speaks, shows digits or words on the face screen, makes
sounds, moves the body and watches through the camera; while it runs, the mind's own wandering pauses.
Ideas from the pet robots that people love: hide-and-seek (Loona, aibo look for their people), «тише едешь —
дальше будешь» (Red Light Green Light: the robot closes its eyes on the screen instead of turning away),
a photo booth with a countdown, a high five with the button on his back.

Any skill of a game can be stopped by «стоп» or by someone starting a conversation: the game then ends.
"""
from __future__ import annotations

import asyncio
import math
import random
import time

from . import screen

HIDE_COUNT = 20
SEARCH_S = 180.
CALLS = 2                    # «Ку-ку! Где ты?» at most this many times in one search
ANSWER_S = 2.5               # the answer is listened for this long after the call
RLGL_ROUNDS = 8


class GameOver(Exception):
    pass


class Game:
    name = ''
    title = ''

    def __init__(self, play, who, args=None):
        self.play, self.life, self.who = play, play.life, who or 'человек'
        self.args = dict(args or {})
        self.started = time.time()

    # ------------------------------------------------------------ what a game can do
    async def say(self, text, wait=True):
        self.life.publish('voice.say', {'text': text, 'id': 'game', 'tts_only': True})
        if wait:
            await self.spoken(len(text))

    async def spoken(self, chars):
        """Wait until the voice has said it (speaking on, then off), or a fair estimate passed."""
        world = self.life.world
        deadline = time.monotonic() + 2.5 + chars * .09
        began = False
        while time.monotonic() < deadline:
            speaking = bool(world.voice.get('speaking'))
            if speaking:
                began = True
            elif began:
                return
            await asyncio.sleep(.1)
            self.check()

    def face(self, state):
        self.life.face_now(state)

    def sfx(self, emotion):
        self.life.sfx(emotion, force=True)

    def screen(self, text, sub='', seconds=2.):
        self.life.io.submit(_quiet, lambda: screen.show(screen.card(text, sub), seconds))

    async def body(self, skill, args, timeout=60.):
        """Run a body skill and wait for its end: 'done' / 'failed' / 'cancelled' / 'timeout'."""
        return await self.play.body(skill, args, timeout)

    def stop_body(self):
        self.life.publish('body.stop', {'reason': 'game'})

    def people(self, fresh=.6):
        world = self.life.world
        if time.time() - world.eyes_t > fresh:
            return []
        return [i for i in world.items if i['label'] == 'person' and i.get('score', 0) >= .5]

    def check(self):
        if self.play.cancelled:
            raise GameOver('stopped')

    async def sleep(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            await asyncio.sleep(min(.1, end - time.monotonic()))
            self.check()

    async def watch_for_person(self, seconds, min_frames=2):
        """Poll the camera for a person (seen at least `min_frames` frames in a row by the tracker)."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            for item in self.people():
                if item.get('frames', 1) >= min_frames:
                    return item
            await asyncio.sleep(.15)
            self.check()
        return None

    async def run(self):                     # pragma: no cover - interface
        raise NotImplementedError


class HideAndSeek(Game):
    name, title = 'hide_seek', 'прятки'

    async def run(self):
        self.face('happy')
        self.sfx('excited')
        await self.say(f'Играем в прятки! Я считаю до {HIDE_COUNT}, а ты прячься.')
        turning = asyncio.ensure_future(self.body('rotate', {'angle': math.pi}, timeout=20.))
        self.face('sleepy')                        # eyes closed: no peeking
        for n in range(HIDE_COUNT, 0, -1):
            self.screen(str(n), 'прятки', 1.3)
            if n in (10, 5):
                await self.say(f'{n}!', wait=False)
            await self.sleep(1.)
        await turning
        self.face('curious')
        self.sfx('curious')
        await self.say('Кто не спрятался — я не виноват! Иду искать.')
        found = await self.search()
        if found is None:
            self.face('concerned')
            self.sfx('sad')
            await self.say('Сдаюсь! Выходи, я тебя не нашёл.')
            found = await self.watch_for_person(25.)
            if found is not None:
                self.face('happy')
                await self.say('Вот ты где! Хорошо спрятался.')
            return 'не нашёл — сдался' if found is None else 'не нашёл сам, вышел ко мне'
        self.stop_body()
        who = found.get('name') or self.who
        seconds = int(time.time() - self.search_started)
        self.face('laughing')
        self.sfx('found')
        self.screen('Нашёл!', f'за {seconds // 60} мин {seconds % 60} с' if seconds >= 60 else f'за {seconds} с', 5.)
        await self.say('Нашёл! Ку-ку!' if who == 'человек' else f'{who}, нашёл тебя!')
        await self.body('track', {'who': 'человек', 'mode': 'follow', 'seconds': 6., 'distance': 1.}, timeout=10.)
        await self.body('emote', {'kind': 'wiggle'}, timeout=6.)
        return f'нашёл {who} за {seconds} с'

    async def search(self):
        """Look round here; not here — «Ку-ку! Где ты?» and go where the answer came from (the body's ears, C52);
        then go round this room and the rooms one can get to; stop at the first person."""
        self.search_started = time.time()
        world = self.life.world
        plan = [('look_around', {})]
        plan.append(('explore', {'room': world.room, 'viewpoints': 3}))
        reach = world.reachable()
        others = [name for name in world.rooms if name != world.room and name in reach]
        random.shuffle(others)
        for name in others[:2]:
            point = world.room_point(name)
            if point:
                plan += [('go', {'xy': list(point[1]), 'tolerance': .5}), ('look_around', {})]
        calls = 0
        while plan:
            skill, args = plan.pop(0)
            if time.time() - self.search_started > SEARCH_S:
                break
            task = asyncio.ensure_future(self.body(skill, args, timeout=SEARCH_S))
            while not task.done():
                person = await self.watch_for_person(.3)
                if person is not None:
                    task.cancel()
                    return person
                if time.time() - self.search_started > SEARCH_S:
                    task.cancel()
                    return None
            if skill == 'look_around' and calls < CALLS:
                calls += 1
                heard = await self.call_out()
                if heard is not None:
                    plan = self.toward_voice(heard) + plan
        return None

    async def call_out(self):
        """«Ку-ку! Где ты?» — and where the answer came from: the ears' direction (dict with 'bearing') or None."""
        if not hasattr(self.play, 'locate'):
            return None
        await self.say('Ку-ку! Где ты?')
        await self.sleep(ANSWER_S)                   # «тут!», «ку-ку!»
        heard = await self.play.locate(ANSWER_S + 1.)
        if not heard or heard.get('bearing') is None or float(heard.get('bearing_conf') or 0.) < .3:
            return None
        self.face('happy')
        await self.say('Слышу! Иду!', wait=False)
        return heard

    def toward_voice(self, heard):
        """Body steps to where the answer came from: the room behind a doorway, or this way here, then a look."""
        world = self.life.world
        sounds = getattr(world, 'sounds', None)
        ray = sounds.latest(('voice',), within=10.) if sounds is not None else None
        if ray and ray.get('via') in ('door', 'open') and ray.get('toward') not in (None, world.room):
            point = world.room_point(ray['toward'])
            if point:
                return [('go', {'xy': list(point[1]), 'tolerance': .5}), ('look_around', {})]
        steps = [('rotate', {'angle': float(heard['bearing'])})]
        pose = (world.body or {}).get('pose') if hasattr(world, 'body') else None
        free = float(ray.get('free') or 0.) if ray else 0.
        if ray and pose and free > 1.2:
            step = min(1.5, free - .7)
            steps.append(('go', {'xy': [round(pose[0] + step * math.cos(ray['heading']), 2),
                                        round(pose[1] + step * math.sin(ray['heading']), 2)], 'tolerance': .4}))
        return steps + [('look_around', {})]


class TisheEdesh(Game):
    """Red Light, Green Light: eyes closed on the screen — come closer; eyes open — freeze."""
    name, title = 'red_light', 'тише едешь — дальше будешь'
    MOVED = .06              # the person's box shifted this share of the frame width while «red»
    GREW = .10               # ... or grew this much (stepped closer)

    async def run(self):
        self.life.publish('mind.eyes', {'mode': 'focus', 'seconds': 240})
        self.face('happy')
        self.sfx('excited')
        await self.say('Играем в «Тише едешь — дальше будешь»! Встань подальше. Когда закрою глаза — иди ко мне, '
                       'когда открою — замри!')
        player = await self.watch_for_person(15.)
        if player is None:
            await self.say('Тебя не вижу. Встань напротив меня, и сыграем.')
            return 'не увидел игрока'
        caught = 0
        for round_ in range(RLGL_ROUNDS):
            self.face('sleepy')
            self.screen('Зелёный!', 'иди', 1.5)
            await self.say('Зелёный!', wait=False)
            await self.sleep(random.uniform(2., 4.5))
            self.face('surprised')
            self.sfx('curious')
            self.screen('Красный!', 'замри', 2.5)
            await self.say('Красный!', wait=False)
            await self.sleep(.5)                  # a moment to stop is fair
            result = await self.watch_freeze(2.)
            if result == 'arrived':
                self.face('laughing')
                self.sfx('found')
                self.screen('Победа!', '', 4.)
                await self.say('Ты дошёл до меня — победа!')
                return f'игрок дошёл за {round_ + 1} раунд(а), пойман {caught} раз'
            if result == 'moved':
                caught += 1
                self.face('laughing')
                self.sfx('laughing')
                await self.say('Вижу-вижу, ты двигался! Шаг назад.')
            elif result == 'gone':
                await self.say('Эй, ты куда пропал?')
        await self.say('Хорошо поиграли!')
        return f'сыграли {RLGL_ROUNDS} раундов, пойман {caught} раз'

    async def watch_freeze(self, seconds):
        start = None
        end = time.monotonic() + seconds
        width = 480.
        while time.monotonic() < end:
            people = self.people(.4)
            if people:
                p = max(people, key=lambda i: (i['box'][3] - i['box'][1]))
                x1, y1, x2, y2 = p['box']
                cx, height = (x1 + x2) / 2 / width, (y2 - y1)
                if (p.get('distance') or 9.) < .9 or height > 560:
                    return 'arrived'
                if start is None:
                    start = (cx, height)
                elif abs(cx - start[0]) > self.MOVED or height > start[1] * (1 + self.GREW):
                    return 'moved'
            await asyncio.sleep(.08)
            self.check()
        return 'still' if start is not None else 'gone'


class PhotoBooth(Game):
    name, title = 'photo', 'фото'

    async def run(self):
        self.face('happy')
        await self.say('Смотри на меня и улыбайся!')
        for n in (3, 2, 1):
            self.screen(str(n), 'фото', 1.2)
            await self.say(f'{n}', wait=False)
            await self.sleep(1.)
        self.sfx('found')
        looked = await self.play.photo('фото на память')
        if not looked:
            await self.say('Ой, камера не успела. Давай ещё раз попозже.')
            return 'камера не дала кадр'
        try:
            with open(looked, 'rb') as handle:
                jpeg = handle.read()
            self.life.io.submit(_quiet, lambda: screen.show(jpeg, 6))
        except OSError:
            pass
        await self.say('Готово! Вот что получилось — сохранил в альбом.')
        return 'сделал фото на память'


class HighFive(Game):
    name, title = 'high_five', 'дай пять'

    async def run(self):
        self.face('happy')
        self.screen('Дай пять!', 'нажми кнопку у меня на спине', 8.)
        await self.say('Дай пять! Нажми кнопку у меня на спине.')
        touched = await self.play.wait_touch(8.)
        if touched:
            self.face('laughing')
            self.sfx('happy')
            await self.say('Есть пять!')
            await self.body('emote', {'kind': 'wiggle'}, timeout=6.)
            return 'дали пять'
        self.face('idle')
        await self.say('Ладно, в другой раз.')
        return 'пять не дали'


class Errand(Game):
    """Find someone in the house: here, then the rooms they are usually in, then the other reachable rooms."""
    SEARCH_S = 150.

    def named(self, who):
        for item in self.people():
            if who in (None, '', 'человек') or (item.get('name') or '').casefold() == str(who).casefold():
                return item
        return None

    async def find(self, who):
        found = self.named(who) or (await self.watch_for_person(1.5) if who in (None, '', 'человек') else None)
        if found is not None:
            return found
        world = self.life.world
        reach = world.reachable()
        habits = self.life.mind.drives.habits(time.time()) if hasattr(self.life, 'mind') else {}
        usual = list((habits.get(who) or {}).get('rooms', []))
        rooms = [r for r in usual + list(world.rooms) if r in reach]
        rooms = [world.room] + [r for i, r in enumerate(rooms) if r != world.room and r not in rooms[:i]]
        started = time.time()
        for room in rooms:
            if time.time() - started > self.SEARCH_S:
                break
            if room != world.room:
                point = world.room_point(room)
                if not point:
                    continue
                going = asyncio.ensure_future(self.body('go', {'xy': list(point[1]), 'tolerance': .5}, timeout=90.))
                while not going.done():
                    found = self.named(who)
                    if found is not None:
                        going.cancel()
                        return found
                    await asyncio.sleep(.3)
            looking = asyncio.ensure_future(self.body('look_around', {}, timeout=30.))
            while not looking.done():
                found = self.named(who)
                if found is not None:
                    looking.cancel()
                    return found
                await asyncio.sleep(.2)
        return None


class Deliver(Errand):
    """«Скажи Марине, что ужин готов» (Astro's message delivery): find her by face and say it."""
    name, title = 'deliver', 'передать сообщение'

    async def run(self):
        to, text = self.args.get('to') or 'человек', str(self.args.get('text') or '').strip()
        sender = self.args.get('from')
        if not text:
            return 'нечего передавать'
        person = await self.find(to)
        address = '' if to in ('человек', '') else f'{to}, '
        message = f'{address}тебе сообщение{" от " + sender if sender and sender != to else ""}: {text}'
        if person is None:
            self.face('concerned')
            await self.say(f'{to} не нашёл. Скажу вслух: {message}' if address else f'Скажу вслух: {message}')
            return f'не нашёл {to}, сказал вслух'
        self.stop_body()
        await self.body('track', {'who': 'человек', 'mode': 'follow', 'seconds': 5., 'distance': 1.}, timeout=8.)
        self.face('happy')
        self.sfx('hello')
        await self.say(message)
        return f'передал {to}: «{text[:60]}»'


class WakeUp(Errand):
    """An alarm with legs: go to the person and wake them, up to three calls (stops when they answer)."""
    name, title = 'wake', 'разбудить'

    async def run(self):
        to = self.args.get('to') or self.who
        text = str(self.args.get('text') or 'Доброе утро! Пора вставать.')
        world = self.life.world
        room = self.args.get('room')
        if room and room != world.room and room in world.reachable():
            point = world.room_point(room)
            if point:
                await self.body('go', {'xy': list(point[1]), 'tolerance': .6}, timeout=120.)
        for attempt in range(3):
            self.face('happy')
            self.sfx('hello' if attempt == 0 else 'excited')
            await self.say((f'{to}, ' if to and to != 'человек' else '') + text)
            if await self.answered(40.):
                self.sfx('happy')
                return f'разбудил {to}'
        return f'будил {to} трижды — не ответил'

    async def answered(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.life.world.live:
                return True                      # someone talks to me: awake
            if self.play.touched_since(self.started):
                return True
            await asyncio.sleep(.3)
            self.check()
        return False


class Timer(Game):
    """A kitchen timer: «засеки 10 минут» — at the time, a sound, a card and words."""
    name, title = 'timer', 'таймер'

    async def run(self):
        label = str(self.args.get('label') or 'таймер')
        for attempt in range(2):
            self.sfx('found')
            self.face('surprised')
            self.screen('Время!', label, 10.)
            await self.say(f'Время вышло: {label}!')
            if self.people() or self.life.world.live:
                break
            await self.sleep(20.)
        return f'прозвенел таймер «{label}»'


class Album(Game):
    """A slideshow of my photos on the face, with what they are (Vector shows its photos; ours tells about them)."""
    name, title = 'album', 'альбом'

    async def run(self):
        album = [p for p in self.life.state.get('album', []) if p.get('path')][-6:]
        if not album:
            await self.say('Альбом пока пустой — давай сфотографируемся!')
            return 'альбом пуст'
        self.face('happy')
        await self.say(f'Вот мои снимки — {len(album)} последних.')
        shown = 0
        for photo in album:
            try:
                with open(photo['path'], 'rb') as handle:
                    jpeg = handle.read()
            except OSError:
                continue
            self.life.io.submit(_quiet, lambda data=jpeg: screen.show(data, 6))
            caption = str(photo.get('caption') or '').split(' (в кадре')[0]
            if caption:
                await self.say(caption, wait=False)
            shown += 1
            await self.sleep(5.)
        return f'показал {shown} снимков'


GAMES = {cls.name: cls for cls in (HideAndSeek, TisheEdesh, PhotoBooth, HighFive, Deliver, WakeUp, Timer, Album)}
ALIASES = {'прятки': 'hide_seek', 'hide': 'hide_seek', 'тише едешь': 'red_light', 'светофор': 'red_light',
           'red light': 'red_light', 'замри': 'red_light', 'море волнуется': 'red_light', 'фото': 'photo',
           'фотобудка': 'photo', 'сфотографируй': 'photo', 'дай пять': 'high_five', 'пять': 'high_five',
           'альбом': 'album', 'снимки': 'album', 'фотки': 'album'}


def resolve(name):
    text = str(name or '').casefold().strip()
    if text in GAMES:
        return text
    return next((key for alias, key in ALIASES.items() if alias in text), None)


class Play:
    """One game at a time on the Life loop."""

    def __init__(self, life):
        self.life = life
        self.task = None
        self.game = None
        self.cancelled = False
        self.waiting: dict[str, asyncio.Future] = {}
        self.serial = 0
        self.touch_future = None

    @property
    def active(self):
        return self.task is not None and not self.task.done()

    def start(self, name, who=None, args=None):
        key = resolve(name)
        if key is None:
            return False, 'такой игры не знаю: ' + ', '.join(cls.title for cls in GAMES.values())
        if self.active:
            return False, f'уже занят: {self.game.title}'
        self.cancelled = False
        self.game = GAMES[key](self, who, args)
        self.task = self.life.loop.create_task(self._run(self.game))
        return True, f'начинаю игру: {self.game.title}'

    def stop(self):
        if self.active:
            self.cancelled = True
            self.life.publish('body.stop', {'reason': 'game_over'})

    async def _run(self, game):
        result = 'игра прервана'
        try:
            result = await game.run()
        except GameOver:
            result = 'игру остановили'
        except Exception as error:                 # a broken game must not take life down
            result = f'игра сломалась: {type(error).__name__}'
            self.life.archive.event('game_error', {'game': game.name, 'error': f'{type(error).__name__}:{error}'[:200]})
        finally:
            for future in self.waiting.values():
                if not future.done():
                    future.cancel()
            self.waiting.clear()
            if hasattr(self.life, 'face_calm'):
                self.life.face_calm()           # the game's faces were held: back to the waiting face
        self.life.world.event('game_played', f'сыграли в «{game.title}»: {result}', who=game.who, game=game.name)

    async def body(self, skill, args, timeout):
        self.serial += 1
        skill_id = f'game-{self.serial}'
        future = self.life.loop.create_future()
        self.waiting[skill_id] = future
        self.life.publish('body.do', {'skill': skill, 'args': args, 'origin': 'game', 'id': skill_id})
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            self.life.publish('body.stop', {'reason': 'game_timeout'})
            return 'timeout'
        except asyncio.CancelledError:
            if not self.cancelled:
                self.life.publish('body.stop', {'reason': 'game_next'})
            raise
        finally:
            self.waiting.pop(skill_id, None)

    async def locate(self, seconds):
        """Ask the ears where the voice of the last `seconds` came from (voice.locate → event «located»): the event's
        fields, or None if the voice does not answer in time."""
        self.serial += 1
        ask_id = f'locate-{self.serial}'
        future = self.life.loop.create_future()
        self.waiting[ask_id] = future
        self.life.publish('voice.locate', {'id': ask_id, 'seconds': seconds})
        try:
            return await asyncio.wait_for(future, 3.)
        except asyncio.TimeoutError:
            return None
        finally:
            self.waiting.pop(ask_id, None)

    def on_located(self, data):
        future = self.waiting.get(str(data.get('id')))
        if future is not None and not future.done():
            future.set_result(data)

    def on_skill(self, event):
        future = self.waiting.get(str(event.get('id')))
        if future is not None and not future.done() and event.get('status') in ('done', 'failed', 'cancelled'):
            future.set_result(event['status'])
            if event['status'] == 'cancelled' and event.get('reason') in ('voice_stop', 'conversation', 'mind',
                                                                          'autonomy_off'):
                self.cancelled = True              # «стоп» or someone talking to me ends the game

    async def photo(self, caption):
        photo_id = f'photo-{int(time.time())}'
        future = self.life.loop.create_future()
        self.life.looks[photo_id] = future
        self.life.captions[photo_id] = caption
        self.life.publish('eyes.look', {'id': photo_id, 'keep': True, 'caption': caption})
        try:
            looked = await asyncio.wait_for(future, 3.)
        except asyncio.TimeoutError:
            return None
        return looked.get('path')

    async def wait_touch(self, seconds):
        self.touch_future = self.life.loop.create_future()
        try:
            await asyncio.wait_for(self.touch_future, seconds)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            self.touch_future = None

    def on_touch(self):
        self.touched_at = time.time()
        if self.touch_future is not None and not self.touch_future.done():
            self.touch_future.set_result(True)

    def touched_since(self, since):
        return getattr(self, 'touched_at', 0.) > since


def _quiet(fn):
    try:
        return fn()
    except Exception:
        return None
