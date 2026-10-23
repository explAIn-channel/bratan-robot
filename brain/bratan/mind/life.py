"""Life: ties the senses, the mind, the voice and the body together, and keeps the robot safe.

Runs on the supervisor's asyncio loop. Model calls run in worker threads; their
results come back through `self.publish` (thread-safe).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ..common.config import STATE, read_json, secret, write_json
from ..face import client as face
from .brain import ASIDES_IN_ROW, LOOK_WORDS, NOT_FOR_ME, Mind, accusative
from .executive import Executive
from .games import Play
from .llm import LLM, LLMError, direct_llm
from .memory import Archive, MindState
from .weather import Weather
from .phone import ChargeLimit
from .world import World

FACE_MIN_GAP_S = .6
LOW_BATTERY_HOLD_S = 10.
CHARGED_AFTER_S = 600.     # on the contacts the charger lifts the reading at once (08.10 00:43: «зарядился» 2 s after docking)
GAZE_EVERY_S = .4
GAZE_HOLD_S = 1.2          # the face app holds a requested gaze 1.8 s, then returns to its own idle glances
LOST_CONFIRM_S = 3.        # a loss of place that heals by itself within this is not worth turning around for
# A reflex on the face before any thinking: what the ears just made out.
SOUND_REACTIONS = {'dog': 'happy', 'laughter': 'laughing', 'music': 'happy', 'doorbell': 'curious', 'knock': 'curious',
                   'bang': 'surprised', 'door': 'curious', 'clapping': 'happy', 'whistle': 'curious',
                   'sneeze': 'curious', 'crying': 'concerned', 'alarm': 'curious'}
# A whole feeling at once (owner 08.10): the body moves, the mood lamp shimmers (the face app colours the lamp by
# the face state: happy yellow, laughing magenta, curious cyan, affectionate pink, surprised white...), and a sound
# goes with it when someone is there to hear it.
# The lamp is an LED panel (ELK-BLEDOM) with its own dynamic modes (owner 08.10 12:35): a feeling runs one of them
# for a few seconds (face app 87+: /mood-light/effect) while the face shows it; the colour steps below are the
# fallback for an older face app.
EXPRESSION_LIGHTS = {'wiggle': ('rainbow', 'happy'), 'tremble': ('party', 'laughing'), 'bounce': ('warm', 'happy'),
                     'spin': ('party', 'happy'), 'shiver': ('flash', 'surprised'), 'nod': ('warm', 'happy'),
                     'shake': ('no', 'concerned'), 'bow': ('love', 'affectionate'), 'love': ('love', 'affectionate')}
EFFECT_S = 4
# A strong reaction to something real shimmers briefly too; the calm waiting face stays a quiet steady glow.
REACTION_EFFECTS = {'laughing': 'party', 'surprised': 'flash', 'affectionate': 'love', 'happy': 'warm'}
REACTION_EFFECT_S = 3
REACTION_EFFECT_GAP_S = 8.
EXPRESSIONS = {
    'wiggle': (('happy', 'laughing', 'happy', 'curious', 'happy', 'laughing'), ('happy', 'excited')),
    'tremble': (('laughing', 'curious', 'happy', 'affectionate', 'surprised', 'happy', 'laughing'), ('excited', 'laughing')),
    'bounce': (('happy', 'curious', 'laughing', 'happy', 'curious'), ('excited', 'happy')),
    'spin': (('curious', 'happy', 'laughing', 'affectionate', 'curious', 'happy', 'laughing'), ('excited',)),
    'shiver': (('surprised', 'idle', 'surprised', 'idle'), ('surprised',)),
    'nod': (('happy', 'curious', 'happy'), ('yes',)),
    'shake': (('concerned', 'idle', 'concerned'), ('no',)),
    'bow': (('affectionate', 'happy', 'affectionate'), ('hello', 'affectionate')),
    'love': (('affectionate', 'happy', 'affectionate', 'happy', 'affectionate'), ('affectionate',)),
}
LIGHT_STEP_S = .35

# Reflex sounds: only what people do to or around him (and a sharp bang); alone he stays silent (sfx()).
REFLEX_SOUNDS = {'touch': 'affectionate', 'carried': 'surprised', 'startle': 'surprised', 'boop': 'laughing',
                 'handled': 'curious'}
SOUND_REFLEX_SOUNDS = {'laughter': 'laughing', 'bang': 'surprised', 'sneeze': 'curious', 'clapping': 'happy'}
# Sounds the eyes and then the body turn to when the ears know where they came from (C52; startle always did).
ORIENT_SOUNDS = ('dog', 'doorbell', 'knock', 'bang', 'laughter', 'clapping', 'whistle', 'crying', 'alarm', 'sneeze')
REACTIONS = {'dog_appeared': 'happy', 'startle': 'surprised', 'boop': 'laughing', 'bump': 'surprised', 'carried': 'surprised', 'touch': 'affectionate',
             'handled': 'surprised',
             'object_gone': 'curious', 'found': 'happy', 'door_opened': 'curious'}


class Life:
    def __init__(self, hub, loop, config):
        self.hub, self.loop, self.config = hub, loop, config
        self.state = MindState(STATE / 'mind.json')
        self.archive = Archive(STATE / 'memory.sqlite')
        rooms = read_json(config['rooms_json'], {}) or {}
        room_map = None
        try:
            import numpy as np
            from ..body.grid import Grid
            room_map = (Grid(config['map_yaml']), np.load(config['rooms_labels']))
        except Exception as error:
            print('room_map_unavailable', type(error).__name__, error, flush=True)
        self.world = World(rooms, config.get('pets', {}).get('dog'), self.archive, room_map, state=self.state)
        self.world.person_debounce = float(config.get('person_debounce_s', self.world.person_debounce))
        if room_map is not None:
            from .world import MOVING_LABELS
            removed, placed, missing = self.archive.tidy(self.world.room_at, MOVING_LABELS)
            self.archive.event('objects_tidy', {'removed_moving': removed, 'rooms_set': placed, 'without_room': missing})
        key = secret(config['llm_key'])
        direct = direct_llm(config)
        self.llm = LLM(config['llm_base_url'], key, config['llm_fast'], pool=3, fallback=config.get('llm_smart'),
                       providers=config.get('llm_providers', ()), direct=direct(3))
        self.vision = LLM(config['llm_base_url'], key, config['llm_vision'], pool=1, direct=direct(1))
        self.mind = Mind(self.state, self.world, self.archive, self.llm, config, self.act)
        self.executive = Executive(self)       # what the body is doing, and is it still worth it (10 Hz)
        self.weather = Weather(config)
        self.mind.weather = self.weather
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='mind')
        # Face/light/gaze calls are small and frequent: their own workers, never queued behind a model call.
        self.io = ThreadPoolExecutor(max_workers=2, thread_name_prefix='face-io')
        self.captions: dict[str, str] = {}
        self.events_seen = time.time()      # time of the newest world event already reacted to
        self.gaze = (None, 0.)
        self.charge = ChargeLimit(STATE / 'phone-charge-held.json', config['phone_charge_limit_percent'],
                                  config['phone_charge_resume_percent'])
        self.looks: dict[str, asyncio.Future] = {}
        self.last_face = (None, 0.)
        self.low_since = None
        self.departure_announced = False
        self.docked_since = None
        self.context_text = ''
        self.thinking = None
        self.started = time.time()
        self.display_awake = None          # unknown until the first decision
        self.last_activity = time.time()
        self.play = Play(self)
        self.peek = {'state': 'none', 'at': 0., 'last': 0.}
        self.cancelled_asks: list = []

    # ------------------------------------------------------------ bus glue
    def publish(self, topic, data):
        """Thread-safe publish onto the hub."""
        if threading_main(self.loop):
            self.hub.publish(topic, data)
        else:
            self.loop.call_soon_threadsafe(self.hub.publish, topic, data)

    def start(self):
        self.loop.call_later(3., self.face_calm)      # a restart may leave an old held face on the screen
        self.hub.subscribe(('body.state', 'body.event', 'eyes.seen', 'eyes.looked', 'eyes.enrolled',
                            'voice.state', 'voice.event', 'voice.ask', 'mind.play', 'eyes.things', 'eyes.gesture',
                            'voice.cancel'),
                           self.on_message)
        self.pool.submit(self.llm.warm)
        self.loop.create_task(self.run())

    def on_message(self, topic, data):
        if topic == 'body.state':
            self.world.on_body(data)
            self.on_touch(data)
        elif topic == 'body.event':
            self.world.on_body_event(data)
            self.on_body_event(data)
        elif topic == 'eyes.seen':
            self.world.on_eyes(data)
        elif topic == 'eyes.things':
            self.world.on_things(data)
        elif topic == 'eyes.gesture':
            self.on_gesture(data)
        elif topic == 'eyes.looked':
            future = self.looks.pop(str(data.get('id')), None)
            if future is not None and not future.done():
                future.set_result(data)
            self.on_photo(data)
        elif topic == 'eyes.enrolled':
            self.world.event('learned_face', f'запомнил лицо: {data.get("name")} ({data.get("samples")} снимков)')
        elif topic == 'voice.state':
            was = self.world.live
            self.world.voice = data
            if data.get('live') and not was:
                self.on_conversation_start()
        elif topic == 'voice.event':
            self.on_voice_event(data)
        elif topic == 'voice.ask':
            self.loop.create_task(self.answer(data))
        elif topic == 'voice.cancel':
            # The person went on talking before we answered: that answer is not said and not acted on.
            self.cancelled_asks.append(str(data.get('id')))
            del self.cancelled_asks[:-20]
        elif topic == 'mind.play':
            # A game started from outside (the owner's CLI, tests): `cli pub mind.play '{"game": "прятки"}'`.
            if data.get('express'):
                self.express(str(data['express']), origin='owner')     # a feeling on demand (tests, the owner)
            elif data.get('stop'):
                self.play.stop()
            else:
                ok, text = self.play.start(data.get('game'), data.get('who') or (self.world.visible() or [None])[0])
                self.archive.event('play_request', {'game': data.get('game'), 'ok': ok, 'text': text})

    # ------------------------------------------------------------ actions for the mind
    def act(self, kind, **data):
        if kind == 'quiet':
            return self.quiet()
        if kind == 'playing':
            return self.play.active
        if kind == 'play':
            return self.play.start(data.get('game'), data.get('who'), data.get('args'))
        if kind == 'may_leave_dock':
            voltage = self.world.body.get('voltage') or 0
            threshold = self.config['departure_voltage']
            # Charging voltage wobbles around the threshold: once ready was announced, keep it.
            return voltage >= threshold or (self.departure_announced and voltage >= threshold - .2)
        if kind == 'face':
            self.face(data['state'])
        elif kind == 'say':
            self.wake_face()
            if data.get('aloud'):
                # A thought said aloud while driving: the plain voice, never a live session for a line like this.
                self.publish('voice.say', {'text': data['text'], 'id': 'aloud', 'tts_only': True})
            elif data.get('quiet'):
                self.publish('voice.say', {'text': data['text'], 'id': 'self'})
            elif data.get('answer'):
                # An answer that came later (the internet, C52): said in the plain voice as part of the talk — the
                # ears listen for what follows it.
                self.publish('voice.say', {'text': data['text'], 'local': True, 'dialog': True, 'live': True})
            else:
                self.publish('voice.say', {'text': data['text'], 'live': bool(data.get('live'))})
        elif kind == 'body_do':
            sent = f'm{int(time.time() * 1000) % 10**9}'
            self.publish('body.do', {'skill': data['skill'], 'args': data.get('args', {}),
                                     'origin': data.get('origin', 'mind'), 'id': sent})
            return sent                         # a step of my own skill waits for the end of this id
        elif kind == 'body_stop':
            self.publish('body.stop', {'reason': 'mind'})
        elif kind == 'photo':
            photo_id = f'photo-{int(time.time())}'
            self.captions[photo_id] = str(data.get('caption') or '')[:80]
            self.publish('eyes.look', {'id': photo_id, 'keep': True, 'caption': data.get('caption', '')})
        elif kind == 'show_photo':
            return self.show_photo(data.get('id'))
        elif kind == 'eyes_focus':
            self.publish('mind.eyes', {'mode': 'focus', 'seconds': 60})
        elif kind == 'step':
            self.publish('mind.step', data)
        elif kind == 'sfx':
            # The mind's sound keeps the 30 s gap: right after a reflex sound (a wave, a hello) it is dropped.
            self.sfx(data.get('emotion'))
        elif kind == 'express':
            return self.express(data.get('emote', 'wiggle'))
        elif kind == 'reflex':
            if data.get('tone') == 'praise':
                self.react('affectionate')
                self.express('love' if self.world.body.get('docked') else random.choice(('wiggle', 'tremble')),
                             origin='reflex')
            elif data.get('tone') == 'scold':
                self.react('concerned')
                self.sfx('sad', force=True)
        elif kind == 'enroll':
            self.publish('eyes.enroll', {'name': data['name']})
        return True

    def face(self, state):
        """A feeling on the face for a few seconds (the app's reaction layer expires by itself back to the calm
        waiting face). Never the 60 s event layer: a stray thought's face held there looked like a mood."""
        name, at = self.last_face
        now = time.monotonic()
        if state == name and now - at < 10 or now - at < FACE_MIN_GAP_S:
            return
        self.last_face = (state, now)
        self.io.submit(_quiet_call, face.post, f'/emotion?state={state}&source=reaction&event_id=f{int(now * 1000)}',
                       'face.write')

    def face_calm(self):
        """Back to the waiting face now: clear whatever a game or a show held."""
        self.last_face = ('idle', time.monotonic())
        for layer in ('event', 'reaction'):
            self.io.submit(_quiet_call, face.post, f'/emotion?state=idle&source={layer}', 'face.write')

    def face_now(self, state):
        """A face for a game: shown at once, no throttle."""
        self.last_face = (state, time.monotonic())
        self.wake_face()
        self.io.submit(_quiet_call, face.post, f'/emotion?state={state}&source=event', 'face.write')

    def dog_moment_tick(self, now):
        world = self.world
        if now - world.eyes_t > .6 or world.dark or self.quiet() or not self.config.get('autonomy'):
            return
        close = [i for i in world.items if i['label'] == 'dog' and i.get('score', 0) >= .6
                 and (i['box'][3] - i['box'][1]) > 200 and i.get('frames', 1) >= 4]
        if not close or now - self.state.get('dog_photo_at', 0) < 3600:
            return
        self.state.set('dog_photo_at', now)
        name = self.config.get('pets', {}).get('dog') or 'пёс'
        photo_id = f'photo-{int(now)}'
        self.captions[photo_id] = f'{name} позирует'
        self.publish('eyes.look', {'id': photo_id, 'keep': True, 'caption': f'{name} позирует'})
        world.event('dog_photo', f'{name} подошёл близко — сфотографировал его для альбома')

    def peekaboo_tick(self, now):
        """Ку-ку: someone close hides the face (hands) and shows it again — a laugh (babies and pets love it)."""
        world = self.world
        if (now - world.eyes_t > .6 or self.play.active or world.live or self.quiet()
                or not self.config.get('autonomy')):
            return
        close = [i for i in world.items if i['label'] == 'person' and (i['box'][3] - i['box'][1]) > 260]
        if not close:
            self.peek['state'] = 'none'
            return
        face_seen = any(i.get('face') for i in close)
        peek = self.peek
        if face_seen:
            if peek['state'] == 'hidden' and .7 < now - peek['at'] < 6. and now - peek['last'] > 30.:
                peek['last'] = now
                self.react('laughing')
                if not world.body.get('docked'):        # on the dock: a laugh on the face and the lamp, silently
                    self.sfx('laughing', force=True)
                    self.publish('voice.say', {'text': 'Ку-ку!', 'id': 'peekaboo', 'tts_only': True})
                world.event('peekaboo', 'со мной играли в ку-ку: лицо спрятали и показали', who=(world.visible() or [None])[0])
            if peek['state'] != 'face':
                peek.update(state='face', at=now)
        elif peek['state'] == 'face' and now - peek['at'] > 1.:
            peek.update(state='hidden', at=now)

    def react(self, state):
        """A quick, short-lived reaction on the face and the light (surprise at a bump, joy at Taffa)."""
        self.wake_face()
        self.io.submit(_quiet_call, face.post, f'/emotion?state={state}&source=reaction&event_id=r{int(time.time() * 1000)}',
                       'face.write')
        effect = REACTION_EFFECTS.get(state)
        now = time.time()
        if (effect and not getattr(self, 'showing', False) and not self.quiet()
                and now - getattr(self, 'reaction_effect_at', 0.) > REACTION_EFFECT_GAP_S):
            self.reaction_effect_at = now
            self.io.submit(self.light_effect, effect, REACTION_EFFECT_S)

    def light_effect(self, name, seconds):
        """The panel's own dynamic mode for a few seconds (blocking: run it in a worker). True if shown; an
        older face app (no such call) is remembered and the colour steps are used instead."""
        if getattr(self, 'effects_unsupported', False):
            return False
        try:
            face.post(f'/mood-light/effect?name={name}&seconds={int(seconds)}', 'face.write')
            return True
        except Exception as error:
            if getattr(error, 'code', None) in (400, 404):
                self.effects_unsupported = True
            return False

    SFX_GAP_S = 30.

    def with_people(self):
        world = self.world
        return bool(world.visible() or world.available() or world.live or self.play.active)

    def sfx(self, emotion, force=False, serious=False, asked=False):
        """A sound without words (voice/sfx.py). Owner 08.10: rare and social — only with people around (to call
        them, react to them, play) or at something really sharp (`serious`: a loud bang); never alone, never to
        voice an inner state; never in the quiet hours or when asked to be quiet. On the dock — silent
        (owner 08.10 12:35: a sound each time he saw me from the base got tiresome): sounds are for when he is
        out and about, in a game, or when asked for (`asked`)."""
        now = time.time()
        if not emotion or self.quiet() or self.state.get('hush_until', 0) > now:
            return
        if self.world.body.get('docked') and not (asked or self.play.active):
            return
        if not (serious or asked) and not self.with_people():
            return
        if not force and now - getattr(self, 'sfx_at', 0.) < self.SFX_GAP_S:
            return
        self.sfx_at = now
        self.publish('voice.sfx', {'emotion': emotion})

    def wiggle(self, kind='wiggle', sound=None):
        """Joy as a reflex, when free to: not during a talk or a game, at most once a minute. True if shown."""
        if self.world.live or self.quiet() or not self.config.get('autonomy') or self.play.active:
            return False
        if time.time() - getattr(self, 'wiggled_at', 0.) < 60:
            return False
        self.wiggled_at = time.time()
        self.express(kind, origin='reflex', sound=sound)
        return True

    def express(self, kind, origin='mind', sound=None):
        """A feeling with everything he has: the body (off the dock and free), the lamp, a sound with people."""
        lights, sounds = EXPRESSIONS.get(kind, EXPRESSIONS['wiggle'])
        effect, feeling = EXPRESSION_LIGHTS.get(kind, EXPRESSION_LIGHTS['wiggle'])
        if sound:
            sounds = (sound,)
        body = self.world.body
        moved = False
        if kind in ('wiggle', 'tremble', 'bounce', 'spin', 'shiver', 'nod', 'shake', 'bow') and not body.get('docked') \
                and not body.get('moving') and not (body.get('skill') and origin == 'reflex'):
            self.publish('body.do', {'skill': 'emote', 'args': {'kind': kind}, 'origin': origin, 'id': f'emote-{kind}'})
            moved = True
        self.sfx(random.choice(sounds), force=True, asked=origin == 'owner')    # asked for: heard even unseen
        self.loop.create_task(self.light_show(lights, effect, feeling))
        return moved

    async def light_show(self, states, effect=None, feeling=None):
        """The panel runs its own mode of the feeling (rainbow, jumping colours, white blinks...) while the face
        shows it; without the mode (an older face app, the panel not ready) the face and the lamp step through
        a few colours of the feeling. Then back to the calm waiting face."""
        if getattr(self, 'showing', False):
            return
        self.showing = True
        try:
            asleep = self.display_awake is not True
            self.wake_face()
            if asleep:
                await asyncio.sleep(.5)             # the panel takes no mode while the screen is dark
            if effect and await self.loop.run_in_executor(self.io, self.light_effect, effect, EFFECT_S):
                self.io.submit(_quiet_call, face.post,
                               f'/emotion?state={feeling or states[0]}&source=reaction&event_id=e{int(time.time() * 1000)}',
                               'face.write')
                await asyncio.sleep(EFFECT_S)
                return
            for index, state in enumerate(states):
                self.io.submit(_quiet_call, face.post,
                               f'/emotion?state={state}&source=reaction&event_id=s{int(time.time() * 1000)}{index}',
                               'face.write')
                await asyncio.sleep(LIGHT_STEP_S)
        finally:
            self.showing = False
            self.face_calm()                    # the show is over: the calm waiting face, not a «mood» face

    def look_at(self, x, y):
        self.io.submit(_quiet_call, face.post, f'/gaze?x={x:.2f}&y={y:.2f}', 'face.write')

    def gaze_tick(self, now):
        """Eyes on whoever is in front (a face first), else toward where the body is turning.

        Gaze at the person is the strongest cue that a robot is alive and attentive (warmth, intention);
        without a target the face app keeps its own idle glances."""
        if not self.display_awake or now - self.gaze[1] < GAZE_EVERY_S:
            return
        world = self.world
        target = None
        if now - world.eyes_t < 1.:
            people = [i for i in world.items if i['label'] == 'person' and i.get('score', 0) >= .5]
            dogs = [i for i in world.items if i['label'] == 'dog' and i.get('score', 0) >= .45]
            pick = [i for i in people if i.get('face')] or people or dogs
            if pick:
                target = max(pick, key=lambda i: (i['box'][2] - i['box'][0]) * (i['box'][3] - i['box'][1]))
        if target is not None and target.get('bearing') is not None:
            # bearing > 0: to the robot's left = to the right of the screen as the person sees it
            x = max(-1., min(1., float(target['bearing']) / .45))
            y = -.25 if target['label'] == 'person' else .35
        else:
            twist = world.body.get('twist') or (0., 0.)
            if not world.body.get('moving') or abs(twist[1]) < .15:
                return
            x, y = max(-.8, min(.8, twist[1] / .5)), 0.
        last, at = self.gaze
        if last is not None and abs(last[0] - x) < .08 and abs(last[1] - y) < .08 and now - at < GAZE_HOLD_S:
            return
        self.gaze = ((x, y), now)
        self.look_at(x, y)

    def expression_tick(self):
        """Immediate reactions to what just happened, before the mind has even thought about it."""
        fresh = [e for e in list(self.world.events) if e['t'] > self.events_seen]
        if not fresh:
            return
        self.events_seen = fresh[-1]['t']
        for event in fresh[-3:]:
            state = REACTIONS.get(event['kind'])
            if event['kind'] in ('sound', 'sound_note'):
                state = SOUND_REACTIONS.get(event.get('sound'))
            named = event['kind'] == 'person_appeared' and event.get('who') not in (None, 'человек')
            if named:
                state = 'happy'
            if state:
                self.react(state)
            # The same moment as a sound without words, and joy with the whole body.
            sound = REFLEX_SOUNDS.get(event['kind']) or (SOUND_REFLEX_SOUNDS.get(event.get('sound'))
                                                         if event['kind'] == 'sound' else None)
            if named:
                sound = 'hello'
            joy = event['kind'] == 'dog_appeared' or named and ('сегодня вижу впервые' in event.get('text', '')
                                                                or 'не виделись' in event.get('text', ''))
            if joy and self.wiggle('wiggle', sound=sound):
                continue                            # the wiggle carried the sound: one sound, not two
            if sound:
                self.sfx(sound, serious=event['kind'] == 'startle' or event.get('sound') == 'bang')

    def on_cue(self, event):
        """A step of a number has started in the body: its words, sound and face now, with the motion."""
        steps = self.mind.performances.get(str(event.get('id')))
        index = event.get('index')
        if not steps or not isinstance(index, int) or not 0 <= index < len(steps):
            return
        step = steps[index]
        if step.get('face'):
            self.react(step['face'])
        if step.get('sound'):
            self.sfx(step['sound'], force=True, asked=True)
        if step.get('say'):
            self.wake_face()
            self.publish('voice.say', {'text': step['say'], 'id': 'perform', 'tts_only': True})

    def on_touch(self, state):
        """The buttons on the body: a press is a touch (petting); the dock button means «go home»."""
        count = state.get('touch_events')
        if count is None:
            return
        last, self.touch_count = getattr(self, 'touch_count', None), count
        if last is None or count <= last:
            return
        buttons = state.get('buttons') or [0, 0]
        if self.play.active:
            self.play.on_touch()                # «дай пять» and other games take the touch first
        now = time.time()
        presses = [t for t in getattr(self, 'presses', []) if now - t < 1.5] + [now]
        self.presses = presses
        if len(presses) >= 2 and not self.play.active and not self.quiet():
            # Pressed twice quickly: «listen to me», like calling him by name (Vector's back button).
            self.presses = []
            self.react('curious')
            self.publish('voice.listen', {})
            self.archive.event('touch', {'buttons': buttons, 'double': True})
            return
        if len(buttons) > 1 and buttons[1] and not state.get('docked'):
            self.world.event('touch', 'нажали кнопку «домой» на мне — просят ехать на базу')
            self.publish('body.do', {'skill': 'dock', 'origin': 'owner', 'id': 'button-dock'})
        else:
            who = (self.world.available() or self.world.visible() or [None])[0]
            self.world.event('touch', 'меня погладили — нажали кнопку на спине', who=who)
        self.archive.event('touch', {'buttons': buttons})

    def on_gesture(self, data):
        """Body language (eyes/gesture.py): a wave is answered at once — joy on the face and the lamp, a hello
        and a wiggle of the whole body (no arms to wave back with); a smile is mirrored (Moxie); a sad face is
        noticed quietly, for the mind to ask «всё хорошо?» if it is right to. Games read their own gestures."""
        kind = data.get('kind')
        who = data.get('who') or 'человек'
        name = '' if who == 'человек' else who
        if kind == 'wave':
            self.world.event('waved', f'{name or "человек"} помахал(а) мне рукой', who=who)
            if self.play.active:
                return
            self.react('happy')
            if not self.wiggle('wiggle', sound='hello'):
                self.sfx('hello')
        elif kind == 'hands_up':
            self.world.event('cheered', f'{name or "человек"} поднял(а) обе руки вверх — радуется', who=who)
            if self.play.active:
                return
            self.react('laughing')
            self.wiggle('tremble')
        elif kind == 'smile':
            self.world.event('smiled', f'{name or "человек"} улыбается мне', who=who)
            if not self.play.active:
                self.react('happy')                 # a smile for a smile, silently
        elif kind == 'sad':
            self.world.event('looks_sad', f'у {name or "человека"} грустное лицо', who=who)
            if not self.play.active:
                self.react('affectionate')
        elif kind == 'surprise' and not self.play.active:
            self.react('surprised')

    def on_photo(self, data):
        if not data.get('path') or not str(data.get('id', '')).startswith('photo-'):
            return
        from .brain import RU
        album = self.state.setdefault('album', [])
        # The mind's own words for the picture (07.10: the album showed «person» and empty captions).
        caption = self.captions.pop(str(data.get('id')), '')
        seen = ', '.join(sorted({RU(i['label']) for i in data.get('items', [])}))
        album.append({'id': Path(data['path']).stem, 'path': data['path'], 't': time.time(),
                      'caption': (caption + (f' (в кадре: {seen})' if seen else '') if caption else seen)[:120]})
        del album[:-150]

    def show_photo(self, photo_id):
        item = next((p for p in self.state.get('album', []) if p['id'] == photo_id), None)
        if item is None or not os.path.exists(item['path']):
            return False
        with open(item['path'], 'rb') as handle:
            jpeg = handle.read()
        self.io.submit(_quiet_call, face.post, '/photo?seconds=8', 'face.write', jpeg, 3., 'image/jpeg')
        return True

    # ------------------------------------------------------------ conversation
    def on_conversation_start(self):
        """A live conversation begins: the mind stops wandering and looks at the person."""
        self.mind.version += 1
        self.mind.asides_in_row = 0
        skill = self.world.body.get('skill') or {}
        if skill and skill.get('origin') == 'mind' and skill.get('kind') not in ('dock',):
            self.publish('body.stop', {'reason': 'conversation'})
        self.publish('mind.eyes', {'mode': 'focus', 'seconds': 600})
        self.face('happy')
        self.archive.event('conversation_start', {})
        self.turn_to_voice(skill)

    VOICE_BEARING_S = 8.          # the direction of «Братан» heard by the body's ears is fresh this long
    VOICE_BEARING_CONF = .3

    def heard_from(self, kinds=('wake',), within=None):
        """Bearing of the last sound of `kinds` the body's ears placed (voice/array.py): radians (+ left) or None."""
        heard = getattr(self.world, 'heard_bearing', None)
        within = self.VOICE_BEARING_S if within is None else within
        if not heard or heard['kind'] not in kinds or time.time() - heard['t'] > within \
                or heard['conf'] < self.VOICE_BEARING_CONF:
            return None
        return heard['bearing']

    def turn_to_voice(self, skill):
        """Called and nobody in view: turn to the voice — the strongest «he heard me» cue there is. Since 17.10 the
        body's three microphones say where «Братан» came from (LOVOT turns to a voice with its microphone array too);
        without them — where someone was last seen or something last moved. On the dock and during the owner's errand
        only the eyes go there."""
        world = self.world
        if world.visible() or world.dog_visible():
            return
        hint = self.heard_from(('wake',))
        if hint is None:
            hint = world.where_last('человек', within=60.)
        if hint is None or abs(hint) < .45:
            return                                  # nothing known, or ahead already: the camera sees it
        if self.display_awake:
            self.look_at(max(-1., min(1., hint / .45)), -.2)
        if world.body.get('docked') or (skill and skill.get('origin') == 'owner'):
            return
        self.publish('body.do', {'skill': 'rotate', 'args': {'angle': hint}, 'origin': 'reflex', 'id': 'turn-to-voice'})

    def on_voice_event(self, data):
        kind = data.get('kind')
        if kind != 'sound':                         # a sound is kept below, with where it came from
            self.archive.event('voice_' + str(kind), data)
        if data.get('bearing') is not None:
            self.world.heard_bearing = {'t': time.time(), 'bearing': float(data['bearing']),
                                        'conf': float(data.get('bearing_conf') or 0.), 'side': data.get('side'),
                                        'kind': 'wake' if kind == 'wake' else data.get('sound') or kind}
            if kind in ('wake', 'local_heard', 'located'):
                # My name, called, or words to me: a ray on the map too — «ко мне» from another room goes there (C52).
                self.world.sounds.add('wake' if kind == 'wake' else 'voice',
                                      'меня позвали' if kind == 'wake' else 'со мной говорят', data['bearing'],
                                      data.get('bearing_conf'))
        if kind == 'located':
            self.play.on_located(data)              # a game asked where the answer came from («Ку-ку! Где ты?»)
            return
        if kind == 'sound':
            self.world.on_sound(data)
            ray = self.world.sounds.latest((data.get('sound'),), within=2.)
            where = self.world.sounds.words(ray)
            # In the memory with its room (C52): the diary and the dreams say «лаял Таффа (10:05 со стороны …)».
            self.archive.event('heard_sound', {**data, **({'toward': ray['toward']} if where else {}),
                                               'here': self.world.room})          # where I was: the sound habits
            if data.get('sound') == 'startle' or (data.get('urgent') and data.get('sound') in ORIENT_SOUNDS):
                bearing = self.heard_from((data.get('sound'),), within=3.)
                if bearing is not None:
                    # A bang, a bark, the doorbell beside or behind: the eyes go there, then the body — the same
                    # orienting reflex as a rustle the lidar saw, with the same limits (not on the dock, not at
                    # night, not in a talk). The ears point the eyes (C52): what it was the camera then tells.
                    self.orient(bearing, data.get('side') or '')
            return
        if kind == 'wake':
            self.wake_face()
            self.face('curious')
        elif kind == 'local_started':
            self.world.event('voice_local', 'облачный голос не работает — разговариваю запасным голосом')
        elif kind == 'unavailable':
            self.face('concerned')
            reason = 'закончились деньги на голосовом сервисе' if 'credit' in str(data.get('code')) else 'сервис не отвечает'
            self.world.event('voice_down', f'меня позвали, но голос не работает ({reason}) — ответить вслух не могу')
        elif kind == 'live_closed':
            now = time.time()
            heard, said = data.get('heard', ''), data.get('said', '')
            if heard:
                who = self.world.speaker_by_voice(within=60.) or (self.world.available() or ['человек'])[0]
                self.state.heard(now, who, heard[-400:])
            if said:
                self.state.heard(now, self.config.get('name', 'Братан'), said[-400:])
            self.state.set('open_calls', 0)
            self.publish('mind.eyes', {'mode': 'normal', 'seconds': 1})
            self.world.event('conversation_ended', 'поговорили' + (f': «{heard[-120:]}»' if heard else ''))

    async def answer(self, ask):
        """Delegated request from the live voice: look if needed, think, answer."""
        text = ask.get('text', '')
        if self.mind.WEATHER_WORDS.search(text) and self.weather.current:
            from . import screen
            current = dict(self.weather.current)
            self.io.submit(_quiet_call, lambda: screen.show(screen.weather_card(current), 8))
        look = None
        if LOOK_WORDS.search(text):
            look = await self.look_and_describe(text)
        ask_id = str(ask.get('id') or '')
        # A long phrase without the name in the middle of a talk may be meant for someone else: its answer waits
        # for the whole reply (no sentence-by-sentence), so a «не понял, повтори» to it can be held back.
        aside = bool(ask.get('follow_up')) and not ask.get('named') and len(text.split()) >= 6
        if ask.get('named'):
            self.mind.asides_in_row = 0             # called by name: it is me again
        # The offline conversation speaks sentence by sentence as the model writes them.
        stream = ask_id.startswith('local-')
        parts = []

        def on_spoken(part):
            parts.append(part)
            self.publish('voice.answer_part', {'id': ask_id, 'text': part})
        spoken, decision = await self.loop.run_in_executor(
            self.pool, lambda: self.mind.converse(text, ask.get('said', ''), look=look,
                                                  on_spoken=on_spoken if stream and not aside else None,
                                                  cancelled=lambda: ask_id in self.cancelled_asks, aside=aside))
        # «streamed» only if parts really went out: a failed call returns a whole fallback line to say.
        self.publish('voice.answer', {'id': ask.get('id'), 'text': spoken, 'silent': not spoken,
                                      'streamed': stream and bool(parts)})
        if NOT_FOR_ME.search(text) or self.mind.asides_in_row >= ASIDES_IN_ROW:
            # «там я вообще не тебе, меня не слушай и не отвечай» (09.10 19:20), and the next phrase got an answer
            # again; 19:29-19:31 a phone call heard phrase by phrase: the talk is over until my name.
            self.publish('voice.end', {})

    async def look_and_describe(self, question):
        request_id = f'look-{int(time.time() * 1000)}'
        future = self.loop.create_future()
        self.looks[request_id] = future
        self.publish('eyes.look', {'id': request_id})
        try:
            looked = await asyncio.wait_for(future, 1.5)
        except asyncio.TimeoutError:
            return None
        if not looked.get('path'):
            return None
        try:
            return await asyncio.wait_for(self.loop.run_in_executor(
                self.pool, lambda: self.mind.describe_view(
                    looked['path'], question,
                    lambda content: self.vision.chat('Ты — зрение робота.', content, max_tokens=180,
                                                     temperature=.2, timeout=5.))), 5.5)
        except (asyncio.TimeoutError, LLMError):
            labels = ', '.join(sorted({i['label'] for i in looked.get('items', [])}))
            return f'(подробного описания нет) детектор видит: {labels or "ничего определённого"}'

    async def look_closer(self, event):
        """Arrived at an object: really look at it, so the mind knows what is there instead of guessing."""
        self.publish('mind.eyes', {'mode': 'focus', 'seconds': 5})
        await asyncio.sleep(.8)             # the body still sways right after stopping: a sharp frame needs a moment
        from .brain import RU
        target = getattr(self.mind, 'last_approach', None)
        name = RU(target[1]) if target else 'предмет'
        seen = await self.look_and_describe(
            f'Робот подъехал рассмотреть предмет, который по памяти считает «{name}». Первая строка ответа — '
            f'одно слово: «да», если перед ним правда {name}, «нет», если это что-то другое, «не видно», если '
            'кадр тёмный или неразборчивый. Дальше — что это и какие детали видны.')
        text = seen or 'камера не дала кадр'
        head, _, rest = text.strip().partition('\n')
        verdict = head.strip(' .,:!«»').casefold()
        detail = ' '.join((rest if verdict in ('да', 'нет', 'не видно') or verdict.startswith(('да', 'нет'))
                           else text).split())
        if target and verdict.startswith('нет'):
            # The memory was wrong (a wardrobe taken for a fridge): forget that object.
            self.archive.forget(target[0])
            self.archive.event('object_forgotten', {'id': target[0], 'label': target[1], 'seen': text[:200]})
            text = f'это оказался не {name} — забыл ошибочную запись. ' + detail
        elif target and verdict.startswith('да'):
            text = f'да, {name}: ' + detail       # the day summary keeps what was seen, not the bare «да»
        else:
            text = detail or text
        self.world.event('action_done', 'approach: подъехал и рассмотрел вблизи — ' + text,
                         id=event.get('id'), skill='approach', origin='mind')

    # ------------------------------------------------------------ body
    def on_body_event(self, event):
        kind = event.get('kind')
        if kind == 'motion':
            self.on_motion(event)
            return
        if kind == 'cue':
            self.on_cue(event)
            return
        if (kind == 'skill' and event.get('status') in ('done', 'failed') and
                (event.get('skill') == 'look_around' or str(event.get('reason', '')) in ('not_found', 'lost_sight'))):
            # The body has just looked all round (a look round, a search's full turn): the mind need not order
            # another one at once (09.10 11:12: «не нашёл» after a full turn, then «осмотрюсь как следует»).
            self.state.set('spun_at', time.time())
        intent = self.mind.intents.pop(event.get('id'), None) if kind == 'skill' and event.get('status') in (
            'done', 'failed', 'cancelled') else None
        if intent and event.get('status') == 'done':
            self.loop.create_task(self.carry_out(intent, event))
        orienting = getattr(self, 'orienting', None)
        if (kind == 'skill' and orienting and event.get('id') == orienting['id']
                and event.get('status') in ('done', 'failed', 'cancelled')):
            self.orienting = None
            if event.get('status') == 'done':
                self.loop.create_task(self.look_after_orienting(orienting))
        if kind == 'lost':
            # 07.10: lost and found again in the same second, but the turn-around search had already replaced
            # the running action (a trip was dropped). Search only if the place is still unknown a moment later.
            self.loop.call_later(LOST_CONFIRM_S, self._relocalize_if_lost)
        if (kind == 'skill' and event.get('status') == 'failed' and event.get('origin') == 'mind'
                and self.mind.last_body_do):
            self.mind.failed.setdefault(self.mind.last_body_do, []).append(time.time())
        if (kind == 'skill' and event.get('status') == 'failed' and event.get('reason') == 'not_there'
                and event.get('skill') == 'approach' and getattr(self.mind, 'last_approach', None)):
            # Ten frames in good view and light, never seen: the memory is wrong now (09.10 17:17 the same missing
            # chair was driven to again two seconds later).
            oid, label = self.mind.last_approach
            self.archive.forget(oid)
            self.archive.event('object_forgotten', {'id': oid, 'label': label, 'seen': 'не увидел на месте, подъезжая'})
        if (kind == 'skill' and event.get('status') == 'failed' and event.get('skill') == 'go'
                and str(event.get('reason', '')).startswith('no_route')      # an obstacle by the robot is not a closed way
                and getattr(self.mind, 'last_go_target', None)):
            closed = self.state.get('closed_ways', {})
            closed[self.mind.last_go_target] = time.time()      # do not try the same closed way again and again
            self.state.set('closed_ways', closed)
        if (kind == 'skill' and event.get('skill') == 'explore' and event.get('origin') == 'mind'
                and getattr(self.mind, 'last_explore', None)):
            room, status = self.mind.last_explore, event.get('status')
            if status == 'done':
                explored = self.state.get('explored', {})
                explored[room] = time.time()
                self.state.set('explored', explored)
            elif status == 'failed' and str(event.get('reason', '')).startswith('no_route'):
                closed = self.state.get('closed_ways', {})
                closed[room.casefold()] = time.time()
                self.state.set('closed_ways', closed)
            if status in ('done', 'failed', 'cancelled'):
                self.mind.last_explore = None
        if kind == 'skill' and str(event.get('id', '')).startswith('game-'):
            self.play.on_skill(event)
        if (kind == 'skill' and event.get('status') == 'done' and event.get('skill') == 'approach'
                and event.get('origin') == 'mind'):
            self.loop.create_task(self.look_closer(event))
        if kind == 'skill' and event.get('status') in ('done', 'failed', 'cancelled') and self.mind.skill_run:
            # Only the end of the step's own command moves my skill on (09.10 audit).
            self.loop.run_in_executor(self.pool, self.mind.next_skill_step, dict(event))
        if (kind == 'skill' and event.get('status') in ('done', 'failed', 'cancelled') and event.get('origin') == 'owner'
                and getattr(self.mind, 'then', None)):
            self.loop.run_in_executor(self.pool, self.mind.run_then, dict(event))      # «…, потом вернись на базу»

    ORIENT_SECTOR = .5          # rad: «the same place» for the inhibition of return (Vector: ±30°)
    ORIENT_REPEAT_S = 10.       # ...for this long, at most ORIENT_MAX turns there (Vector: 2 in 10 s)
    ORIENT_MAX = 2
    ORIENT_GAP_S = 3.           # and never two turns closer than this (no ping-pong between two rustles)
    ORIENT_LOOK_S = .7          # after the turn: the camera settles and the detector gets a frame or two
    MOTION_NEWS_S = 20.
    # Own activities a glance aside does not spoil (an errand of the owner's, a trip, a game are not interrupted).
    ORIENT_INTERRUPTS = ('wait', 'look_around', 'explore')

    def on_motion(self, event):
        """Something moved beside or behind me (lidar): the eyes go there at once, then the body turns, like a
        pet at a rustle (the orienting reflex: the superior colliculus turns eyes and head before one knows what it
        was; Vector: eyes first, then the body). Not twice to the same place in a few seconds (inhibition of
        return), not during the owner's errand, a talk or a game, never on the dock."""
        bearing, distance = float(event.get('bearing', 0.)), float(event.get('distance', 9.))
        side = 'сзади' if abs(bearing) > 2.3 else ('слева' if bearing > 0 else 'справа')
        if abs(bearing) < .5:
            return                                  # ahead: the camera sees it anyway
        body = self.world.body
        pose = body.get('pose') or [0., 0., 0.]
        heading = math.atan2(math.sin(pose[2] + bearing), math.cos(pose[2] + bearing))
        now = time.time()
        self.world.last_motion = (now, heading)
        # News for the mind only now and then, and not when someone is in view anyway (09.10 drive: 13 rustles
        # in 145 s with Taffa trotting along — the reflex takes every one, the thoughts need not).
        if now - getattr(self, 'motion_news_at', 0.) > self.MOTION_NEWS_S and not (
                self.world.visible() or self.world.dog_visible()):
            self.motion_news_at = now
            self.world.event('motion_seen', f'что-то шевельнулось {side}, в {distance:.1f} м', bearing=bearing,
                             heading=round(heading, 3))
        self.orient(bearing, side)

    def orient(self, bearing, side):
        """The orienting reflex itself (a rustle the lidar saw, a bang the body's ears placed): eyes first, then
        the body, with every limit of on_motion."""
        if abs(bearing) < .5:
            return                                  # ahead: the camera sees it anyway
        if self.quiet() or not self.config.get('autonomy') or self.world.live or self.play.active:
            return
        body = self.world.body
        pose = body.get('pose') or [0., 0., 0.]
        heading = math.atan2(math.sin(pose[2] + bearing), math.cos(pose[2] + bearing))
        if self.display_awake:
            self.look_at(max(-1., min(1., bearing / .45)), 0.)     # eyes first
        skill = body.get('skill') or {}
        free = not skill or (skill.get('origin') == 'mind' and skill.get('kind') in self.ORIENT_INTERRUPTS)
        if body.get('docked') or not free:
            return
        if self.world.visible() or self.world.dog_visible():
            # Someone is in view already, most likely the one moving about (09.10 17:25: four turns and four alike
            # thoughts after Taffa padding round the robot): the eyes follow, the body stays with them.
            return
        now = time.time()
        log = [(t, h) for t, h in getattr(self, 'orient_log', []) if now - t < self.ORIENT_REPEAT_S]
        same = sum(1 for t, h in log if abs(math.atan2(math.sin(h - heading), math.cos(h - heading))) < self.ORIENT_SECTOR)
        if same >= self.ORIENT_MAX or (log and now - log[-1][0] < self.ORIENT_GAP_S):
            self.orient_log = log
            return
        self.orient_log = log + [(now, heading)]
        self.react('curious')
        orient_id = f'orient-{int(now * 1000) % 10**9}'
        known = set(self.world.visible()) | ({'собака'} if now - self.world.dog.get('seen', -1e9) < 20. else set())
        self.orienting = {'id': orient_id, 'side': side, 'heading': heading, 'interrupted': skill.get('kind'),
                          'known': known}
        self.publish('body.do', {'skill': 'rotate', 'args': {'angle': bearing}, 'origin': 'reflex', 'id': orient_id})

    async def carry_out(self, intent, event):
        """Arrived where I went to someone: do what I went for, at once (09.10, the owner: «нашёл Таффу и смотрит на
        него — когда находишь, надо что-то делать; у него должны быть намерения»). Then the mind hears what was done
        and goes on from there."""
        who, kind = intent['who'], intent['kind']
        world, now = self.world, time.time()
        dog = who in ('собака', 'собаку')
        pet = self.config.get('pets', {}).get('dog') or 'пёс'
        name = pet if dog else who
        there = world.dog_visible() if dog else bool(world.visible())
        if event.get('reason') in ('time', 'lost_sight') and not there:
            return                                  # it slipped away before I got there: nothing to do it with
        hushed = self.quiet() or self.state.get('hush_until', 0) > now
        asker = intent.get('asker')
        found_asker = not dog and asker and (who == asker or who in ('человек', 'кто-то'))
        did = []
        self.react('happy')
        emote = {'play': 'bounce', 'report': 'wiggle', 'call': 'wiggle'}.get(kind, 'wiggle')
        sound = {'play': 'excited', 'report': 'found', 'call': 'found'}.get(kind, 'hello')
        self.express(emote, origin='reflex', sound=sound)
        did.append('обрадовался' + (' и вильнул' if not world.body.get('docked') else ''))
        line = None
        if kind in ('report', 'call') and asker and intent.get('owner'):
            line = 'Вот ты где!' if found_asker else (f'Нашёл {accusative(name)}!' if kind == 'report'
                                                      else f'Иди посмотри — тут {name}!')
        elif kind == 'tell' and intent.get('words') and not dog:
            line = intent['words']
        elif kind == 'play' and not dog:
            line = 'Поиграем?'
        elif kind == 'greet' and not dog and intent.get('owner'):
            line = 'Привет!'
        if line and not hushed:
            self.publish('voice.say', {'text': line, 'live': True})
            did.append(f'сказал «{line}»')
        if kind == 'play' and dog:
            await asyncio.sleep(2.)
            self.express('spin', origin='reflex', sound='laughing')
            did.append('поиграл с ним: подпрыгнул и покружился')
        elif kind == 'photo':
            ok, text = await self.loop.run_in_executor(
                self.pool, self.mind.do, {'tool': 'photo', 'args': {'caption': name}}, bool(intent.get('owner')))
            did.append('сфотографировал' if ok else f'снимок не вышел: {text}')
        elif kind == 'company':
            self.publish('body.do', {'skill': 'track', 'args': {'who': who, 'mode': 'watch', 'seconds': 60},
                                     'origin': 'mind', 'id': f'company-{int(now)}'})
            did.append('остался рядом, смотрю на него')
        world.event('arrived', f'нашёл {accusative(name)} и подъехал ({"по просьбе" if intent.get("owner") else "сам"}): '
                               f'{", ".join(did)}; дальше реши: поиграть, сфотографировать, побыть рядом, '
                               'позвать кого-то посмотреть или заняться своим', who=who)

    async def look_after_orienting(self, orienting):
        """Turned to the rustle: what is there? The mind hears it at once («обернулся на шорох справа — там
        Таффа»); nobody there is only context."""
        await asyncio.sleep(self.ORIENT_LOOK_S)
        world = self.world
        if time.time() - world.eyes_t > 1.5:
            return
        dog = world.dog_visible()
        people = world.visible()
        names = [p for p in people if p != 'человек']
        found = ([self.config.get('pets', {}).get('dog') or 'собака'] if dog else []) + (names or (['человек'] if people else []))
        known = orienting.get('known') or set()
        new = [f for f in found if not ((f == (self.config.get('pets', {}).get('dog') or 'собака') and 'собака' in known)
                                        or f in known)]
        if found and not new:
            # The one I was with anyway: context, not a reason for a new thought.
            world.event('oriented_nothing', f'обернулся на движение {orienting["side"]} — там {", ".join(found)}, '
                                            'он(а) и был(а) рядом')
        elif found:
            world.event('oriented', f'обернулся на движение {orienting["side"]} — там {", ".join(found)}',
                        who=found[0])
        elif orienting.get('interrupted'):
            # My own look round was cut short for it: the mind decides whether to go on.
            world.event('oriented', f'обернулся на движение {orienting["side"]} — никого не увидел; '
                                    f'«{orienting["interrupted"]}» прервал')
        else:
            world.event('oriented_nothing', f'обернулся на движение {orienting["side"]} — никого не увидел')

    def _relocalize_if_lost(self):
        body = self.world.body
        if body.get('localized') is False and not self.world.docked and not self.world.live \
                and (body.get('skill') or {}).get('kind') != 'relocalize':
            self.publish('body.do', {'skill': 'relocalize', 'origin': 'system', 'id': 'relocalize'})

    def quiet(self):
        hour = datetime.now(ZoneInfo(self.config['timezone'])).hour
        start, end = self.config['quiet_start_hour'], self.config['quiet_end_hour']
        if start == end:
            return True                     # equal hours: asleep all day, until the owner says otherwise (11.10)
        return (hour >= start or hour < end) if start > end else start <= hour < end

    def energy(self):
        body = self.world.body
        voltage = body.get('voltage')
        if not voltage:
            return
        now = time.time()
        if body.get('docked') and self.docked_since is None:
            self.docked_since = now
            self.low_told = False
        if not body.get('docked'):
            self.departure_announced = False
            self.docked_since = None
            if voltage < self.config['board_reserve_voltage']:
                self.low_since = self.low_since or now
                skill = body.get('skill') or {}
                if now - self.low_since > LOW_BATTERY_HOLD_S and skill.get('kind') != 'dock':
                    if not getattr(self, 'low_told', False):
                        # Once per empty pack: 10.10 11:55-12:43 «батарея села — еду на базу» every 70 s while lost.
                        self.world.event('battery_low', 'батарея села — еду на базу')
                        self.low_told = True
                    self.publish('body.do', {'skill': 'dock', 'origin': 'energy', 'id': 'energy-dock'})
                    self.low_since = now + 60
            else:
                self.low_since = None
            if self.quiet() and not self.world.live and not (body.get('skill') or {}).get('kind') == 'dock':
                if now - getattr(self, 'night_dock_at', 0) > 300:
                    self.night_dock_at = now
                    self.publish('body.do', {'skill': 'dock', 'origin': 'night', 'id': 'night-dock'})
        elif (voltage >= self.config['departure_voltage'] and not self.departure_announced and not self.quiet()
              and now - (self.docked_since or now) >= CHARGED_AFTER_S):
            self.departure_announced = True
            self.world.event('departure_ready', 'зарядился — можно выезжать')

    # ------------------------------------------------------------ screen
    IDLE_DARK_S = 30.

    def wake_face(self):
        # Asleep in the quiet hours: a sound, a reaction or someone coming in does not light the screen
        # (owner 09.10 01:02: «захожу в комнату — он загорается, а должен спать»); a talk does.
        if self.quiet() and not self.world.live:
            return
        self.last_activity = time.time()
        if self.display_awake is not True:
            self.set_display(True)

    def set_display(self, awake):
        changed = awake != self.display_awake
        self.display_awake = awake
        self.io.submit(_quiet_call, face.post, f'/display?awake={int(awake)}', 'face.write')
        if changed:
            self.archive.event('display', {'awake': awake})

    def display_tick(self, now):
        """Dark screen when idle on the dock; awake while out, busy, talking or with someone near."""
        body, world = self.world.body, self.world
        busy = (not body.get('docked') or body.get('skill') or world.live or world.voice.get('speaking')
                or world.voice.get('activity', 'idle') != 'idle'          # heard «Братан»: awake at once (19.10)
                or (not self.quiet() and (world.available() or world.visible())))
        if busy:
            self.last_activity = now
        # At night the screen goes dark a few seconds after a talk (and after a restart), not half a minute.
        awake = now - self.last_activity < (5. if self.quiet() else self.IDLE_DARK_S)
        # Re-assert once a minute: a restarted face app wakes its screen (and the lamp) by default.
        if awake != self.display_awake or now - getattr(self, 'display_said_at', 0) > 60:
            self.display_said_at = now
            self.set_display(awake)

    # ------------------------------------------------------------ main loop
    async def run(self):
        last_context = 0.
        last_slow = 0.
        while True:
            await asyncio.sleep(.1)
            now = time.time()
            try:
                self.world.tick()
                self.display_tick(now)
                self.expression_tick()
                self.gaze_tick(now)
                self.peekaboo_tick(now)
                self.dog_moment_tick(now)
                self.executive.tick(now)
                if self.config.get('autonomy') and self.thinking is None and self.world.body:
                    trigger = self.mind.due(now, self.quiet())
                    if trigger:
                        self.thinking = self.loop.run_in_executor(self.pool, self.mind.step, trigger)
                if self.thinking is not None and self.thinking.done():
                    error = self.thinking.exception()
                    if error:
                        self.archive.event('mind_crash', {'error': f'{type(error).__name__}:{error}'[:200]})
                    self.thinking = None
                if self.world.available() and not self.world.live and now - getattr(self, 'prewarmed', 0) > 20:
                    self.prewarmed = now
                    self.publish('voice.prewarm', {})
                if now - last_context > 3.:
                    last_context = now
                    text = self.mind.live_context()
                    if text != self.context_text:
                        self.context_text = text
                        self.publish('mind.context.state', {'text': text})
                if now - last_slow > 5.:
                    last_slow = now
                    self.day_rhythm(now)
                    if self.weather.due(now):
                        self.weather.tried = now
                        self.loop.run_in_executor(self.pool, self.weather.fetch)
                    if self.config.get('autonomy'):
                        self.energy()
                        self.idle_home(now)
                        self.maybe_reflect(now)
                        self.maybe_dream(now)
                    if now - getattr(self, 'storage_at', 0) > 600:
                        self.storage_at = now
                        self.loop.run_in_executor(self.pool, self.storage_tick, now)
                        self.keep_localized(now)
                    self.publish('mind.state', {'autonomy': self.config.get('autonomy'), 'quiet': self.quiet(),
                                                'thinking': self.thinking is not None,
                                                'llm': self.llm.stats, 'phone': self.charge.poll()})
                    if self.quiet() and self.world.docked and not self.world.live:
                        self.publish('mind.eyes', {'mode': 'rest', 'seconds': 30})
                    self.state.save()
            except Exception as error:
                self.archive.event('life_error', {'error': f'{type(error).__name__}:{error}'[:300]})

    def day_rhythm(self, now):
        """Falling asleep and waking up are moments: a sleepy wind-down at the start of the quiet hours,
        a hello when they end (aibo's morning stretch, Vector waking on his charger)."""
        quiet = self.quiet()
        was = getattr(self, 'was_quiet', None)
        self.was_quiet = quiet
        if was is None or quiet == was or self.world.live:
            return
        if quiet:
            self.react('sleepy')                    # no sound: falling asleep is an inner state (owner 08.10)
            self.world.event('falling_asleep', 'начались тихие часы — засыпаю')
        else:
            self.wake_face()
            self.react('happy')
            self.sfx('hello')                       # good morning — to someone, out and about, not to the room
            self.world.event('woke_up', 'тихие часы кончились — проснулся, новый день')

    def keep_localized(self, now):
        """Lost while talking (or the first attempt failed): search again once the robot is free."""
        body = self.world.body
        if (body.get('localized') is False and not body.get('docked') and not body.get('skill')
                and not body.get('recovering') and not self.world.live and now - getattr(self, 'relocalized_at', 0) > 30):
            self.relocalized_at = now
            self.publish('body.do', {'skill': 'relocalize', 'origin': 'system', 'id': 'relocalize-retry'})

    def maybe_dream(self, now):
        """Once a night, asleep on the dock: the cleverer model reshapes the character from the day."""
        if (not self.quiet() or self.world.live or not self.world.body.get('docked')
                or now - self.state.get('dreamed_at', 0) < 12 * 3600 or getattr(self, 'dreaming', False)):
            return
        hour = datetime.now(ZoneInfo(self.config['timezone'])).hour
        if not 1 <= hour < 6:                     # deep night: everyone asleep, nothing to miss
            return
        self.dreaming = True

        async def run():
            try:
                record = await self.loop.run_in_executor(self.pool, self.mind.dream, now)
                if record:
                    self.world.event('dreamed', 'выспался и видел сон')
            finally:
                self.dreaming = False
        self.loop.create_task(run())

    def storage_tick(self, now):
        """The phone's memory must never fill up: keep our own data in bounds, report the rest."""
        from ..common import storage
        freed = storage.trim_log(Path('/opt/robotbrain/bratan/logs/main.log'))
        freed += storage.prune_trips(now)
        freed += storage.prune_album(self.state)
        if now - self.state.get('archive_pruned_at', 0) > 86400:
            self.state.set('archive_pruned_at', now)
            rows = storage.prune_archive(self.archive, now)
            if rows:
                self.archive.event('archive_pruned', {'rows': rows})
        free = storage.free_bytes()
        level = storage.level_of(free)
        self.publish('system.storage.state', {'free_gb': round(free / 2 ** 30, 1), 'level': level,
                                              'freed_mb': round(freed / 2 ** 20, 1), 't': time.time()})
        if level != getattr(self, 'storage_level', 'ok') and level != 'ok':
            self.archive.event('storage_low', {'free_gb': round(free / 2 ** 30, 1), 'level': level})
            self.world.event('storage_low', f'на телефоне мало места: {free / 2 ** 30:.1f} ГБ свободно'
                             + (' — новые видео не пишу' if level == 'critical' else '')
                             + '. Стоит сказать Сергею.')
        self.storage_level = level

    REFLECT_EVERY_S = 3 * 3600.

    def maybe_reflect(self, now):
        """Generative Agents' reflection: every few hours, resting on the dock, think the time over."""
        if (self.quiet() or self.world.live or not self.world.body.get('docked')
                or now - self.state.get('reflected_at', 0) < self.REFLECT_EVERY_S):
            return
        self.state.set('reflected_at', now)

        async def run():
            insight = await self.loop.run_in_executor(self.pool, self.mind.reflect, now)
            if insight:
                self.world.event('reflected', f'обдумал прошедшие часы: {insight}')
        self.loop.create_task(run())

    IDLE_HOME_S = 120.

    def idle_home(self, now):
        """Standing around off the dock with nothing happening.

        After 2 minutes the mind is asked to choose (somewhere to go, something to look at,
        or home); if it again chooses nothing for another 2 minutes, the body goes home
        instead of draining the battery (owner, 06.10).
        """
        body = self.world.body
        skill = (body.get('skill') or {}).get('kind')
        busy = (body.get('docked') or self.world.live or self.world.visible() or self.world.dog_visible()
                or (skill and skill != 'wait') or body.get('moving'))
        if busy:
            self.idle_since = None
            self.idle_asked = False
            return
        self.idle_since = getattr(self, 'idle_since', None) or now
        if now - self.idle_since <= self.IDLE_HOME_S:
            return
        self.idle_since = now
        if not getattr(self, 'idle_asked', False):
            self.idle_asked = True
            self.world.event('idle_choice', 'стою без дела вне базы уже 2 минуты — пора выбрать: поехать куда-то, '
                                            'поискать Таффу или людей, рассмотреть что-то — или вернуться на базу')
        else:
            self.idle_asked = False
            self.world.event('idle_home', 'ещё 2 минуты без дела — поехал на базу')
            self.publish('body.do', {'skill': 'dock', 'origin': 'mind', 'id': 'idle-home'})

    def close(self):
        self.state.save(force=True)
        try:
            self.charge.release()
        except OSError:
            pass


def threading_main(loop):
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def _quiet_call(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None
