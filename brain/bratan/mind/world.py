"""The mind's picture of the moment, built from bus messages, plus the events that wake it."""
from __future__ import annotations

import math
import time
from collections import deque

from ..eyes.detect import RU

PERSON_DEBOUNCE_S = 600.      # «вижу Сергея» again only after 10 min apart
GONE_AFTER_S = 15.
DARK_LIGHT = 25             # mean frame brightness below this: the room is dark (kitchen at night: 10)
FACE_FRESH_S = 10.      # a thought takes ~5 s: the face seen just before it still counts
SEEN_FRESH_S = 6.
PRESENT_HITS = 2            # frames within PRESENT_WINDOW_S before someone counts as really here
PRESENT_WINDOW_S = 3.
# 11.10 11:30:40 «вижу человека», 11:31:37 «вижу: Сергей», then «человек ушёл» and «Сергей ушёл»: one man, two beings.
UNNAMED_GRACE_S = 3.        # an unnamed person is news after this long: the face is usually named first
NAME_CARRY_S = 8.           # a named person out of the face's view for this long keeps the name on a new track
NAME_CARRY_M = 1.2          # ... if the new track is this close to where they were
# 19.10, the owner: «стоит мне чуток лицо закрыть — он меня не узнаёт; пусть предполагает, что это тот же человек, если
# он на том же месте в течение 5 минут». Someone at the very place a known person was seen this recently is them.
PLACE_CARRY_S = 300.
PLACE_CARRY_M = 1.
MERGE_S = 20.               # a «человек» seen this recently, recognised now, was this person all along
LOST_CONFIRM_S = 3.         # a loss of place shorter than this is not news (07.10: lost+found in one second)
DOCK_CONFIRM_S = 2.         # the charge contact chatters while settling on the dock (docked/undocked/docked)
VOLTAGE_WINDOW_S = 45.      # under load the pack sags 0.3-0.5 V: the honest level is the recent maximum
VISIT_SAVE_S = 30.
ABSENT_LOOKS = 12           # seconds of looking right at a remembered thing without seeing it: one strike
ABSENT_STRIKES_APART_S = 1200.   # two strikes on different occasions before a thing counts as taken away
# Only things people carry about can be «taken away»; furniture stays put and the detector often misses it up
# close or half in view (08.10: the bed, 510 sightings, the sofa, 391, were «not there» and erased).
PORTABLE_LABELS = {'book', 'bottle', 'cup', 'backpack', 'suitcase', 'teddy bear', 'sports ball', 'vase', 'handbag',
                   'bowl', 'umbrella', 'laptop', 'chair', 'potted plant', 'clock', 'bicycle'}
OBJECT_LABELS = {'chair', 'couch', 'bed', 'dining table', 'tv', 'laptop', 'potted plant', 'refrigerator',
                 'book', 'bottle', 'cup', 'backpack', 'suitcase', 'teddy bear', 'sports ball', 'clock', 'vase',
                 'handbag', 'bowl', 'oven', 'microwave', 'sink', 'umbrella', 'bicycle'}
# Animals and people move: they are sightings, never fixed things on the map
# (the detector sometimes calls our dog a cat — three «cats» had piled up in the bedroom).
# Why the body could not do something, in words the mind understands.
REASONS = {'no_route': 'не нашёл дороги — проход закрыт (может, закрыта дверь)',
           'blocked': 'путь перекрыт, проехать не смог', 'no_maneuver': 'не смог развернуться в тесном месте',
           'motion_inhibited': 'тело сейчас не может двигаться', 'lost_sight': 'потерял из виду',
           'not_localized': 'не понимал, где нахожусь', 'timeout': 'не успел', 'cancelled': 'отменил',
           'turn_blocked': 'тесно — развернуться не дало препятствие рядом', 'undock_blocked': 'съехать с базы мешает препятствие',
           'contact_unstable': 'контакт с базой неустойчивый', 'lost': 'потерялся на карте',
           'dock_blocked': 'к базе не подъехать — рядом с ней что-то стоит (стул?), нужно убрать',
           'dock_unreachable': 'не смог доехать до базы', 'dock_failed': 'не получилось встать на контакты базы',
           'board_disconnected': 'пропала связь с телом (платой) — сейчас восстановится',
           'board_restarted': 'плата тела перезапустилась',
           'passage_blocked': 'в дверь не проехал — три попытки, каждый раз отъезжал назад; похоже, в проёме что-то стоит',
           'no_viewpoint': 'не нашёл, откуда посмотреть', 'not_found': 'огляделся вокруг — не нашёл', 'not_seen': 'не увидел цель',
           'not_there': 'подъезжая, не увидел его на месте — похоже, убрали (или загорожено)'}
MOVING_LABELS = ('cat', 'dog', 'person', 'bird', 'horse', 'sheep', 'cow', 'pet')
CAMERA_HALF_FOV = .33       # rad: a remembered thing within this of the heading is right in view
DOOR_SHUT_TRUSTED_S = 3 * 3600.   # a door seen shut this recently is shut; older news may be stale
# 09.10 17:12-17:54 the hall door «closed» and «opened» again within 2-5 s six times — someone passing through the
# doorway looks like a shut door to the lidar. Shut counts once it has held this long; so does door news.
DOOR_STABLE_S = 8.   # 10.10: «closed» for 5-22 s a dozen times — someone in the doorway, not the door
# The last half hour survives a restart as context (09.10 audit: 93 restarts in 54 h, each a small amnesia).
RESTORE_S = 1800.
RESTORED_KINDS = ('action_done', 'action_failed', 'battery_low', 'bump', 'carried', 'cheered', 'conversation_ended',
                  'departure_ready', 'docked', 'undocked', 'dog_appeared', 'dog_left', 'dog_photo', 'door_closed',
                  'door_opened', 'dreamed', 'errand', 'falling_asleep', 'woke_up', 'found', 'game_played',
                  'idle_choice', 'idle_home', 'learned_face', 'looks_sad', 'smiled', 'waved', 'lost', 'motion_seen',
                  'object_gone', 'peekaboo', 'boop', 'touch', 'praise', 'scold', 'person_appeared', 'person_left',
                  'recognized',
                  'plan_due', 'quiet_note', 'recall_result', 'reflected', 'refused', 'repeat', 'sound', 'sound_note',
                  'startle', 'storage_low')


def ago(seconds):
    if seconds is None:
        return 'никогда'
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f'{seconds} с назад'
    if seconds < 7200:
        return f'{seconds // 60} мин назад'
    if seconds < 172800:
        return f'{seconds // 3600} ч назад'
    return f'{seconds // 86400} дн назад'


def reason_text(reason):
    reason = str(reason or '')
    return next((text for key, text in REASONS.items() if reason.startswith(key)), reason or 'причина неизвестна')


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class World:
    def __init__(self, rooms: dict, pet_name=None, archive=None, room_map=None, state=None):
        self.rooms = {r['name']: r for r in (rooms or {}).get('rooms', [])}
        self.room_map = room_map          # (Grid, room label raster) or None
        self.pet = pet_name
        # «вижу Сергея» again only this long apart (config person_debounce_s; 21.10, the video shoot: a greeting at
        # every take of «узнал лицо», 20 s)
        self.person_debounce = PERSON_DEBOUNCE_S
        self.archive = archive
        self.state = state                # MindState: where I was is remembered across restarts
        self.body: dict = {}
        self.voice: dict = {}
        self.presence: dict[str, dict] = {}
        self.dog = {'seen': -1e9, 'hits': deque(maxlen=6), 'announced': -1e9}
        self.events: deque = deque(maxlen=40)
        self.room = None
        self.room_since = time.time()
        saved = (state.get('room_visits') if state is not None else None) or {}
        self.room_visits: dict[str, float] = {k: float(v) for k, v in saved.items() if k in self.rooms}
        light = (state.get('room_light') if state is not None else None) or {}
        self.room_light: dict[str, tuple] = {k: tuple(v) for k, v in light.items() if k in self.rooms}
        if not self.room_visits and archive is not None:
            self.room_visits = self._visits_from_archive()
        self.visits_saved_at = 0.
        self.light_saved_at = 0.
        self.eyes_t = -1e9
        self.items: list = []
        self.last_saved_object: dict = {}
        self.volts: deque = deque(maxlen=600)
        self.charge_log: deque = deque(maxlen=60)    # (t, V) every 30 s on the charger: how fast it fills
        self.lost_pending = None          # (time, pose) of a loss not yet reported
        self.lost_reported = False
        self.dock_reported = None         # the dock state the mind was last told
        self.dock_changed_at = 0.
        self.near_objects = []
        self.near_at = 0.
        self.misses: dict[int, tuple] = {}
        self.doors: list = []             # [{rooms, state, seen}] from the body (lidar)
        self.door_since: dict = {}        # rooms -> (state, since when) as the body reports it
        self.door_pending: dict = {}      # rooms -> (state, since): a change not yet told to the mind
        self.door_announced: dict = {}    # rooms -> the state the mind was last told
        self.heard: dict[str, tuple] = {}  # sound kind -> (time, words): what the ears made out lately
        self.heard_where: dict[str, str] = {}   # sound kind -> «справа, со стороны «Коридор»» (C52)
        from .hearing import SoundMap
        self.sounds = SoundMap(self)       # the rays of the last minutes on the map: where sounds come from
        self.listeners: list = []          # callables(kind, data) told about every event (the mood)
        self.strikes: dict[int, list] = {}
        if archive is not None:
            self.events.extend(self._restore_events())

    def _restore_events(self):
        """What happened in the last half hour, back after a restart: context for the thoughts, not news — no
        reaction and no thought is woken by it (`recent` skips it)."""
        try:
            rows = self.archive.events(time.time() - RESTORE_S, RESTORED_KINDS, limit=30)
        except Exception:
            return []
        return [{**data, 't': t, 'kind': kind, 'restored': True}
                for t, kind, data in reversed(rows) if isinstance(data, dict) and data.get('text')]

    def _visits_from_archive(self):
        """First start with this memory: where was I lately, from my own past trips (not «давно» everywhere)."""
        visits = {}
        try:
            steps = self.archive.events(time.time() - 3 * 86400, ('mind_step',), limit=400)
        except Exception:
            return visits
        for t, _, step in steps:
            effect = str(step.get('effect') or '')
            args = ((step.get('do') or {}).get('args') or {})
            for name in self.rooms:
                if f'«{name}»' in effect or args.get('room') == name or args.get('to') == name:
                    visits[name] = max(visits.get(name, 0.), t)
        return visits

    # ------------------------------------------------------------- events
    def event(self, kind, text, **data):
        self.events.append({'t': time.time(), 'kind': kind, 'text': text, **data})
        if self.archive is not None:
            self.archive.event(kind, {'text': text, **data})
        for listener in self.listeners:
            try:
                listener(kind, data)
            except Exception:
                pass

    def recent(self, since):
        return [e for e in self.events if e['t'] > since and not e.get('restored')]

    # ------------------------------------------------------------- inputs
    def on_body(self, state):
        previous = self.body
        self.body = state
        now = time.time()
        if state.get('localized') is False and state.get('pose'):
            self.lost_track = list(state['pose'][:2])        # where the wheels say I am while lost
        voltage = state.get('voltage')
        if voltage:
            self.volts.append((now, float(voltage)))
            if state.get('docked') and (not self.charge_log or now - self.charge_log[-1][0] >= 30.):
                self.charge_log.append((now, float(voltage)))
            elif not state.get('docked'):
                self.charge_log.clear()
        room = state.get('room')
        if room and room != self.room:
            if self.room:
                self._visit(self.room, now, force=True)
            self.room, self.room_since = room, now
        if self.room and state.get('localized') and now - self.visits_saved_at > VISIT_SAVE_S:
            self._visit(self.room, now)
        if state.get('doors') is not None:
            for door in state['doors']:
                key = tuple(door.get('rooms') or ())
                if key not in self.door_since:
                    self.door_since[key] = (door.get('state'), 0.)      # known before: nothing to wait out
                elif self.door_since[key][0] != door.get('state'):
                    self.door_since[key] = (door.get('state'), now)
            self.doors = state['doors']
        docked = state.get('docked')
        if docked is not None and docked != previous.get('docked'):
            self.dock_changed_at = now
        if self.dock_reported is None and docked is not None:
            self.dock_reported = bool(docked)

    def _visit(self, room, now, force=False):
        self.room_visits[room] = now
        if self.state is not None and (force or now - self.visits_saved_at > VISIT_SAVE_S):
            self.visits_saved_at = now
            self.state.set('room_visits', dict(self.room_visits))

    @property
    def voltage(self):
        """The pack's level without the sag of driving: the highest reading of the last 45 s."""
        now = time.time()
        recent = [v for t, v in self.volts if now - t <= VOLTAGE_WINDOW_S]
        return max(recent) if recent else self.body.get('voltage')

    def charge_eta_text(self, departure=16.3):
        """«выеду примерно через 12 мин» from the last 10 minutes of charging, or ''."""
        log = [(t, v) for t, v in self.charge_log if self.charge_log[-1][0] - t <= 600]
        if len(log) < 4:
            return ''
        (t0, v0), (t1, v1) = log[0], log[-1]
        if v1 >= departure:
            return 'заряда уже хватает'
        if v1 - v0 < .02:
            return 'набирается медленно'
        minutes = (departure - v1) / ((v1 - v0) / (t1 - t0)) / 60
        return f'выеду примерно через {max(1, round(minutes))} мин'

    def on_body_event(self, event):
        kind = event.get('kind')
        now = time.time()
        if (kind == 'skill' and event.get('status') == 'done' and event.get('skill') == 'approach'
                and event.get('origin') == 'mind'):
            pass   # life looks at the object first and reports what was actually seen
        elif kind == 'skill' and event.get('status') in ('done', 'failed'):
            if event.get('origin') in ('reflex', 'game'):
                return                    # a wag of joy or a game's own move is not news to think about
            if event.get('skill') == 'relocalize' and event.get('origin') == 'system' and event['status'] == 'done' \
                    and not self.lost_reported:
                return                    # a search for a loss the mind never heard of: no news either
            text = f"{event.get('skill')}: {'получилось' if event['status'] == 'done' else 'не вышло — ' + reason_text(event.get('reason'))}"
            if event['status'] == 'done' and event.get('reason') == 'lost_sight':
                text = f"{event.get('skill')}: {reason_text('lost_sight')}"      # not «получилось» when it slipped away
            if event['status'] == 'failed' and self.visible() and str(event.get('reason', '')).startswith(('blocked', 'no_route')):
                text += ' (рядом человек — можно попросить подвинуться или открыть дверь)'
            self.event('action_done' if event['status'] == 'done' else 'action_failed', text,
                       id=event.get('id'), skill=event.get('skill'), reason=event.get('reason'), origin=event.get('origin'))
        elif kind == 'lost':
            if self.lost_pending is None:
                self.lost_pending = (now, list((self.body.get('pose') or [0, 0])[:2]))
        elif kind == 'found':
            pose = (event.get('pose') or self.body.get('pose') or [0, 0])[:2]
            pending, self.lost_pending = self.lost_pending, None
            reported, self.lost_reported = self.lost_reported, False
            # Compared with where the wheels say I am now, not where I was lost: driving on while lost is not being
            # carried (10.10 17:47: «меня перенесли на руках» after driving home lost).
            before = (getattr(self, 'lost_track', None) or pending[1]) if pending else None
            self.lost_track = None
            if before and math.dist(pose, before) > 1.:
                where = 'на базу' if self.body.get('docked') else f'в «{self.body.get("room") or "другое место"}»'
                self.event('carried', f'меня перенесли на руках {where} — я теперь здесь, а не там, где был')
            elif reported:
                self.event('found', 'снова понимаю, где я')
        elif kind == 'handled':
            # Hands turned or pushed me while I stood (body/feel.py; GrowBot felt being rocked and picked up).
            from ..body.feel import feeling_text
            self.event('handled', feeling_text(event), who=(self.visible() or self.available() or [None])[0])
        elif kind == 'bumper':
            still = not self.body.get('moving') and not self.body.get('skill')
            if still and self.visible():
                # Standing still and someone is here: they tapped my nose (Cozmo's fist bump, Eilik's tap).
                self.event('boop', 'меня легонько ткнули в нос — играют со мной', who=(self.visible() or [None])[0])
            else:
                self.event('bump', 'упёрся во что-то бампером')
        elif kind == 'door':
            # Told once the change has held DOOR_STABLE_S (tick): a flicker there and back is no news.
            self.door_pending[tuple((event.get('rooms') or ['?', '?'])[:2])] = (event.get('state'), now)

    def on_sound(self, sound):
        """A sound the ears named (voice/sounds.py): news if it is worth a reaction, else just context."""
        now = time.time()
        kind, words = sound.get('sound') or sound.get('kind', ''), sound.get('words', '')
        self.heard[kind] = (now, words)
        # Where it came from, when the body's ears placed it surely enough (17.10, voice/array.py): «слышу слева: лай»;
        # out through a doorway on the map, the room behind it (C52): «слышу справа, со стороны «Коридор»: лай».
        where = f' {sound["side"]}' if sound.get('side') and (sound.get('bearing_conf') or 0.) >= .3 else ''
        extra = {'bearing': sound['bearing']} if where else {}
        ray = self.sounds.add(kind, words, sound.get('bearing'), sound.get('bearing_conf')) if where else None
        room = self.sounds.words(ray)
        if room:
            where += f', {room}'
            extra['toward'] = ray['toward']
        self.heard_where[kind] = where.strip()
        if kind == 'startle':
            self.event('startle', f'вздрогнул: резкий громкий звук{where}', sound=kind, **extra)
        elif sound.get('urgent'):
            self.event('sound', f'слышу{where}: {words}', sound=kind, **extra)
        else:
            self.event('sound_note', f'слышу{where}: {words}', sound=kind, **extra)

    def on_things(self, data):
        """Small things the eyes named, with what they lie on: remembered for «где ключи?»."""
        if self.archive is None:
            return
        now = time.time()
        for item in data.get('items', []):
            xy = item.get('map_xy')
            if not xy:
                continue
            key = ('thing', item['label'], round(xy[0] * 2), round(xy[1] * 2))
            if now - self.last_saved_object.get(key, 0) < 60:
                continue
            self.last_saved_object[key] = now
            self.archive.saw(item['label'], xy, self.room_at(xy) or self.room, now, place=item.get('place'))

    def hearing_text(self, now=None, within=180.):
        now = time.time() if now is None else now
        recent = sorted(((t, words, self.heard_where.get(kind, '')) for kind, (t, words) in self.heard.items()
                         if kind != 'startle' and now - t < within), reverse=True)
        return ', '.join(f'{words} ({where + ", " if where else ""}{ago(now - t)})' for t, words, where in recent)

    def on_eyes(self, seen):
        now = time.time()
        self.eyes_t = now
        self.light = (seen.get('stats') or {}).get('light', getattr(self, 'light', 128))
        if self.room and now - self.room_light.get(self.room, (0, 0))[0] > 30:
            self.room_light[self.room] = (now, self.light)
            if self.state is not None and now - self.light_saved_at > 120:
                self.light_saved_at = now
                self.state.set('room_light', {k: list(v) for k, v in self.room_light.items()})
        self.items = seen.get('items', [])
        people = [i for i in self.items if i['label'] == 'person' and i.get('score', 0) >= .5]
        named_now = {i['name'] for i in people if i.get('name')}
        for item in people:
            who = item.get('name') or self.carried_name(item, named_now, now) or 'человек'
            corrected = None
            if who != 'человек':
                corrected = self.recognized_late(who, item, len(people), now)
            entry = self.presence.setdefault(who, {'seen': -1e9, 'face': -1e9, 'announced': -1e9, 'since': now,
                                                   'hits': deque(maxlen=8)})
            if now - entry['seen'] > GONE_AFTER_S:
                entry['since'] = now
                entry.pop('real_at', None)
            if corrected:
                entry['since'] = min(entry['since'], corrected['since'])
            entry['seen'] = now
            entry['track'] = item.get('track')
            entry['hits'].append(now)
            entry['xy'] = item.get('map_xy')
            entry['bearing'] = item.get('bearing')
            if item.get('face') and item.get('facing', True):
                entry['face'] = now
            # A single flicker of the detector is not someone arriving (219 «ушёл из виду» a day before).
            real = sum(1 for t in entry['hits'] if now - t <= PRESENT_WINDOW_S) >= PRESENT_HITS or item.get('name')
            if real:
                entry['present'] = True
                entry.setdefault('real_at', now)
            if who == 'человек' and real and now - entry['real_at'] < UNNAMED_GRACE_S:
                continue                            # a moment for the face to be named before «вижу человека»
            if corrected and corrected.get('told') and now - entry['announced'] <= self.person_debounce:
                # Already greeted by name a while ago: only the correction is news.
                self.event('recognized', f'тот «человек», которого я видел {ago(now - corrected["since"])}, — это '
                           f'{who}: узнал по лицу', who=who)
            if real and now - entry['announced'] > self.person_debounce:
                entry['announced'] = now
                entry['told'] = True               # «ушёл» is said only after an «вижу» (10.10: 81 left, 28 came)
                if who == 'человек':
                    text = 'вижу человека' + (', лицо не узнаю' if item.get('face_unknown') else '')
                    self.event('person_appeared', text, who=who, unknown_face=bool(item.get('face_unknown')))
                else:
                    # A meeting has a history: back after hours, or the first time today (a morning hello).
                    last = self.archive.last_seen(who) if self.archive is not None else None
                    since, news = '', {}
                    if last and now - last[0] > 3 * 3600:
                        midnight = now - (time.localtime(now).tm_hour * 3600 + time.localtime(now).tm_min * 60)
                        since = (' — сегодня вижу впервые' if last[0] < midnight
                                 else f' — не виделись {ago(now - last[0])[:-6]}')
                        try:
                            # Welcome home: what happened while they were away, for the mind to tell in a word.
                            news = {'news': self.archive.diary(max(last[0], now - 86400), now)[:400], 'since': since}
                        except Exception:
                            news = {}
                    if corrected and corrected.get('told'):
                        since += f' (тот «человек» {ago(now - corrected["since"])} — это и есть {who}, узнал по лицу)'
                    self.event('person_appeared', f'вижу: {who}{since}', who=who, **news)
                if self.archive is not None:
                    self.archive.sighting(who, self.room, now)
        dogs = [i for i in self.items if i['label'] == 'dog' and i.get('score', 0) >= .40]   # the eyes' own bar
        if dogs:
            self.dog['hits'].append(now)
            hits = [t for t in self.dog['hits'] if now - t < 3.]
            if len(hits) >= 2:
                if now - self.dog['seen'] > PERSON_DEBOUNCE_S:
                    self.dog['told'] = self.dog['present'] = True
                    self.event('dog_appeared', f'вижу собаку{" " + self.pet if self.pet else ""}')
                    if self.archive is not None:
                        self.archive.sighting('собака', self.room, now)
                self.dog['seen'] = now
                self.dog['xy'] = dogs[0].get('map_xy')
                self.dog['bearing'] = dogs[0].get('bearing')
        if self.archive is not None:
            for item in self.items:
                if item['label'] in OBJECT_LABELS and item.get('map_xy') and item.get('frames', 1) >= 3:
                    key = (item['label'], round(item['map_xy'][0] * 2), round(item['map_xy'][1] * 2))
                    if now - self.last_saved_object.get(key, 0) > 60:
                        self.last_saved_object[key] = now
                        # The room of the thing itself: seen through a doorway it is not in our room.
                        self.archive.saw(item['label'], item['map_xy'], self.room_at(item['map_xy']) or self.room, now)
            self._absence(now)

    def carried_name(self, item, named_now, now):
        """The name of someone just seen by face, for a person without a face in view now: the same track, or the
        only named person gone from view a moment ago, close to where they were (the robot turned and the track broke)."""
        track = item.get('track')
        recent = [(name, entry) for name, entry in self.presence.items()
                  if name != 'человек' and name not in named_now and now - entry['seen'] < NAME_CARRY_S]
        for name, entry in recent:
            if track is not None and entry.get('track') == track:
                return name
        xy = item.get('map_xy')
        if len(recent) == 1:
            name, entry = recent[0]
            was = entry.get('xy')
            if not (xy and was and math.dist(xy[:2], was[:2]) > NAME_CARRY_M):
                return name
        # The face covered or turned away for longer: the only known person seen right here in the last minutes.
        if not xy:
            return None
        here = [name for name, entry in self.presence.items()
                if name != 'человек' and name not in named_now and now - entry['seen'] < PLACE_CARRY_S
                and entry.get('xy') and math.dist(xy[:2], entry['xy'][:2]) <= PLACE_CARRY_M]
        return here[0] if len(here) == 1 else None

    def recognized_late(self, who, item, in_view, now):
        """A «человек» seen a moment ago turns out to be `who`: one being. Its record is folded into the name, the
        events and talk lines written as «человек» since then are rewritten (11.10: «ты что, меня не узнаёшь?»).
        Returns the stranger's entry if there was one."""
        stranger = self.presence.get('человек')
        if not stranger or now - stranger['seen'] > MERGE_S:
            return None
        same = item.get('track') is not None and stranger.get('track') == item.get('track')
        if not same and in_view > 1:
            return None                             # two people in view: the stranger may be someone else
        self.presence.pop('человек', None)
        since = stranger['since'] - 1.
        for event in self.events:
            if event['t'] >= since and event.get('who') == 'человек':
                event['who'] = who
                if event['kind'] == 'person_appeared':
                    event['text'] = f'вижу: {who} (сначала не узнал)'
                elif event['kind'] == 'person_left':
                    event['text'] = f'{who} ушёл из виду'
        if self.state is not None:
            dialogue = self.state.get('dialogue', [])
            if any(d.get('who') == 'человек' and d.get('t', 0) >= since for d in dialogue):
                self.state.set('dialogue', [dict(d, who=who) if d.get('who') == 'человек' and d.get('t', 0) >= since
                                            else d for d in dialogue])
        if self.archive is not None:
            self.archive.event('recognized_late', {'who': who, 'after_s': round(now - stranger['since'], 1),
                                                   'told': bool(stranger.get('told')), 'same_track': same})
        return stranger

    def _absence(self, now):
        """A remembered thing right in front of the camera, in good light, that the detector does not see.

        The owner (ЗАДУМКА): a thing taken away must not stay in memory for ever, a moved one gets its new
        place from new sightings. Every ~12 s of looking at the old place without seeing it costs the memory
        some trust; an object nobody sees there any more is forgotten (and told to the mind).
        """
        body = self.body
        pose = body.get('pose')
        twist = body.get('twist') or (0., 0.)
        if (not pose or not body.get('localized') or self.dark or abs(twist[1]) > .25
                or abs(twist[0]) > .2 or now - self.eyes_t > 1.):
            return
        if now - self.near_at > 10.:
            self.near_at = now
            self.near_objects = [o for o in self.archive.objects(min_count=1, limit=40, near=tuple(pose[:2]))
                                 if o['x'] is not None and math.dist((o['x'], o['y']), pose[:2]) < 3.]
        labels = {i['label'] for i in self.items}
        for item in self.near_objects:
            if item['label'] not in PORTABLE_LABELS:
                continue
            distance = math.dist((item['x'], item['y']), pose[:2])
            if not .9 < distance < 2.6:
                continue
            bearing = _wrap(math.atan2(item['y'] - pose[1], item['x'] - pose[0]) - pose[2])
            if abs(bearing) > CAMERA_HALF_FOV:
                continue
            if item['label'] in labels:
                self.misses.pop(item['id'], None)
                continue
            # Behind a wall, in the next room, the camera does not see it at all (08.10: the fridge and the
            # sofa through the bedroom wall were «not there»).
            if (self.room_at((item['x'], item['y'])) or item.get('room')) != self.room or \
                    not self.line_of_sight(pose[:2], (item['x'], item['y'])):
                continue
            count, at = self.misses.get(item['id'], (0, -1e9))
            if now - at < 1.:
                continue
            count += 1
            self.misses[item['id']] = (count, now)
            if count < ABSENT_LOOKS:
                continue
            self.misses.pop(item['id'], None)
            strikes = [t for t in self.strikes.get(item['id'], []) if now - t < 6 * 3600]
            if not strikes or now - strikes[-1] >= ABSENT_STRIKES_APART_S:
                strikes.append(now)
            self.strikes[item['id']] = strikes
            if len(strikes) < 2:
                continue                       # seen missing once: could be hidden by someone; look again later
            self.strikes.pop(item['id'], None)
            gone = self.archive.fade(item['id'], amount=1 << 30)     # missing twice: taken away (hidden)
            if gone is not None:
                self.near_at = 0.
                self.event('object_gone', f'смотрю на место, где был {RU.get(gone["label"], gone["label"])} '
                           f'[{gone["id"]}], — его там нет уже не первый раз: наверное, убрали', id=gone['id'])

    def line_of_sight(self, a, b):
        """No structural wall between two map points (the last 20 cm at the thing may touch a wall)."""
        if self.room_map is None:
            return True
        grid = self.room_map[0]
        length = math.dist(a, b)
        steps = max(2, int(length / (grid.resolution * .5)))
        for k in range(1, steps):
            f = k / steps
            if length * (1 - f) < .2:
                break
            cell = grid.cell((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
            if not grid.inside(cell) or grid.structural[cell]:
                return False
        return True

    def tick(self):
        """Departures, and the news that needs a moment to be sure (lost, docked)."""
        now = time.time()
        for who, entry in self.presence.items():
            if entry.get('present') and now - entry['seen'] > GONE_AFTER_S:
                entry['present'] = False
                if entry.pop('told', False):
                    self.event('person_left', f'{who} ушёл из виду', who=who)
        # The dog leaves too: without this the mind kept believing «Таффа рядом» long after he went.
        if self.dog.get('present') and now - self.dog['seen'] > GONE_AFTER_S:
            self.dog['present'] = False
            if self.dog.pop('told', False):
                # Only after an «вижу собаку» (10.10: 78 «ушёл из виду», 13 «вижу» — each a thought «мелькнул и пропал»).
                self.event('dog_left', f'{self.pet or "собака"} ушёл из виду')
        elif now - self.dog['seen'] <= GONE_AFTER_S:
            self.dog['present'] = True
        if self.lost_pending and not self.lost_reported and now - self.lost_pending[0] >= LOST_CONFIRM_S:
            if self.body.get('localized'):
                self.lost_pending = None          # healed by itself (e.g. settled on the dock): nothing to tell
            else:
                self.lost_reported = True
                self.lost_at = self.lost_pending[1]
                self.event('lost', 'потерялся на карте — не понимаю, где я')
        for rooms, (state, since) in list(self.door_pending.items()):
            if now - since < DOOR_STABLE_S:
                continue
            del self.door_pending[rooms]
            if self.door_announced.get(rooms) == state:
                continue                          # it flickered and came back: nothing changed
            self.door_announced[rooms] = state
            a, b = rooms
            if state == 'open':
                self.event('door_opened', f'дверь между «{a}» и «{b}» открыли — туда теперь можно', rooms=[a, b])
                if self.state is not None:
                    # «No way there» was about this door: it is open now (17:41 the mind was refused the hall
                    # five times over 4 minutes after the door had opened again).
                    closed = self.state.get('closed_ways') or {}
                    kept = {k: t for k, t in closed.items() if k not in (a.casefold(), b.casefold())}
                    if kept != closed:
                        self.state.set('closed_ways', kept)
            else:
                self.event('door_closed', f'дверь между «{a}» и «{b}» закрыли', rooms=[a, b])
        docked = self.body.get('docked')
        if docked is not None and self.dock_reported is not None and bool(docked) != self.dock_reported \
                and now - self.dock_changed_at >= DOCK_CONFIRM_S:
            self.dock_reported = bool(docked)
            if docked:
                self.event('docked', 'встал на базу, заряжаюсь')
            else:
                self.event('undocked', 'съехал с базы')

    # ------------------------------------------------------------- queries
    def available(self):
        """Who can be spoken to right now (a face seen within 3 s)."""
        now = time.time()
        # Face recognized recently and the person still in view.
        return [who for who, e in self.presence.items()
                if now - e['face'] <= FACE_FRESH_S and now - e['seen'] <= SEEN_FRESH_S]

    def visible(self):
        now = time.time()
        return [who for who, e in self.presence.items() if now - e['seen'] <= SEEN_FRESH_S and e.get('present')]

    def where_last(self, who, within=120.):
        """Which way (rad from the bow, now) `who` was last seen or something last moved, or None: a search starts
        there (a person turns first where the voice or the rustle came from)."""
        now = time.time()
        pose = self.body.get('pose')
        if not pose:
            return None
        if who in ('собака', 'собаку'):
            seen, xy = self.dog.get('seen', -1e9), self.dog.get('xy')
        else:
            entries = [e for name, e in self.presence.items() if name == who or who == 'человек' or name == 'человек']
            entry = max(entries, key=lambda e: e.get('seen', -1e9), default={})
            seen, xy = entry.get('seen', -1e9), entry.get('xy')
        if xy and now - seen < within:
            return round(_wrap(math.atan2(xy[1] - pose[1], xy[0] - pose[0]) - pose[2]), 3)
        moved = getattr(self, 'last_motion', None)          # (time, map heading) of the last rustle
        if moved is not None and now - moved[0] < 15.:
            return round(_wrap(moved[1] - pose[2]), 3)
        return None

    def speaker_by_voice(self, within=10., cone=.45):
        """Which of the people in view the last words came from (C52): the one standing where the ears heard the voice
        (within `cone` rad of its ray), or None. With two of ours in view the first one seen was taken for the speaker;
        the voice's direction tells them apart."""
        ray = self.sounds.latest(('voice', 'wake'), within=within)
        if not ray:
            return None
        now = time.time()
        best = None
        for who, entry in list(self.presence.items()):
            xy = entry.get('xy')
            if who == 'человек' or not xy or not entry.get('present') or now - entry.get('seen', -1e9) > SEEN_FRESH_S:
                continue
            off = abs(_wrap(math.atan2(xy[1] - ray['y'], xy[0] - ray['x']) - ray['heading']))
            if off <= cone and (best is None or off < best[0]):
                best = (off, who)
        return best[1] if best else None

    def recent_person(self, within=120.):
        """The one of ours whose face was recognized last, within `within` s: who we are most likely talking
        to when no face is in view right now (08.10: Марина turned away and became «человек» mid-talk)."""
        now = time.time()
        named = [(e['face'], who) for who, e in self.presence.items()
                 if who != 'человек' and now - e.get('face', -1e9) <= within]
        return max(named)[1] if named else None

    @property
    def dark(self):
        return getattr(self, 'light', 128) < DARK_LIGHT

    def room_dark(self, name, now=None, within=5400.):
        """The room was dark when I was last there (lately): looking around there now is no use."""
        now = time.time() if now is None else now
        seen = self.room_light.get(name)
        return bool(seen and now - seen[0] < within and seen[1] < DARK_LIGHT)

    def dog_visible(self):
        return time.time() - self.dog['seen'] <= SEEN_FRESH_S

    @property
    def docked(self):
        return bool(self.body.get('docked'))

    @property
    def live(self):
        return bool(self.voice.get('live'))

    def things_in_view(self):
        names = {}
        for item in self.items:
            if item['label'] in ('person', 'dog'):
                continue
            ru = RU.get(item['label'], item['label'])
            names[ru] = names.get(ru, 0) + 1
        return ', '.join(f'{k}' + (f' ×{v}' if v > 1 else '') for k, v in names.items())

    def situation(self, config):
        body = self.body
        now = time.time()
        parts = []
        if not body:
            return 'Тело пока не отвечает.'
        if body.get('localized'):
            parts.append(f'Я в комнате «{self.room}».' if self.room else 'Я между комнатами.')
        else:
            parts.append('Я не понимаю, где нахожусь на карте.')
        voltage = self.voltage
        reserve = config['board_reserve_voltage']
        if body.get('docked'):
            ready = voltage is not None and voltage >= config['departure_voltage']
            eta = '' if ready else self.charge_eta_text(config['departure_voltage'])
            parts.append('Стою на базе, заряжаюсь' + ('; заряда достаточно, могу выезжать. Тело исправно: съехать с '
                                                      'базы можно (если куда-то не проехать — это закрытая дверь, '
                                                      'а не поломка).' if ready
                                                     else '; выезжать пока рано, батарея набирается.'))
            if voltage:
                # 10.10 15:45 «какой процент зарядки?» -> «точных данных о батарее у меня нет»: the pack's voltage is known.
                parts.append(f'Батарея: {voltage:.1f} В, выезжаю с {config["departure_voltage"]:.1f} В'
                             + (f' — {eta}.' if eta else '.'))
        else:
            # Said plainly: with only «вернётся на базу при 15 В» the mind thought «я на базе» standing on the floor
            # (08.10 12:53, twice).
            parts.append('Я НЕ на базе — ' + ('еду.' if body.get('moving') else 'стою на полу.'))
            if voltage:
                parts.append(('Заряда хватает (%.1f В) — тело само вернётся на базу при %.1f В.' % (voltage, reserve))
                             if voltage > reserve + .2 else 'Батарея почти села (%.1f В) — скоро на базу.' % voltage)
        skill = body.get('skill')
        if skill:
            why = {'energy': ' — села батарея', 'night': ' — ночь, пора спать', 'system': '',
                   'owner': ' — попросил человек'}.get(skill.get('origin'), '')
            if skill.get('id') == 'idle-home':
                why = ' — долго стоял без дела'
            if skill.get('kind') == 'dock':
                parts.append(f'Тело везёт меня на базу{why}.')
            else:
                parts.append(f'Тело занято: {skill.get("kind")} ({skill.get("stage")}){why}.')
        if body.get('blocked') in ('obstacle', 'front_ir', 'bumper'):
            parts.append('Впереди препятствие.')
        if self.live:
            parts.append('Сейчас идёт живой разговор голосом.')
        seen = [w for w in self.visible()]
        if seen:
            parts.append('Вижу: ' + ', '.join(seen) + '.')
        if self.dog_visible():
            parts.append(f'Вижу пса{" " + self.pet if self.pet else ""}.')
        heard = self.hearing_text(now)
        if heard:
            parts.append('Слышу: ' + heard + '.')
        if now - self.eyes_t < 10 and self.dark:
            parts.append('Здесь темно — камера почти ничего не видит: рассматривать вещи бесполезно.')
        things = self.things_in_view() if now - self.eyes_t < 5 else ''
        if things:
            parts.append('В кадре: ' + things + '.')
        elif now - self.eyes_t > 10:
            parts.append('Свежего кадра с камеры нет.')
        if self.pet:
            if self.dog_visible():
                gone = ''
            elif self.dog['seen'] > 0:
                gone = f' Сейчас его в кадре нет (последний раз видел {ago(now - self.dog["seen"])}).'
            else:
                gone = ' Сейчас его не вижу.'
            parts.append(f'Нашего пса зовут {self.pet}, это мальчик.' + gone)
        return ' '.join(parts)

    def rooms_facts(self):
        now = time.time()
        facts = []
        reach = self.reachable(now)
        for name, room in self.rooms.items():
            last = self.room_visits.get(name)
            here = name == self.room
            dark = ', было темно' if not here and self.room_dark(name, now) else ''
            shut = '' if here or name in reach else f' — сейчас не попасть: {reach.get("_why", {}).get(name, "закрыта дверь")}'
            facts.append(f'{name}{" (я здесь)" if here else ""}: был {ago(now - last) if last else "давно"}{dark}{shut}')
        return '; '.join(facts)

    def door_shut(self, door, now=None):
        now = time.time() if now is None else now
        since = self.door_since.get(tuple(door.get('rooms') or ()), ('shut', 0.))
        return (door.get('state') == 'shut' and now - float(door.get('seen') or 0) < DOOR_SHUT_TRUSTED_S
                and (since[0] != 'shut' or now - since[1] >= DOOR_STABLE_S))

    def reachable(self, now=None):
        """Rooms I can drive to from here: {name: True, '_why': {unreachable name: reason}}.

        A door seen shut lately blocks its way; an unseen or long-unseen door is assumed open."""
        now = time.time() if now is None else now
        start = self.room
        if not start or not self.doors:
            return {**{name: True for name in self.rooms}, '_why': {}}
        links = {}
        for door in self.doors:
            rooms = door.get('rooms') or []
            if len(rooms) == 2 and not self.door_shut(door, now):
                links.setdefault(rooms[0], set()).add(rooms[1])
                links.setdefault(rooms[1], set()).add(rooms[0])
        seen, todo = {start}, [start]
        while todo:
            for other in links.get(todo.pop(), ()):
                if other not in seen:
                    seen.add(other)
                    todo.append(other)
        why = {}
        shut = [d for d in self.doors if self.door_shut(d, now)]
        for name in self.rooms:
            if name not in seen:
                mine = [d for d in shut if start in d['rooms']]
                door = mine[0] if mine else (shut[0] if shut else None)
                why[name] = (f'закрыта дверь между «{door["rooms"][0]}» и «{door["rooms"][1]}»' if door
                             else 'нет пути')
        return {**{name: True for name in seen}, '_why': why}

    def doors_text(self, now=None):
        now = time.time() if now is None else now
        parts = []
        for door in self.doors:
            rooms = door.get('rooms') or ['?', '?']
            state = door.get('state')
            if state is None:
                parts.append(f'{rooms[0]} — {rooms[1]}: не знаю, не видел')
                continue
            word = 'открыта' if state == 'open' else 'закрыта'
            parts.append(f'{rooms[0]} — {rooms[1]}: {word} (видел {ago(now - float(door.get("seen") or 0))})')
        return '; '.join(parts) or 'пока не знаю'

    def room_at(self, xy):
        """Room containing a map point (the same room raster the body uses), or None."""
        if self.room_map is None:
            return None
        grid, labels = self.room_map
        row, col = grid.cell(xy)
        if not (0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]):
            return None
        label = int(labels[row, col])
        for name, room in self.rooms.items():
            if room.get('label') == label:
                return name
        return None

    def room_point(self, name):
        if not name:
            return None
        text = name.casefold()
        for room_name, room in self.rooms.items():
            if room_name.casefold() == text or text in room_name.casefold() or room_name.casefold().startswith(text[:4]):
                return room_name, room.get('visit') or room.get('centroid')
        return None
