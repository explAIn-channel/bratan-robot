"""Motivation: needs that grow with time and are satisfied by doing things.

The owner (06.10): «у людей всегда какая-то мотивация есть — ничего не делать странный выбор».
Built on what worked in research and in real robots:
- Kismet (Breazeal, MIT): each need has a comfortable band; left alone it drifts to «lacking»
  (seek), overdone it goes to «too much» (turn to something else) — after a long talk the
  robot does not push for more talk;
- Psi / MicroPsi (Dörner, Bach): self-depleting needs (company, energy) plus cognitive ones —
  competence (recent failures make «something that surely works» attractive) and certainty
  (novelty, things not seen up close);
- D2A (ICLR 2025): a need's target comes from the character, the gap is the motive, and
  levels are given to the language model in words; candidate activities come with each need;
- Oudeyer: curiosity goes to what is new but doable (closed ways are skipped);
- Anki Vector (shipped code): stimulation from people, talk and touch decays with time — low
  stimulation means napping on the charger, high means exploring and engaging; moods decay
  towards a friendly default, Confident/Frustrated follows successes and failures, and
  activities have explicit cooldowns rather than being purely emergent.
Everything is computed from the archive and the world, so it survives restarts.
"""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime
from statistics import median
from zoneinfo import ZoneInfo

from .world import ago

CURIOUS_FULL_S = 40 * 60       # nothing new looked at for this long: the need is at its peak
SOCIAL_FULL_S = 90 * 60
DOG_FULL_S = 60 * 60
HOME_FULL_S = 3 * 3600
REST_FULL_S = 45 * 60
TALK_ENOUGH_LINES = 25         # this many spoken lines in the last hour: company is satisfied for now
STIM_HALF_LIFE_S = 15 * 60
HEARD_TOWARD_S = 600.          # a sound placed in another room says where someone is for this long (C52)
SOUND_WORD = {'speech': 'голос', 'voice': 'голос', 'wake': 'голос', 'laughter': 'смех', 'tv': 'телевизор',
              'music': 'музыка'}
# The sounds of the house over the days (C52, owner: «вечером в зале телевизор → там люди»). Only sounds that say
# something about life at home; a sneeze or a bang are heard too seldom and too often wrongly to make a habit of.
HABIT_SOUNDS = {'speech': 'голоса', 'music': 'музыка', 'tv': 'телевизор', 'dog': 'лай', 'laughter': 'смех',
                'kitchen': 'звуки кухни', 'water': 'шум воды', 'vacuum': 'пылесос', 'door': 'хлопают двери',
                'knock': 'стук', 'doorbell': 'звонок в дверь'}
PEOPLE_SOUNDS = ('speech', 'music', 'tv', 'laughter', 'kitchen', 'vacuum')
HABIT_DAYS = 14
HABIT_MIN_DAYS = 3             # heard on this many different days before it is «обычно»

WORDS = {   # need -> words for low / some / strong
    'любопытство': ('новое пока не тянет', 'хочется чего-то нового', 'очень хочется нового, засиделся'),
    'знать дом': ('дом знаю хорошо', 'давно не заглядывал в одну комнату', 'тянет проверить, что там в других комнатах'),
    'скучаю по своим': ('с людьми наобщался', 'хочется увидеть своих', 'соскучился по своим'),
    'скучаю по Таффе': ('Таффу видел недавно', 'интересно, где Таффа', 'соскучился по Таффе'),
    'уверенность': ('всё получается', 'было пару неудач', 'много неудач — хочется сделать что-то, что точно выйдет'),
    'отдых': ('сил хватает', 'немного устал', 'устал, тянет на базу'),
    'хочу внимания': ('пообщались', 'хочется, чтобы со мной пообщались', 'свои рядом, а со мной никто — очень хочется внимания'),
}


def _clip(x):
    return max(0., min(1., x))


def _word(name, level):
    # A need without its words must not break all the others (08.10: «хочу внимания» failed every thought
    # with people around).
    low, some, strong = WORDS.get(name, (name, name, name))
    return strong if level >= .7 else some if level >= .35 else low


PARTS = ((6, 12, 'утром'), (12, 18, 'днём'), (18, 23, 'вечером'))
ARRIVAL_GAP_S = 3 * 3600       # seen again after this long away: an arrival (home from work, woke up)


def part_of_day(hour):
    return next((name for start, end, name in PARTS if start <= hour < end), 'ночью')


class Drives:
    def __init__(self, archive, world, state, config):
        self.archive, self.world, self.state, self.config = archive, world, state, config
        self.tz = ZoneInfo(config.get('timezone', 'Europe/Moscow'))
        self._habits = (0., {})

    def habits(self, now):
        """Where and when the household is usually seen, counted from my own sightings in code (PreHeat-style
        statistics; the model only reads the result). {who: {'room', 'part', 'rooms', 'arrives', 'seen'}}"""
        at, cached = self._habits
        if now - at < 600:
            return cached
        result = {}
        for who in list(self.config.get('people', [])) + ['собака']:
            rows = self.archive.habits(who, days=14)
            if len(rows) < 3:
                continue
            hours = [(t, datetime.fromtimestamp(t, self.tz).hour, room) for t, room in rows]
            part = part_of_day(datetime.fromtimestamp(now, self.tz).hour)
            here_now = Counter(room for _, hour, room in hours if room and part_of_day(hour) == part)
            anywhere = Counter(room for _, _, room in hours if room)
            if not anywhere:
                continue
            arrivals, last, days = [], None, set()
            for t, _, _ in hours:
                if last is None or t - last > ARRIVAL_GAP_S:
                    local = datetime.fromtimestamp(t, self.tz)
                    if 7 <= local.hour < 23:
                        arrivals.append(local.hour * 60 + local.minute)
                        days.add(local.date())
                last = t
            arrives = None
            if len(arrivals) >= 3 and len(days) >= 3:      # a habit needs several days, not one busy afternoon
                minute = int(median(arrivals))
                arrives = f'{minute // 60:02d}:{minute % 60:02d}'
            result[who] = {'room': (here_now or anywhere).most_common(1)[0][0], 'part': part,
                           'rooms': [r for r, _ in anywhere.most_common(3)], 'arrives': arrives, 'seen': len(rows)}
        self._habits = (now, result)
        return result

    def habits_text(self, now):
        lines = []
        for who, h in self.habits(now).items():
            name = (self.world.pet or 'Таффа') if who == 'собака' else who
            text = f'{name}: {h["part"]} чаще в «{h["room"]}»'
            others = [r for r in h['rooms'] if r != h['room']]
            if others:
                text += ', бывает в ' + ', '.join(f'«{r}»' for r in others)
            if h['arrives']:
                text += f'; после долгого перерыва обычно появляется около {h["arrives"]}'
            lines.append(text + f' (замечал {h["seen"]} раз за 2 недели)')
        return '; '.join(lines) or 'пока мало наблюдений — замечаю, кто где бывает'

    def sound_habits(self, now):
        """When and from where each sound of the house is usually heard, from my own ears over two weeks (counted in
        code like the sightings). {kind: {'words', 'days', 'span', 'parts', 'room', 'usual_now'}}"""
        at, cached = getattr(self, '_sounds', (0., {}))
        if now - at < 600:
            return cached
        rows = self.archive.events(now - HABIT_DAYS * 86400, ('heard_sound',), limit=5000)
        by_kind = {}
        for t, _, data in rows:
            kind = data.get('sound')
            if kind in HABIT_SOUNDS:
                by_kind.setdefault(kind, []).append((t, data.get('toward')))
        part_now = part_of_day(datetime.fromtimestamp(now, self.tz).hour)
        result = {}
        for kind, heard in by_kind.items():
            local = [(datetime.fromtimestamp(t, self.tz), toward) for t, toward in heard]
            days = {d.date() for d, _ in local}
            if len(days) < HABIT_MIN_DAYS:
                continue
            span = max(1, (max(days) - min(days)).days + 1)
            # The parts of the day it belongs to: counted by days, not by sounds (one loud evening is not a habit).
            per_part = Counter()
            for part in {(d.date(), part_of_day(d.hour)) for d, _ in local}:
                per_part[part[1]] += 1
            parts = [p for p, n in per_part.most_common() if n >= max(2, .25 * len(days))]
            # Its room only from directions placed on two days at least (18.10: two rays of one afternoon made
            # «voices usually from the corridor» out of 1224 voices).
            rooms = Counter(toward for _, toward in local if toward)
            room = None
            if rooms:
                best, n = rooms.most_common(1)[0]
                on_days = {d.date() for d, toward in local if toward == best}
                if n >= 3 and len(on_days) >= 2 and n >= .5 * sum(rooms.values()):
                    room = best
            words = (self.world.pet or 'Таффа') + ' лает' if kind == 'dog' else HABIT_SOUNDS[kind]
            result[kind] = {'words': words, 'days': len(days), 'span': span, 'parts': parts, 'room': room,
                            'usual_now': part_now in parts}
        self._sounds = (now, result)
        return result

    def sound_habits_text(self, now):
        lines = []
        for kind, h in sorted(self.sound_habits(now).items(), key=lambda kv: -kv[1]['days']):
            share = h['days'] / h['span']
            often = 'почти каждый день' if share >= .7 else 'через день-два' if share >= .4 else 'иногда'
            text = f'{h["words"]} — {often}'
            if h['parts']:
                text += ', чаще ' + ' и '.join(h['parts'][:2])
            if h['room']:
                text += f', обычно со стороны «{h["room"]}»'
            lines.append(text)
        return '; '.join(lines) or 'пока мало наблюдений — прислушиваюсь, что и когда звучит дома'

    def heard_usually(self, now, kinds=PEOPLE_SOUNDS):
        """A sound of people that is usual for this part of the day and comes from another room: (room, words)."""
        here = self.world.room
        for kind, h in sorted(self.sound_habits(now).items(), key=lambda kv: -kv[1]['days']):
            if kind in kinds and h['usual_now'] and h['room'] and h['room'] != here:
                return h['room'], h['words']
        return None

    def bonds(self, now):
        """Attachment grows with attention (aibo, LOVOT: the ones who talk to me and are around are close)."""
        lines = []
        for who in self.config.get('people', []):
            owner = (self.config.get('people') or [None])[0]
            talks = self.archive.talks_with(who, now - 14 * 86400, default_who=owner)
            seen = len(self.archive.habits(who, days=14))
            if not talks and not seen:
                lines.append(f'{who} — пока мало знаю')
                continue
            kind = self.archive.kinds_by(who, now - 14 * 86400, ('praise', 'touch', 'game_played', 'waved'))
            warmth = kind.get('praise', 0) + kind.get('touch', 0) + 2 * kind.get('game_played', 0) + kind.get('waved', 0)
            score = talks + warmth
            closeness = 'очень близкий' if score >= 15 else 'близкий' if score >= 4 or seen >= 20 else 'знакомый'
            extra = f', хвалил {kind["praise"]}' if kind.get('praise') else ''
            extra += f', играли {kind["game_played"]}' if kind.get('game_played') else ''
            lines.append(f'{who} — {closeness} (разговоров за 2 недели {talks}, встреч {seen}{extra})')
        if self.world.pet:
            dog = len(self.archive.habits('собака', days=14))
            lines.append(f'{self.world.pet} — наш пёс, мой приятель (встреч за 2 недели {dog})')
        return '; '.join(lines)

    def heard_toward(self, kinds, now, within=HEARD_TOWARD_S):
        """The freshest sound of `kinds` the ears placed out through a doorway: (ray, «со стороны «X»») or None."""
        sounds = getattr(self.world, 'sounds', None)
        if sounds is None:
            return None
        for ray in reversed(sounds.snapshot()):
            if now - ray['t'] > within:
                return None
            if ray['kind'] in kinds:
                where = sounds.words(ray)
                if where:
                    return ray, where
        return None

    def _traits(self):
        return (self.state.get('character') or {}).get('traits') or {}

    def _last(self, kinds, since, match=None):
        for t, _, data in self.archive.events(since, kinds, limit=300):
            if match is None or match(data):
                return t
        return None

    def stimulation(self, now):
        """Vector's «stim»: what happened around me lately (people, dog, talk, touch), decaying."""
        weights = {'person_appeared': .5, 'dog_appeared': .4, 'conversation_start': 1., 'touch': .6,
                   'voice_wake': .3, 'action_done': .15}
        total = 0.
        for t, kind, _ in self.archive.events(now - 3 * STIM_HALF_LIFE_S, tuple(weights), limit=200):
            total += weights[kind] * .5 ** ((now - t) / STIM_HALF_LIFE_S)
        return _clip(total / 2.)

    def needs(self, now, quiet=False):
        """[(name, strength 0..1, words, why, how)], strongest first."""
        from .brain import RU
        w, since = self.world, now - 12 * 3600
        body = w.body or {}
        traits = self._traits()
        needs = []

        # Curiosity (certainty): new things up close, rooms explored, photos.
        new = self._last(('action_done',), since, lambda d: d.get('skill') in ('approach', 'explore'))
        photos = [p['t'] for p in self.state.get('album', []) if p.get('t')]
        last_new = max([t for t in (new, max(photos) if photos else None) if t] or [now - CURIOUS_FULL_S])
        level = _clip((now - last_new) / CURIOUS_FULL_S * (.5 + .5 * traits.get('curious', .7)) / .85)
        looked = self.archive.day(now - 6 * 3600)
        objects = self.archive.objects(near=tuple(body.get('pose', (0, 0))[:2]), limit=12)
        fresh = [o for o in objects if o['label'] and RU(o['label']) != o['label']      # named in Russian
                 and RU(o['label']) not in looked][:2]
        how = ('подъехать рассмотреть ' + ', '.join(f'[{o["id"]}] {RU(o["label"])}' for o in fresh)
               if fresh else 'осмотреть комнату или сфотографировать что-то для альбома')
        if getattr(w, 'dark', False):
            how = 'здесь темно — рассматривать бесполезно; поехать туда, где светло, или поискать своих'
        needs.append(('любопытство', level, f'новое рассматривал {ago(now - last_new)}', how))

        # Knowing the home: the room not visited for the longest time (closed ways are skipped).
        closed = {k for k, t in (self.state.get('closed_ways') or {}).items() if now - t < 600}
        reach = w.reachable(now) if hasattr(w, 'reachable') else {}
        closed |= {name.casefold() for name in w.rooms if reach and name not in reach}     # a door seen shut
        others = [name for name in w.rooms if name != w.room]
        confined = bool(others) and all(name.casefold() in closed for name in others)
        if confined:
            # Doors shut: the robot's world is this room — calm, but still alive in it (08.10: with only «rest»
            # left he sat on the dock all morning). Rest is one choice among the things to do here.
            needs.append(('отдых', .35, f'немного устал; двери в другие комнаты закрыты ({", ".join(others)})',
                          'отдохнуть на базе — или заняться чем-то здесь: рассмотреть вещи, станцевать под музыку, '
                          'поиграть с тем, кто рядом; дверь откроют — поеду смотреть'))
        # A room found dark lately counts as freshly seen: in the dark there is nothing to look at
        # (07.10 evening: «давно не был в зале» four times, the hall dark every time).
        visits = [(w.room_visits.get(name, 0) + (3 * 3600 if w.room_dark(name, now) else 0), name) for name in w.rooms
                  if name != w.room and name.casefold() not in closed]
        if visits:
            _, name = min(visits)
            last = w.room_visits.get(name, 0)
            age = now - last if last else HOME_FULL_S
            level = _clip(age / HOME_FULL_S * (.6 + .4 * traits.get('tidy', .7)))
            dark = w.room_dark(name, now)
            if dark:
                level *= .3
            needs.append(('знать дом', level, f'в «{name}» был {ago(age) if last else "давно"}'
                          + (', там было темно' if dark else ''), f'поехать в «{name}» и посмотреть, что там'))

        # Our people (Kismet's band: much talk lately satisfies it for a while; asleep at night).
        talk = self._last(('conversation_start',), since) or now - SOCIAL_FULL_S
        lines = len(self.archive.events(now - 3600, ('voice_spoke',), limit=300))
        social = 0. if quiet or lines > TALK_ENOUGH_LINES else _clip(
            (now - talk) / SOCIAL_FULL_S * (.6 + .6 * traits.get('talkative', .5)))
        habits = self.habits(now)
        voices = self.heard_toward(('speech', 'laughter', 'voice', 'wake', 'tv', 'music'), now)
        if w.available():
            how = f'заговорить с {w.available()[0]}'
        elif w.visible():
            how = 'подъехать к человеку, которого вижу'
        elif voices:
            # People are where the house sounds from (C52): voices, laughter, a TV beat yesterday's habits.
            ray, where = voices
            how = (f'{where} слышно «{ray["words"]}» ({ago(now - ray["t"])}) — там кто-то есть: пойти на звук '
                   f'(to_sound {{"sound": "{SOUND_WORD.get(ray["kind"], "голос")}"}})')
        else:
            usual = [(who, h['room']) for who, h in habits.items() if who != 'собака' and h['room'] != w.room
                     and h['room'].casefold() not in closed]
            sounds = self.heard_usually(now)
            if usual:
                how = f'поехать в «{usual[0][1]}» — {usual[0][0]} {habits[usual[0][0]]["part"]} обычно там'
            elif sounds and sounds[0].casefold() not in closed:
                # Nobody's sightings yet, but the ears know the evenings (C52): where the TV usually is, people are.
                how = (f'в это время обычно слышно «{sounds[1]}» со стороны «{sounds[0]}» — свои, наверное, там: '
                       f'поехать в «{sounds[0]}»')
            else:
                how = 'подождать своих здесь или заглянуть в другие комнаты'
        needs.append(('скучаю по своим', social, f'разговаривал {ago(now - talk)}', how))

        # Taffa.
        if not w.dog_visible():
            seen = self.archive.last_seen('собака')
            last_dog = seen[0] if seen else now - DOG_FULL_S
            where = f', последний раз в «{seen[1]}»' if seen and seen[1] else ''
            usual = (habits.get('собака') or {}).get('room') or (seen[1] if seen else None)
            # Spinning on the spot for a dog who is in another room failed 14 times a day: go where he usually is.
            how = (f'поехать в «{usual}» — он обычно там' if usual and usual != w.room and usual.casefold() not in closed
                   else 'подождать его здесь: он сам прибегает')
            barked = self.heard_toward(('dog',), now)
            if barked:
                # He barked and the ears know where from (C52): better than where he usually is.
                ray, where = barked
                how = f'{w.pet or "Таффа"} лаял {where} ({ago(now - ray["t"])}) — пойти на лай (to_sound {{"sound": "лай"}})'
            needs.append(('скучаю по Таффе', 0. if quiet else _clip((now - last_dog) / DOG_FULL_S),
                          f'видел его {ago(now - last_dog)}{where}', how))

        # Attention (Vector, LOVOT, EMO: a pet that is ignored comes and asks): people around, nobody with me.
        if w.visible() and not quiet and not w.docked:          # on the dock nobody is approached (owner 08.10)
            last = self._last(('conversation_start', 'praise', 'touch', 'game_played', 'waved', 'smiled'),
                              now - 6 * 3600) \
                or now - 3600
            level = _clip((now - last - 600) / 1800)
            if level > 0:
                needs.append(('хочу внимания', level, f'со мной не общались {ago(now - last)}, а свои рядом',
                              'подъехать к ним, издать звук (sound), предложить поиграть — прятки, загадки, '
                              '«тише едешь», показать трюк'))

        # Competence (Psi; Vector's Confident/Frustrated): recent failures make a sure thing attractive.
        recent = self.archive.events(now - 1800, ('action_done', 'action_failed', 'refused'), limit=60)
        # Only the body's failures count (a line held back as a repeat is not a failure).
        failed = sum(1 for _, kind, data in recent if kind == 'action_failed'
                     or (kind == 'refused' and str(data.get('text', '')).startswith('не получилось')))
        done = sum(1 for _, kind, _ in recent if kind == 'action_done')
        if failed:
            needs.append(('уверенность', _clip(failed / max(3., failed + done) * 1.5),
                          f'за полчаса неудач {failed}, удач {done}',
                          'вернуться на базу или туда, где светло' if getattr(w, 'dark', False) else
                          'сделать то, что точно получится: сфотографировать, осмотреться, подъехать к близкому'))

        # Rest: long off the dock, a tired battery, night, nothing going on (Vector's low stim -> nap).
        if not body.get('docked'):
            off = self._last(('undocked',), since) or now
            voltage = w.voltage or 16.
            needs.append(('отдых', max(_clip((now - off) / REST_FULL_S), _clip((16.2 - voltage) / 1.4),
                                       .9 if quiet else 0.),
                          f'съехал с базы {ago(now - off)}', 'поехать на базу отдохнуть'))

        needs.sort(key=lambda n: -n[1])
        return [(name, level, _word(name, level), why, how) for name, level, why, how in needs]

    def mood(self, needs, stim=None):
        """What I miss most right now, in words ('' if nothing). How I feel is the affect's one line (09.10 audit:
        two moods side by side — «спокойный (бодрость 0,2); бодрый» — contradicted each other)."""
        strong = [n for n in needs if n[1] >= .7]
        if strong and strong[0][0] == 'отдых':
            return 'устал'
        return strong[0][2] if strong else ''

    def goals(self, now):
        """Long goals with progress from memory: something to work towards every day."""
        local = time.localtime(now)
        midnight = now - (local.tm_hour * 3600 + local.tm_min * 60 + local.tm_sec)
        day = self.archive.day(midnight)
        rooms = sum(1 for name in self.world.rooms if name in day)
        photos_today = sum(1 for p in self.state.get('album', []) if p.get('t', 0) >= midnight)
        talks = len(self.archive.events(midnight, ('conversation_start',), limit=200))
        return [
            f'знать весь дом: сегодня побывал в {rooms} из {len(self.world.rooms)} комнат',
            f'собрать альбом нашего дома и семьи: снимков {len(self.state.get("album", []))}, сегодня {photos_today}',
            f'быть рядом со своими: сегодня разговоров {talks}',
        ]

    def text(self, now, quiet=False):
        """(needs, goals, mood) as prompt lines."""
        needs = self.needs(now, quiet)
        lines = [f'{name}: {words} ({why}) → {how}' for name, level, words, why, how in needs if level >= .2]
        return ('; '.join(lines) or 'всё утолено — можно отдохнуть на базе или заняться чем угодно',
                '; '.join(self.goals(now)), self.mood(needs))
