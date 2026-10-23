"""System 1 between the mind and the body: what the body is doing, what for, and is it still worth it.

Checked ten times a second (life's loop), so the robot changes course in a fraction of a second when the world
changes under it — not when the model next thinks (2–3 s later). Owner 09.10: «он должен на ходу соображать».

From the research (reports/Братан органы чувств и реакции.md): a reaching hand corrects course within 100–150 ms
and stops when the cup is taken away; behaviour trees re-check their conditions on every tick and pre-empt what no
longer holds; Vector's reactions may interrupt its own activities but never the owner's command. Each rule here is
one such check of a running activity against the world; when it fires, the mind is told why (an event), so its next
thought starts from the new situation.

What the body checks itself, faster still (body/skills.py): a thing gone from where an approach goes, a person
or the dog moving while followed or sought, a turn that is blocked.
"""
from __future__ import annotations

import time

from .brain import accusative

DARK_HOLD_S = 3.          # this long in the dark while looking round: the camera sees nothing, stop
FOUND_HITS_S = .5         # the one searched for seen for this long (not a flicker) before changing course
WHEN_HOLD_S = .5          # a plan's «when I see …» needs them in view this long (not a flicker)
PLAN_AFTER_S = 3.         # a plan's «доехал / не вышло» is judged this long after its command ended


class Executive:
    def __init__(self, life):
        self.life = life
        self.fired = {}            # (rule, skill id) -> time: each rule acts once on a running activity
        self.dark_since = None
        self.found_since = None
        self.when_since = {}       # (plan id, rule) -> since when its «when I see» has held

    def tick(self, now):
        life = self.life
        world = life.world
        body = world.body or {}
        skill = body.get('skill') or {}
        if world.live or life.play.active or not life.config.get('autonomy'):
            self.dark_since = self.found_since = None
            return
        try:
            self.contingency(now, skill)
        except Exception as error:              # a check must never stop the loop
            life.archive.event('executive_error', {'rule': 'contingency', 'error': f'{type(error).__name__}:{error}'[:200]})
        if not skill:
            self.dark_since = self.found_since = None
            return
        for rule in (self.found_on_the_way, self.way_closed, self.blind):
            try:
                rule(now, skill)
            except Exception as error:          # a check must never stop the loop
                life.archive.event('executive_error', {'rule': rule.__name__, 'error': f'{type(error).__name__}:{error}'[:200]})

    def once(self, rule, skill):
        key = (rule, skill.get('id'))
        if key in self.fired:
            return False
        self.fired = {k: t for k, t in self.fired.items() if time.time() - t < 600}
        self.fired[key] = time.time()
        return True

    # ------------------------------------------------------------------ rules
    def contingency(self, now, skill):
        """The running command's plan («если» of the decision that started it): the first condition that holds is
        acted on at once — the body does not wait 2-3 s for the next thought (10.10: Taffa gone by then)."""
        mind = self.life.mind
        plan = getattr(mind, 'plan_if', None)
        if not plan:
            if self.when_since:
                self.when_since.clear()
            return
        running = skill.get('id') == plan['id']
        if not running:
            if skill and skill.get('origin') == 'owner':
                mind.plan_if = None              # the owner's errand took over
                return
            plan.setdefault('ended', now)
            if now - plan['ended'] > PLAN_AFTER_S:
                mind.plan_if = None              # the command is over and nothing of the plan happened
                return
        for index, rule in enumerate(plan['rules']):
            if self.holds(rule, plan, index, running, now):
                mind.plan_if = None
                mind.reacted_at = mind.body_order_at = time.time()
                if rule['when'] in ('dog', 'person', 'name'):
                    mind.seeking = None
                from .brain import REFLEX_TOOL_WORDS, WHEN_TEXT
                arg = f' {rule["arg"]}' if rule.get('arg') else ''
                self.life.world.event('reacted', f'на ходу: {WHEN_TEXT[rule["when"]]}{arg} — '
                                                 f'{REFLEX_TOOL_WORDS.get(rule["do"]["tool"], rule["do"]["tool"])}',
                                      when=rule['when'])
                self.life.loop.run_in_executor(self.life.pool, mind.do, rule['do'], False)
                return

    def holds(self, rule, plan, index, running, now):
        world = self.life.world
        when, arg = rule['when'], rule.get('arg')
        if when in ('dog', 'person', 'name'):
            seen = (world.dog_visible() if when == 'dog' else
                    arg in world.visible() if when == 'name' else bool(world.visible()))
            key = (plan['id'], index)
            if not (running and seen):
                self.when_since.pop(key, None)
                return False
            return now - self.when_since.setdefault(key, now) >= WHEN_HOLD_S
        if when == 'dark':
            return running and world.dark
        mine = [e for e in list(world.events) if e['t'] >= plan['t']]
        own = [e for e in mine if e.get('id') == plan['id']]
        if when == 'arrived':
            return any(e['kind'] == 'action_done' and e.get('reason') not in ('lost_sight',) for e in own)
        if when == 'failed':
            return any(e['kind'] == 'action_failed' for e in own)
        if when == 'lost_sight':
            return any(e['kind'] in ('action_done', 'action_failed') and e.get('reason') in ('lost_sight', 'not_found')
                       for e in own)
        if when == 'door_closed':
            return any(e['kind'] == 'door_closed' or (e['kind'] == 'refused' and 'закрыта дверь' in e.get('text', ''))
                       or (e['kind'] == 'action_failed' and str(e.get('reason', '')).startswith('no_route')) for e in mine
                       if e['kind'] != 'action_failed' or e.get('id') == plan['id'])
        if when == 'sound':
            from .brain import SOUND_FORMS
            forms = SOUND_FORMS.get(arg, (arg,)) if arg else ()
            return any(e['kind'] in ('sound', 'sound_note') and (not forms or any(f in e.get('text', '').casefold()
                                                                                  for f in forms)) for e in mine)
        return False

    def found_on_the_way(self, now, skill):
        """Going to another room to look for someone — and they are here, on the way: go to them now."""
        mind = self.life.mind
        run = getattr(mind, 'skill_run', None)
        seeking = getattr(mind, 'seeking', None)
        if run and str(run.get('name', '')).startswith('поиск: ') and skill.get('origin') in ('mind', 'owner'):
            who, intent = run['name'][len('поиск: '):], None
        elif seeking and seeking.get('id') == skill.get('id') and skill.get('origin') == 'mind':
            # A trip «поищу Таффу» (10.10 21:23: she came in view, the mind took 3 s and chose a photo, she was gone).
            who, intent, run = seeking['who'], seeking.get('intent'), None
        else:
            self.found_since = None
            return
        world = self.life.world
        pet = self.life.config.get('pets', {}).get('dog')
        dog = who.casefold() in ('собака', 'собаку') or (pet and who.casefold().startswith(pet.casefold()[:4]))
        seen = world.dog_visible() if dog else (who in world.visible() or (who == 'человек' and world.visible()))
        if not seen:
            self.found_since = None
            return
        self.found_since = self.found_since or now
        if now - self.found_since < FOUND_HITS_S or not self.once('found', skill):
            return
        owner = bool(run.get('owner')) if run else False
        if run:
            mind.drop_skill_run('увидел по дороге')
        mind.seeking = None
        target = 'собака' if dog else who
        name = pet if dog else who
        world.event('found_on_the_way', f'искал {accusative(name)} — увидел по дороге, подъезжаю', who=target)
        args = {'who': target, **({'intent': intent} if intent else {})}
        self.life.loop.run_in_executor(self.life.pool, mind.do, {'tool': 'approach', 'args': args}, owner)

    def way_closed(self, now, skill):
        """Driving on my own to a room whose door has just been seen shut: stop at once, say why."""
        if skill.get('kind') != 'go' or skill.get('origin') != 'mind':
            return
        world, mind = self.life.world, self.life.mind
        target = getattr(mind, 'last_go_target', None)
        room = world.room_point(target) if target else None
        if not room or room[0] == world.room:
            return
        why = world.reachable().get('_why', {}).get(room[0])
        if not why or not self.once('closed', skill):
            return
        self.life.publish('body.stop', {'reason': 'way_closed'})
        closed = self.life.state.get('closed_ways', {})
        closed[target] = time.time()
        self.life.state.set('closed_ways', closed)
        world.event('refused', f'ехал в «{room[0]}» — по дороге увидел: {why}; остановился, туда сейчас не попасть')

    def blind(self, now, skill):
        """Looking round in the dark: the camera sees nothing, so it is no use — stop and say so."""
        looking = skill.get('kind') in ('look_around', 'explore') or (
            skill.get('kind') == 'track' and (skill.get('args') or {}).get('mode') == 'search')
        if not looking or skill.get('origin') != 'mind' or not self.life.world.dark:
            self.dark_since = None
            return
        self.dark_since = self.dark_since or now
        if now - self.dark_since < DARK_HOLD_S or not self.once('dark', skill):
            return
        self.life.publish('body.stop', {'reason': 'dark'})
        self.life.world.event('refused', 'здесь темно — камера ничего не видит, осматриваться бесполезно; '
                                         'посмотрю, когда включат свет, или займусь другим')
