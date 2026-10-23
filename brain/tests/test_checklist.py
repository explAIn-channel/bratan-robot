"""The checklist (owner 10.10: «составь чек-лист на тысячу-полторы и прогони»): every action with ordinary, odd and
absurd arguments, asked by the owner or thought up by the mind itself, in nine circumstances — no crash, a human
refusal, a sane body command, and the house rules (night, lost, an empty pack, «побудь там», a talk) kept.

python -m pytest tests/test_checklist.py -q
"""
import json
import math

import pytest

from bratan.body import skills
from bratan.mind import brain as brainmod
from tests.test_mind import FakeLLM, clock, world  # noqa: F401  (fixtures)
from bratan.common.config import DEFAULTS

ROOMS = ['Зал', 'Кухня', 'Коридор', 'Спальня и кабинет']
VARIANTS = {
    'go': [{'to': r} for r in ROOMS] + [{'to': 'зал'}, {'to': 'кухню'}, {'to': 'база'}, {'to': 'домой'}, {'to': 'вперёд'},
                                         {'to': 'назад'}, {'to': 'Марс'}, {'to': 'магазин'}, {'to': ''}, {'to': None},
                                         {'to': 123}, {'room': 'Зал'}, {'place': 'кухня'}, {'to': ['Зал']}, {}],
    'approach': [{'id': 'THING'}, {'id': 99999}, {'id': '12'}, {'id': -1}, {'who': 'собака'}, {'who': 'Таффа'},
                 {'who': 'меня'}, {'who': 'Сергей'}, {'who': 'Вася'}, {'who': 'человек'}, {'target': 'кровать'},
                 {'who': 'собака', 'intent': 'сфотографировать'}, {'who': 'человек', 'intent': 'сказать: привет'}, {},
                 {'who': None}, {'who': 5}, {'id': None}],
    'look_around': [{}, {'angle': 9}, {'x': 'y'}],
    'explore_room': [{}, {'room': 'Кухня'}, {'room': 'Луна'}, {'room': None}, {'room': 7}],
    'follow': [{'who': 'собака'}, {'who': 'человек', 'seconds': 10}, {'who': 'Сергей', 'seconds': -5},
               {'who': 'собака', 'seconds': 1e9}, {'who': 'собака', 'seconds': 'долго'}, {}],
    'watch': [{'who': 'собака', 'seconds': 20}, {'who': 'человек'}, {'who': 'Марина', 'seconds': 0}, {}],
    'search': [{'who': 'собака'}, {'who': 'Таффа'}, {'who': 'Сергей'}, {'who': 'человек'}, {'who': 'ключи'}, {}],
    'to_sound': [{}, {'sound': 'лай'}, {'sound': 'голос'}, {'sound': 'музыка'}, {'sound': 'стук'}, {'sound': 'нечто'},
                 {'sound': None}, {'sound': 5}, {'what': 'звон'}],
    'photo': [{}, {'caption': 'Таффа'}, {'caption': 'x' * 500}, {'caption': None}],
    'show_photo': [{}, {'id': 'p1'}, {'id': 'нет такого'}, {'id': None}],
    'rest': [{}, {'now': True}],
    'stop': [{}],
    'dance': [{}, {'seconds': 10}, {'seconds': -1}, {'seconds': 1e6}, {'seconds': 'долго'}],
    'play': [{'game': g} for g in ('прятки', 'тише едешь', 'фото', 'дай пять', 'альбом', 'шахматы', '', None)],
    'emote': [{'kind': k} for k in ('wiggle', 'bounce', 'tremble', 'shiver', 'nod', 'shake', 'spin', 'bow', 'fly', '', None)],
    'wait': [{'seconds': s} for s in (0, -10, 5, 30, 600, 1e9, 'abc', None)],
    'move': [{'meters': m} for m in (.5, -.5, 1, -1, 100, -100, 0, 'abc', None)] + [{'distance': .3}, {}],
    'alarm': [{'at': '08:00'}, {'at': '25:99'}, {'at': ''}, {'at': None}, {}],
    'timer': [{'minutes': 5, 'label': 'чай'}, {'minutes': -1}, {'minutes': 'x'}, {'minutes': 1e9}, {}],
    'hush': [{'minutes': 10}, {'minutes': 0}, {'minutes': -1}, {'minutes': 'x'}, {'minutes': 1e9}, {}],
    'learn_face': [{'name': 'Петя'}, {'name': ''}, {}],
    'plan': [{'what': 'объехать зал', 'when': 'вечером'}, {'what': 'спеть', 'in_minutes': 30}, {'what': ''}, {}],
    'want': [{'text': 'поиграть с Таффой'}, {'text': ''}, {}],
    'note': [{'text': 'на кухне скользко'}, {}],
    'recall': [{'about': 'кровать'}, {'about': ''}, {}],
    'name_thing': [{'id': 'THING', 'name': 'стул Сергея'}, {'id': 99999, 'name': 'x'}, {}],
    'remember_place': [{'name': 'у окна'}, {}],
    'new_skill': [{'name': 'обход', 'steps': [{'tool': 'look_around'}]}, {'name': 'x', 'steps': 'не список'},
                  {'name': '', 'steps': []}, {}],
    'use_skill': [{'name': 'нет такого'}, {}],
    'drop_plan': [{'index': 0}, {'index': 99}, {'index': -1}, {}],
    'drop_want': [{'index': 0}, {'index': 'x'}, {}],
    'self': [{'text': 'я спокойный робот'}, {}],
    'tell': [{'who': 'Сергей', 'text': 'Таффа на кухне'}, {}],
    'forget_skill': [{'name': 'обход'}, {}],
    # not tools at all: what a model might reach for
    'dock': [{}], 'charge': [{}], 'forward': [{'meters': .3}], 'turn': [{}], 'fly': [{}], 'jump': [{}],
    'eat': [{}], 'open_door': [{}], 'cook': [{'what': 'кофе'}], 'call': [{'who': 'полиция'}], 'GO': [{'to': 'Зал'}],
    'perform': [{'steps': [{'emote': 'bow', 'say': 'Встречайте!'}, {'turn': 360, 'sound': 'excited'}]},
                {'steps': [{'move': 5}, {'move': -5}, {'move': 5}, {'move': 5}]}, {'steps': [{'say': 'просто слова'}]},
                {'steps': [{'turn': 'много'}, {'emote': 'мяу'}, {'face': 'злой'}]}, {'steps': 'станцуй'}, {'steps': []},
                {'steps': [{}] * 30}, {'steps': [{'pause': 99}]}, {'steps': [{'turn': 1e9, 'say': 'а' * 500}]}, {},
                {'steps': {'emote': 'spin'}}, {'steps': [None, 3, 'x', {'move': None}]}],
    'show': [{'steps': [{'emote': 'nod', 'say': 'Да!'}]}],
    '': [{}],
}
CIRCUMSTANCES = ('normal', 'docked', 'docked_empty', 'lost', 'dark', 'staying', 'quiet', 'talk', 'hushed')
MOVING = {'go', 'go_room', 'approach', 'explore', 'track', 'drive', 'dance', 'emote', 'look_around', 'rotate'}
KNOWN = set(brainmod.BODY_TOOLS) | set(brainmod.MIND_TOOLS) | set(brainmod.TOOL_ALIASES)
CASES = [(tool, i, who, c) for tool, variants in VARIANTS.items() for i in range(len(variants))
         for who in ('owner', 'mind') for c in CIRCUMSTANCES]


def make(world, clock, circumstance):
    acts = []
    flags = {'quiet': circumstance == 'quiet', 'may_leave_dock': circumstance != 'docked_empty'}

    def act(kind, **data):
        acts.append((kind, data))
        if kind in flags:
            return flags[kind]
        if kind == 'playing':
            return False
        if kind == 'body_do':
            return f'm{len(acts)}'
        if kind == 'show_photo':
            return data.get('id') in ('p1', 'p2')
        return True
    config = dict(DEFAULTS, people=['Сергей', 'Марина'])
    mind = brainmod.Mind(world.state, world, world.archive, FakeLLM(''), config, act)
    world.state.set('album', [{'id': 'p1', 't': clock.now, 'caption': 'Таффа'}])
    body = {**world.body}
    if circumstance in ('docked', 'docked_empty'):
        body.update(docked=True, voltage=16.4 if circumstance == 'docked' else 15.1, pose=[-.09, -.05, .2])
    if circumstance == 'lost':
        body.update(localized=False)
    world.on_body(body)
    if circumstance == 'docked_empty':
        clock(50)                                   # the honest level is the last 45 s: let the empty pack settle
        world.on_body(body)
    if circumstance == 'dark':
        for _ in range(3):
            world.on_eyes({'items': [], 'stats': {'light': 3}})
    if circumstance == 'staying':
        world.state.set('stay', {'until': clock.now + 600, 'room': 'Спальня и кабинет'})
    if circumstance == 'talk':
        world.voice = {'live': True}
    if circumstance == 'hushed':
        world.state.set('hush_until', clock.now + 600)
    if circumstance != 'lost':
        # What the ears placed lately (C52): a bark, music, a knock — for «to_sound» to go to.
        for kind, words, bearing in (('dog', 'лает собака', .4), ('music', 'музыка', -1.2), ('knock', 'стук', 2.)):
            world.sounds.add(kind, words, bearing, .8)
    return mind, acts


def sane(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(sane(v) for v in value)
    if isinstance(value, dict):
        return all(sane(v) for v in value.values())
    return False


@pytest.mark.parametrize('tool,index,who,circumstance', CASES,
                         ids=[f'{t or "empty"}-{i}-{w}-{c}' for t, i, w, c in CASES])
def test_checklist(world, clock, tool, index, who, circumstance):
    mind, acts = make(world, clock, circumstance)
    thing = world.archive.saw('chair', (3.0, .4), 'Спальня и кабинет', clock.now, count=4)
    args = json.loads(json.dumps(VARIANTS[tool][index]))
    for key, value in list(args.items()):
        if value == 'THING':
            args[key] = next(o['id'] for o in world.archive.objects(min_count=1)) if thing is not False else 1
    owner = who == 'owner'
    result = mind.do({'tool': tool, 'args': args}, owner=owner)                        # 1. never crashes
    assert isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool)
    ok, text = result
    if not ok:
        assert isinstance(text, str) and text.strip(), 'a refusal says why'
        spoken = brainmod.said_reason(text)
        assert spoken and not any(ch in spoken for ch in '{}[]"'), f'refusal said as «{spoken}»'
    if owner and tool in KNOWN:
        assert not str(text or '').startswith('не умею'), f'the owner asked «{tool}»: {text}'
    body = [d for k, d in acts if k == 'body_do']
    for command in body:                                                               # 2. a sane body command
        assert command['skill'] in skills.SKILLS, command
        assert sane(command.get('args')), command
        a = command.get('args') or {}
        if command['skill'] == 'drive':
            assert abs(a.get('distance', 0)) <= 1.0
        if command['skill'] in ('wait', 'track', 'dance'):
            assert 0 <= float(a.get('seconds', 1)) <= 900
    moving = [c for c in body if c['skill'] in MOVING]
    if not owner:                                                                      # 3. the house rules
        if circumstance in ('quiet', 'docked_empty'):
            assert not moving, f'{circumstance}: moved on its own: {moving}'
        if circumstance == 'lost':                  # looking round helps to find the place; trips by the map do not
            assert not [c for c in body if c['skill'] in ('go', 'go_room', 'approach', 'explore', 'track', 'drive')], \
                f'lost: drove on its own: {body}'
        if circumstance == 'staying':
            assert not any(c['skill'] == 'go' and c['args'].get('xy') and c['args']['xy'][0] > 2.6 for c in body), \
                'left the room it was asked to stay in'
        if circumstance == 'talk':
            pass                                                                       # a talk is handled by life
    if owner and circumstance == 'docked_empty' and tool in ('go', 'explore_room', 'follow', 'approach', 'move'):
        assert not moving, 'an empty pack: the owner is told, the body does not leave'
