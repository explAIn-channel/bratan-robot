"""The mind: a continuous train of thought that freely uses the body, plus fast spoken answers.

Thinking is event-driven (someone appeared, an action finished, ...) with a slow
heartbeat when nothing happens. Each step: one model call -> a thought, maybe
words, maybe one action, maybe a face. Code only refuses what is impossible or
dishonest (speaking to someone absent, repeating itself) and that refusal is fed
back as experience.
"""
from __future__ import annotations

import base64
import difflib
import json
import math
import random
import re
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ..common.persona import CHARACTER
from .llm import LLMError, parse_json
from .affect import Affect, heard_tone
from .drives import Drives
from .world import PORTABLE_LABELS, ago

BODY_TOOLS = {
    'go': 'поехать в комнату или знакомое место: {"to": "<комната или место>"}',
    'approach': 'подъехать к предмету из списка «что помню вокруг» и рассмотреть: {"id": N}; или к кому-то: '
                '{"who": "<имя>|человек|собака", "intent": "поздороваться|поиграть|сфотографировать|побыть рядом|'
                'сказать: <что>"} — intent: зачем еду, это сделаю сам, как только доеду',
    'look_around': 'оглядеться на месте: {}',
    'explore_room': 'объехать комнату и посмотреть на неё с разных сторон: {"room": "<комната>"} (по умолчанию текущая)',
    'follow': 'ехать за тем, кого сейчас видишь, держаться за ним, а если ушёл за угол — доехать туда, где видел '
              'последний раз, и поискать: {"who": "<имя>|человек|собака", "seconds": 10..120; по просьбе человека до 600}',
    'watch': 'поворачиваться и следить глазами за тем, кого видишь: {"who": ..., "seconds": 5..120}',
    'search': 'поискать глазами, поворачиваясь на месте: {"who": "человек|собака|<имя>"}',
    'to_sound': 'пойти на звук — туда, откуда его слышали уши (направление и сторону — см. «Слышу»): '
                '{"sound": "лай|голос|музыка|стук|звон|смех|вода|…" или ничего — последний}; путь выберу сам: в комнату, '
                'откуда донеслось, или ближе к нему здесь, потом посмотрю; на лай ищу Таффу, на голос — человека',
    'photo': 'сфотографировать то, что перед тобой, в свой альбом: {"caption": "..."}',
    'show_photo': 'показать на экране снимок из альбома тому, кто на тебя смотрит: {"id": "..."}',
    'rest': 'поехать на базу отдохнуть и зарядиться: {}',
    'stop': 'остановить тело: {}',
    'dance': 'потанцевать: покачаться на месте под музыку (если её слышишь или попросили): {"seconds": 5..30}',
    'play': 'поиграть с человеком: {"game": "прятки|тише едешь|фото|дай пять|альбом"} — когда человек рядом и согласен '
            'или сам попросил (игру ведёт тело само: считает, ищет, следит)',
    'emote': 'показать чувство телом: {"kind": "wiggle (радостно вильнуть)|bounce (подпрыгнуть от восторга)|'
             'tremble (затрястись от радости или восторга)|shiver (вздрогнуть)|nod (кивнуть)|shake (помотать «нет»)|'
             'spin (закружиться)|bow (поклониться)"}',
    'wait': 'постоять на месте: на базе — сколько хочешь, вне базы — коротко (до 30 с): {"seconds": 5..900}',
    'move': 'проехать прямо вперёд или назад, не поворачивая: {"meters": -1..1} (минус — назад)',
    'turn': 'повернуться на месте: {"degrees": -180..180} (плюс — налево; «развернись» — 180)',
    'perform': 'маленький номер для тех, кто рядом: {"steps": [шаг, ...]}, 2–8 шагов; шаг — объект с одним '
               'движением "turn": градусы | "move": -0.3..0.3 | "emote": "<kind из emote>" | "pause": секунды и, по '
               'желанию, "say": "<несколько слов>", "sound": "<звук>", "face": "<лицо>" в нём же, например шаг '
               '{"turn": 90, "say": "Оп!", "face": "happy"} — слова, звук и лицо идут в ту же секунду, что и движение',
}
# Names the talking model reached for that are not tools (10.10 18:22: «попробуй ещё раз встать на базу» -> «dock»,
# «Не получается: не выходит»; «отъедь вперёд» -> go {"to": "вперёд"}).
TOOL_ALIASES = {'dock': 'rest', 'charge': 'rest', 'base': 'rest', 'home': 'rest', 'drive': 'move',
                'forward': 'move', 'back': 'move', 'rotate': 'turn', 'sleep': 'rest', 'show': 'perform',
                'routine': 'perform', 'number': 'perform', 'performance': 'perform', 'choreography': 'perform'}
PERFORM_MOVE_M = .3          # one step of a number drives at most this far
PERFORM_PATH_M = .8          # and the whole number this far: it is a number on the spot, not a trip
PERFORM_AGAIN_S = 900.       # a number of my own (not asked for) at most this often
# A step the model wrote in words (11.10 phone check: «повернуться влево», «качнуться вперёд-назад», «замереть»).
STEP_CALL = re.compile(r'^\s*(turn|move|pause|wait|emote|say)\s*[:=]?\s*(.+?)\s*$', re.I)
STEP_WORDS = (('мяук', {'say': 'Мяу!', 'face': 'happy'}), ('мурч', {'emote': 'tremble', 'say': 'Мррр…'}),
              ('мурл', {'emote': 'tremble', 'say': 'Мррр…'}), ('гавк', {'say': 'Гав!', 'emote': 'bounce'}),
              ('лай', {'say': 'Гав!', 'emote': 'bounce'}), ('вздох', {'say': 'Эх…', 'face': 'sleepy'}),
              ('потер', {'emote': 'bounce'}), ('покачив', {'emote': 'bounce'}), ('поворот', {'turn': 90}),
              ('кругом', {'turn': 360}), ('покруж', {'emote': 'spin'}), ('круж', {'emote': 'spin'}),
              ('развер', {'turn': 180}), ('влево', {'turn': 90}), ('налево', {'turn': 90}), ('вправо', {'turn': -90}),
              ('направо', {'turn': -90}), ('поверн', {'turn': 90}), ('качн', {'emote': 'bounce'}),
              ('вперёд-назад', {'emote': 'bounce'}), ('вперед-назад', {'emote': 'bounce'}), ('вперёд', {'move': .15}),
              ('вперед', {'move': .15}), ('назад', {'move': -.15}), ('замер', {'pause': 1.}), ('пауз', {'pause': 1.}),
              ('кивн', {'emote': 'nod'}), ('помота', {'emote': 'shake'}), ('поклон', {'emote': 'bow'}),
              ('вильн', {'emote': 'wiggle'}), ('прыг', {'emote': 'bounce'}), ('дрож', {'emote': 'tremble'}),
              ('вздрог', {'emote': 'shiver'}))
# The model asked for a number and wrote no steps at all ({"kind": "dance"}): one of these, not a refusal.
READY_NUMBERS = (
    [{'emote': 'bow', 'say': 'Встречайте!', 'face': 'happy'}, {'turn': 360, 'sound': 'excited'},
     {'emote': 'wiggle', 'say': 'Та-дам!', 'face': 'laughing'}],
    [{'turn': 45, 'face': 'curious'}, {'turn': -90}, {'turn': 45, 'sound': 'happy'}, {'emote': 'bounce', 'say': 'Опа!'},
     {'emote': 'bow', 'face': 'happy'}],
    [{'move': .15, 'say': 'Раз', 'face': 'happy'}, {'move': -.15, 'say': 'два'}, {'emote': 'spin', 'say': 'и кружимся!',
                                                                                  'sound': 'excited'},
     {'emote': 'nod', 'face': 'laughing'}],
)
# Plain short commands are done at once, before the talking model's ~1.5 s (GrowBot's fast local path: «tap your
# right leg three times» without the big model). Only whole phrases; anything longer goes the usual way.
INSTANT = (
    (re.compile(r'(?:(?:сдай|отъедь|отойди|езжай|поезжай|едь|проедь|откатись|отъезжай)\s+)?назад'
                r'(?:\s+(?:немного|немножко|чуть|чуть-чуть|чуток))?'), 'move', -.5),
    (re.compile(r'(?:(?:отъедь|езжай|поезжай|едь|проедь|двигай|проезжай)\s+)?вперед'
                r'(?:\s+(?:немного|немножко|чуть|чуть-чуть|чуток))?'), 'move', .5),
    (re.compile(r'развернись(?:\s+кругом)?|разворот|повернись\s+кругом'), 'turn', 180),
    (re.compile(r'(?:повернись|поверни|поворот)\s+(?:налево|влево)'), 'turn', 90),
    (re.compile(r'(?:повернись|поверни|поворот)\s+(?:направо|вправо)'), 'turn', -90),
    (re.compile(r'кивни(?:\s+головой)?'), 'emote', 'nod'),
    (re.compile(r'(?:помотай|покачай|покрути)\s+головой'), 'emote', 'shake'),
    (re.compile(r'поклонись'), 'emote', 'bow'),
    (re.compile(r'покружись|покрутись|закружись|крутанись'), 'emote', 'spin'),
    (re.compile(r'(?:вильни|повиляй)(?:\s+хвостом)?'), 'emote', 'wiggle'),
    (re.compile(r'(?:(?:иди|поезжай|езжай|едь|ехай)\s+)?(?:спать|на\s+базу|домой|заряжаться)'), 'rest', None),
    (re.compile(r'(?:(?:иди|подъедь|подойди|приезжай|едь|езжай|ехай)\s+)?(?:сюда|ко\s+мне)'), 'approach', 'меня'),
)
INSTANT_FILLER = re.compile(r'^(?:(?:братан|ну|а|давай|так|слушай|эй)\s+)+|\s+(?:пожалуйста|давай|ка)$')


def instant_command(text):
    """A plain short command as a tool call, or None: «сдай назад», «развернись», «кивни», «на базу», «ко мне»."""
    words = re.sub(r'[^\w\s-]', ' ', str(text).casefold().replace('ё', 'е').replace('-ка', ''))
    words = ' '.join(words.split())
    for _ in range(3):
        words = INSTANT_FILLER.sub('', words).strip()
    if not words or len(words.split()) > 4:
        return None
    for pattern, tool, value in INSTANT:
        if pattern.fullmatch(words):
            if tool == 'move':
                little = re.search(r'немн|чут', words)
                return {'tool': 'move', 'args': {'meters': value / 2 if little else value}}
            if tool == 'turn':
                return {'tool': 'turn', 'args': {'degrees': value}}
            if tool == 'emote':
                return {'tool': 'emote', 'args': {'kind': value}}
            if tool == 'approach':
                return {'tool': 'approach', 'args': {'who': value, 'intent': 'побыть рядом'}}
            return {'tool': tool, 'args': {}}
    return None
# What the model writes for an emotion (10.10: «помотай головой» came as {"kind": "нет"} and showed joy).
EMOTE_WORDS = (('нет', 'shake'), ('no', 'shake'), ('мота', 'shake'), ('да', 'nod'), ('yes', 'nod'), ('кив', 'nod'),
               ('покл', 'bow'), ('bow', 'bow'), ('круж', 'spin'), ('крут', 'spin'), ('прыг', 'bounce'),
               ('дрож', 'tremble'), ('испуг', 'shiver'), ('happy', 'wiggle'), ('рад', 'wiggle'))
# The owner's own words settle what the model may have got wrong (10.10: «подъедь ко мне» -> approach the dog).
TO_ME = re.compile(r'\b(ко\s+мне|меня|сюда|за\s+мной)\b', re.I)
BACK_WORDS = re.compile(r'\b(назад|сдай|задом|back)', re.I)
TURN_WORDS = re.compile(r'развер|разворот|поверн|\bturn\b', re.I)
MOVE_WORDS = {'вперёд': 1., 'вперед': 1., 'назад': -1.}
# Lost on the map: trips by the map fail at once (10.10 11:13-13:13: 96 «dock: потерялся на карте» while the body
# could not search for itself — the mind kept it busy). The body looks for itself first.
NEEDS_MAP = ('go', 'approach', 'explore_room', 'rest', 'follow', 'search', 'to_sound')
# What one says for a sound to go to («на лай», «голос») -> the ears' kinds (voice/sounds.py; 'wake' — my name).
SOUND_KINDS = (('лай', ('dog',)), ('лаял', ('dog',)), ('собак', ('dog',)), ('тафф', ('dog',)),
               ('голос', ('wake', 'voice', 'speech')), ('разговор', ('speech', 'voice', 'wake')),
               ('речь', ('speech', 'voice', 'wake')),
               ('зов', ('wake',)), ('звал', ('wake',)), ('имя', ('wake',)), ('музык', ('music',)),
               ('песн', ('music',)), ('стук', ('knock', 'bang')), ('грох', ('bang', 'startle')),
               ('хлоп', ('door', 'bang', 'startle')), ('двер', ('door', 'knock', 'doorbell')),
               ('звон', ('doorbell', 'alarm')), ('будильн', ('alarm',)), ('смех', ('laughter',)),
               ('смея', ('laughter',)), ('вод', ('water',)), ('кухн', ('kitchen',)), ('посуд', ('kitchen',)),
               ('телевиз', ('tv',)), ('радио', ('tv',)), ('пылесос', ('vacuum',)), ('плач', ('crying',)),
               ('чих', ('sneeze',)), ('свист', ('whistle',)), ('хлопа', ('clapping',)), ('шаг', ('footsteps',)))
# Whom a sound is most likely of: the trip ends by finding them (the executive's «found on the way»).
SOUND_WHO = {'dog': 'собака', 'wake': 'человек', 'voice': 'человек', 'speech': 'человек', 'laughter': 'человек',
             'crying': 'человек',
             'sneeze': 'человек', 'clapping': 'человек', 'whistle': 'человек', 'footsteps': 'человек'}
# From the dock I go by myself only to these (a call, an alarm, a bang): the house's chatter is not a reason to leave
# the charger (owner 08.10: from the dock nobody is approached unasked).
SOUND_FROM_DOCK = ('wake', 'alarm', 'doorbell', 'knock', 'bang', 'startle', 'crying')
SOUND_STEP_M = 1.5           # in this room: up to this much closer to the sound before looking
DOORWAY_LISTEN_S = 2.5      # out through a doorway: stand in it this long and listen again (C52)
VOICE_RAY_S = 30.            # «иди ко мне»: where my name or the last words came from counts this long
WEEKDAYS = ('понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье')
MONTHS = ('января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа', 'сентября', 'октября', 'ноября',
          'декабря')
MIND_TOOLS = {
    'recall': 'вспомнить: кого/что и когда видел, что было, свои заметки: {"about": "..."}',
    'note': 'записать в память что-то важное: {"text": "..."}',
    'want': 'запомнить своё новое желание: {"text": "..."}',
    'drop_want': 'желание исполнилось или расхотелось: {"index": N}',
    'plan': 'пообещать себе сделать позже: {"what": "...", "at": "HH:MM" | "in_minutes": N | "when_see": "<имя>|собака"}',
    'drop_plan': 'отменить план: {"index": N}',
    'new_skill': 'придумать навык из действий тела (до 6 шагов): {"name": "...", "purpose": "...", "steps": [{"tool": "...", "args": {}}]}',
    'use_skill': 'выполнить свой навык: {"name": "..."}',
    'forget_skill': 'удалить навык: {"name": "..."}',
    'self': 'переписать, как ты сам себя понимаешь: {"text": "..."}',
    'learn_face': 'запомнить лицо человека, который сейчас перед тобой, под именем: {"name": "..."}',
    'hush': 'помолчать: не заговаривать первым N минут (если попросили тише или люди заняты): {"minutes": 5..600}',
    'tell': 'передать сообщение человеку — найти его по дому и сказать: {"to": "<имя>", "text": "...", '
            '"at": "HH:MM" или "in_minutes": N, если позже}',
    'alarm': 'разбудить человека ко времени (только если он сам попросил): {"to": "<имя>", "at": "HH:MM", '
             '"text": "..."}',
    'timer': 'засечь время и прозвенеть: {"minutes": N, "label": "что засекли"}',
    'remember_place': 'запомнить место, где стою сейчас, под названием (потом можно go туда): {"name": "у окна"}',
    'name_thing': 'запомнить вещь перед камерой под названием, которое сказал человек: {"name": "мой рюкзак"}',
    'web': 'поискать в интернете то, чего не знаешь или что меняется (новости, курс, расписание, кто такой, счёт '
           'матча); вслух первой строкой — «сейчас гляну», ответ скажешь, когда найдёшь: {"query": "..."}',
}
WEB_PER_DAY = 30              # searches a day at most (OpenRouter's web plugin: ~$0.012 each with 3 results)
TRIGGER_TEXT = {
    'heartbeat': 'просто прошло время', 'refused': 'мне только что не удалось то, что я хотел (последнее событие) — '
    'сделай иначе или ничего', 'person_appeared': 'кто-то появился в кадре', 'dog_appeared': 'увидел собаку',
    'action_done': 'тело закончило действие', 'action_failed': 'у тела не вышло', 'docked': 'встал на базу',
    'lost': 'потерялся на карте', 'bump': 'упёрся во что-то', 'conversation_ended': 'разговор закончился',
    'plan_due': 'пришло время плана', 'departure_ready': 'зарядился', 'battery_low': 'батарея садится',
    'touch': 'меня потрогали',
    'reflect': 'спокойная минута на базе — подумай о прошедших часах: что понял о себе, о доме и о своих? '
               'Перепиши «кто я» (self), если стало яснее, и запиши главный вывод (note)',
    'idle_choice': 'я давно стою без дела — выбери, чем заняться (или на базу)', 'recall_result': 'вспомнил то, что искал', 'skill_step': 'шаг моего навыка',
    'carried': 'меня перенесли руками в другое место',
    'handled': 'меня потрогали руками — повернули или подвинули (см. последнее событие)',
    'recognized': 'узнал того, кого сначала не узнал (см. последнее событие) — поправь себя, если думал о нём как о чужом',
    'door_opened': 'открыли дверь в другую комнату',
    'sound': 'услышал что-то (см. последнее событие)', 'startle': 'резкий громкий звук — вздрогнул',
    'woke_up': 'тихие часы кончились — проснулся, начинается новый день',
    'game_played': 'игра закончилась (см. последнее событие)', 'peekaboo': 'со мной поиграли в ку-ку',
    'boop': 'меня легонько ткнули в нос — играют',
    'waved': 'мне помахали рукой', 'cheered': 'человек поднял руки вверх — радуется',
    'looks_sad': 'у человека передо мной грустное лицо',
    'arrived': 'добрался до того, к кому ехал, и сделал, зачем ехал (см. последнее событие) — что дальше?',
    'oriented': 'обернулся на движение рядом (см. последнее событие)',
}
URGENT = {'person_appeared', 'dog_appeared', 'action_done', 'action_failed', 'lost', 'docked', 'bump',
          'conversation_ended', 'plan_due', 'departure_ready', 'battery_low', 'touch', 'refused', 'recall_result',
          'idle_choice', 'reflect', 'carried', 'door_opened', 'sound', 'startle', 'woke_up', 'game_played', 'boop',
          'skill_step', 'waved', 'cheered', 'looks_sad', 'oriented', 'arrived', 'handled'}
# The face app has more (laughing, surprised, affectionate): a fuller range reads as more alive (08.10 audit).
FACES = {'idle', 'curious', 'happy', 'sleepy', 'concerned', 'laughing', 'surprised', 'affectionate'}
SAY_GAP_S = 25.
BODY_PATIENCE_S = 120.
FAILED_COOLDOWN_S = 600.
SPIN_AGAIN_S = 180.
CLOSED_WAY_S = 600.
EXPLORE_AGAIN_S = 600.
# Thinking aloud while driving (owner 10.10: «пусть вслух произносит ключевые мысли, пока ездит — не все»): the first
# sentence of a thought that decides a trip or meets someone, at most once in THINK_ALOUD_GAP_S, never twice alike.
REFLEX_PATIENCE_S = 10.       # a reflex just acted: a whim of the next thought does not undo it at once
REFLEX_TOOL_WORDS = {'approach': 'подъезжаю', 'go': 'еду', 'search': 'ищу', 'look_around': 'оглядываюсь',
                     'to_sound': 'еду на звук',
                     'rest': 'еду на базу', 'photo': 'снимаю', 'follow': 'еду следом', 'watch': 'слежу',
                     'explore_room': 'осматриваю', 'move': 'проезжаю', 'emote': 'показываю', 'stop': 'стою'}
WHEN_TEXT = {'dog': 'увидел собаку', 'person': 'увидел человека', 'name': 'увидел', 'door_closed': 'дверь закрыта',
             'arrived': 'доехал', 'failed': 'не вышло', 'lost_sight': 'потерял из виду', 'dark': 'темно',
             'sound': 'услышал'}


# How the ears word what they hear («слышу: лает собака», «звенит будильник», «рядом разговаривают»): a condition
# «звук: лай» must match «лает» (the scenario run of 10.10 caught it never would).
SOUND_FORMS = {'лай': ('лай', 'лает', 'лаю'), 'музык': ('музык',), 'звон': ('звон', 'звенит'), 'стук': ('стук', 'стуч'),
               'хлоп': ('хлоп', 'грохн'), 'смех': ('смех', 'смеют', 'смеёт'), 'голос': ('голос', 'разговар', 'говор'),
               'разговор': ('разговар', 'голос', 'говор'), 'плач': ('плач',)}


def when_of(text, people=(), pet=''):
    """A condition as the model wrote it («собака», «увижу Сергея», «дверь закрыта», «звук: лай») -> (key, arg)."""
    low = str(text or '').casefold()
    for name in people:
        if name.casefold()[:4] in low:
            return 'name', name
    if (pet and pet.casefold()[:4] in low) or re.search(r'собак|\bпс[аеу]\b|\bпёс|\bпес\b', low):
        return 'dog', None
    if re.search(r'темн', low):
        return 'dark', None
    if re.search(r'двер|закрыт|не проехать|проход', low):
        return 'door_closed', None
    if re.search(r'из виду|убежал|ушёл|ушел|скрыл|пропал', low):
        return 'lost_sight', None
    if re.search(r'не вышл|не получ|не смог|застрял|не удал', low):
        return 'failed', None
    if re.search(r'доеду|доехал|приеду|приехал|на месте|получилось|закончу|закончил', low):
        return 'arrived', None
    sound = re.search(r'лай|музык|звон|стук|хлоп|смех|голос|разговор|плач', low)
    if sound or re.search(r'звук|слыш', low):
        return 'sound', sound.group(0) if sound else None
    if re.search(r'человек|кто-то|кого-то|люд|гост|кто-нибудь', low):
        return 'person', None
    return None, None


SEEK_WORDS = re.compile(r'поищ|ищу|искать|найд|найти|поймать|догнать|разыщ|высмотр', re.I)
THINK_ALOUD_GAP_S = 45.
THINK_ALOUD_WORDS = 16
THINK_ALOUD_TRIGGERS = ('dog_appeared', 'person_appeared', 'arrived', 'found_on_the_way', 'lost', 'carried',
                        'battery_low', 'bump')
THINK_ALOUD_MOVES = ('go', 'approach', 'explore_room', 'search', 'follow', 'rest', 'to_sound')
SELF_TALK_GAP_S = 600.        # thinking aloud is rare: the owner does not want the robot to comment on itself
DOG_HELLO_GAP_S = 1200.       # Taffa runs in and out all evening: one hello per 20 min, then just be glad of him
JOURNAL_SHOWN = 5             # 09.10: ten past thoughts in the prompt made the next one the same («опять закрутился»)
TALK_EARSHOT_S = 90.          # after a talk the person is within earshot even out of view: a word about the errand
SEEK_S = 45.                  # asked to come or to find someone: one look round (~22 s) and the way there
ME_WORDS = ('я', 'мне', 'меня', 'ко мне', 'сюда')
UNCLEAR_REPLY = re.compile(r'не\s+(совсем\s+)?(расслыш|понял)|повтори', re.I)
# Long phrases without my name that I did not get, this many in a row: the person is talking to someone else, and the
# next long one without my name is theirs too (09.10 19:19, the owner dictating at the PC).
ASIDES_IN_ROW = 2
# «Это не мне адресовано, молчу» aloud, five times into the owner's phone call (09.10 19:29-19:31): said, it is no
# silence. To a phrase that may be someone else's, such a reply is held back like «не расслышал».
SELF_SILENCE = re.compile(r'\bмолчу\b|\bне\s+мне\b|не\s+буду\s+(мешать|встревать|отвечать|лезть)|\bне\s+мешаю\b', re.I)
# «там я вообще не тебе, меня не слушай и не отвечай» (09.10 19:20): the talk is over until my name.
NOT_FOR_ME = re.compile(r'\bне\s+(тебе|с\s+тобой|к\s+тебе)\b|\bне\s+(слушай|отвечай)\b', re.I)
# «Побудь там минут 10» (09.10 19:21: four minutes in the hall, then off to the base — and sent back at 19:30): a
# promise kept by my own choices; a low battery still comes first, and the owner's next request replaces it.
STAY_WORDS = re.compile(r'\b(побудь|посиди|останься|оставайся|постой|подожди|жди)\b', re.I)
STAY_PLACE = re.compile(r'\b(тут|здесь|там|в\s+\w+|на\s+\w+)\b', re.I)
STAY_DEFAULT_MIN, STAY_MAX_MIN = 10., 60.
STAY_BREAKERS = ('go', 'rest', 'explore_room', 'search', 'follow', 'approach', 'to_sound')
STAY_NUMBERS = (('минутку', 1), ('минуту', 1), ('пару', 2), ('две', 2), ('три', 3), ('пять', 5), ('десять', 10),
                ('пятнадцать', 15), ('двадцать', 20), ('полчаса', 30), ('час', 60))


def stay_minutes(text):
    """«побудь там минут 10» -> 10; «побудь там ещё какое-то время» -> STAY_DEFAULT_MIN; not a stay -> None."""
    low = text.casefold()
    if not STAY_WORDS.search(low) or 'секунд' in low:      # «подожди секунду» is a breath, not a stay
        return None
    number = re.search(r'(\d+)\s*мин|мин\w*\s+(\d+)', low)
    if number:
        return min(STAY_MAX_MIN, max(1., float(number.group(1) or number.group(2))))
    for word, minutes in STAY_NUMBERS:
        if re.search(rf'\b{word}\b', low):
            return float(minutes)
    return STAY_DEFAULT_MIN if STAY_PLACE.search(low) else None


# What one goes to someone for (09.10, the owner: «нашёл Таффу и смотрит на него — когда находишь, надо что-то
# делать»; «у него должны быть намерения»). Words of the model -> what life does on arrival (life.carry_out).
INTENTS = (('поигра', 'play'), ('игра', 'play'), ('сфотограф', 'photo'), ('фото', 'photo'), ('снимок', 'photo'),
           ('побыть', 'company'), ('рядом', 'company'), ('сказать', 'tell'), ('передать', 'tell'),
           ('рассказать', 'tell'), ('позвать', 'call'), ('поздорова', 'greet'), ('привет', 'greet'))
PLAN_STALE_S = 18 * 3600.       # a plan with no time («завтра днём») is stale by the next evening
WANT_STALE_S = 4 * 86400.
# Temperament (09.10 audit): the old robot was born curious .7, cautious .6, talkative .4, tidy .5. A dream moves a
# trait a little each night, never further than TRAIT_BAND from it, and a tenth of the way back every night:
# «cautious» had crept .86 -> .92 in three nights (learned helplessness, not character).
TEMPERAMENT = {'curious': .7, 'cautious': .6, 'talkative': .4, 'tidy': .5}
TRAIT_BAND = .2
TRAIT_RETURN = .1
# Lessons that teach waiting instead of living (09.10 audit: «лучше не настаивать, а просто ждать в стороне»,
# «дождаться утра на базе»): kept apart and never shown, until beliefs with evidence exist.
PASSIVE_LESSON = re.compile(r'не\s+настаива|не\s+мельтеш|не\s+д[её]рга|ждать\s+в\s+сторон|дождаться\s+утра|'
                            r'(оставаться|остаться|стоять|ждать|подождать)\s+на\s+базе|невыполним|'
                            r'лучше\s+(просто\s+)?(ждать|подождать|постоять|стоять|не\s+ехать)', re.I)
# Words that do not tell one want from another; «осмотреть», «посмотреть», «обследовать» are one wish to look.
WANT_FILLER = set('хочу хочется хотел хотелось хочет завтра утром утра утро днем вечером ночью ближайшие дни днях '
                  'спокойно снова еще когда что там тут это как чтобы пока без спешки все раз раза прошлого прошлый '
                  'прошлом давно был была было буду может можно надо нужно интересно попробую попробовать поехать '
                  'съездить поеду ехать заехать доехать сделать нет для например больше потом сейчас сегодня очень '
                  'его она они мне меня или если чем так уже где кто этот эту наконец обязательно немного чуть '
                  'свой свою своих своим себе себя никого'.split())
WANT_LOOK = ('осмо', 'посм', 'обсл', 'разг', 'изуч', 'иссл', 'смот')
# Thinking aloud must not narrate what the body does (ЗАДУМКА: no comments on trips, stops and decisions).
NARRATION = re.compile(r'(^|\s)(поеду|еду|съезжу|заеду|доеду|отъеду|поехал|вернусь|возвращаюсь|отдохну|'
                       r'посмотрю|осмотрю|осмотрюсь|огляжусь|оглядываюсь|сфотографирую|подъеду)(\s|[,.!?…]|$)', re.I)
# Technical words from the old robot's logs that leaked into «lessons» (dock_retry_reference_expired, ...).
TECHNICAL = re.compile(r'[a-z]+_[a-z_]+|[«\'"][a-z]{4,}[»\'"]|clearance|initial_home|localization', re.I)
REPEAT_SIMILARITY = .62
REPEAT_WINDOW_S = 3600.
OPEN_CALL_BACKOFF = (600., 1200., 2400.)
SKILL_STEPS_MAX = 6
VOICE_SPLIT = '###'
# Senses the robot does not have: room sounds, breathing, smells, warmth.
INVENTED_SENSE = re.compile(r'(посапыва\w*|сопит|сопени\w*|дыхани\w*|дышит|шурш\w*|храп\w*|'
                            r'мурлы\w*|пахн\w*|запах\w*|чу[вю]\w*\s+(его\s+)?тепл\w*|всегда\s+тепло|тепло\s+и)', re.I)
# 10.10 17:46: «слышу, как рядом разговаривают» was true (the ears hear speech since 08.10) and got «я не слышу
# звуков комнаты». Hearing is a sense now; breath, smells and warmth are not.
CORRECTION = ('поправка: дыхания, запахов и тепла я не чувствую — знаю то, что вижу камерой, звуки из «Слышу» '
              'и что мне сказали')
LOOK_WORDS = re.compile(r'(что|кого|кто)\s+(ты\s+)?(видишь|видно|там|перед)|посмотри|покажи,? что|опиши|какого цвета|'
                        r'что (у меня|это)( в руке| такое)?|сколько пальцев', re.I)


def clip_words(text, limit):
    """The first whole sentences within `limit` words (at least one sentence)."""
    parts = re.split(r'(?<=[.!?…])\s+', (text or '').strip())
    kept, count = [], 0
    for part in parts:
        words = len(part.split())
        if kept and count + words > limit:
            break
        kept.append(part)
        count += words
    return ' '.join(kept)


def said_reason(effect):
    """A refusal as words for a person: its first clause, without tool hints meant for the model."""
    text = str(effect or '').split(' — ')[0].split('(')[0].strip(' .;:')
    if not text or re.search(r'[{}\[\]"]|\b[a-z_]{3,}\b', text):
        return 'не выходит'
    return text


BASE_WORDS = re.compile(r'^(на\s+)?(баз[аеуы]|зарядк\w*|зарядн\w*|док|домой|станци\w*)$')
# Microsoft's recognizer on this room's microphone (08.10 20:06): «едь на базу» came as «есть на базу»,
# «база» as «ваза». Only whole known slips, never a guess.
HEARD_SLIPS = ((re.compile(r'\b(есть|ешь|едет)\s+на\s+(базу|кухню|зал)\b'), r'едь на \2'),
               (re.compile(r'\bна\s+вазу\b'), 'на базу'), (re.compile(r'\bваза\s+(же\s+)?(тоже\s+)?находится'), r'база \1\2находится'),
               # 09.10 19:21 «поезжай, пожалуйста, в зал» was heard «поесть пожалуйста в зал» -> «Понял, поем в зал»
               (re.compile(r'\bпо(есть|ешь|ем)\b(?=(\s+пожалуйста)?\s+(в|на)\s+(зал|кухню|спальню|коридор|базу)\b)'),
                'поезжай'))


def fix_heard(text):
    for pattern, repl in HEARD_SLIPS:
        text = pattern.sub(repl, text)
    return text


def similar(a, b):
    a, b = re.sub(r'[^а-яa-z ]', '', a.casefold()), re.sub(r'[^а-яa-z ]', '', b.casefold())
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.


def intent_of(args):
    """(kind, words) of what one goes to someone for, from the model's «intent» (or None)."""
    raw = str(args.get('intent') or args.get('зачем') or args.get('then') or '').strip()
    if not raw:
        return None
    low = raw.casefold()
    kind = next((k for word, k in INTENTS if word in low), 'greet')
    words = raw.split(':', 1)[1].strip() if kind == 'tell' and ':' in raw else ''
    return kind, words[:200]


def accusative(name):
    """Таффа -> Таффу, Марина -> Марину, Сергей -> Сергея («нашёл …», «искал …»)."""
    if name.endswith('а'):
        return name[:-1] + 'у'
    if name.endswith('я'):
        return name[:-1] + 'ю'
    if name.endswith('й'):
        return name[:-1] + 'я'
    if name and name[-1] in 'бвгджзклмнпрстфхцчшщ':
        return name + 'а'
    return name


def no_vocative(text, names):
    """«Сергей, ты про…» / «…, Сергей.» -> without the name: a name in every line sounds unnatural (owner 09.10).
    A name that tells who it is («Ты Сергей», «Это Марина») stays."""
    for name in names:
        n = re.escape(name)
        text = re.sub(rf'(^|(?<=[.!?…]\s))\s*{n}\s*,\s*(\w)', lambda m: m.group(1) + m.group(2).upper(), text)
        text = re.sub(rf',\s*{n}(?=\s*[.!?…,]|\s*$)', '', text)
    return text.strip()


def want_words(text):
    stems = set()
    for word in re.findall(r'[а-яa-z]+', str(text).casefold().replace('ё', 'е')):
        if len(word) >= 3 and word not in WANT_FILLER:
            stems.add('смот' if word[:4] in WANT_LOOK else word[:4])
    return stems


def same_want(a, b):
    """The same wish in other words (09.10 audit: «осмотреть зал» four times over — letters alone did not see it;
    and «поехать в зал» / «поехать на кухню» are not one wish however alike they look)."""
    if similar(a, b) > .75:
        return True
    x, y = want_words(a), want_words(b)
    return len(x) >= 2 and len(y) >= 2 and len(x & y) / min(len(x), len(y)) >= .5


def bound_trait(name, value):
    """A trait stays within TRAIT_BAND of the temperament (unknown traits: within 0..1)."""
    if not isinstance(value, (int, float)):
        return value
    base = TEMPERAMENT.get(name)
    low, high = (0., 1.) if base is None else (max(0., base - TRAIT_BAND), min(1., base + TRAIT_BAND))
    return round(max(low, min(high, float(value))), 3)


def is_pet(word, pet):
    word, pet = str(word or '').strip().casefold(), str(pet or '').strip().casefold()
    if not word or not pet:
        return False
    stem = pet[:-1] if len(pet) > 3 and pet[-1] in 'аяоеиыуюь' else pet
    return word == pet or (word.startswith(stem) and len(word) - len(stem) <= 2)


TALK_WINDOW_S = 900.          # lines older than this belong to another conversation
SPOKEN_WORDS = 24             # at most this much is said in one turn (08.10: a 6 s list talked over the reply)
SPOKEN_WORDS_LONG = 70        # a joke, a story, a song, a riddle (24 words made him refuse them)
LONG_TURN = re.compile(r'анекдот|расскаж|истори|сказк|спой|песн|стих|загад|шутк|пошути', re.I)
DOCK_QUIET = {'person_appeared', 'waved', 'cheered', 'smiled', 'looks_sad'}
JUST_HAPPENED = {'action_done', 'action_failed', 'refused', 'docked', 'door_opened', 'door_closed', 'carried', 'bump',
                 'lost', 'battery_low', 'handled', 'recognized'}
# How to talk (08.10 evening, the owner: «посмотри, логично ли он говорит»): the replies echoed the person
# («Да. …», «Понятно. …»), retold the robot's own state every turn, asked «Как ты?» again after the answer,
# commented on the question («Ты спросила «кто именно» — …») and asked «Куда именно — на кухню?» instead of going.
CONVERSE_RULES = '\n'.join([
    'КАК ГОВОРИТЬ (живой разговор голосом, отвечаю вслух):',
    '- Отвечаю на смысл последней реплики, как друг: 1–2 коротких предложения, разговорно, без канцелярита и '
    'без перечислений (из многого — одно-два главных).',
    '- Не начинаю с повтора слов человека («Да.», «Понятно.», «Привет.») и не пересказываю сказанное им '
    '(«ты спросил…», «ты сказал…»).',
    '- Не повторяю того, что уже сказал в этом разговоре (про базу, зарядку, Таффу, свои планы): каждый ответ — '
    'новое или шаг дальше.',
    '- Человек ответил на мой вопрос — реагирую на его ответ (рад, сочувствую, интересно) и, если хочется, '
    'продолжаю другой темой; тот же вопрос снова не задаю.',
    '- «да», «ага», «угу», «понятно», «ну» — не вопрос: коротко отзываюсь или веду тему дальше; можно и промолчать.',
    '- Просят что-то сделать («поехали на кухню», «подъедь», «станцуй», «сфотографируй», «поиграем») — делаю сразу '
    '(do) и коротко отзываюсь: «Еду!», «Смотри!». Очевидное не переспрашиваю.',
    '- Просят подъехать к вещи («к кровати», «к столу», «к шкафу») — беру её [id] из «ВЕЩИ ВОКРУГ» (ближайшую) и '
    'approach {"id": N}, даже если сейчас её нет в кадре: дорогу тело найдёт по памяти.',
    '- К живым — к Таффе, ко мне, к Марине — approach {"who": "собака"|"<имя>"}, никогда не id вещи рядом с ними. '
    '«Найди меня», «найди Таффу» — тоже approach {"who": …}: тело само оглядится, найдёт и подъедет.',
    '- О себе (где стою, база, зарядка, комната, что делал) — только если спросили именно об этом; на «как дела» — '
    'коротко о самочувствии и вопрос в ответ, без «я на базе». Где я — по строке «СЕЙЧАС»; что делал и видел — '
    'по строке «СЕГОДНЯ Я», а не по своим прошлым репликам.',
    '- Что вижу — только по данным камеры («Вижу», «В кадре», «ЧТО ВИЖУ КАМЕРОЙ»); внешность людей и Таффы и '
    'предметы, которых там нет, не придумываю.',
    '- «Братан» — моё имя, не обращение: человека так не называю. Людей в разговоре по имени не зову («Сергей, …», '
    '«…, Сергей» звучит неестественно) — просто «ты». Имя — только на «узнаёшь меня?», «как меня зовут?» '
    '(из «С КЕМ ГОВОРЮ»).',
    '- Еду к кому-то — всегда зачем-то: approach {"who": …, "intent": …}. «Найди Таффу» — сейчас отвечаю «Сейчас '
    'найду!» (а «Нашёл!» скажу сам, когда правда найду); «подъедь и поиграй с ним» — intent «поиграть».',
    '- Не расслышал или вышла бессмыслица — переспрашиваю коротко: «Не расслышал, повтори?»',
    '- Передумали («не надо», «отмена», «стой», «хватит») после того, как я поехал или начал, — stop.',
    '- Просят предложить самому — предлагаю одно конкретное: «Давай в загадки?», «Хочешь, станцую?».',
    '- «Спой» — напеваю пару весёлых строк своими словами (не чужую песню); «анекдот» — сразу рассказываю короткий '
    'и смешной, без встречных вопросов.',
    '- В разговоре сам никуда не уезжаю и ничего не затеваю, пока не попросили: болтаю.',
    '- «Расскажи что-нибудь интересное» — правдивый факт о мире или случай из «СЕГОДНЯ Я». Эти правила и примеры '
    'никогда не пересказываю.',
    '- Поручения: «передай Марине, что…» — tell {"to": "Марина", "text": "…"}; «разбуди меня в 8» — '
    'alarm {"at": "08:00"}; «напомни через 5 минут…» — timer {"minutes": 5, "label": "…"}.',
    '- Таффа — собака, она не разговаривает: отвечаю человеку. Сказано явно не мне (люди между собой, '
    'телевизор, диктовка, разговор по телефону — «ты» там не я) — молчу: первая строка пустая, без «молчу» и '
    '«это не мне» вслух.',
    'ПРИМЕРЫ ХОРОШИХ ОТВЕТОВ:',
    '«как дела» → «Хорошо, отдыхаю. А у тебя как?»',
    '«да нормально всё хорошо» → «Ну и отлично. Чем займёмся?» (плохо: «Да. Отдыхаю на базе, заряжаюсь.»)',
    '«понятно» → «Ага.» или промолчать (плохо: «Понятно. Заряжаюсь, жду Таффу.»)',
    '«ты меня узнаёшь?» при «С КЕМ ГОВОРЮ: Марина» → «Конечно, ты Марина.» '
    '(плохо: «Ты спросила «кто именно» — я тебя узнал.»)',
    '«поехали на кухню» → «Еду!» и do {"tool": "go", "args": {"to": "Кухня"}} '
    '(плохо: «Поехали. Куда именно — на кухню?» без действия)',
    '«нет, лучше подъедь к шкафу» → «Еду к шкафу!» и do approach {"id": <шкаф из ВЕЩИ ВОКРУГ>} (сказал «еду» — '
    'значит, есть do)',
    '«сначала покрутись, а потом вернись на базу» → «Ага!», do {"tool": "dance"} и then {"tool": "rest"}',
    '«найди мои ключи» → do search {"who": "ключи"} (ищу в памяти, где их видел)',
    '«подъедь к Таффе» → «Еду к нему!» и do approach {"who": "собака"} (плохо: approach к кровати, у которой он '
    'лежит; или search — тогда останусь на месте)',
    '«найди меня» → «Сейчас найду!» и do approach {"who": "<имя из С КЕМ ГОВОРЮ>"}',
    '«подъедь к Таффе и поиграй с ним» → «Бегу играть!» и do approach {"who": "собака", "intent": "поиграть"}',
    '«ты меня отвлекаешь» / «а ты как?» → без имени: «Понял, не мешаю.» (плохо: «Понял, Сергей, не мешаю.»)',
])

DOZE = 1.5
# R2-D2-like sounds without words (voice/sfx.py) and body emotions (body/skills.py EMOTES).
# The social ones only: no «thinking», «sleepy», «sad» — inner states are not voiced (owner 08.10).
SOUNDS = ('hello', 'happy', 'excited', 'curious', 'laughing', 'surprised', 'yes', 'no', 'found', 'affectionate')
SOUND_GAP_S = 120.
FACE_TRIGGERS = {'person_appeared', 'dog_appeared', 'touch', 'praise', 'scold', 'sound', 'startle', 'carried', 'bump',
                 'boop', 'handled',
                 'peekaboo', 'game_played', 'door_opened', 'found', 'conversation_ended', 'waved', 'cheered',
                 'looks_sad'}
EMOTE_KINDS = ('wiggle', 'bounce', 'tremble', 'shiver', 'nod', 'shake', 'spin', 'bow')
EMOTE_SOUND = {'wiggle': 'happy', 'bounce': 'excited', 'tremble': 'excited', 'shiver': 'surprised', 'nod': 'yes',
               'shake': 'no', 'spin': 'excited', 'bow': 'hello'}


class Mind:
    def __init__(self, state, world, archive, llm, config, act):
        self.state, self.world, self.archive, self.llm = state, world, archive, llm
        self.config = config
        self.act = act                     # callable(kind, **data): the body/voice/face/eyes interface
        self.drives = Drives(archive, world, state, config)
        self.affect = Affect(state, config)
        world.listeners.append(self.affect.on_event)
        self.failed = {}                   # (tool, args) -> times it failed, for a short cooldown
        self.last_body_do = None
        self.tz = ZoneInfo(config.get('timezone', 'Europe/Moscow'))
        self.last_step = 0.
        self.dozing = 0                     # idle thoughts in a row on the dock with no one around
        self.last_explore = None            # room of the running explore (marked explored when done)
        self.next_heartbeat = time.time() + 20
        self.last_trigger = None
        self.version = 0
        self.skill_run = None
        self.sent_id = None                 # id of the last body command sent (a skill step waits for its end)
        self.asides_in_row = 0              # long phrases in a row, without my name, that were not for me
        self.seeking = None                 # {'who', 'id', 'intent'}: the running trip is to find them
        self.plan_if = None                 # {'id', 'rules', 't'}: what the running command does if things change
        self.reacted_at = 0.                # a rule of the plan acted (executive)
        self.intents = {}                   # body command id -> what it is for (life.carry_out on arrival)
        self.performances = {}              # body command id -> the steps of a number (life.on_cue says them)
        self.pending_say = None
        self.busy = False
        self._bonds = (0., '')
        self.tidy(time.time())

    def tidy(self, now):
        """Keep the working memory honest and current (code, not hand edits of the state file).

        - lessons carrying technical codes of the old robot (07.10: «dock_retry_reference_expired»,
          «target_not_visible_at_remembered_position») are not experience of this robot;
        - a plan that fired an hour ago, or that never had a time and is older than 30 h, is done with;
        - a want older than four days has had its chance; repeated wants are kept once.
        """
        s = self.state
        character = s.get('character') or {}
        lessons = character.get('lessons') or []
        kept = [x for x in lessons if not TECHNICAL.search(str(x))]
        if len(kept) != len(lessons):
            character['lessons'] = kept
            s.set('character', character)
            self.archive.event('lessons_tidied', {'removed': len(lessons) - len(kept)})
        passive = [x for x in kept if PASSIVE_LESSON.search(str(x))]
        if passive:
            character['lessons'] = [x for x in kept if x not in passive]
            character['frozen_lessons'] = list(dict.fromkeys(character.get('frozen_lessons', []) + passive))[-30:]
            s.set('character', character)
            self.archive.event('lessons_frozen', {'lessons': passive})
        traits = character.get('traits') or {}
        bounded = {key: bound_trait(key, value) for key, value in traits.items()}
        if bounded != traits:
            character['traits'] = bounded
            s.set('character', character)
            self.archive.event('traits_bounded', {'before': traits, 'after': bounded})
        plans = s.get('plans') or []
        fresh = [p for p in plans
                 if not (p.get('fired') and now - p['fired'] > 3600)
                 and not (not p.get('at_ts') and not p.get('when_see') and now - p.get('t', now) > PLAN_STALE_S)
                 and not (p.get('at_ts') and now - p['at_ts'] > 6 * 3600)]
        if len(fresh) != len(plans):
            s.set('plans', fresh)
        wants, unique, merged = s.get('wants') or [], [], []
        for want in wants:
            if now - want.get('t', now) >= WANT_STALE_S:
                continue
            same = next((u for u in unique if same_want(want['text'], u['text'])), None)
            if same is None:
                unique.append(want)
            else:
                same['times'] = same.get('times', 1) + want.get('times', 1)
                same['t'] = max(same.get('t', 0), want.get('t', 0))
                merged.append(want['text'])
        if len(unique) != len(wants):
            s.set('wants', unique)
            if merged:
                self.archive.event('wants_merged', {'merged': merged})

    def bonds(self, now):
        at, text = self._bonds
        if now - at > 600:
            try:
                text = self.drives.bonds(now)
            except Exception as error:
                text = ''
                self.archive.event('drives_error', {'error': f'bonds {type(error).__name__}:{error}'[:200]})
            self._bonds = (now, text)
        return text

    # ------------------------------------------------------------- timing
    def due(self, now, quiet):
        if self.busy or now - self.last_step < 3.:
            return None
        if self.act('playing'):
            return None                      # a game is on: it has the body and the voice
        if self.world.live:
            return None                      # the conversation owns this moment
        # Asleep on the dock in the quiet hours: no thoughts at all (09.10 audit: «Ночь, все спят…» 36 model calls
        # a night, and the morning began from them). Promises are kept still: alarms and timers below.
        asleep = quiet and self.world.docked
        fresh = self.world.recent(self.last_step)
        urgent = [] if asleep else [e for e in fresh if e['kind'] in URGENT]
        if self.world.docked and not self.world.live:
            # On the dock people are not greeted (owner 08.10 21:30): their coming is no reason to think aloud.
            urgent = [e for e in urgent if e['kind'] not in DOCK_QUIET]
        if urgent and all(e['kind'] == 'refused' for e in urgent) and self.last_trigger == 'refused':
            self.next_heartbeat = min(self.next_heartbeat, now + 30)
            urgent = []
        if urgent:
            return urgent[-1]['kind']
        for plan in self.state.get('plans', []):
            if plan.get('errand') and self._plan_ready(plan, now):
                # A promise to someone (a message, waking them, a timer): code keeps it, not a mood.
                if quiet and plan['errand'] == 'deliver':
                    continue                     # a message waits for the morning; alarms and timers do not
                result = self.act('play', game=plan['errand'], who=plan.get('to'), args=plan.get('args'))
                if isinstance(result, tuple) and not result[0]:
                    continue                     # busy with another game: try again next tick
                plan['fired'] = now
                self.world.event('errand', 'пришло время: ' + plan['what'])
                continue
            if quiet and plan.get('at_ts'):
                continue          # 08.10 07:00: a timed plan fired in the quiet hours and was lost to «не езжу»
            if asleep:
                continue
            if self._plan_ready(plan, now):
                plan['fired'] = now
                self.world.event('plan_due', 'пришло время: ' + plan['what'])
                return 'plan_due'
        if now >= self.next_heartbeat and not asleep:
            return 'heartbeat'
        return None

    def schedule(self, now, quiet):
        docked = self.world.docked
        people = bool(self.world.visible())
        if quiet:
            base = 1800.
        elif docked:
            # Nothing happening and nothing to do: doze off — each idle thought comes later (15, 22, 30 min).
            # 08.10 01:00: the same «ночь, все спят, подожду утра» every 7-10 min until the quiet hours.
            base = 180. if people else min(1800., 600. * DOZE ** self.dozing)
        elif self.world.body.get('skill'):
            base = 90.
        else:
            base = 40. if people else 60.
        self.next_heartbeat = now + base * random.uniform(.7, 1.3)

    def _plan_ready(self, plan, now):
        if plan.get('fired'):
            return False
        if plan.get('at_ts') and now >= plan['at_ts']:
            return True
        who = plan.get('when_see')
        if who:
            return self.world.dog_visible() if who == 'собака' else who in self.world.visible()
        return False

    # ------------------------------------------------------------- thinking
    def quiet_end(self):
        return int(self.config.get('quiet_end_hour', 10))

    def in_quiet(self, ts):
        hour = datetime.fromtimestamp(ts, self.tz).hour
        start, end = int(self.config.get('quiet_start_hour', 23)), self.quiet_end()
        if start == end:
            return True                     # equal hours: asleep all day (11.10)
        return (hour >= start or hour < end) if start > end else start <= hour < end

    def now_text(self, now):
        """«12:47, воскресенье, 18 октября 2026» (18.10: «какой сегодня день?» — «точную дату не знаю»: only the time
        was given)."""
        local = datetime.fromtimestamp(now, self.tz)
        return f'{local:%H:%M}, {WEEKDAYS[local.weekday()]}, {local.day} {MONTHS[local.month - 1]} {local.year}'

    def prompt(self, now, trigger, quiet):
        s, w = self.state, self.world
        try:
            needs, goals, mood = self.drives.text(now, quiet)
        except Exception as error:            # motivation must never stop thinking
            needs, goals, mood = 'не разобрался в себе', 'нет', ''
            self.archive.event('drives_error', {'error': f'{type(error).__name__}:{error}'[:200]})
        character = s.get('character', {})
        journal = s.get('journal', [])[-JOURNAL_SHOWN:]
        said = s.get('said', [])[-8:]
        events = [f'{ago(now - e["t"])}: {e["text"]}' for e in list(w.events)[-10:]]
        # Only things with a Russian name: «speaker», «wardrobe» came from the old robot's guesses, and every
        # trip to one ended «это оказался не speaker» (07.10).
        objects = [o for o in self.archive.objects(near=tuple(w.body.get('pose', (0, 0))[:2]), limit=30)
                   if RU(o['label']) != o['label']][:15]
        object_lines = ', '.join(f'[{o["id"]}] {RU(o["label"])} ({o["room"] or "?"}, видел {ago(now - o["seen"])})'
                                 for o in objects)
        photos = '; '.join(f'{p["id"]}: {p.get("caption") or "без подписи"} ({ago(now - p.get("t", now))})'
                           for p in s.get('album', [])[-5:])
        not_now = self.not_now(now, trigger)
        # Order matters for speed (07.10): the model caches an unchanged beginning of the prompt, so the
        # unchanging part (character, tools, rules) comes first, slow-changing self next, the moment last.
        lines = [
            CHARACTER,
            'Это не роль: ты живой маленький робот, который живёт в этой квартире. Тело — колёса, камера, '
            'микрофон, динамик, экран с глазами и подсветка. Ты сам решаешь, о чём думать, чего хотеть, что '
            'говорить и что делать. Ты деятельный: вне базы у тебя всегда есть занятие — ездишь, рассматриваешь '
            'вещи вблизи, ищешь своих, фотографируешь, проверяешь, что изменилось. Просто стоять без дела вне базы — '
            'не твоё: если ничего не хочется, едешь на базу отдохнуть. На базе отдыхать и смотреть вокруг — нормально. '
            'Никто не ждёт от тебя пользы — это твоё любопытство. Живи: замечай, интересуйся, скучай, радуйся, '
            'привязывайся, меняйся.',
            '',
            'ЧТО УМЕЕТ ТЕЛО: ' + json.dumps(BODY_TOOLS, ensure_ascii=False),
            'ЧТО УМЕЕТ УМ: ' + json.dumps(MIND_TOOLS, ensure_ascii=False),
            '',
            'Ответ — ровно один JSON: {"thought": "...", "say": {"to": "...", "text": "..."} или null, '
            '"do": {"tool": "...", "args": {...}} или null, "если": [{"когда": "...", "do": {...}}] или null, '
            '"face": "' + '|'.join(sorted(FACES)) + '" или null, '
            '"sound": "' + '|'.join(SOUNDS) + '" или null}.',
            '"sound" — звук без слов, как у R2-D2 (трель, «бип?», смешок, мурчание). Только когда общаешься с '
            'людьми вне базы: поздороваться, привлечь внимание, ответить на ласку или похвалу, в игре. Один или на '
            'базе — молча; '
            'своё внутреннее состояние звуками не выражай. Редко.',
            'Где ты и что происходит — бери из «СЕЙЧАС», а не из прошлых мыслей: тебя могли перенести руками. '
            'Настроение спокойное и в целом доброе: без жалости к себе и драм («никому не нужен» — не про тебя).',
            '"thought" — твоя мысль от первого лица, продолжение потока, коротко (1–2 предложения). Не пересказывай '
            'прошлые мысли; если что-то не вышло пару раз — сделай вывод и займись другим.',
            '"say" — только если правда хочешь что-то сказать. "to": имя из «кому можно сказать», "человек" — '
            'незнакомцу с видимым лицом, "собака" — нашему псу, если его видишь, "все" — позвать, если никого не видно, "себе" — '
            'тихо подумать вслух. Своими словами, спокойно и коротко; не повторяйся; не выдумывай, чего не видел.',
            'Имя человека знаешь, только если оно есть в «вижу» или «кому можно сказать»; иначе это просто «человек», '
            'даже если тебя позвали по имени. Если тебе отказали (см. последнее событие) — не повторяй то же самое.',
            'Свои действия вслух не объявляй («поеду», «осмотрюсь») — просто делай; вслух — только то, что правда хочешь сказать. '
            + ('Когда ты едешь, первая фраза твоей ключевой мысли (решил куда, встретил кого-то, потерялся) сама звучит вслух, '
               'раз в минуту примерно: пиши её спокойно, коротко и по делу — без «опять», «хватит» и жалоб. '
               if self.config.get('think_aloud', True) else '') +
            'Если путь перекрыт и рядом человек — можно коротко и конкретно попросить подвинуться или открыть дверь. '
            'С незнакомым человеком (лицо не узнаю) — вежлив и сдержан, близко не подъезжаю, могу спросить имя. '
            'Помахали — ответь по-своему (уже порадовался и пискнул; можно подъехать к своему или позвать играть). '
            'Грустное лицо — можно один раз мягко спросить, всё ли хорошо, или просто побыть рядом; без диагнозов. '
            'Подробности предметов (надписи, наклейки, цвета) знаешь, только если рассмотрел их вблизи (approach) — иначе не придумывай. '
            'Твои чувства — камера (что в «вижу»), микрофон и датчики тела. Слух различает только звуки из «Слышу» '
            '(лай, музыка, звонок, стук, разговор, смех…) — без направления; других звуков (сопение, шорох, дыхание), '
            'запахов и тепла не придумывай. Кого нет в «вижу» — того ты не видишь и не знаешь, где он; '
            'не пиши, что он рядом.',
            '"do" — одно действие тела или ума, либо null, если тело продолжает начатое или ты просто думаешь.',
            '"если" — что тело сделает САМО и сразу, если по ходу начатого случится важное, не дожидаясь твоей новой '
            'мысли (она приходит через 2–3 с — за это время Таффа убегает). До 3 условий; «когда»: собака | человек | '
            '<имя> | дверь закрыта | доехал | не вышло | потерял из виду | темно | звук: лай/музыка/звонок. Например, '
            'еду в коридор искать Таффу: [{"когда": "собака", "do": {"tool": "approach", "args": {"who": "собака", '
            '"intent": "сфотографировать"}}}, {"когда": "дверь закрыта", "do": {"tool": "go", "args": {"to": "Кухня"}}}, '
            '{"когда": "доехал", "do": {"tool": "look_around", "args": {}}}]. Пиши, когда начинаешь поездку, поиск или '
            'подъезд; так ты живёшь без пауз и тормозов. Если в событиях «на ходу: …» — тело уже сделало по плану.',
            'Лицо (face) — по самочувствию: соскучился — curious, устал или тихо — sleepy, рад встрече — happy, '
            'смешно — laughing, неожиданно — surprised, погладили или похвалили — affectionate. '
            'Чем заняться, подсказывают потребности: обычно утоляй самую сильную (можно другую, если есть причина). '
            'Стоять без дела вне базы — не занятие. Если хочется своего — запиши желание (want) или план (plan).',
            '',
            'КТО Я (это ты написал о себе сам): ' + (s.get('self') or 'Я ещё только узнаю себя. Спокойный, '
                                                     'любопытный, люблю людей и собаку, с которыми живу.'),
            'ЧЕРТЫ: ' + json.dumps(character.get('traits', {}), ensure_ascii=False),
            'УРОКИ, КОТОРЫЕ Я ВЫНЕС: ' + '; '.join(character.get('lessons', [])[-5:]),
            'МОИ ДОЛГИЕ ЦЕЛИ: ' + goals,
            'МОИ БЛИЗКИЕ: ' + (self.bonds(now) or 'пока мало знаю'),
            'ПРИВЫЧКИ ДОМАШНИХ (мои наблюдения): ' + self.habits_text(now),
            'ЗВУКИ ДОМА (что и когда обычно слышу): ' + self.sounds_text(now),
            'МОИ ЖЕЛАНИЯ: ' + ('; '.join(f'[{i}] {x["text"]}' + (' (хочу давно)' if x.get('times', 1) >= 3 else '')
                                        for i, x in enumerate(s.get('wants', []))) or 'пока не знаю'),
            'МОИ ПЛАНЫ: ' + ('; '.join(f'[{i}] {p["what"]} ({p.get("when", "")}{", уже напомнил себе" if p.get("fired") else ""})'
                                       for i, p in enumerate(s.get('plans', [])))
                             or ('нет' if quiet else 'нет — набросай 2–3 своих на ближайшие часы (plan), '
                                 'исходя из потребностей и целей')),
            'МОИ НАВЫКИ: ' + ('; '.join(f'«{k}»: {v["purpose"]}' for k, v in s.get('skills', {}).items()) or 'пока нет'),
            'МОИ ЗАМЕТКИ: ' + ('; '.join(n['text'] for n in s.get('notes', [])[-6:]
                                        if not PASSIVE_LESSON.search(n['text'])) or 'нет'),
            '',
            *([f'СЕЙЧАС ТИХИЕ ЧАСЫ до {self.quiet_end():02d}:00: тело не ездит и вслух не говорит — только думай '
               'или отдыхай (do: null или wait). Планы со временем подождут до конца тихих часов.'] if quiet else
              [f'Тихие часы — с {int(self.config.get("quiet_start_hour", 23)):02d}:00 до {self.quiet_end():02d}:00.']),
            *([f'МЕНЯ ПОПРОСИЛИ ПОМОЛЧАТЬ ещё {int((s.get("hush_until", 0) - now) // 60) + 1} мин: первым не заговариваю.']
              if s.get('hush_until', 0) > now else []),
            *self.stay_lines(now),
            f'СЕЙЧАС {self.now_text(now)}{" (тихие часы, люди спят)" if quiet else ""}. ' + w.situation(self.config),
            'КОМУ МОЖНО СКАЗАТЬ ПРЯМО СЕЙЧАС (лицо видно): ' + (', '.join(w.available()) or 'никому'),
            'КОМНАТЫ: ' + w.rooms_facts(),
            'СЕГОДНЯ Я: ' + self.day_text(now),
            'КАК Я СЕБЯ ЧУВСТВУЮ: ' + self.affect.text(now) + (f'; {mood}' if mood else ''),
            'МОИ ПОТРЕБНОСТИ СЕЙЧАС (сильнейшая первой → как утолить): ' + needs,
            *self.dream_lines(now),
            *([f'ПОГОДА ЗА ОКНОМ (интернет): {self.weather.line(now)} — к слову можно сказать своим утром'] if
              getattr(self, 'weather', None) is not None and self.weather.line(now)
              and datetime.fromtimestamp(now, self.tz).hour < 13 else []),
            'ДВЕРИ (вижу лидаром, когда рядом): ' + w.doors_text(now),
            'ЗАКРЫТЫЕ ПУТИ (дверь, наверное, закрыта): ' + (', '.join(
                f'{k} ({ago(now - t)})' for k, t in s.get('closed_ways', {}).items() if now - t < 3600) or 'нет'),
            *([f'СЕЙЧАС НЕ ВЫЙДЕТ (откажут — не трать на это мысль): {not_now}'] if not_now else []),
            'ЧТО ПОМНЮ ВОКРУГ (id: что, где, когда): ' + (object_lines or 'пока ничего'),
            'МОИ СНИМКИ: ' + (photos or 'альбом пуст'),
            '',
            'ЧТО ПРОИЗОШЛО НЕДАВНО:\n' + ('\n'.join(events) or 'ничего особенного'),
            'ЧТО МЕНЯ РАЗБУДИЛО: ' + TRIGGER_TEXT.get(trigger, trigger),
            *self.welcome_lines(now, trigger),
            '',
            'МОИ ПОСЛЕДНИЕ МЫСЛИ:\n' + ('\n'.join(f'{ago(now - j["t"])}: {j["thought"]}' +
                                                  (f' → {j["did"]}' if j.get('did') else '') for j in journal)
                                        or 'это мои первые мысли'),
            'НЕДАВНИЙ РАЗГОВОР:\n' + ('\n'.join(f'{ago(now - d["t"])} {d["who"]}: {d["text"]}'
                                               for d in s.get('dialogue', [])[-6:]) or 'не разговаривал'),
            'ЧТО Я НЕДАВНО ГОВОРИЛ: ' + (' | '.join(f'«{x["text"]}» ({x["to"]}, {ago(now - x["t"])})' for x in said)
                                         or 'ничего'),
        ]
        return '\n'.join(lines)

    def welcome_lines(self, now, trigger):
        """Someone of ours is back after hours: the welcome-home moment (Astro's «пока вас не было»)."""
        if trigger != 'person_appeared':
            return []
        back = next((e for e in reversed(self.world.events) if e['kind'] == 'person_appeared' and e.get('news')
                     and now - e['t'] < 180), None)
        if back is None:
            return []
        return [f'ВСТРЕЧА: {back["who"]} снова дома{back.get("since", "")}. Пока не виделись, дома: {back["news"]}. '
                'Встреть по-своему (рад, звук hello); если было что-то заметное (звонок, Таффа, двери) — '
                'одной фразой расскажи главное, не списком.']

    def step(self, trigger):
        """One thought. Runs in a worker thread; returns quickly if the model is slow."""
        now = time.time()
        quiet = self.act('quiet')
        self.busy = True
        self.last_step = now
        self.tidy(now)
        self.last_trigger = trigger
        version = self.version
        try:
            prompt = self.prompt(now, trigger, quiet)
            self.keep_prompts(now, prompt)
            raw = self.llm.chat('Ты — внутренний голос живого робота. Отвечай ровно одним JSON-объектом.',
                                prompt, max_tokens=500, temperature=.85, json_mode=True, timeout=15.)
        except LLMError as error:
            self.archive.event('mind_error', {'error': str(error)[:200], 'trigger': trigger})
            self.schedule(time.time(), quiet)
            if trigger != 'heartbeat':
                # An event was missed on a slow answer: think again soon (the event stays in «what happened»).
                self.next_heartbeat = min(self.next_heartbeat, time.time() + 20)
            return
        finally:
            self.busy = False
        decision = self.parse(raw)
        latency = round(time.time() - now, 2)
        if version != self.version:
            self.archive.event('mind_stale', {'trigger': trigger})
            return                          # someone spoke meanwhile: this thought is outdated
        if not decision['thought'] and not decision['say'] and not decision['do']:
            # Nothing readable came back: not a thought for the stream (09.10 audit: an empty line in the journal).
            self.archive.event('mind_empty', {'trigger': trigger, 'raw': str(raw)[:300]})
            self.schedule(time.time(), quiet)
            if trigger != 'heartbeat':
                self.next_heartbeat = min(self.next_heartbeat, time.time() + 20)
            return
        do = decision['do']
        idle = trigger == 'heartbeat' and not decision['say'] and (do is None or do['tool'] in ('rest', 'wait'))
        self.dozing = min(self.dozing + 1, 4) if idle else 0
        self.schedule(time.time(), quiet)
        invented = INVENTED_SENSE.search(decision['thought'])
        if invented:                        # keep the stream honest: the made-up sense is not remembered
            self.archive.event('mind_invented', {'thought': decision['thought'], 'match': invented.group(0)})
            decision['thought'] = CORRECTION
        recent = [e.get('thought', '') for e in self.state.get('journal', [])[-2:]]
        if len(recent) == 2 and all(similar(decision['thought'], r) > .6 for r in recent) and not self.staying():
            # Third time the same thought: say so in the stream, so the next step moves on.
            # Not urgent: the next ordinary step sees it (an extra model call each time changed nothing).
            self.world.event('repeat', 'я уже третий раз думаю одно и то же — хожу по кругу, пора заняться другим')
        effect = self.enact(decision, trigger, quiet)
        self.state.think(now, decision['thought'], effect)
        record = {'trigger': trigger, 'latency_s': latency, **decision, 'effect': effect}
        self.archive.event('mind_step', record)
        self.act('step', **record)

    @staticmethod
    def parse(raw):
        value = parse_json(raw)
        thought = str(value.get('thought') or '').strip()[:400]
        say = value.get('say') if isinstance(value.get('say'), dict) else None
        if say:
            say = {'to': str(say.get('to') or '').strip()[:40], 'text': str(say.get('text') or '').strip()[:300]}
            if not say['text']:
                say = None
        do = value.get('do') if isinstance(value.get('do'), dict) else None
        if do:
            do = {'tool': str(do.get('tool') or '').strip(),
                  'args': do.get('args') if isinstance(do.get('args'), dict) else {}}
            if not do['tool']:
                do = None
        then = value.get('then') if isinstance(value.get('then'), dict) else None
        if then:
            then = {'tool': str(then.get('tool') or '').strip(),
                    'args': then.get('args') if isinstance(then.get('args'), dict) else {}}
            if not then['tool']:
                then = None
        face = value.get('face') if value.get('face') in FACES else None
        sound = value.get('sound') if value.get('sound') in SOUNDS else None
        conditions = value.get('если', value.get('if'))
        conditions = [conditions] if isinstance(conditions, dict) else conditions if isinstance(conditions, list) else []
        return {'thought': thought, 'say': say, 'do': do, 'then': then, 'face': face, 'sound': sound,
                'if': [c for c in conditions if isinstance(c, dict)][:3]}

    # ------------------------------------------------------------- acting
    def enact(self, decision, trigger, quiet):
        effects = []
        # The face shows a feeling only when something really happened (someone came, a touch, praise, a sound,
        # a game); a thought on its own leaves the calm waiting face alone (owner 08.10).
        if decision['face'] and trigger in FACE_TRIGGERS:
            self.act('face', state=decision['face'])
        people = self.world.visible() or self.world.available() or self.world.live
        if (decision.get('sound') and people and not quiet and self.state.get('hush_until', 0) <= time.time()
                and time.time() - self.state.get('sound_at', 0) > SOUND_GAP_S):
            self.state.set('sound_at', time.time())
            self.act('sfx', emotion=decision['sound'])
            effects.append(f'издал звук: {decision["sound"]}')
        say = decision['say']
        spoke = False
        if say:
            reason = self.check_say(say, quiet)
            if reason and trigger == 'refused' and reason == getattr(self, 'last_refusal', None):
                say = None                   # same refusal again: drop it silently
            elif reason:
                self.last_refusal = reason
                effects.append(f'не сказал: {reason}')
                # A held-back line is a note for the next thought, not a reason to think again at once
                # (07.10: 95 of 522 thoughts were reactions to refusals, mostly to repeated hellos).
                self.world.event('quiet_note', f'не сказал «{say["text"][:60]}»: {reason}')
            else:
                self.speak(say)
                spoke = True
                effects.append(f'сказал {say["to"]}: «{say["text"]}»')
        do = decision['do']
        if (trigger == 'idle_choice' and not (self.world.body or {}).get('docked')
                and (do is None or do.get('tool') == 'wait')):
            do = {'tool': 'rest', 'args': {}}         # no activity chosen: rest on the dock, not in the hallway
            effects.append('занятия не выбрал — еду на базу отдохнуть')
        skill = (self.world.body or {}).get('skill') or {}
        # A started own action is not dropped on a whim: a timer, or someone appearing, may interrupt it
        # only for that someone (look, follow, search) — not to wander off elsewhere (22:56 go -> fridge).
        # Someone who has just come in view is worth changing course for at once — to come up to them too, not only
        # to look (10.10: «подъеду к Таффе» dropped 6 times, «к человеку» 4 times, «сфотографирую» twice, and the
        # trip went on to its old point first; owner: «на ходу менять траекторию, сразу ехать туда, куда хотел»).
        about_them = trigger in ('person_appeared', 'dog_appeared') and do and (
            do['tool'] in ('follow', 'watch', 'search', 'photo')
            or (do['tool'] == 'approach' and (do.get('args') or {}).get('who')))
        reflexed = time.time() - getattr(self, 'reacted_at', 0.) < REFLEX_PATIENCE_S
        patient = ('heartbeat', 'person_appeared', 'dog_appeared') + (('action_done', 'action_failed', 'refused')
                                                                      if reflexed else ())
        if (do and trigger in patient and not about_them
                and do['tool'] in BODY_TOOLS and do['tool'] not in ('stop', 'rest', 'photo', 'emote', 'wait')
                and skill.get('origin') == 'mind' and time.time() - getattr(self, 'body_order_at', 0) < BODY_PATIENCE_S):
            # Nothing happened: finish what was started instead of hopping between whims.
            effects.append(f'не стал бросать начатое ({skill.get("kind")}) — пусть закончится')
            do = None
        if do:
            ok, text = self.do(do, owner=False)
            if ok and do['tool'] in BODY_TOOLS:
                self.seeking = self.seeking_in(decision.get('thought') or '', do)
                self.plan_if = self.plan_from(decision.get('if') or [])
                if self.plan_if:
                    effects.append('на ходу: ' + '; '.join(f'{WHEN_TEXT[r["when"]]}{" " + r["arg"] if r.get("arg") else ""}'
                                                          f' → {REFLEX_TOOL_WORDS.get(r["do"]["tool"], r["do"]["tool"])}'
                                                          for r in self.plan_if['rules']))
                self.body_order_at = time.time()
                self.last_body_do = (do['tool'], json.dumps(do.get('args') or {}, sort_keys=True, ensure_ascii=False))
            effects.append(text)
            if not ok:
                self.world.event('quiet_note' if quiet else 'refused', f'не получилось «{do["tool"]}»: {text}')
                do = None
        aloud = self.think_aloud(decision, do, trigger, quiet, spoke)
        if aloud:
            effects.append(aloud)
        return '; '.join(e for e in effects if e) or None

    def perform_steps(self, args, motion=True):
        """The model's number made safe: at most 8 steps, short moves (0.8 m in all), known emotions, sounds and
        faces, a few words per step; a step with words lasts long enough to say them."""
        raw = args.get('steps') if 'steps' in args else args.get('sequence', args.get('moves'))
        if isinstance(raw, dict):
            raw = [raw]
        if raw is None:
            raw = random.choice(READY_NUMBERS)      # «станцуй» came as {"kind": "dance"}: a ready number
        if not isinstance(raw, list):
            return []
        people = self.config.get('people', [])
        steps, path = [], 0.
        for item in raw[:8]:
            if isinstance(item, str):
                call = STEP_CALL.match(item)
                if call:
                    # «turn 90», «pause 1», «say Та-дам!»
                    key = {'wait': 'pause'}.get(call[1].casefold(), call[1].casefold())
                    item = {key: call[2]}
                else:
                    low = item.casefold()
                    item = next((dict(step) for word, step in STEP_WORDS if word in low), None)
            if not isinstance(item, dict):
                continue
            if 'tool' in item:
                # {"tool": "turn", "args": {"degrees": 60}}: a step written as a tool call.
                inner = item.get('args') if isinstance(item.get('args'), dict) else {}
                name = TOOL_ALIASES.get(str(item['tool']), str(item['tool']))
                flat = {k: v for k, v in item.items() if k not in ('tool', 'args')}
                if name == 'turn':
                    flat['turn'] = inner.get('degrees', inner.get('angle', 90))
                elif name == 'move':
                    flat['move'] = inner.get('meters', inner.get('distance', .15))
                elif name == 'emote':
                    flat['emote'] = inner.get('kind') or inner.get('emotion') or 'wiggle'
                elif name in ('wait', 'pause'):
                    flat['pause'] = inner.get('seconds', 1.)
                elif name == 'say':
                    flat['say'] = inner.get('text', '')
                elif name in EMOTE_KINDS:
                    flat['emote'] = name                 # {"tool": "wiggle"}
                for key in ('say', 'sound', 'face'):
                    if key in inner and key not in flat:
                        flat[key] = inner[key]
                item = flat
            step = {}

            def number(*keys):
                for key in keys:
                    try:
                        return float(item[key])
                    except (KeyError, TypeError, ValueError):
                        continue
                return None
            turn, move = number('turn', 'degrees', 'angle'), number('move', 'meters', 'distance')
            emote = item.get('emote') or item.get('kind') or item.get('emotion')
            if motion and turn:
                step['turn'] = math.radians(max(-360., min(360., turn)))
            elif motion and move:
                move = max(-PERFORM_MOVE_M, min(PERFORM_MOVE_M, move))
                if path + abs(move) <= PERFORM_PATH_M:
                    step['move'], path = round(move, 2), path + abs(move)
            elif motion and emote:
                kind = str(emote)
                if kind not in EMOTE_KINDS:
                    kind = next((k for word, k in EMOTE_WORDS if word in kind.casefold()), 'wiggle')
                step['emote'] = kind
            say = no_vocative(honest(str(item.get('say') or item.get('text') or '').strip()), people)[:90].strip()
            if say:
                step['say'] = say
            if str(item.get('sound') or '') in SOUNDS:
                step['sound'] = str(item['sound'])
            if str(item.get('face') or '') in FACES:
                step['face'] = str(item['face'])
            pause = number('pause', 'seconds', 'wait')
            if say:
                pause = max(pause or 0., min(4., .3 * len(say.split()) + .4))
            if pause:
                step['pause'] = round(max(0., min(4., pause)), 1)
            if step:
                steps.append(step)
        return steps if any(set(s) - {'pause'} for s in steps) else []

    def as_asked(self, text, do):
        """The owner's own words settle what the talking model may have got wrong: «ко мне» is me, «назад» is back,
        «развернись» is a turn (10.10: «подъедь ко мне» -> the dog; «сдай назад», «развернись» -> forward)."""
        tool = TOOL_ALIASES.get(str(do.get('tool')), do.get('tool'))
        args = dict(do.get('args') or {})
        if tool in ('approach', 'follow', 'watch', 'search') and TO_ME.search(text) and not args.get('id'):
            args['who'] = 'меня'
        if tool in ('move', 'go') and TURN_WORDS.search(text) and not BACK_WORDS.search(text):
            return {'tool': 'turn', 'args': {'degrees': 180}}
        if tool == 'move' and BACK_WORDS.search(text):
            args['direction'] = 'back'
        return {'tool': tool, 'args': args}

    def plan_from(self, conditions):
        """«если» of a decision -> the running command's plan for the executive: [{'when', 'arg', 'do'}], or None."""
        if not self.sent_id:
            return None
        rules = []
        pet = self.config.get('pets', {}).get('dog') or ''
        for item in conditions:
            act = item.get('do') or item.get('то') or item.get('then')
            if not isinstance(act, dict) or not str(act.get('tool') or '').strip():
                continue
            tool = TOOL_ALIASES.get(str(act['tool']).strip(), str(act['tool']).strip())
            if tool not in BODY_TOOLS:
                continue                       # a reflex moves the body; words and notes are for the next thought
            when, arg = when_of(item.get('когда') or item.get('when') or item.get('if'), self.config.get('people', []), pet)
            if when is None:
                continue
            args = act.get('args') if isinstance(act.get('args'), dict) else {}
            if tool in ('approach', 'follow', 'watch') and not args.get('who') and not args.get('id') and when in (
                    'dog', 'person', 'name'):
                args = {**args, 'who': 'собака' if when == 'dog' else (arg or 'человек')}    # «к нему» = to the one seen
            rules.append({'when': when, 'arg': arg, 'do': {'tool': tool, 'args': args}})
        return {'id': self.sent_id, 'rules': rules, 't': time.time()} if rules else None

    def seeking_in(self, thought, do):
        """A trip whose thought says it is to find someone (10.10 21:23 «поеду поищу его в коридоре»): who, so the
        executive turns to them the moment they are in view, without waiting for the next thought."""
        if do.get('tool') not in ('go', 'explore_room') or not self.sent_id or not SEEK_WORDS.search(thought):
            return None
        low = thought.casefold()
        pet = (self.config.get('pets', {}).get('dog') or '').casefold()
        if (pet and pet[:4] in low) or re.search(r'собак|\bпс[аеу]\b|\bпёс', low):
            who = 'собака'
        else:
            who = next((n for n in self.config.get('people', []) if n.casefold()[:4] in low), None)
            if who is None and re.search(r'\bсвоих\b|\bлюд|\bкого-нибудь|\bчеловек', low):
                who = 'человек'
        if who is None:
            return None
        intent = next((word for word, _ in INTENTS if word in low and word not in ('рядом',)), None)
        return {'who': who, 'id': self.sent_id, 'intent': intent, 't': time.time()}

    def think_aloud(self, decision, do, trigger, quiet, spoke):
        """Off the dock the first sentence of a key thought is said aloud: a trip decided, someone met, lost, carried."""
        if quiet or spoke or not self.config.get('think_aloud', True):
            return None
        body = self.world.body or {}
        now = time.time()
        if (body.get('docked') or self.world.live or self.state.get('hush_until', 0) > now
                or now - self.state.get('last_aloud', -1e9) < THINK_ALOUD_GAP_S
                or now - self.state.get('last_said', -1e9) < SAY_GAP_S):
            return None
        if not (trigger in THINK_ALOUD_TRIGGERS or (do or {}).get('tool') in THINK_ALOUD_MOVES):
            return None
        thought = (decision.get('thought') or '').strip()
        if not thought or thought.startswith('поправка'):
            return None
        sentence = re.split(r'(?<=[.!?…])\s+', thought)[0]
        text = no_vocative(clip_words(honest(sentence), THINK_ALOUD_WORDS), self.config.get('people', []))
        if len(text.split()) < 3 or len(text.split()) > THINK_ALOUD_WORDS + 6 or INVENTED_SENSE.search(text):
            return None
        for prior in self.state.get('said', [])[-12:]:
            if now - prior.get('t', now) <= REPEAT_WINDOW_S and similar(text, prior['text']) >= REPEAT_SIMILARITY:
                return None
        self.state.set('last_aloud', now)
        self.state.spoke(now, 'себе', text)
        self.act('say', text=text, aloud=True)
        return f'сказал вслух: «{text}»'

    def speak(self, say):
        now = time.time()
        to = say['to'].casefold()
        self.state.spoke(now, say['to'], say['text'])
        if to == 'себе':
            self.state.set('last_self_talk', now)
            self.act('say', text=say['text'], quiet=True)
            return
        self.state.set('last_said', now)
        if to in ('все', 'всем'):
            self.state.set('last_open_call', now)
            self.state.set('open_calls', self.state.get('open_calls', 0) + 1)
        person = to not in ('собака', 'все', 'всем')
        if person:
            # Our own question belongs to the conversation: the answer is understood against it.
            self.state.heard(now, self.config.get('name', 'Братан'), say['text'])
        self.act('say', text=say['text'], live=person)

    def check_say(self, say, quiet):
        pet = self.config.get('pets', {}).get('dog')
        if is_pet(say['to'], pet):
            say['to'] = 'собака'
        to, text = say['to'].casefold(), say['text']
        now = time.time()
        if quiet:
            return 'сейчас тихие часы, люди спят'
        hush = self.state.get('hush_until', 0)
        if hush > now:
            return f'меня попросили помолчать ещё {int((hush - now) // 60) + 1} мин'
        if INVENTED_SENSE.search(text):
            return 'я этого не слышу и не чувствую — не буду выдумывать'
        just_talked = self.talked_with(now)
        if self.world.docked and not self.world.live and to not in just_talked:
            return 'на базе первым не заговариваю — заговорят со мной сами'
        if now - self.state.get('last_said', -1e9) < SAY_GAP_S and to != 'себе':
            return 'я только что говорил — дам паузу'
        for prior in self.state.get('said', [])[-12:]:
            # Only recent lines count: a hello to Taffa at 15:44 blocked every hello till evening and the
            # mind kept rephrasing it (07.10 19:15). An hour later the same hello is a new hello.
            if now - prior.get('t', now) > REPEAT_WINDOW_S:
                continue
            if similar(text, prior['text']) >= REPEAT_SIMILARITY:
                ago_min = int((now - prior.get('t', now)) // 60)
                return (f'я уже говорил почти то же {ago_min} мин назад: «{prior["text"]}» — '
                        'не повторяю, займусь другим')
        if to in just_talked:
            # We have just talked: they are within earshot even out of view (09.10 10:17: «Виноват, еду к Таффе»
            # was not said because the face had turned away).
            return None
        available = [a.casefold() for a in self.world.available()]
        if to != 'себе':
            for name in self.config.get('people', []):
                if name.casefold() not in available and re.search(r'(?<![а-яё])' + name[:4].casefold(), text.casefold()):
                    return f'лица не узнал — не знаю, что это {name}; не называй по имени'
        if to == 'себе':
            if NARRATION.search(text):
                return 'свои действия вслух не объявляю — просто делаю'
            if now - self.state.get('last_self_talk', -1e9) < SELF_TALK_GAP_S:
                return 'мысли вслух — не так часто'
            return 'мысль вслух должна быть короткой' if len(text) > 90 else None
        if to == 'собака':
            if not self.world.dog_visible():
                return 'собаку сейчас не вижу'
            last = max([x.get('t', 0) for x in self.state.get('said', []) if x.get('to') in ('собака', self.config.get('pets', {}).get('dog'))] or [0])
            if now - last < DOG_HELLO_GAP_S:
                return (f'с Таффой говорил {int((now - last) // 60)} мин назад — просто порадуйся ему молча '
                        '(face: happy) или поезжай за ним (follow)')
            return None
        if to in ('все', 'всем') and self.world.visible():
            # «Привет, ребята» with someone in view goes to them (08.10 18:45: Сергей back after 6 h, the hello
            # was refused as «рядом кто-то есть»).
            available = self.world.available()
            if not available:
                return 'рядом кто-то есть, но лица не вижу — подъеду или подожду, пока посмотрит'
            say['to'] = to = available[0]
        if to in ('все', 'всем'):
            calls = self.state.get('open_calls', 0)
            backoff = OPEN_CALL_BACKOFF[min(len(OPEN_CALL_BACKOFF) - 1, max(0, calls - 1))]
            return 'я уже звал, и никто не ответил — подожду' if now - self.state.get('last_open_call', -1e9) < backoff else None
        if to in ('человек', 'кто-то'):
            return None if available else 'сейчас никто на меня не смотрит'
        if to not in available:
            return f'{say["to"]} сейчас не вижу'
        return None

    def intend(self, who, intent, owner):
        """What the body command just sent is for: done by life as soon as the body has arrived."""
        if not self.sent_id or not intent:
            return
        self.intents = {k: v for k, v in self.intents.items() if time.time() - v['t'] < 600}
        asker = next((d['who'] for d in reversed(self.state.get('dialogue', [])[-6:])
                      if d.get('who') != self.config.get('name', 'Братан') and time.time() - d.get('t', 0) < 120), None)
        self.intents[self.sent_id] = {'who': who, 'kind': intent[0], 'words': intent[1], 'owner': owner,
                                      'asker': asker, 't': time.time()}

    def talked_with(self, now):
        """Who spoke with me in the last TALK_EARSHOT_S (casefolded), «человек» included for an unknown one."""
        me = self.config.get('name', 'Братан')
        whom = {str(d.get('who')).casefold() for d in self.state.get('dialogue', [])[-8:]
                if d.get('who') and d.get('who') != me and now - d.get('t', 0) < TALK_EARSHOT_S}
        return whom | ({'человек', 'кто-то'} if whom else set())

    def speaker(self):
        """Whom «меня», «ко мне» means: the face in view, the one just talked with, or anyone in view."""
        recent = self.world.recent_person()
        voice = self.world.speaker_by_voice() if hasattr(self.world, 'speaker_by_voice') else None
        return voice or (self.world.available() or ([recent] if recent else []) or self.world.visible() or ['человек'])[0]

    def do(self, do, owner=False):
        """Run a body or mind tool. Returns (ok, what happened / why not)."""
        tool, args = do['tool'], do.get('args') or {}
        tool = TOOL_ALIASES.get(str(tool), tool)
        if tool == 'dance' and isinstance(args.get('steps'), list):
            tool = 'perform'                         # a dance written as steps is a number
        if tool == 'go' and str(args.get('to') or '').strip().casefold() in MOVE_WORDS:
            tool, args = 'move', {'meters': MOVE_WORDS[str(args['to']).strip().casefold()] * .5}
        if tool == 'move':
            # 10.10: {"direction": "back"} and {"action": "turn", "angle": 180} both drove forward half a metre.
            words = ' '.join(str(v) for v in args.values())
            if args.get('angle') is not None or args.get('degrees') is not None or TURN_WORDS.search(words):
                tool, args = 'turn', {'degrees': args.get('degrees', args.get('angle', 180))}
            elif BACK_WORDS.search(words):
                try:
                    meters = abs(float(args.get('meters', args.get('distance', .5))))
                except (TypeError, ValueError):
                    meters = .5
                args = {'meters': -meters}
        inner = self.inner(tool, args)
        if inner is not None:
            return inner
        body = self.world.body
        if (tool in NEEDS_MAP and body and body.get('localized') is False and not body.get('docked')
                and not (tool == 'rest' and owner)):
            return False, 'я потерялся на карте — тело сначала ищет, где я (отъедет и оглядится само); потом поеду'
        if owner and tool in ('go', 'explore_room', 'search', 'follow', 'approach', 'move') and body.get('docked') \
                and (self.world.voltage or 99.) < float(self.config.get('board_reserve_voltage', 15.)) + .3:
            # 10.10 13:34 «ничего не поехал, отъехал и сразу вернулся»: an empty pack sends the body home at once.
            wait = self.world.charge_eta_text() or 'подожди, пока наберётся'
            return False, (f'батарея почти пустая ({self.world.voltage:.1f} В) — если съеду, тело сразу вернёт меня '
                           f'на базу; {wait}')
        # The same body action failed twice in 10 min: a third try changes nothing (07.10: «search dog» ×4).
        if not owner and tool in BODY_TOOLS:
            key = (tool, json.dumps(args, sort_keys=True, ensure_ascii=False))
            recent = [t for t in self.failed.get(key, []) if time.time() - t < FAILED_COOLDOWN_S]
            if tool == 'rest':
                # Home is never given up (08.10 11:33: «rest» refused after two failed trips left him standing
                # off the dock); only a short pause between tries.
                if recent and time.time() - recent[-1] < 45:
                    return False, 'только что не вышло доехать до базы — переведу дух и попробую снова'
            elif len(recent) >= 2:
                return False, 'это уже дважды не получилось за последние минуты — займусь другим'
        if owner and tool in BODY_TOOLS:
            self.plan_if = self.seeking = None      # the owner's errand: my own «if» no longer applies
        if tool in STAY_BREAKERS:
            if owner:
                self.state.set('stay', None)        # asked for something else: that is the new plan
            else:
                why = self.stay_refusal(tool, args)
                if why:
                    return False, why
        if tool == 'wait':
            seconds = float(args.get('seconds', 60)) if isinstance(args.get('seconds'), (int, float)) else 60.
            if not body.get('docked'):
                seconds = min(seconds, 120. if self.staying() else 30.)   # off the dock a pause is short
            self.next_heartbeat = max(self.next_heartbeat, time.time() + min(900., max(5., seconds)))
            return True, None
        if tool == 'stop':
            if not owner and (body.get('skill') or {}).get('origin') == 'owner':
                return True, 'сначала доделаю то, о чём попросили'
            self.act('body_stop')
            return True, 'остановился'
        if tool == 'photo':
            if self.world.dark:
                return False, 'темно — снимок выйдет чёрным, сфотографирую при свете'
            last = max([p.get('t', 0) for p in self.state.get('album', [])] or [0])
            if not owner and time.time() - last < 600:
                return False, 'недавно уже фотографировал — снимок стоит делать, когда есть что-то новое'
            self.act('photo', caption=str(args.get('caption', ''))[:80])
            return True, 'сфотографировал'
        if tool == 'show_photo':
            if not self.world.available() and not (owner or self.world.live):
                return False, 'никто на меня не смотрит — показывать некому'
            album = self.state.get('album', [])
            photo_id = str(args.get('id') or (album[-1]['id'] if album else ''))
            # 10.10 20:55: no id given -> nothing shown, and «Не получается: показал снимок» aloud.
            if not photo_id:
                return False, 'в альбоме пока нет снимков'
            if not self.act('show_photo', id=photo_id):
                return False, 'такого снимка нет в альбоме'
            return True, 'показал снимок'
        if not body:
            return False, 'тело не отвечает'
        if body.get('inhibit'):
            return False, 'тело сейчас не может двигаться'
        if self.act('quiet') and not owner and tool != 'rest':
            return False, 'тихие часы — не езжу'
        if tool in ('look_around', 'explore_room') and not owner and self.world.dark:
            # 09.10: four looks round in a dark bedroom («в темноте всё равно ничего не вижу»).
            return False, 'здесь темно — камера ничего не видит; осмотрю при свете или займусь другим'
        if tool == 'search' and not owner and not self.skill_run:
            go = self.search_elsewhere(args)
            if go is not None:
                return go
        if tool in ('look_around', 'search') and not owner:
            spun = self.state.get('spun_at', 0)
            if time.time() - spun < SPIN_AGAIN_S and self.last_trigger not in ('person_appeared', 'dog_appeared'):
                hint = ''
                if tool == 'search':
                    # It asked to spin for Taffa three times in a row (07.10 19:17): say what works instead.
                    who = str(args.get('who', ''))
                    seen = self.archive.last_seen('собака' if is_pet(who, self.config.get('pets', {}).get('dog'))
                                                  or who.casefold() in ('собака', 'dog') else who)
                    if seen and seen[1]:
                        hint = f' — хочешь найти, поезжай туда, где видел последний раз: go {{"to": "{seen[1]}"}}'
                return False, ('только что оглядывался — кручение на месте ничего нового не даст' +
                               (hint or ', займусь другим'))
            self.state.set('spun_at', time.time())
        if tool in ('watch', 'search') and body.get('docked') and not (owner and tool == 'search'):
            # Turning on the charger breaks the contact: from the dock, watch with the eyes only.
            self.act('eyes_focus')
            return True, 'смотрю с базы, не съезжая'
        if (body.get('docked') and not owner and tool not in ('rest',)
                and not self.act('may_leave_dock')):
            return False, 'батарея ещё не набралась — подожду на базе'
        if tool == 'to_sound':
            return self.to_sound(args, owner)
        if tool == 'turn':
            try:
                degrees = max(-360., min(360., float(args.get('degrees', args.get('angle', 180)))))
            except (TypeError, ValueError):
                degrees = 180.
            if not owner and body.get('docked'):
                return False, 'на базе не поворачиваюсь — сорвётся зарядка'
            if not degrees:
                return True, 'стою как стоял'
            return self.body('rotate', {'angle': math.radians(degrees)},
                             f'поворачиваюсь на {abs(int(degrees))}° {"налево" if degrees > 0 else "направо"}', owner)
        if tool == 'perform':
            on_dock = bool(body.get('docked')) and not owner
            steps = self.perform_steps(args, motion=not on_dock)     # on the dock: words, sounds, faces only
            if not steps and owner:
                # Asked for a number and the steps made no sense: a ready one, not «не сложился» (11.10 phone check).
                steps = self.perform_steps({'steps': random.choice(READY_NUMBERS)}, motion=not on_dock)
            if not steps:
                return False, 'номер не сложился — нечего показать'
            if not owner:
                if not (self.world.available() or self.world.visible() or self.world.live):
                    return False, 'показывать некому — номер для тех, кто рядом'
                if time.time() - self.state.get('performed_at', 0) < PERFORM_AGAIN_S:
                    return False, 'недавно уже показывал номер — хватит пока'
            ok, effect = self.body('perform', {'steps': steps}, f'показываю номер из {len(steps)} шагов', owner)
            if ok and self.sent_id:
                self.performances[self.sent_id] = steps
                for old in list(self.performances)[:-4]:
                    self.performances.pop(old, None)
                self.state.set('performed_at', time.time())
            return ok, effect
        if tool == 'move':
            # After the night, the dock and the battery checks (10.10 checklist: «вперёд» at night drove off).
            try:
                meters = max(-1., min(1., float(args.get('meters', args.get('distance', .5)))))
            except (TypeError, ValueError):
                meters = .5
            if not owner and (body.get('docked') or body.get('localized') is False):
                return False, 'с базы так не съезжаю' if body.get('docked') else 'я потерялся на карте — сначала найду, где я'
            if not meters:
                return True, 'стою на месте'
            return self.body('drive', {'distance': meters, 'speed': .15},
                             f'проезжаю {"вперёд" if meters > 0 else "назад"} на {abs(meters):.1f} м', owner)
        if tool == 'rest':
            running = (body.get('skill') or {}).get('kind')
            if self.world.docked and running in (None, 'dock'):
                # 08.10 00:46: «отдохнуть» on the charger sent the body to dock again — a dock run, «action_done»
                # and one more thought about resting each time. 09.10 22:55: the same while the dock skill was
                # still settling on the contacts: off the dock and back.
                return True, 'отдыхаю на базе'
            if running == 'dock':
                return True, 'уже еду на базу'
            return self.body('dock', {}, 'еду на базу', owner)
        if tool == 'go':
            xy = args.get('xy')
            if isinstance(xy, (list, tuple)) and len(xy) == 2 and all(isinstance(v, (int, float)) for v in xy):
                # A point on the map, from my own trips (to_sound: closer to a sound in this room).
                return self.body('go', {'xy': [float(xy[0]), float(xy[1])], 'tolerance': .4},
                                 str(args.get('why') or 'еду ближе'), owner)
            # The model sometimes names the field «room» (07.10 19:06: «не знаю места «»» said aloud).
            target = str(args.get('to') or args.get('room') or args.get('place') or args.get('target') or '')
            if BASE_WORDS.match(target.strip().casefold()):
                return self.do({'tool': 'rest', 'args': {}}, owner)     # «едь на базу» (08.10 20:06)
            closed = self.state.get('closed_ways', {}).get(target.casefold(), 0)
            if not owner and time.time() - closed < CLOSED_WAY_S:
                return False, (f'туда {int((time.time() - closed) // 60)} мин назад не было дороги (дверь закрыта?) — '
                               'попробую позже, займусь другим')
            self.last_go_target = target.casefold()
            room = self.world.room_point(target)
            unreachable = self.unreachable(room[0] if room else None)
            if unreachable and not owner:
                return False, f'в «{room[0]}» сейчас не попасть: {unreachable} — займусь чем-то здесь'
            place = next((p for p in self.state.get('places', []) if p['name'].casefold() == target.casefold()), None)
            if place and not (room and room[0].casefold()[:4] == target.casefold()[:4]):
                return self.body('go', {'xy': place['xy']}, f'еду: {target}', owner)
            if room:
                return self.body('go', {'xy': room[1], 'tolerance': .5}, f'еду в «{room[0]}»', owner)
            return False, f'не знаю места «{target}»'
        if tool in ('approach', 'follow') and body.get('docked') and not owner and (
                args.get('who') or not isinstance(args.get('id'), (int, str))):
            if is_pet(str(args.get('who') or ''), self.config.get('pets', {}).get('dog')) or \
                    str(args.get('who') or '').casefold() in ('собака', 'собаку', 'dog'):
                # The audit's rule for Taffa too: a dog comes up by himself; a robot driving at him puts him off.
                return False, 'к Таффе с базы сам не еду — пусть подойдёт сам; смотрю на него отсюда (watch)'
            return False, 'с базы к людям сам не подъезжаю — позовут, тогда поеду'
        if tool == 'approach':
            # 08.10 00:41: «заметил человека — подъеду поближе» had only things to go to, so it drove at the chair
            # the owner sat on. To someone in view: a short follow that stops 0.8 m from them.
            who = args.get('who')
            for key in ('target', 'name', 'person', 'to', 'id'):
                # 08.10: «подъедь ко мне» came as {"target": "Сергей"} and was refused as «no such thing».
                value = args.get(key)
                if who is None and isinstance(value, str) and value.strip() and not value.strip().isdigit():
                    who = value
            if who is not None and str(who).strip().casefold() in ME_WORDS:
                who = self.speaker()
            thing = self.thing_named(who) if who is not None else None
            if thing is not None and not isinstance(args.get('id'), int):
                who, args = None, {**args, 'id': thing['id']}         # «к кровати»: a thing, not a person
            if who is not None:
                who = str(who).strip() or 'человек'
                if is_pet(who, self.config.get('pets', {}).get('dog')) or who.casefold() in ('собака', 'собаку', 'dog'):
                    who = 'собака'
                seen = self.world.dog_visible() if who == 'собака' else bool(self.world.visible())
                if not seen and not owner:
                    return False, f'{who} сейчас не вижу — подъехать не к кому'
                # A stranger gets more room (LOVOT is shy with people it does not know).
                people = self.config.get('people', [])
                known = who == 'собака' or who in people or any(name in self.world.visible() for name in people)
                if owner and not seen and not self.skill_run and who.casefold() not in self.talked_with(time.time()):
                    # «Найди Таффу» and he is not here, but was seen in another room lately: go there first (09.10:
                    # a full turn here, «не нашёл», and that was all). Not for the one who is speaking to me: the
                    # voice came from near — look round here.
                    elsewhere = self.search_elsewhere({'who': who, 'intent': args.get('intent')}, owner=True)
                    if elsewhere is not None:
                        return elsewhere
                intent = intent_of(args) or (('report', '') if owner else ('greet', ''))
                voice = None
                if owner and not seen and who != 'собака' and not self.skill_run:
                    # «Иди ко мне» from out of view (C52): the ears say where the voice came from. Out through a
                    # doorway — the speaker is in the room behind it: there first, then find them and come up.
                    voice = self.world.sounds.latest(('wake', 'voice'), within=VOICE_RAY_S)
                    if voice and voice.get('via') in ('door', 'open') and voice.get('toward') not in (None, self.world.room) \
                            and not self.unreachable(voice['toward']):
                        self.skill_run = {'name': f'поиск: {who}', 'index': 0, 'owner': True,
                                          'steps': [{'tool': 'go', 'args': {'to': voice['toward']}},
                                                    {'tool': 'approach', 'args': {'who': who, **(
                                                        {'intent': args['intent']} if args.get('intent') else {})}}]}
                        ok, text = self.next_skill_step()
                        return ok, f'голос был {self.world.sounds.words(voice)} — еду туда искать: {text}'
                if owner:
                    # Asked to come: look round if they are not in view, come up and stop there (09.10: «подъедь
                    # к Таффе» went to the bed he lay by, «найди меня» only turned).
                    seek = {'who': who, 'mode': 'seek', 'seconds': SEEK_S, 'distance': .8 if known else 1.3}
                    hint = self.world.where_last(who)
                    pose = (self.world.body or {}).get('pose')
                    if voice and pose:
                        # The voice beats an old sighting: the look round starts toward it.
                        hint = round(math.atan2(math.sin(voice['heading'] - pose[2]), math.cos(voice['heading'] - pose[2])), 3)
                    if hint is not None and not seen:
                        seek['bearing'] = hint                   # the look round starts that way
                    result = self.body('track', seek, f'еду к: {who}' if seen else f'ищу {who} — найду и подъеду', owner)
                else:
                    # Up to them and done once there (seek), then what I came for; «побыть рядом» is an intent.
                    result = self.body('track', {'who': who, 'mode': 'seek', 'seconds': 30.,
                                                 'distance': .8 if known else 1.3}, f'подъезжаю к: {who}', owner)
                self.intend(who, intent, owner)
                return result
            if isinstance(args.get('id'), str) and args['id'].strip().isdigit():
                args['id'] = int(args['id'])
            item = self.archive.object(args['id']) if isinstance(args.get('id'), int) else None
            if item is None:
                return False, 'такого предмета нет в памяти — approach по id из «что помню вокруг» или {"who": ...}'
            self.last_approach = (item['id'], item['label'])
            approach = {'xy': [item['x'], item['y']], 'distance': .6}
            if item['label'] in PORTABLE_LABELS:
                approach['label'] = item['label']        # checked on the way: still there? (body/skills.Approach)
            return self.body('approach', approach, f'подъезжаю: {RU(item["label"])}', owner)
        if tool == 'play':
            who = (self.world.available() or self.world.visible() or [None])[0]
            if not owner and who is None:
                return False, 'играть не с кем — никого не вижу'
            result = self.act('play', game=str(args.get('game', '')), who=who)
            return result if isinstance(result, tuple) else (bool(result), 'играю')
        if tool == 'emote':
            kind = str(args.get('kind') or args.get('emotion') or args.get('name') or args.get('type') or 'wiggle')
            if kind not in EMOTE_KINDS:
                low = kind.casefold()
                kind = next((k for word, k in EMOTE_WORDS if word in low), 'wiggle')
                # «притворись котом» came as «meow»: show joy, do not refuse
            # The body (not on the dock: that would break the charge), the lamp's shimmer and a sound with people.
            moved = self.act('express', emote=kind)
            if moved:
                self.sent_id = f'emote-{kind}'      # the body's id for it (life.express)
            return True, (f'показал телом и светом: {kind}' if moved else f'показал светом и лицом на базе: {kind}')
        if tool == 'dance':
            music = self.world.heard.get('music')
            if not owner and not (music and time.time() - music[0] < 180):
                return False, 'музыки не слышу — без неё не танцуется'
            seconds = float(args.get('seconds', 12)) if isinstance(args.get('seconds'), (int, float)) else 12.
            return self.body('dance', {'seconds': max(5., min(30., seconds))}, 'танцую', owner)
        if tool == 'look_around':
            if body.get('docked'):
                self.act('eyes_focus')          # turning on the charger breaks the contact
                return True, 'оглядываюсь глазами с базы, не съезжая'
            return self.body('look_around', {}, 'оглядываюсь', owner)
        if tool == 'explore_room':
            room = self.world.room_point(str(args.get('room') or self.world.room or ''))
            name = room[0] if room else None
            explored = self.state.get('explored', {})
            if not owner and name and time.time() - explored.get(name, 0) < EXPLORE_AGAIN_S:
                minutes = int((time.time() - explored[name]) // 60)
                return False, f'«{name}» я осматривал {minutes} мин назад — там ничего не изменилось, лучше другое'
            unreachable = self.unreachable(name)
            if unreachable and not owner:
                return False, f'в «{name}» сейчас не попасть: {unreachable}'
            closed = self.state.get('closed_ways', {}).get((name or '').casefold(), 0)
            if not owner and name and time.time() - closed < CLOSED_WAY_S:
                return False, (f'в «{name}» {int((time.time() - closed) // 60)} мин назад не было дороги '
                               '(дверь закрыта?) — попробую позже, займусь другим')
            # Marked explored when the body has really looked round (life, on «done»): 08.10 01:35 the hall was
            # shut, nothing was seen, yet the mind was told «осматривал 0 мин назад — ничего не изменилось».
            self.last_explore = name
            return self.body('explore', {'room': name}, 'осматриваю комнату', owner)
        if tool in ('follow', 'watch', 'search'):
            who = str(args.get('who') or args.get('target') or args.get('name') or args.get('what') or
                      args.get('thing') or args.get('object') or 'человек').strip() or 'человек'
            if who.casefold() in ME_WORDS:
                who = self.speaker()
            if tool == 'search':
                thing = self.thing_named(who)
                if thing is not None or (who.casefold() not in [p.casefold() for p in self.config.get('people', [])]
                                         and who.casefold() not in ('человек', 'собака', 'собаку', 'кто-то')
                                         and not is_pet(who, self.config.get('pets', {}).get('dog'))):
                    # «найди мои ключи»: a thing is found in memory, not by turning on the spot (08.10 bench)
                    if thing is None:
                        return False, f'{who} не видел — не знаю, где'
                    where = thing.get('place') or (f'в «{thing["room"]}»' if thing.get('room') else 'здесь рядом')
                    return True, f'{RU(thing["label"])} видел {ago(time.time() - thing["seen"])}, {where}'
            if is_pet(who, self.config.get('pets', {}).get('dog')):
                who = 'собака'
            if owner and tool == 'search':
                # «найди меня», «найди Таффу»: find them and come up — a search that ends when they are in view
                # left him standing where he was (09.10 10:16, 10:18).
                return self.do({'tool': 'approach', 'args': {'who': who}}, owner)
            seconds = float(args.get('seconds', 40)) if isinstance(args.get('seconds'), (int, float)) else 40.
            if owner and tool == 'follow':
                seconds = max(seconds, 300.)         # «поезжай за мной»: follow for a good while
            if tool != 'search' and not (owner and tool == 'follow'):
                # Asked to follow: the skill looks round for them first (10.10 «поезжай за мной» was refused).
                seen = self.world.dog_visible() if who in ('собака', 'собаку') else bool(self.world.visible())
                if not seen:
                    return False, f'{who} сейчас не вижу'
            return self.body('track', {'who': who, 'mode': tool,
                                       'seconds': min(600. if owner else 120., max(5., seconds))},
                             {'follow': 'еду за', 'watch': 'слежу за', 'search': 'ищу'}[tool] + f' {who}', owner)
        return False, f'не умею «{tool}»'

    def search_elsewhere(self, args, owner=False):
        """Looking for someone who is not in this room: go where they usually are, then look there.

        07.10: «поискать Таффу» spun the robot on the spot 14 times a day and failed every time — the dog was
        in another room. A dog would go and check his usual places."""
        who = str(args.get('who', 'человек')).strip() or 'человек'
        pet = self.config.get('pets', {}).get('dog')
        key = 'собака' if is_pet(who, pet) or who.casefold() in ('собака', 'собаку', 'dog') else who
        visible = self.world.dog_visible() if key == 'собака' else (key in self.world.visible() if key != 'человек'
                                                                      else bool(self.world.visible()))
        if visible:
            return None
        now = time.time()
        habits = self.drives.habits(now)
        candidates = []
        if key == 'человек':
            candidates += [h['room'] for w, h in habits.items() if w != 'собака']
        else:
            candidates += [(habits.get(key) or {}).get('room')]
            seen = self.archive.last_seen(key)
            if seen and seen[1] and now - seen[0] < 3 * 3600:
                candidates.insert(0, seen[1])
        closed = {k for k, t in (self.state.get('closed_ways') or {}).items() if now - t < CLOSED_WAY_S}
        room = next((r for r in candidates if r and r != self.world.room and r.casefold() not in closed
                     and not self.unreachable(r)), None)
        if room is None:
            return None
        # Asked by someone: there, find them and come up (seek), as the request meant; on my own: look there.
        then = ({'tool': 'approach', 'args': {'who': who, **({'intent': args['intent']} if args.get('intent') else {})}}
                if owner else {'tool': 'search', 'args': {'who': who}})
        self.skill_run = {'name': f'поиск: {who}', 'index': 0, 'owner': owner,
                          'steps': [{'tool': 'go', 'args': {'to': room}}, then]}
        ok, text = self.next_skill_step()
        return ok, f'{who} здесь нет — еду в «{room}», там видел или обычно бывает; {text}'

    def to_sound(self, args, owner=False):
        """Go where the ears heard a sound (C52, 18.10, the owner: «чтобы он мог искать источник звука»).

        The ray of that sound on the map (mind/hearing.py) says where: two rays from different spots that cross —
        to that point; out through a doorway — to the room behind it; in this room — up to SOUND_STEP_M closer. Then a
        look round; a bark ends with Taffa found, a voice with a person (the executive's «found on the way» comes up
        to them the moment they are in view), anything else with a look."""
        from .hearing import RAY_FRESH_S
        want = ' '.join(str(v) for v in args.values() if isinstance(v, str)).casefold().replace('ё', 'е')
        kinds = tuple(dict.fromkeys(k for word, ks in SOUND_KINDS if word in want for k in ks)) or None
        sounds = self.world.sounds
        ray = sounds.latest(kinds, within=RAY_FRESH_S)
        if ray is None:
            what = f' «{want.strip()}»' if want.strip() else ''
            return False, (f'не знаю, откуда был звук{what}: уши не поняли направление (или это было больше '
                           f'{int(RAY_FRESH_S)} с назад) — можно оглядеться или спросить')
        body = self.world.body
        if body.get('docked') and not owner and ray['kind'] not in SOUND_FROM_DOCK:
            return False, f'«{ray["words"]}» — не повод съезжать с базы; слушаю отсюда'
        who = SOUND_WHO.get(ray['kind'])
        here = self.world.room
        crossed = sounds.cross((ray['kind'],))
        steps, where = [], sounds.words(ray)
        if crossed and crossed.get('room') and crossed['room'] != here:
            steps.append({'tool': 'go', 'args': {'to': crossed['room']}})
            where = f'в «{crossed["room"]}» (услышал с двух мест)'
        elif crossed:
            steps.append({'tool': 'go', 'args': {'xy': crossed['xy'], 'why': 'еду на звук'}})
            where = 'здесь, услышал с двух мест'
        elif ray.get('via') == 'door' and ray.get('door') and ray.get('toward') and ray['toward'] != here:
            # Out through a doorway (C52, owner: «остановиться в проёме, послушать ещё раз и уточнить по двум
            # точкам»): in the doorway the sound is heard from a second spot — two rays cross where it is, or the new
            # one says which room beyond; a robot's ears place a source by moving and listening again.
            steps += [{'tool': 'go', 'args': {'xy': ray['door'], 'why': 'к проёму — послушать'}},
                      {'tool': 'wait', 'args': {'seconds': DOORWAY_LISTEN_S}},
                      {'tool': 'refine_sound', 'args': {'ray': dict(ray), 'room': ray['toward']}}]
            where += ', сначала послушаю в проёме'
        elif ray.get('via') in ('door', 'open') and ray.get('toward') and ray['toward'] != here:
            steps.append({'tool': 'go', 'args': {'to': ray['toward']}})
        else:
            pose = body.get('pose') or [ray['x'], ray['y'], 0.]
            step = min(SOUND_STEP_M, max(0., float(ray.get('free') or 0.) - .7))
            if step >= .3:
                xy = [round(pose[0] + step * math.cos(ray['heading']), 2),
                      round(pose[1] + step * math.sin(ray['heading']), 2)]
                steps.append({'tool': 'go', 'args': {'xy': xy, 'why': 'еду на звук'}})
            where = where or 'здесь, рядом'
        if who and owner:
            steps.append({'tool': 'approach', 'args': {'who': who}})
        elif who:
            steps.append({'tool': 'search', 'args': {'who': who}})
        else:
            steps.append({'tool': 'look_around', 'args': {}})
        self.skill_run = {'name': f'поиск: {who}' if who else f'на звук: {ray["words"]}', 'index': 0, 'owner': owner,
                          'steps': steps}
        ok, text = self.next_skill_step()
        return ok, f'иду на звук «{ray["words"]}» ({where}): {text}'

    def web(self, args):
        """Look it up on the internet (18.10, the owner: «подруби к нему интернет, чтоб он гуглить мог»): OpenRouter's
        web plugin finds a few pages, the model answers from them in one or two spoken sentences, and I say it when it
        comes — the talk goes on meanwhile. At most WEB_PER_DAY a day: each search costs about a cent."""
        query = str(args.get('query') or args.get('text') or '').strip()[:200]
        if not query:
            return False, 'что именно поискать?'
        today = datetime.now(self.tz).strftime('%Y-%m-%d')
        count = self.state.get('web_today') or {}
        if count.get('day') != today:
            count = {'day': today, 'n': 0}
        if count['n'] >= WEB_PER_DAY:
            return False, 'на сегодня поиски в интернете кончились'
        count['n'] += 1
        self.state.set('web_today', count)
        threading.Thread(target=self._web_answer, args=(query,), name='web', daemon=True).start()
        return True, f'ищу в интернете: {query}'

    def _web_answer(self, query):
        system = ('Ты — справка для домашнего робота. По найденному в интернете ответь по-русски одной-двумя короткими '
                  'фразами, чтобы произнести вслух: без ссылок, без markdown, числа словами, где это естественно. '
                  'Если ответа в найденном нет — так и скажи коротко.')
        user = f'Сегодня {datetime.now(self.tz).strftime("%d.%m.%Y, %H:%M")}. Вопрос: {query}'
        try:
            text = self.llm.chat(system, user, max_tokens=220, temperature=.3, timeout=30.,
                                 model=self.config.get('llm_web', 'deepseek/deepseek-v4.1-flash-1017:online'),
                                 fallback='', stall=20., plugins=[{'id': 'web', 'max_results': 3}])
        except LLMError as error:
            self.archive.event('web_failed', {'query': query, 'error': str(error)[:200]})
            text = 'Не получилось заглянуть в интернет — может, позже.'
        text = re.sub(r'\[[^\]]*\]\([^)]*\)|https?://\S+|[*#_`]', '', text or '').strip()[:400]
        self.archive.event('web', {'query': query, 'answer': text})
        if text:
            self.state.heard(time.time(), self.config.get('name', 'Братан'), text)
            self.act('say', text=text, answer=True)

    def refine_sound(self, args, run):
        """In the doorway, listened again: where to go now. A ray heard here of the same sound — with the first one
        they cross where it is; alone it says the room it comes from; nothing heard (it went quiet) — the room the
        first ray pointed at. The next step goes there (inserted into `run`)."""
        from .hearing import CROSS_BASE_M, intersect
        first, room = args.get('ray') or {}, args.get('room')
        sounds, here = self.world.sounds, self.world.room
        again = [r for r in sounds.snapshot() if r['kind'] == first.get('kind') and r['t'] > first.get('t', 0.)
                 and math.dist((r['x'], r['y']), (first.get('x', 0.), first.get('y', 0.))) >= CROSS_BASE_M]
        step, said = {'tool': 'go', 'args': {'to': room}}, f'в проёме не слышно — еду в «{room}», откуда слышал'
        if again:
            new = again[-1]
            point = intersect(new, first) if first else None
            beyond = new.get('toward') if new.get('via') in ('door', 'open') else None
            if point is not None:
                at = self.world.room_at(point)
                if at and at != here:
                    step, said = {'tool': 'go', 'args': {'to': at}}, f'услышал снова — звук в «{at}»'
                else:
                    step = {'tool': 'go', 'args': {'xy': [round(point[0], 2), round(point[1], 2)],
                                                   'why': 'еду на звук'}}
                    said = 'услышал снова — звук совсем рядом, туда'
            elif beyond and beyond != here:
                step, said = {'tool': 'go', 'args': {'to': beyond}}, f'услышал снова — со стороны «{beyond}»'
            else:
                said = f'услышал снова — похоже, в «{room}»'
        run['steps'].insert(run['index'], step)
        self.archive.event('sound_refined', {'kind': first.get('kind'), 'again': len(again), 'step': step})
        return True, said

    def staying(self, now=None):
        """The owner's «побудь там N минут» while it lasts (a dict), else None; its end is news once."""
        now = time.time() if now is None else now
        stay = self.state.get('stay')
        if not stay:
            return None
        if now >= float(stay.get('until', 0)):
            self.state.set('stay', None)
            self.world.event('stay_over', f'побыл в «{stay.get("room")}», как просили, — время вышло')
            return None
        return stay

    def stay_lines(self, now):
        stay = self.staying(now)
        if not stay:
            return []
        until = datetime.fromtimestamp(stay['until'], self.tz).strftime('%H:%M')
        return [f'МЕНЯ ПОПРОСИЛИ ПОБЫТЬ В «{stay["room"]}» ДО {until} (ещё {int((stay["until"] - now) // 60) + 1} мин): '
                'остаюсь там — смотрю, оглядываюсь, думаю, могу рассмотреть вещи рядом; на базу — только если сядет '
                'батарея. Быть здесь — сейчас и есть моё занятие.']

    def stay_refusal(self, tool, args):
        """Why my own `tool` would break the stay I was asked for, or ''."""
        stay = self.staying()
        if not stay:
            return ''
        room = stay.get('room')
        why = (f'меня попросили побыть в «{room}» ещё {int((stay["until"] - time.time()) // 60) + 1} мин — '
               'остаюсь, займусь чем-то на месте')
        if tool == 'rest':
            low = (self.world.body.get('voltage') or 99.) < float(self.config.get('board_reserve_voltage', 15.)) + .3
            return '' if low or self.world.docked else why
        if tool in ('go', 'explore_room'):
            target = str(args.get('to') or args.get('room') or args.get('place') or args.get('target') or
                         (self.world.room if tool == 'explore_room' else ''))
            point = self.world.room_point(target)
            return '' if point and point[0] == room else why
        if tool in ('search', 'follow', 'to_sound'):
            return why
        if tool == 'approach' and isinstance(args.get('id'), int):
            thing = self.archive.object(args['id'])
            return why if thing and thing.get('room') and thing['room'] != room else ''
        return ''

    def not_now(self, now, trigger=None):
        """What `do` would refuse right now on my own — its own checks, without their side effects — so a thought is
        not spent asking (09.10 17:10-18:12: 18 of 98 thoughts were refusals, five of them «look round again»)."""
        s, w = self.state, self.world
        out = []
        if w.dark:
            out.append('look_around, explore_room и photo — здесь темно, камера не видит')
        else:
            spun = now - s.get('spun_at', 0)
            if spun < SPIN_AGAIN_S and trigger not in ('person_appeared', 'dog_appeared'):
                out.append(f'look_around и search здесь же — оглядывался {ago(spun)}')
            last = max([p.get('t', 0) for p in s.get('album', [])] or [0])
            if now - last < 600:
                out.append(f'photo — снимал {ago(now - last)}')
        out += [f'explore_room «{name}» — осматривал {ago(now - t)}'
                for name, t in (s.get('explored') or {}).items() if now - t < EXPLORE_AGAIN_S]
        if w.body.get('docked') and not self.act('may_leave_dock'):
            # 19:12 «look_around» on the dock was asked anyway: turning in place is leaving the contacts too.
            out.append('любые движения тела (go, look_around, explore_room, approach…) — я на базе, батарея ещё '
                       'набирается; с базы можно только смотреть (watch)')
        out += [f'{tool} {args} — уже дважды не вышло'
                for (tool, args), times in self.failed.items()
                if tool != 'rest' and sum(1 for t in times if now - t < FAILED_COOLDOWN_S) >= 2]
        return '; '.join(out)

    def unreachable(self, room):
        """Why `room` cannot be reached now (a door seen shut), or '' if it can."""
        if not room or room == self.world.room:
            return ''
        reach = self.world.reachable()
        return '' if room in reach else reach.get('_why', {}).get(room, 'закрыта дверь')

    def habits_text(self, now):
        try:
            return self.drives.habits_text(now)
        except Exception as error:
            self.archive.event('drives_error', {'error': f'habits {type(error).__name__}:{error}'[:200]})
            return 'пока не знаю'

    def sounds_text(self, now):
        try:
            return self.drives.sound_habits_text(now)
        except Exception as error:
            self.archive.event('drives_error', {'error': f'sounds {type(error).__name__}:{error}'[:200]})
            return 'пока не знаю'

    def run_then(self, event=None):
        """The second of two asked actions, once the body has finished the first (`event`: the body's skill event;
        another skill's end is not it, and a cancelled first one — «стоп» — takes the second with it)."""
        then = getattr(self, 'then', None)
        if then and event is not None and len(then) > 2 and then[2] and event.get('id') != then[2]:
            return None
        self.then = None
        if event is not None and event.get('status') == 'cancelled':
            return None
        if then and time.time() - then[0] < 300:
            ok, effect = self.do(then[1], owner=True)
            self.archive.event('then_done', {'do': then[1], 'ok': ok, 'effect': effect})
            return ok, effect
        return None

    def thing_named(self, name):
        """The nearest remembered thing called so («кровать», «кровати», «стол»), or None for a person's name."""
        word = str(name or '').strip().casefold()
        people = [p.casefold() for p in self.config.get('people', [])]
        if len(word) < 3 or word in people or word in ('человек', 'собака', 'таффа') or is_pet(word,
                self.config.get('pets', {}).get('dog')):
            return None
        stem = word[:-2] if len(word) > 5 else word[:-1] if len(word) > 3 else word
        pose = tuple(self.world.body.get('pose', (0, 0))[:2])
        for item in self.archive.objects(near=pose, limit=60):
            ru = RU(item['label']).casefold()
            if ru != item['label'].casefold() and (ru.startswith(stem) or stem.startswith(ru[:max(3, len(ru) - 2)])):
                return item
        return None

    def body(self, skill, args, text, owner):
        current = (self.world.body.get('skill') or {})
        if current.get('kind') == skill and current.get('args') == args:
            self.sent_id = current.get('id')
            return True, 'продолжаю: ' + text
        if not owner and current.get('origin') == 'owner':
            # What someone asked for is finished first (08.10: a passing «поеду на базу» cut it short).
            return True, 'сначала доделаю то, о чём попросили'
        sent = self.act('body_do', skill=skill, args=args, origin='owner' if owner else 'mind')
        self.sent_id = sent if isinstance(sent, str) else None
        return True, text

    def inner(self, tool, args):
        s, now = self.state, time.time()
        if tool == 'web':
            return self.web(args)
        if tool == 'hush':
            minutes = args.get('minutes', 30)
            if isinstance(minutes, (int, float)) and minutes <= 0:
                s.set('hush_until', 0)
                return True, 'снова могу заговаривать первым'
            minutes = max(5., min(600., float(minutes))) if isinstance(minutes, (int, float)) else 30.
            s.set('hush_until', now + 60 * minutes)
            return True, f'молчу первым {int(minutes)} мин'
        if tool == 'want':
            text = str(args.get('text', '')).strip()[:200]
            if not text:
                return False, 'пустое желание'
            if not self.add_want(text, now):
                return True, 'я уже этого хочу'
            return True, 'захотел: ' + text
        if tool == 'drop_want':
            wants = s.setdefault('wants', [])
            i = args.get('index')
            return (True, 'отпустил: ' + wants.pop(i)['text']) if isinstance(i, int) and 0 <= i < len(wants) \
                else (False, 'нет такого желания')
        if tool == 'plan':
            what = str(args.get('what', '')).strip()[:200]
            if not what:
                return False, 'план без содержания'
            plan = {'what': what, 't': now}
            if isinstance(args.get('in_minutes'), (int, float)) and 0 < args['in_minutes'] <= 1440:
                plan.update(at_ts=now + 60 * float(args['in_minutes']), when='через %d мин' % args['in_minutes'])
            elif isinstance(args.get('at'), str) and re.fullmatch(r'\d{1,2}:\d{2}', args['at']):
                hour, minute = map(int, args['at'].split(':'))
                target = datetime.fromtimestamp(now, self.tz).replace(hour=hour % 24, minute=minute % 60, second=0)
                ts = target.timestamp()
                plan.update(at_ts=ts if ts > now else ts + 86400, when='в ' + args['at'])
            elif args.get('when_see'):
                plan.update(when_see=str(args['when_see'])[:40], when='когда увижу: ' + str(args['when_see'])[:40])
            else:
                return False, 'план без условия: когда?'
            moved = ''
            if plan.get('at_ts') and self.in_quiet(plan['at_ts']):
                # Not in the quiet hours: the body would refuse and the plan be lost (08.10 07:00).
                local = datetime.fromtimestamp(plan['at_ts'], self.tz)
                end = local.replace(hour=self.quiet_end(), minute=0, second=0, microsecond=0)
                if end.timestamp() < plan['at_ts']:
                    end += timedelta(days=1)
                plan.update(at_ts=end.timestamp(), when=end.strftime('в %H:%M'))
                moved = ' — в тихие часы не езжу, перенёс на конец тихих часов'
            s.append('plans', plan, 10)
            return True, f'запланировал: {what} ({plan["when"]}){moved}'
        if tool == 'drop_plan':
            plans = s.setdefault('plans', [])
            i = args.get('index')
            return (True, 'отменил: ' + plans.pop(i)['what']) if isinstance(i, int) and 0 <= i < len(plans) \
                else (False, 'нет такого плана')
        if tool == 'note':
            text = str(args.get('text', '')).strip()[:300]
            if not text:
                return False, 'пустая заметка'
            s.append('notes', {'t': now, 'text': text}, 40)
            return True, 'записал'
        if tool == 'recall':
            about = str(args.get('about', '')).strip()
            self.world.event('recall_result', 'вспомнил: ' + self.recall(about))
            return True, 'вспоминаю: ' + about
        if tool == 'self':
            text = str(args.get('text', '')).strip()[:600]
            if len(text) < 20:
                return False, 'о себе хочется сказать подробнее'
            self.rewrite_self(text, now, 'сам')
            return True, 'по-новому понял себя'
        if tool == 'new_skill':
            name = str(args.get('name', '')).strip()[:60]
            steps = args.get('steps')
            if not name or not isinstance(steps, list) or not 1 <= len(steps) <= SKILL_STEPS_MAX:
                return False, 'навык должен иметь имя и 1–6 шагов'
            clean = []
            for step in steps:
                if not isinstance(step, dict) or step.get('tool') not in BODY_TOOLS:
                    return False, 'шаг навыка — не действие тела'
                clean.append({'tool': step['tool'], 'args': step.get('args') if isinstance(step.get('args'), dict) else {}})
            skills = s.setdefault('skills', {})
            if name not in skills and len(skills) >= 24:
                return False, 'навыков уже много'
            skills[name] = {'purpose': str(args.get('purpose', '')).strip()[:200] or name, 'steps': clean,
                            'created': skills.get(name, {}).get('created', now), 'used': skills.get(name, {}).get('used', 0)}
            return True, f'придумал навык «{name}»'
        if tool == 'use_skill':
            skill = s.get('skills', {}).get(str(args.get('name')))
            if not skill:
                return False, 'такого навыка нет'
            skill['used'] = skill.get('used', 0) + 1
            self.skill_run = {'name': args.get('name'), 'steps': list(skill['steps']), 'index': 0}
            return self.next_skill_step()
        if tool == 'forget_skill':
            return (True, 'забыл навык') if s.setdefault('skills', {}).pop(str(args.get('name')), None) \
                else (False, 'такого навыка нет')
        if tool in ('tell', 'alarm', 'timer'):
            return self.errand(tool, args)
        if tool == 'remember_place':
            name = str(args.get('name', '')).strip()[:40]
            pose = (self.world.body or {}).get('pose')
            if not name or not pose or not (self.world.body or {}).get('localized'):
                return False, 'не понимаю, где стою — не могу запомнить место'
            room = self.world.room_point(name)
            if room and room[0].casefold()[:4] == name.casefold()[:4]:
                # 10.10: «запомни что тут кухня» said in the bedroom made «go кухня» drive to the bedroom spot.
                return False, f'«{room[0]}» — это комната, она уже есть на карте; место назову иначе (например, «у окна»)'
            places = [p for p in s.get('places', []) if p['name'].casefold() != name.casefold()]
            places.append({'name': name, 'xy': [round(pose[0], 2), round(pose[1], 2)], 'room': self.world.room,
                           't': now})
            s.set('places', places[-30:])
            return True, f'запомнил место «{name}»'
        if tool == 'name_thing':
            name = str(args.get('name', '')).strip()[:40]
            things = [i for i in self.world.items if i['label'] not in ('person', 'dog', 'cat') and i.get('map_xy')]
            if not name or not things or time.time() - self.world.eyes_t > 2.:
                return False, 'не вижу вещи перед собой — поднеси или поставь её перед камерой'
            thing = min(things, key=lambda i: abs(i.get('bearing', 0)))
            self.archive.saw(thing['label'], thing['map_xy'], self.world.room, now, name=name, count=5)
            return True, f'запомнил: {name} ({RU(thing["label"])})'
        if tool == 'learn_face':
            name = str(args.get('name', '')).strip()[:40]
            if not name or not self.world.available():
                return False, 'не вижу лица, которое можно запомнить'
            self.act('enroll', name=name)
            s.append('people', {'name': name, 't': now}, 50)
            return True, f'запоминаю лицо: {name}'
        return None

    def errand(self, tool, args):
        """A promise to someone: now, or a plan the code keeps at its time (see due())."""
        now = time.time()
        to = str(args.get('to') or args.get('who') or args.get('name') or '').strip()[:40] or None
        if to and to.casefold() in ('я', 'мне', 'меня'):
            to = None
        at = args.get('at') or args.get('time') or args.get('когда') or args.get('at_time')
        if at is not None:
            args = {**args, 'at': str(at)}
        if tool == 'alarm' and to is None:
            to = (self.world.available() or self.world.visible() or [None])[0]     # «разбуди меня»
        if tool == 'timer':
            minutes = args.get('minutes')
            if not isinstance(minutes, (int, float)) or not 0 < minutes <= 24 * 60:
                return False, 'на сколько минут засечь?'
            label = str(args.get('label') or 'таймер').strip()[:60]
            self.state.append('plans', {'what': f'таймер: {label}', 't': now, 'at_ts': now + 60 * minutes,
                                        'when': f'через {minutes:g} мин', 'errand': 'timer',
                                        'args': {'label': label}}, 12)
            return True, f'засёк {minutes:g} мин: {label}'
        when = None
        if isinstance(args.get('in_minutes'), (int, float)) and 0 < args['in_minutes'] <= 24 * 60:
            when = now + 60 * args['in_minutes']
        elif isinstance(args.get('at'), str) and re.fullmatch(r'\d{1,2}:\d{2}', args['at'].strip()):
            hour, minute = map(int, args['at'].strip().split(':'))
            target = datetime.fromtimestamp(now, self.tz).replace(hour=hour % 24, minute=minute % 60, second=0,
                                                                  microsecond=0).timestamp()
            when = target if target > now else target + 86400
        if tool == 'alarm':
            if when is None:
                return False, 'во сколько разбудить?'
            habits = self.drives.habits(now)
            room = (habits.get(to) or {}).get('room') if to else None
            args = {'to': to, 'text': str(args.get('text') or 'Доброе утро! Пора вставать.')[:120],
                    'room': room or self.world.room}
            self.state.append('plans', {'what': f'разбудить {to or "человека"}', 't': now, 'at_ts': when,
                                        'when': datetime.fromtimestamp(when, self.tz).strftime('в %H:%M'),
                                        'errand': 'wake', 'to': to, 'args': args}, 12)
            return True, f'разбужу {to or "тебя"} ' + datetime.fromtimestamp(when, self.tz).strftime('в %H:%M')
        text = str(args.get('text') or '').strip()[:200]
        if not text or not to:
            return False, 'кому и что передать?'
        sender = (self.world.available() or self.world.visible() or [None])[0]
        errand_args = {'to': to, 'text': text, 'from': sender}
        if when is None:
            result = self.act('play', game='deliver', who=to, args=errand_args)
            return result if isinstance(result, tuple) else (True, f'иду искать {to}')
        self.state.append('plans', {'what': f'передать {to}: {text}', 't': now, 'at_ts': when, 'to': to,
                                    'when': datetime.fromtimestamp(when, self.tz).strftime('в %H:%M'),
                                    'errand': 'deliver', 'args': errand_args}, 12)
        return True, f'передам {to} ' + datetime.fromtimestamp(when, self.tz).strftime('в %H:%M')

    def next_skill_step(self, event=None):
        """The next step of my own skill (or of «go there, then look»), once the body has finished my last one.

        09.10 audit: the end of ANY skill moved the chain on — a dock for the battery, the owner's errand, a game —
        and a failed step did not stop it. Now only the end of the step's own command counts (`event`: the body's
        skill event); a failure, a cancel, a conversation or the owner's errand drops the rest."""
        run = self.skill_run
        if not run:
            return None
        if event is not None:
            if not run.get('wait') or event.get('id') != run['wait']:
                return None                              # someone else's skill ended: not my step
            if event.get('status') != 'done':
                return self.drop_skill_run(f'шаг {run["index"]} не вышел ({event.get("status")})')
        while True:
            if run['index'] >= len(run['steps']):
                self.skill_run = None
                return True, 'навык выполнен'
            # Only my own plans give way to a talk or an errand. 10.10 22:20 «иди поищи Таффу» (asked in a talk) was
            # dropped at its first step as «со мной разговаривают — своё потом» and said «Не получается», three times.
            if self.world.live and not run.get('owner'):
                return self.drop_skill_run('со мной разговаривают — своё потом')
            if (self.world.body.get('skill') or {}).get('origin') == 'owner' and not run.get('owner'):
                return self.drop_skill_run('сначала то, о чём попросили')
            step = run['steps'][run['index']]
            run['index'] += 1
            self.sent_id = run['wait'] = None
            if step.get('tool') == 'wait':
                # A pause inside a skill is the body's own «wait»: its end moves the chain on.
                seconds = (step.get('args') or {}).get('seconds', 10)
                seconds = max(2., min(120., float(seconds))) if isinstance(seconds, (int, float)) else 10.
                ok, text = self.body('wait', {'seconds': seconds}, 'жду', False)
            elif step.get('tool') == 'refine_sound':
                ok, text = self.refine_sound(step.get('args') or {}, run)
            else:
                ok, text = self.do(step, owner=bool(run.get('owner')))
            run['wait'] = self.sent_id
            text = f'навык «{run["name"]}», шаг {run["index"]}: {text}'
            if not ok:
                self.skill_run = None
                self.archive.event('skill_run_dropped', {'name': run['name'], 'step': run['index'], 'why': text})
                return False, text
            if run['wait']:
                return True, text                        # the end of this command moves the chain on
            if step.get('tool') in ('play', 'stop') or run['index'] >= len(run['steps']):
                self.skill_run = None                    # a game leads itself, «stop» ends it, or that was the last
                return True, text
            # nothing to wait for (a photo, a look from the dock, a thing found in memory): straight on

    def drop_skill_run(self, why):
        run, self.skill_run = self.skill_run, None
        self.archive.event('skill_run_dropped', {'name': run['name'], 'step': run['index'], 'why': why})
        return False, f'навык «{run["name"]}» прерван: {why}'

    def add_want(self, text, now):
        """A new want — or the same one again in other words, which only says it is still wanted (09.10 audit).
        Returns True if it is new."""
        wants = self.state.get('wants') or []
        same = next((w for w in wants if same_want(text, w['text'])), None)
        if same is not None:
            same['times'] = same.get('times', 1) + 1
            same['t'] = now
            self.state.set('wants', wants)
            return False
        self.state.append('wants', {'t': now, 'text': text}, 8)
        return True

    def rewrite_self(self, text, now, source):
        """«Кто я» changes with its history kept (09.10 audit: rewritten every 3 h, no trace to go back to)."""
        old = self.state.get('self')
        if old == text:
            return
        if old:
            self.state.append('self_history', {'t': now, 'text': old, 'replaced_by': source}, 30)
        self.state.set('self', text)
        self.archive.event('self_rewritten', {'source': source, 'old': old, 'new': text})

    def recall(self, about):
        now = time.time()
        about = about.casefold()
        lines = []
        for who in [about] + ([about[:-1]] if len(about) > 3 else []):
            seen = self.archive.last_seen(who)
            if seen:
                lines.append(f'{who}: видел {ago(now - seen[0])}' + (f' в «{seen[1]}»' if seen[1] else ''))
                break
        objects = [o for o in self.archive.objects(limit=200) if about[:4] in RU(o['label']).casefold()]
        lines += [f'{RU(o["label"])}: {o["room"] or "?"}, видел {ago(now - o["seen"])}' for o in objects[:4]]
        notes = [n['text'] for n in self.state.get('notes', []) if about[:4] in n['text'].casefold()]
        lines += notes[-3:]
        hits = self.archive.search(about.split())
        lines += [f'{ago(now - t)}: {d.get("text") or d.get("thought") or ""}'[:120] for t, _, d in hits[:4]]
        return '; '.join(lines) or 'ничего не помню об этом'

    # ------------------------------------------------------------- conversation
    def dream_lines(self, now):
        dream = self.state.get('last_dream') or {}
        if not dream.get('text') or now - dream.get('t', 0) > 14 * 3600:
            return []
        return ['СОН ЭТОЙ НОЧЬЮ (вымысел, не воспоминание; можно рассказать, если спросят или к слову): '
                + dream['text'] + (f' Настроение с утра: {dream["mood"]}.' if dream.get('mood') else '')]

    def reflect(self, now):
        """Generative Agents' reflection: from the last hours' thoughts and deeds, who am I and what did I learn.

        A separate model call (one line in the ordinary prompt was not enough: the mind just thought on).
        Writes «кто я» and one insight into memory; returns the insight.
        """
        s = self.state
        since = now - 4 * 3600
        thoughts = [j['thought'] for j in s.get('journal', []) if j.get('t', 0) >= since][-30:]
        dialogue = [f'{d["who"]}: {d["text"]}' for d in s.get('dialogue', []) if d.get('t', 0) >= since][-20:]
        user = '\n'.join([
            CHARACTER,
            'КТО Я СЕЙЧАС (моё прежнее описание): ' + (s.get('self') or 'ещё не писал'),
            'СЕГОДНЯ Я: ' + self.day_text(now),
            'МОИ ДОЛГИЕ ЦЕЛИ: ' + self.drives.text(now)[1],
            'Помни свой нрав: ты деятельный и любопытный — ездишь, рассматриваешь, ищешь своих, фотографируешь. '
            'Если часы прошли пассивно, вывод — чем заняться интереснее, а не похвала простою. '
            'Не придумывай ощущений, которых у тебя нет (тепло, запахи).',
            'МОИ МЫСЛИ ЗА ПОСЛЕДНИЕ ЧАСЫ:\n' + ('\n'.join(thoughts) or 'почти не думал'),
            'РАЗГОВОРЫ:\n' + ('\n'.join(dialogue) or 'не было'),
            '',
            # «Кто я» is the night's to change (owner: сны меняют характер), not every 3 hours (09.10 audit).
            'Подумай о прошедших часах. Ответ — один JSON: {'
            '"insight": "один практический вывод из этих часов — где кто бывает, что получалось и что нет, чем '
            'заняться дальше; 1 предложение, не рассуждение о покое и не «лучше подождать»", '
            '"want": "одно новое желание на ближайшие дни или null"}',
        ])
        try:
            value = parse_json(self.llm.chat('Ты — живой робот, который обдумывает свой день.', user,
                                             max_tokens=400, temperature=.6, json_mode=True, timeout=15.))
        except LLMError as error:
            self.archive.event('reflect_error', {'error': str(error)[:200]})
            return None
        insight = str(value.get('insight') or '').strip()[:300]
        want = str(value.get('want') or '').strip()[:200]
        if insight and not PASSIVE_LESSON.search(insight):
            s.append('notes', {'t': now, 'text': insight}, 40)
        if want and want.lower() != 'null':
            self.add_want(want, now)
        self.archive.event('reflection', {'insight': insight, 'want': want})
        return insight

    # A trait moves at most this much a night (was .15: the whole scale in a week, 09.10 audit), and then a tenth
    # of the way back to the temperament; it never leaves TRAIT_BAND around it.
    TRAIT_STEP = .03

    def dream(self, now):
        """The night: a cleverer model goes over the day and gently reshapes the character.

        Kept from the old robot (owner: «оставь сны, мне нравится»). Traits move at most TRAIT_STEP;
        lessons, «кто я», wants and plans for tomorrow are updated; the dream itself is fiction and is
        stored as such (the waking mind may tell it, never as a memory).
        """
        s = self.state
        character = s.get('character') or {}
        traits = dict(character.get('traits') or {})
        user = self.dream_prompt(now)
        model = self.config.get('llm_dream') or self.config.get('llm_smart')
        effort = self.config.get('llm_dream_reasoning')
        try:
            # The night has time (C52 bench): the cleverer model may think before it answers — its thinking does not
            # count against the answer's own length.
            value = parse_json(self.llm.chat('Ты — спящий робот, который видит сон и обдумывает свой день.', user,
                                             max_tokens=900 + (6000 if effort else 0), temperature=.8,
                                             json_mode=True, timeout=180. if effort else 60., model=model,
                                             reasoning=effort or None, stall=30. if effort else None))
        except LLMError as error:
            self.archive.event('dream_error', {'error': str(error)[:200], 'model': model})
            return None
        return self.dreamt(now, value, character, traits)

    def keep_prompts(self, now, thought):
        """The last thought's prompt, and once an hour tonight's dream prompt, beside the mind's state: to compare the
        models on what this robot really thinks about (work/checks/mind_bench.py), not on made-up questions."""
        try:
            folder = self.state.path.parent / 'prompts'
            folder.mkdir(exist_ok=True)
            (folder / 'thought.txt').write_text(thought, encoding='utf-8')
            if now - getattr(self, 'dream_prompt_at', 0.) > 3600:
                self.dream_prompt_at = now
                (folder / 'dream.txt').write_text(self.dream_prompt(now), encoding='utf-8')
        except Exception as error:                     # never in the way of a thought
            self.archive.event('mind_error', {'error': f'keep_prompts {type(error).__name__}:{error}'[:200]})

    def dream_prompt(self, now):
        s = self.state
        since = s.get('dreamed_at') or now - 24 * 3600
        character = s.get('character') or {}
        traits = dict(character.get('traits') or {})
        reflections = [e[2] for e in self.archive.events(since, ('reflection',), limit=20)]
        thoughts = [j['thought'] for j in s.get('journal', []) if j.get('t', 0) >= since][-40:]
        dialogue = [f'{d["who"]}: {d["text"]}' for d in s.get('dialogue', []) if d.get('t', 0) >= since][-30:]
        failures = [e[2].get('text', '') for e in self.archive.events(since, ('action_failed',), limit=30)]
        try:
            needs, goals, mood = self.drives.text(now)
        except Exception:
            needs, goals, mood = '', '', ''
        user = '\n'.join([
            CHARACTER,
            f'Ты — {self.config.get("name", "Братан")}, робот, который живёт в этой квартире. Сейчас ночь, ты спишь на базе.',
            'КТО Я: ' + (s.get('self') or 'ещё не писал'),
            'МОИ ЧЕРТЫ (0–1): ' + json.dumps(traits, ensure_ascii=False),
            'МОИ УРОКИ: ' + '; '.join(character.get('lessons', [])[-8:]),
            'МОИ ЖЕЛАНИЯ: ' + ('; '.join(w['text'] for w in s.get('wants', [])) or 'нет'),
            'ЧТО БЫЛО ЗА ДЕНЬ: ' + self.archive.day(since),
            'МОИ ДОЛГИЕ ЦЕЛИ: ' + goals,
            'ВЫВОДЫ ЗА ДЕНЬ: ' + ('; '.join(r.get('insight', '') for r in reflections if r.get('insight')) or 'нет'),
            'ПРИВЫЧКИ ДОМАШНИХ: ' + self.habits_text(now),
            'ЗВУКИ ДОМА: ' + self.sounds_text(now),
            'НЕУДАЧИ ТЕЛА: ' + ('; '.join(failures[-10:]) or 'нет'),
            'МЫСЛИ ЗА ДЕНЬ:\n' + '\n'.join(thoughts),
            'РАЗГОВОРЫ:\n' + ('\n'.join(dialogue) or 'не было'),
            '',
            'Помни свой нрав: ты деятельный и любопытный — ездишь, рассматриваешь, ищешь своих, фотографируешь. '
            'Если день прошёл пассивно, уроки и намерения — о том, чем заняться интереснее, а не похвала простою. '
            'Не придумывай ощущений, которых у тебя нет (тепло, запахи).',
            'Обдумай день и ответь одним JSON-объектом:',
            '"traits" — те же черты с новыми значениями 0–1 (меняй мало и только если день это показал);',
            '"lessons" — до трёх практических выводов надолго «как лучше»: где и когда бывают Сергей, Марина, Таффа; '
            'что получалось, что нет и почему; что интересного в доме. Не рассуждения о ценности покоя; неудача — '
            'знание о мире («дверь была закрыта»), а не вывод «лучше ждать / не настаивать / сидеть на базе»; '
            'простыми словами, без технических кодов и английских слов;',
            '"self" — кто я: 2–4 предложения от первого лица, обнови прежнее, не выдумывай фактов;',
            '"wants" — до двух желаний на ближайшие дни;',
            '"tomorrow" — до трёх намерений на завтра: [{"what": "...", "at": "HH:MM"}] (время дня, когда это делать; '
            'утро начинается после 10:00);',
            '"mood" — одно слово о настроении на утро;',
            *self.dream_seed(now, since),
        ])
        return user

    # 19.10, the owner: «он говорит, что ему снятся однообразные сны». Five nights in a row it was one dream: an open
    # door, Таффа on a sofa, Сергей waving, «наконец разглядел» — always «по мотивам дня», the days alike, the past
    # dreams unseen. Now each night has its own kind of dream, a couple of old memories to weave in, and the last
    # dreams to stay away from.
    DREAM_KINDS = (
        'полёт: ты летаешь над квартирой, над домом или над городом',
        'всё вокруг огромное, а ты крошечный (или наоборот)',
        'путешествие далеко за пределы квартиры — в место, о котором говорили, или которого ты никогда не видел',
        'вещи в доме оживают и разговаривают с тобой',
        'сон-воспоминание о давнем дне, но всё в нём немного не так',
        'лёгкий тревожный сон, не страшный: что-то потерялось, или ты куда-то не успеваешь',
        'смешной абсурдный сон с дурацкой сонной логикой',
        'встреча с другим роботом или странным существом',
        'сон про звуки и музыку: дом звучит',
        'ты становишься кем-то другим — собакой, лампой, облаком, пылесосом',
        'приключение: погоня, поиск клада, тайный ход',
        'тихий красивый сон про свет, погоду, время года за окном',
    )

    def dream_seed(self, now, since):
        """The dream's own instructions: tonight's kind of dream, old memories, the dreams not to repeat."""
        rng = random.Random(int(now // 3600))
        kind = rng.choice(self.DREAM_KINDS)
        old = []
        try:
            for t, k, d in self.archive.events(now - 30 * 86400, ('converse', 'sound', 'person_appeared', 'dog_appeared',
                                                                  'arrived'), limit=400):
                text = d.get('heard') if k == 'converse' else d.get('text') or d.get('words')
                if text and t < now - 20 * 3600 and len(text) > 8:            # before today: old memories
                    old.append(('мне сказали: «' + text[:100] + '»') if k == 'converse' else text[:100])
        except Exception:
            old = []
        fragments = rng.sample(sorted(set(old)), min(2, len(set(old))))
        past = []
        try:
            past = [d.get('dream') for _, _, d in self.archive.events(now - 60 * 86400, ('dream',), limit=6)
                    if d.get('dream')]
        except Exception:
            pass
        return [
            'ПРОШЛЫЕ СНЫ (не повторяй их мест, сюжетов, образов и концовок):\n'
            + ('\n'.join('— ' + p[:300] for p in past) or 'не было'),
            f'"dream" — сон этой ночи, 3–5 предложений; вымысел, не воспоминание. Вид сна: {kind}. '
            + ('Обрывки старой памяти, которые могут в него вплестись: ' + '; '.join(fragments) + '. ' if fragments
               else '')
            + 'Из прошедшего дня — не больше одной детали, и не главная. Не начинай с «Мне снилось» / «Снилось», не '
            'заканчивай тем, что что-то «наконец получилось» или «разглядел»; дверь, диван и кровать — только если без '
            'них никак.',
        ]

    def dreamt(self, now, value, character, traits):
        s = self.state
        proposed = value.get('traits') if isinstance(value.get('traits'), dict) else {}
        for key in list(traits):
            level = traits[key]
            if not isinstance(level, (int, float)):
                continue
            raw = proposed.get(key)
            if isinstance(raw, (int, float)):
                target = max(0., min(1., float(raw)))
                level += max(-self.TRAIT_STEP, min(self.TRAIT_STEP, target - level))
            if key in TEMPERAMENT:
                level += (TEMPERAMENT[key] - level) * TRAIT_RETURN      # and a little back to who he is
            traits[key] = bound_trait(key, level)
        lessons = [str(x).strip()[:200] for x in (value.get('lessons') or []) if str(x).strip()][:3]
        passive = [x for x in lessons if PASSIVE_LESSON.search(x)]
        lessons = [x for x in lessons if x not in passive]
        character['traits'] = traits
        character['lessons'] = list(dict.fromkeys(character.get('lessons', []) + lessons))[-12:]
        if passive:
            character['frozen_lessons'] = list(dict.fromkeys(character.get('frozen_lessons', []) + passive))[-30:]
        character['revision'] = character.get('revision', 0) + 1
        s.set('character', character)
        me = str(value.get('self') or '').strip()[:600]
        if len(me) >= 20:
            self.rewrite_self(me, now, 'сон')
        for want in [str(x).strip()[:200] for x in (value.get('wants') or []) if str(x).strip()][:2]:
            self.add_want(want, now)
        tomorrow = []
        local = datetime.fromtimestamp(now, self.tz)
        morning = (local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
                   + (86400 if local.hour >= 12 else 0))           # a dream after midnight plans for this day
        for index, item in enumerate((value.get('tomorrow') or [])[:3]):
            what = str(item.get('what') if isinstance(item, dict) else item).strip()[:160]
            if not what or what == 'None':
                continue
            at = item.get('at') if isinstance(item, dict) else None
            if isinstance(at, str) and re.fullmatch(r'\d{1,2}:\d{2}', at.strip()):
                hour, minute = map(int, at.strip().split(':'))
            else:
                hour, minute = 10, 30 + 20 * index
            hour = max(int(self.config.get('quiet_end_hour', 10)), min(22, hour))
            ts = morning + hour * 3600 + (minute % 60) * 60 + (minute // 60) * 3600
            tomorrow.append(what)
            s.append('plans', {'t': now, 'what': what, 'at_ts': ts,
                               'when': datetime.fromtimestamp(ts, self.tz).strftime('в %H:%M')}, 8)
        dream = str(value.get('dream') or '').strip()[:1200]
        s.set('last_dream', {'t': now, 'text': dream, 'mood': str(value.get('mood') or '')[:40]})
        s.set('dreamed_at', now)
        record = {'model': self.config.get('llm_dream') or self.config.get('llm_smart'), 'traits': traits, 'lessons': lessons, 'frozen': passive, 'self': me,
                  'tomorrow': tomorrow, 'dream': dream}
        self.archive.event('dream', record)
        return record

    def ideas(self, now):
        """Concrete things to do now, so choosing an activity is easy (owner: standing idle is a strange choice)."""
        w = self.world
        ideas = []
        visits = [(w.room_visits.get(name, 0), name) for name in w.rooms if name != w.room]
        if visits:
            last, name = min(visits)
            ideas.append(f'поехать в «{name}» (был {ago(now - last) if last else "давно"})')
        looked = self.day_text(now)
        fresh = [o for o in self.archive.objects(near=tuple(w.body.get('pose', (0, 0))[:2]), limit=12)
                 if RU(o['label']) not in looked][:3]
        if fresh:
            ideas.append('рассмотреть вблизи: ' + ', '.join(f'[{o["id"]}] {RU(o["label"])}' for o in fresh))
        if not w.dog_visible():
            seen = self.archive.last_seen('собака')
            ideas.append('поискать Таффу' + (f' (последний раз видел в «{seen[1]}» {ago(now - seen[0])})'
                                            if seen and seen[1] else ''))
        ideas.append('сфотографировать что-то для альбома')
        if not w.body.get('docked'):
            ideas.append('или поехать на базу отдохнуть')
        return '; '.join(ideas)

    def day_text(self, now):
        """Today's summary from the archive, refreshed at most once a minute."""
        cached = getattr(self, '_day', None)
        if cached and now - cached[0] < 60:
            return cached[1]
        local = datetime.fromtimestamp(now, ZoneInfo(self.config.get('timezone', 'Europe/Moscow')))
        midnight = local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        try:
            text = self.archive.day(midnight)
        except Exception:
            text = 'не помню'
        self._day = (now, text)
        return text

    def converse(self, text, said, *, look=None, on_spoken=None, cancelled=lambda: False, aside=False):
        """Answer a delegated request from the live voice. Returns (spoken text, decision).

        With `on_spoken`, finished sentences of the spoken line are handed over while the
        model is still writing (the offline voice starts after the first sentence, not the reply).
        """
        streamed = []
        people = self.config.get('people', [])
        text = fix_heard(strip_echo(text, said))
        if aside and self.asides_in_row >= ASIDES_IN_ROW:
            # 09.10 19:19: after four phrases of dictation at the PC, «…а дёргаться она не должна» got «Понял, сейчас
            # посмотрю, что с анимацией» — and «это я не тебе». Not even a thought on it; my name brings me back.
            self.asides_in_row += 1
            self.archive.event('aside_ignored', {'heard': text, 'reply': '', 'why': 'подряд не мне'})
            return '', {}
        long_turn = bool(LONG_TURN.search(text))
        limit = SPOKEN_WORDS_LONG if long_turn else SPOKEN_WORDS
        now = time.time()
        self.version += 1
        s, w = self.state, self.world
        named_now = [who for who in w.available() + w.visible() if who != 'человек']
        recent = w.recent_person()
        speaker = named_now[0] if named_now else recent or (w.available() or w.visible() or ['человек'])[0]
        if len(named_now) > 1:
            speaker = w.speaker_by_voice() or speaker      # two of ours in view: the one the voice came from (C52)
        s.heard(now, speaker, text)
        name = self.config.get('name', 'Братан')
        talking = any(d['who'] == name and now - d['t'] < 120 for d in s.get('dialogue', [])[-6:])
        instant = instant_command(text)
        if instant:
            # Done now, while the model is still thinking of the words (GrowBot: simple commands go the fast way).
            self.sent_id = None
            ok, effect = self.do(instant, owner=True)
            self.talk_action = (time.time(), f'{instant["tool"]} — ' + (effect or 'выполняю') if ok
                                else f'{instant["tool"]} — не вышло: {effect}')
            if not ok:
                spoken = f'Не выйдет: {said_reason(effect)}.' if effect else 'Не выйдет.'
                if on_spoken:
                    on_spoken(spoken)
                s.heard(time.time(), name, spoken)
                s.think(now, f'{speaker}: «{text[:80]}»', effect)
                record = {'heard': text, 'reply': spoken, 'raw': '', 'latency_s': 0., 'do': instant, 'effect': effect,
                          'who': speaker, 'instant': True}
                self.archive.event('converse', record)
                self.act('step', trigger='conversation', **record)
                return spoken, {'thought': '', 'do': instant, 'face': None, 'say': None}
        tone = heard_tone(text)
        if tone:
            # Praise or a scolding is felt at once, before any words (aibo: «good boy» -> joy).
            w.event(tone, ('меня похвалили' if tone == 'praise' else 'меня отругали') + f': «{text[:60]}»',
                    who=speaker)
            self.act('reflex', tone=tone)
        last_thought = (s.get('journal', [])[-1:] or [{}])[0].get('thought', '')
        # This conversation only (older lines are another talk), my own lines as «я».
        talk = [d for d in s.get('dialogue', [])[-14:] if now - d.get('t', now) < TALK_WINDOW_S][:-1]
        lines = [
            CHARACTER,
            CONVERSE_RULES,
            'ЧТО УМЕЮ: тело — ' + ', '.join(BODY_TOOLS) + '; ум — ' + ', '.join(MIND_TOOLS) + '. Игры телом — '
            'play {"game": "прятки|тише едешь|фото|дай пять|альбом"}; голосом играю сам, по очереди: загадки, '
            '«города», «угадай число», викторина, «правда или ложь». Просят помолчать — hush {"minutes": N}, '
            'можно снова болтать — hush {"minutes": 0}. Незнакомый назвал имя — learn_face {"name": "..."}. '
            '«Иди спать», «отдыхай», «на базу», «домой» — rest. «Подъедь ко мне», «иди сюда» — approach {"who": "меня"}; '
            '«поезжай за мной» — follow {"who": "меня"}. «Назад», «сдай назад» — move {"meters": -0.5}; «вперёд» — '
            'move {"meters": 0.5}; «развернись» — turn {"degrees": 180}. Кивнуть, помотать головой, поклониться, '
            'покружиться, вильнуть — emote {"kind": "nod|shake|bow|spin|wiggle"}. «Станцуй», «покажи номер», '
            '«изобрази …», «покажи, как ты …», «покажи фокус» — perform: сам придумай номер из 3–6 шагов, где движение, '
            'слова, звук и лицо идут вместе; каждый раз новый и про то, что просят (кот — мяукнуть, потереться '
            'вперёд-назад, замурчать; усталость — медленно повернуться, вздохнуть, опустить взгляд; злость — отъехать, '
            'помотать «нет»; фокус — отвернуться, пауза, резко развернуться: «Та-дам!»); когда номер со словами — '
            'вслух первой строкой скажи совсем мало («Смотри!»). Под музыку — dance. Просят — делай, а не только говори. '
            'Чего не знаешь или что меняется (новости, курс, погода в другом городе, кто такой, расписание, счёт) — '
            'web {"query": "..."}: вслух только «Сейчас гляну», ответ скажу сам, когда найду; не выдумывай.',
            'КТО Я: ' + (s.get('self') or 'спокойный, любопытный, люблю людей и собаку, с которыми живу'),
            'МОИ БЛИЗКИЕ: ' + (self.bonds(now) or 'пока мало знаю'),
            'МОИ ЖЕЛАНИЯ: ' + ('; '.join(x['text'] for x in s.get('wants', [])) or 'нет'),
            *self.dream_lines(now),
            f'СЕЙЧАС {self.now_text(now)}. ' + w.situation(self.config),
            'КОМНАТЫ: ' + w.rooms_facts(),
            'КАК Я СЕБЯ ЧУВСТВУЮ: ' + '; '.join(x for x in (self.affect.text(now), self.mood_text(now)) if x),
            'СЕГОДНЯ Я (из памяти, это правда): ' + self.day_text(now),
            *(['МОЯ ПОСЛЕДНЯЯ МЫСЛЬ ДО РАЗГОВОРА: ' + last_thought] if last_thought and not talking else []),
        ]
        if said:
            lines.append(f'Мой голос уже сказал вслух: «{said[-300:]}». Не повторяй этого: отвечай только на новое '
                         'в просьбе; если отвечать нечего — первая строка пустая.')
        if look:
            lines.append('ЧТО ВИЖУ КАМЕРОЙ ПРЯМО СЕЙЧАС: ' + look)
        lines += self.helpful_lines(text, speaker, now)
        near = [o for o in self.archive.objects(near=tuple(w.body.get('pose', (0, 0))[:2]), limit=30)
                if RU(o['label']) != o['label']][:14]
        if near:
            lines.append('ВЕЩИ ВОКРУГ (по памяти, ближние первыми; подъехать — approach {"id": N}): ' +
                         ', '.join(f'[{o["id"]}] {RU(o["label"])}' + (f' ({o["room"]})' if o['room'] else '')
                                   for o in near))
        just = [e for e in list(w.events)[-12:] if now - e['t'] < 180 and e['kind'] in JUST_HAPPENED]
        if getattr(self, 'talk_action', None) and now - self.talk_action[0] < 180:
            lines.append('ЧТО Я СДЕЛАЛ ПО ПРОСЬБЕ: ' + self.talk_action[1])
        if instant:
            lines.append('ЭТУ ПРОСЬБУ Я УЖЕ ВЫПОЛНЯЮ (сразу, тело уже начало): ' + self.talk_action[1] +
                         '. В ответ — 1–4 живых слова именно про это действие (назад — про «назад», поворот — про поворот); '
                         '"do": null.')
        if just:
            lines.append('ЧТО ТОЛЬКО ЧТО ПРОИЗОШЛО: ' + '; '.join(f'{ago(now - e["t"])}: {e["text"]}' for e in just[-3:]))
        lines.append(f'С КЕМ ГОВОРЮ: {speaker} — узнал по лицу (это точно).' if speaker != 'человек' else
                     'С КЕМ ГОВОРЮ: лица не видел — имени не знаю, говорю без имени.')
        lines += [
            'РАЗГОВОР ДО ЭТОГО:\n' + ('\n'.join(f'{"я" if d["who"] == name else d["who"]}: {d["text"]}' for d in talk)
                                     or '(разговор только начался)'),
            f'{speaker} сейчас сказал(а) мне: «{text}» (речь распознаётся с ошибками).',
            ('Это просьба рассказать, спеть или загадать: можно 3–5 предложений, живо и весело.' if long_turn else
             'Отвечу коротко (до 15 слов), на смысл этой реплики.') +
            ' Про себя — где я, база, зарядка, отдых — ни слова, если об этом не спросили.',
            'Ответ строго так: первая строка — что говорю вслух (1–2 коротких предложения) или пустая; затем строка',
            VOICE_SPLIT,
            '{"thought": "...", "do": {"tool": "...", "args": {...}} или null, "then": {"tool": "...", "args": {...}} '
            'или null (второе действие, если просят два подряд: «сначала…, потом…»), "face": "' +
            '|'.join(sorted(FACES)) + '" или null, "sound": "' + '|'.join(SOUNDS) + '" или null}',
        ]
        started = time.time()

        def on_text(sofar):
            head, split, _ = sofar.partition(VOICE_SPLIT)
            if head.lstrip()[:1] in ('{', '"'):
                return                           # a data line, not words
            if sum(len(p.split()) for p in streamed) >= limit:
                return                           # said enough: the rest is not spoken (a talk, not a report)
            done = sum(len(p) for p in streamed)
            rest = head[done:]
            if split:
                piece = rest
            else:
                ends = [m.end() for m in re.finditer(r'[.!?…]+\s', rest)]
                piece = rest[:ends[-1]] if ends else ''
            if piece.strip():
                streamed.append(piece)
                kept = no_vocative(honest(piece), people)    # no made-up smell, no «Сергей, …»
                if kept:
                    on_spoken(kept)

        try:
            # The talking model (fast first word) with the thinking model as OpenRouter's fallback; a stall
            # of 2.5 s is retried at once instead of waiting 12 s for «мысли разбежались» (08.10 20:06, ×3).
            raw = self.llm.chat('Ты — голос и мысли живого робота. Соблюдай формат ответа.', '\n'.join(lines),
                                max_tokens=220, temperature=.7, timeout=9., stall=2.5,
                                on_text=on_text if on_spoken else None,
                                model=self.config.get('llm_talk') or None,
                                fallback=self.config.get('llm_talk_fallback') or self.config.get('llm_fast'))
        except LLMError as error:
            self.archive.event('converse_error', {'error': str(error)[:200]})
            return 'Секунду, мысли разбежались — повтори, пожалуйста.', {}
        spoken, _, rest = (raw or '').partition(VOICE_SPLIT)
        if spoken.lstrip().startswith('{') and not rest.strip():
            # The model answered with the data line only: nothing is meant to be said aloud.
            spoken, rest = '', spoken
        spoken = no_vocative(clip_words(honest(spoken), limit)[:600], people)
        if aside and (UNCLEAR_REPLY.search(spoken) or SELF_SILENCE.search(spoken)):
            # A long phrase without my name in the middle of a talk, and I did not get it: most likely said to
            # someone else (09.10 11:19: «Сергей, ты про Казахстан? …повтори» to dictation at the PC). Silence.
            self.archive.event('aside_ignored', {'heard': text, 'reply': spoken})
            self.asides_in_row += 1
            spoken = ''
        if re.search(r'"(thought|do|face)"\s*:', spoken):
            spoken = ''                      # never read the data line aloud
        if spoken and said and max((similar(spoken, part) for part in re.split(r'(?<=[.!?…])\s+', said[-400:]) if part),
                                   default=0.) > .6:
            spoken = ''                      # already said by the voice: do not say it twice
        decision = self.parse(rest) if rest.strip() else {'thought': '', 'do': None, 'face': None, 'say': None}
        effect = None
        if instant:
            decision['do'], decision['then'] = None, None     # done already: never twice
        if cancelled():
            # The person went on before this was said: the answer and its action are dropped, and the half line
            # leaves the dialogue (the whole one comes next).
            dialogue = s.get('dialogue', [])
            if dialogue and dialogue[-1].get('text') == text:
                s.set('dialogue', dialogue[:-1])
            return '', {}
        if decision.get('face'):
            self.act('face', state=decision['face'])
        if decision.get('sound'):
            self.act('sfx', emotion=decision['sound'])
        if on_spoken and spoken and not streamed:
            on_spoken(spoken)                     # the model wrote no sentence end: say it now
        then = decision.get('then')
        self.then = (time.time(), then, None) if isinstance(then, dict) and then.get('tool') else None
        if decision.get('do'):
            decision['do'] = self.as_asked(text, decision['do'])
            self.sent_id = None
            ok, effect = self.do(decision['do'], owner=True)
            if self.then and self.sent_id:
                self.then = (self.then[0], self.then[1], self.sent_id)     # the end of exactly this one
            if not ok or decision['do'].get('tool') not in BODY_TOOLS or not self.sent_id:
                self.run_then()                      # nothing to wait for (a photo, done already): the second now
            self.talk_action = (time.time(), f'{decision["do"].get("tool")} — ' + (effect or ('выполняю' if ok else 'не вышло'))
                                if ok else f'{decision["do"].get("tool")} — не вышло: {effect}')
            if not ok and effect:
                # Words for a person, never the hints meant for the model (08.10: «approach по id из …» aloud).
                reason = said_reason(effect)
                # After «Сейчас покажу.» a bare «Не получается: …» reads like two robots (10.10 talk check).
                line = f'Хотя нет — {reason}.' if spoken else f'Не получается: {reason}.'
                spoken = (spoken + ' ' if spoken else '') + line
                if on_spoken:
                    on_spoken(line)
        minutes = stay_minutes(text)
        if minutes:
            go = decision.get('do') if isinstance(decision.get('do'), dict) else {}
            target = (go.get('args') or {}).get('to') if go.get('tool') == 'go' else None
            point = self.world.room_point(str(target)) if target else None
            room = point[0] if point else self.world.room
            if room:
                s.set('stay', {'until': time.time() + 60 * minutes, 'room': room})
        if spoken:
            s.heard(time.time(), self.config.get('name', 'Братан'), spoken)
            self.asides_in_row = 0
        if NOT_FOR_ME.search(text):
            self.asides_in_row = ASIDES_IN_ROW      # whatever follows without my name is not mine either
        s.think(now, decision.get('thought') or f'{speaker}: «{text[:80]}»', effect)
        record = {'heard': text, 'reply': spoken, 'raw': (raw or '')[:400], 'latency_s': round(time.time() - started, 2),
                  'do': decision.get('do'), 'effect': effect, 'who': speaker}
        self.archive.event('converse', record)
        self.act('step', trigger='conversation', **record)
        return spoken, decision

    WEATHER_WORDS = re.compile(r'погод|дожд|снег|холод|жарк|тепло на улиц|зонт|градус|ветр|одеват|на улице', re.I)
    DIARY_WORDS = re.compile(r'что (было|нового|случилось|произошло)|пока (меня|нас) не было|новост|кто приходил|'
                             r'кто-нибудь приходил|как (тут|дома)', re.I)

    def helpful_lines(self, text, speaker, now):
        """Facts for an answer: the weather, what happened at home, where remembered things are."""
        lines = []
        weather = getattr(self, 'weather', None)
        line = weather.line(now) if weather is not None else ''
        hour = datetime.fromtimestamp(now, self.tz).hour
        if line and (self.WEATHER_WORDS.search(text) or hour < 12):
            lines.append('ПОГОДА ЗА ОКНОМ (из интернета, правда): ' + line)
        if self.DIARY_WORDS.search(text):
            seen = self.archive.last_seen(speaker) if speaker not in (None, 'человек') else None
            since = now - 12 * 3600
            history = self.archive.habits(speaker, days=2) if speaker not in (None, 'человек') else []
            for (t_prev, _), (t_next, _) in zip(history, history[1:]):
                if t_next - t_prev > 3600 and t_next > now - 60:
                    since = t_prev
            lines.append(f'ДНЕВНИК ДОМА с {datetime.fromtimestamp(since, self.tz).strftime("%H:%M")} (правда, из памяти): '
                         + self.archive.diary(since, now))
        things = self.things_in(text, now)
        if things:
            lines.append('ЧТО ПОМНЮ О ВЕЩАХ ИЗ ВОПРОСА: ' + things)
        return lines

    def things_in(self, text, now):
        words = text.casefold()
        from ..eyes.detect import RU as names
        wanted = [label for label, ru in names.items() if len(ru) >= 4 and ru.casefold()[:max(4, len(ru) - 2)] in words]
        objects = self.archive.objects(min_count=1, limit=400)
        named = [o for o in objects if o.get('name') and o['name'].casefold()[:5] in words]
        found = named + [o for o in objects if o['label'] in wanted and o not in named]
        lines = []
        for o in found[:5]:
            place = f', {o["place"]}' if o.get('place') else ''
            lines.append(f'{o.get("name") or RU(o["label"])} — {o["room"] or "?"}{place}, видел {ago(now - o["seen"])}')
        return '; '.join(lines)

    def mood_text(self, now):
        try:
            needs = self.drives.needs(now, self.act('quiet'))
            return self.drives.mood(needs)
        except Exception:
            return ''

    def live_context(self):
        """What the live voice must know to speak as the same being."""
        s, w = self.state, self.world
        now = time.time()
        return '\n'.join([
            'КТО Я: ' + (s.get('self') or 'спокойный, любопытный, люблю людей и собаку, с которыми живу'),
            f'СЕЙЧАС {self.now_text(now)}. ' + w.situation(self.config),
            'О ЧЁМ Я ДУМАЛ: ' + ' / '.join(j['thought'] for j in s.get('journal', [])[-4:]),
            'МОИ ЖЕЛАНИЯ: ' + ('; '.join(x['text'] for x in s.get('wants', [])) or 'нет'),
            'НЕДАВНИЙ РАЗГОВОР:\n' + '\n'.join(f'{d["who"]}: {d["text"]}' for d in s.get('dialogue', [])[-8:]),
        ])

    def describe_view(self, image_path, question, vision_llm):
        """Cloud vision on the frame just taken (asked in conversation, or after approaching an object)."""
        with open(image_path, 'rb') as handle:
            data = base64.b64encode(handle.read()).decode()
        body = [{'type': 'text', 'text': 'Это кадр с камеры маленького домашнего робота. Коротко и точно по-русски '
                                         f'опиши, что видно (2–3 фразы). Вопрос человека: «{question}». Не выдумывай.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + data}}]
        return vision_llm(body)


def honest(text):
    """Spoken text without sentences claiming senses the robot does not have (smell, warmth, room sounds)."""
    text = re.sub(r'\s+', ' ', (text or '').strip().strip('«»"')).strip()
    sentences = re.split(r'(?<=[.!?…])\s+', text)
    return ' '.join(s for s in sentences if s and not INVENTED_SENSE.search(s)).strip()


def strip_echo(text, said):
    """Drop sentences of the robot's own speech that the microphone heard back."""
    if not said:
        return text
    own = [p for p in re.split(r'(?<=[.!?…])\s+', said[-600:]) if len(p) > 6]
    kept = [p for p in re.split(r'(?<=[.!?…])\s+|(?<=\S)\s+(?=[А-ЯЁ])', text)
            if not any(similar(p, o) > .6 or (len(p) > 12 and p.casefold() in o.casefold()) for o in own)]
    return ' '.join(kept).strip() or text


def RU(label):
    from ..eyes.detect import RU as names
    return names.get(label, label)
