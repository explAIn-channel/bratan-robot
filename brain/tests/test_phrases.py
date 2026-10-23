"""Checklist, part 3: how phrases are understood — the owner's words and the model's own (offline).

python -m pytest tests/test_phrases.py -q
"""
import pytest

from bratan.mind import brain as b

PEOPLE = ['Сергей', 'Марина']

STAY = [  # (phrase, minutes or None)
    ('поезжай пожалуйста в зал побудь там минут 10', 10), ('побудь тут пять минут', 5), ('посиди здесь', 10),
    ('останься на кухне', 10), ('оставайся тут минут 20', 20), ('подожди меня тут', 10), ('жди здесь', 10),
    ('побудь там ещё какое-то время', 10), ('постой тут минутку', 1), ('побудь рядом', None),
    ('подожди я сейчас', None), ('подожди на секунду', None), ('подожди секунду', None), ('постой', None),
    ('побудь там полчаса', 30), ('посиди там час', 60), ('побудь там 300 минут', 60), ('побудь там 0 минут', 1),
    ('побудь там две минуты', 2), ('посиди в зале пару минут', 2), ('иди в зал', None), ('стой', None),
    ('сиди тихо', None), ('побудь со мной', None), ('побудь в спальне', 10),
]
NOT_FOR_ME = [
    ('там я вообще не тебе пока меня не слушаю и не отвечаю', True), ('горе братан это я не тебе', True),
    ('не слушай', True), ('не отвечай пожалуйста', True), ('это не с тобой разговор', True), ('я не к тебе', True),
    ('а тебе нравится', False), ('ты молодец', False), ('скажи мне что-нибудь', False), ('тебе не холодно', False),
    ('я тебе говорю', False), ('не тебе решать', True),
]
WHEN = [  # (condition, key, arg)
    ('собака', 'dog', None), ('увижу Таффу', 'dog', None), ('если Таффа', 'dog', None), ('пёс', 'dog', None),
    ('человек', 'person', None), ('кто-то появится', 'person', None), ('люди', 'person', None),
    ('увижу Сергея', 'name', 'Сергей'), ('Марина', 'name', 'Марина'), ('дверь закрыта', 'door_closed', None),
    ('не проехать', 'door_closed', None), ('доехал', 'arrived', None), ('приеду', 'arrived', None),
    ('на месте', 'arrived', None), ('не вышло', 'failed', None), ('не получилось', 'failed', None),
    ('застрял', 'failed', None), ('потерял из виду', 'lost_sight', None), ('убежал', 'lost_sight', None),
    ('темно', 'dark', None), ('стемнело', 'dark', None), ('звук: лай', 'sound', 'лай'), ('слышу музыку', 'sound', 'музык'),
    ('звонок', 'sound', 'звон'), ('стук', 'sound', 'стук'), ('хлопнула дверь', 'door_closed', None),
    ('звук', 'sound', None), ('всегда', None, None), ('', None, None), ('погода', None, None),
]
INVENTED = [
    ('слышу, как рядом разговаривают', False), ('слышу, как Таффа сопит', True), ('пахнет кофе', True),
    ('чувствую тепло', True), ('Таффа храпит', True), ('слышу музыку', False), ('вижу Сергея', False),
    ('шуршит пакет', True), ('дыхание Таффы', True), ('слышу лай', False),
]
VOCATIVE = [
    ('Сергей, ты про Казахстан?', 'Ты про Казахстан?'), ('Привет, Сергей!', 'Привет!'),
    ('Конечно, ты Сергей.', 'Конечно, ты Сергей.'), ('Это Марина пришла.', 'Это Марина пришла.'),
    ('Марина, привет.', 'Привет.'), ('Да, Сергей, понял.', 'Да, понял.'),
]
INTENT = [
    ({'intent': 'сфотографировать'}, 'photo'), ({'intent': 'поиграть'}, 'play'), ({'intent': 'побыть рядом'}, 'company'),
    ({'intent': 'поздороваться'}, 'greet'), ({'intent': 'сказать: ужин готов'}, 'tell'), ({}, None),
]
BASE = [('на базу', True), ('домой', True), ('база', True), ('на зарядку', True), ('в зал', False), ('в базу данных', False)]
SLIPS = [('есть на базу', 'едь на базу'), ('поесть пожалуйста в зал', 'поезжай пожалуйста в зал'),
         ('хочу поесть на кухне', 'хочу поесть на кухне'), ('красивая ваза', 'красивая ваза'),
         ('поем в кухню', 'поезжай в кухню')]
UNCLEAR = [('Не расслышал, повтори?', True), ('Не совсем понял.', True), ('Понял, еду.', False), ('Привет!', False)]
SILENCE = [('Это не мне адресовано, молчу.', True), ('Понял, не мешаю.', True), ('Не буду встревать.', True),
           ('Еду на кухню.', False), ('Мне нравится музыка.', False)]
SEEK = [  # (thought, tool, who)
    ('Поеду поищу Таффу в зале.', 'go', 'собака'), ('Найду Сергея на кухне.', 'go', 'Сергей'),
    ('Поищу своих в спальне.', 'go', 'человек'), ('Поеду в зал, там давно не был.', 'go', None),
    ('Осмотрю спальню, вдруг пёс там.', 'explore_room', None), ('Поищу пса в коридоре.', 'go', 'собака'),
    ('Хочу поймать Таффу в кадр — еду в коридор.', 'go', 'собака'), ('Поищу кровать.', 'go', None),
    ('Найду Марину и поздороваюсь.', 'go', 'Марина'), ('Поищу кого-нибудь.', 'go', 'человек'),
]


@pytest.mark.parametrize('phrase,minutes', STAY)
def test_stay(phrase, minutes):
    assert b.stay_minutes(phrase) == minutes


@pytest.mark.parametrize('phrase,yes', NOT_FOR_ME)
def test_not_for_me(phrase, yes):
    assert bool(b.NOT_FOR_ME.search(phrase)) == yes


@pytest.mark.parametrize('text,key,arg', WHEN)
def test_when(text, key, arg):
    assert b.when_of(text, PEOPLE, 'Таффа') == (key, arg)


@pytest.mark.parametrize('text,invented', INVENTED)
def test_invented(text, invented):
    assert bool(b.INVENTED_SENSE.search(text)) == invented


@pytest.mark.parametrize('said,kept', VOCATIVE)
def test_vocative(said, kept):
    assert b.no_vocative(said, PEOPLE) == kept


@pytest.mark.parametrize('args,kind', INTENT)
def test_intent(args, kind):
    result = b.intent_of(args)
    assert (result[0] if result else None) == kind


@pytest.mark.parametrize('text,yes', BASE)
def test_base(text, yes):
    assert bool(b.BASE_WORDS.match(text)) == yes


@pytest.mark.parametrize('heard,fixed', SLIPS)
def test_slips(heard, fixed):
    assert b.fix_heard(heard) == fixed


@pytest.mark.parametrize('reply,yes', UNCLEAR)
def test_unclear(reply, yes):
    assert bool(b.UNCLEAR_REPLY.search(reply)) == yes


@pytest.mark.parametrize('reply,yes', SILENCE)
def test_self_silence(reply, yes):
    assert bool(b.SELF_SILENCE.search(reply)) == yes


@pytest.mark.parametrize('thought,tool,who', SEEK)
def test_seek(thought, tool, who):
    mind = b.Mind.__new__(b.Mind)
    mind.sent_id, mind.config = 'm1', {'people': PEOPLE, 'pets': {'dog': 'Таффа'}}
    found = mind.seeking_in(thought, {'tool': tool, 'args': {}})
    assert (found or {}).get('who') == who
