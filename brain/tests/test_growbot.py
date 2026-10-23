"""Ideas from GrowBot (11.10, «I Gave ChatGPT a Body»): hands on the body felt, a little number with words in
step with the motion, plain commands done before the model answers: python -m pytest tests/test_growbot.py"""
import math
import types

from bratan.body import skills
from bratan.body.feel import Feel, feeling_text
from bratan.mind import brain as brainmod
from bratan.mind.life import Life
from tests.test_mind import clock, make_mind, world  # noqa: F401  (fixtures)


# ------------------------------------------------------------------ feeling hands
def run_feel(samples, feel=None):
    """samples: (seconds, gyro_w, encoder_w, odom dx per second, commanded, docked)."""
    feel, now, odom, out = feel or Feel(), 0., [0., 0., 0.], []
    for seconds, gyro, enc, speed, commanded, docked in samples:
        for _ in range(int(seconds / .02)):
            now += .02
            odom[0] += speed * .02
            body = {'odom': list(odom), 'gyro_w': gyro, 'encoder_w': enc, 'sent': [0., 0.]}
            got = feel.update(body, commanded, docked, now)
            if got:
                out.append(got)
    return out


def test_turned_by_hand_in_the_air():
    out = run_feel([(1., 0., 0., 0., False, False), (1.2, .8, 0., 0., False, False)])
    assert len(out) == 1 and out[0]['how'] == 'turned' and out[0]['lifted'] and out[0]['degrees'] > 25
    assert 'подняли' in feeling_text(out[0]) and 'налево' in feeling_text(out[0])


def test_turned_on_the_floor_is_not_lifted():
    out = run_feel([(1., 0., 0., 0., False, False), (1.2, -.8, -.8, 0., False, False)])
    assert out and out[0]['how'] == 'turned' and not out[0]['lifted'] and 'направо' in feeling_text(out[0])


def test_pushed_back():
    out = run_feel([(1., 0., 0., 0., False, False), (1., 0., 0., -.1, False, False)])
    assert out and out[0]['how'] == 'pushed' and out[0]['back'] and 'назад' in feeling_text(out[0])


def test_my_own_motion_and_its_tail_are_not_hands():
    # Driving and turning myself, then the wheels' tail just after the stop.
    assert not run_feel([(2., .6, .6, .2, True, False), (.6, .6, .6, .1, False, False), (2., 0., 0., 0., False, False)])
    assert not run_feel([(1., 0., 0., 0., False, True), (2., .8, 0., .1, False, True)])      # on the dock
    assert not run_feel([(3., .02, 0., .001, False, False)])                                  # resting noise


def test_one_feeling_not_a_stream():
    out = run_feel([(1., 0., 0., 0., False, False), (4., .8, 0., 0., False, False)])
    assert len(out) == 1


# ------------------------------------------------------------------ a number in the body
class Ctx:
    def __init__(self):
        self.now, self.dt, self.odom, self.twist = 0., .04, (0., 0., 0.), (0., 0.)    # skills run at 25 Hz

    def blocked_for(self):
        return 0.

    def step(self, skill):
        intent = skill.tick(self) or skills.STOP
        x, y, yaw = self.odom
        yaw = math.atan2(math.sin(yaw + intent.w * self.dt), math.cos(yaw + intent.w * self.dt))
        self.odom = (x + intent.v * self.dt * math.cos(yaw), y + intent.v * self.dt * math.sin(yaw), yaw)
        self.now += self.dt


def perform(steps):
    skill = skills.SKILLS['perform']({'steps': steps}, id='p', origin='owner')
    skill.kind, skill.started = 'perform', 0.
    ctx, cues = Ctx(), []
    while skill.status is None and ctx.now < 60:
        ctx.step(skill)
        cues += [(i, round(ctx.now, 2)) for i in skill.info.pop('cues', [])]
    return skill, ctx, cues


def test_a_number_runs_step_by_step_and_cues_each():
    skill, ctx, cues = perform([{'say': 'Встречайте!', 'pause': 1.}, {'turn': math.pi / 2}, {'move': -.2},
                                {'emote': 'nod'}, {'pause': .5}])
    assert skill.status == 'done' and [i for i, _ in cues] == [0, 1, 2, 3, 4]
    assert cues[1][1] >= 1.                     # the words of the first step had their second
    assert abs(ctx.odom[2] - math.pi / 2) < .1
    assert ctx.now < 30


def test_words_only_number_stays_on_the_dock():
    assert not skills.SKILLS['perform']({'steps': [{'say': 'привет', 'pause': 1}]}).lidar_needed
    assert skills.SKILLS['perform']({'steps': [{'say': 'привет'}, {'turn': 1.}]}).lidar_needed


# ------------------------------------------------------------------ a number from the mind
def test_the_model_number_made_safe(world, clock):
    mind, acts = make_mind(world)
    steps = mind.perform_steps({'steps': [{'turn': 720, 'say': 'Сергей, смотри!'}, {'move': 2}, {'move': -2},
                                          {'move': .3}, {'emote': 'радость', 'sound': 'excited', 'face': 'happy'},
                                          {'sound': 'мяу'}, 'чепуха', {}, {'pause': 9}]})
    assert steps[0]['turn'] == math.radians(360) and 'Сергей' not in steps[0]['say'] and steps[0]['pause'] >= .7
    assert [s.get('move') for s in steps[1:4]] == [.3, -.3, None] or sum(abs(s.get('move', 0)) for s in steps) <= .8
    assert any(s.get('emote') == 'wiggle' and s.get('sound') == 'excited' and s.get('face') == 'happy' for s in steps)
    assert all(s.get('pause', 0) <= 4 for s in steps) and len(steps) <= 8
    assert mind.perform_steps({'steps': [{'pause': 2}]}) == []
    assert mind.perform_steps({'steps': 'танец'}) == []


def test_owner_asks_for_a_number(world, clock):
    mind, acts = make_mind(world)
    ok, why = mind.do({'tool': 'perform', 'args': {'steps': [{'emote': 'bow', 'say': 'Та-дам!'}, {'turn': 180}]}},
                      owner=True)
    sent = [d for k, d in acts if k == 'body_do'][-1]
    assert ok and sent['skill'] == 'perform' and mind.performances[mind.sent_id][0]['say'] == 'Та-дам!'
    ok, why = mind.do({'tool': 'dance', 'args': {'steps': [{'turn': 90, 'say': 'Опа'}]}}, owner=True)
    assert ok and [d for k, d in acts if k == 'body_do'][-1]['skill'] == 'perform'


def test_my_own_number_needs_someone_and_not_too_often(world, clock):
    mind, acts = make_mind(world)
    number = {'tool': 'perform', 'args': {'steps': [{'emote': 'spin', 'say': 'Ух!'}]}}
    assert mind.do(number)[0] is False                  # nobody to show it to
    world.voice = {'live': True}
    assert mind.do(number)[0] is True
    assert mind.do(number)[0] is False                  # again at once
    clock.now += brainmod.PERFORM_AGAIN_S + 1
    assert mind.do(number)[0] is True


def test_on_the_dock_my_own_number_is_words_only(world, clock):
    world.on_body({'pose': [0., 0., 0.], 'localized': True, 'docked': True, 'voltage': 16.2, 'twist': [0., 0.]})
    world.voice = {'live': True}
    mind, acts = make_mind(world)
    assert mind.do({'tool': 'perform', 'args': {'steps': [{'turn': 360, 'say': 'Привет!'}]}})[0]
    steps = [d for k, d in acts if k == 'body_do'][-1]['args']['steps']
    assert 'turn' not in steps[0] and steps[0]['say'] == 'Привет!'


def test_a_cue_says_the_words_of_its_step():
    said, sounds, faces = [], [], []
    fake = types.SimpleNamespace(
        mind=types.SimpleNamespace(performances={'m7': [{'say': 'Раз', 'face': 'happy'}, {'sound': 'excited'}]}),
        react=faces.append, sfx=lambda kind, **kw: sounds.append(kind), wake_face=lambda: None,
        publish=lambda topic, data: said.append(data['text']))
    Life.on_cue(fake, {'id': 'm7', 'index': 0})
    Life.on_cue(fake, {'id': 'm7', 'index': 1})
    Life.on_cue(fake, {'id': 'm7', 'index': 5})
    Life.on_cue(fake, {'id': 'other', 'index': 0})
    assert said == ['Раз'] and faces == ['happy'] and sounds == ['excited']


# ------------------------------------------------------------------ plain commands at once
def test_plain_commands_are_recognised():
    cases = {'сдай назад': ('move', -.5), 'Братан, развернись!': ('turn', 180), 'назад немного': ('move', -.25),
             'кивни-ка': ('emote', 'nod'), 'поверни направо': ('turn', -90), 'иди спать': ('rest', None),
             'подъедь ко мне': ('approach', 'меня'), 'покружись пожалуйста': ('emote', 'spin'),
             'ну давай вперёд чуть-чуть': ('move', .25)}
    for text, (tool, value) in cases.items():
        got = brainmod.instant_command(text)
        assert got and got['tool'] == tool, text
        if value is not None:
            assert value in got['args'].values(), (text, got)
    for text in ('поезжай в зал', 'не назад', 'найди Таффу', 'развернись и поезжай на кухню к Таффе', 'как дела',
                 'назад в будущее смотрел'):
        assert brainmod.instant_command(text) is None, text


class OrderLLM:
    def __init__(self, acts, reply):
        self.acts, self.reply, self.seen = acts, reply, None

    def chat(self, system, prompt, **kw):
        self.seen = [k for k, _ in self.acts]
        self.prompt = prompt
        return self.reply


def test_a_plain_command_moves_before_the_model_answers_and_only_once(world, clock):
    mind, acts = make_mind(world)
    mind.llm = OrderLLM(acts, 'Еду назад!\n' + brainmod.VOICE_SPLIT + '\n{"thought": "", "do": {"tool": "move", '
                                                                      '"args": {"meters": -0.5}}}')
    spoken, decision = mind.converse('сдай назад', '')
    assert 'body_do' in mind.llm.seen                     # the wheels went before the words were written
    assert 'УЖЕ ВЫПОЛНЯЮ' in mind.llm.prompt
    assert len([d for k, d in acts if k == 'body_do']) == 1 and decision['do'] is None
    assert spoken == 'Еду назад!'


def test_a_plain_command_that_cannot_be_done_is_answered_at_once(world, clock):
    world.on_body({'pose': [1.5, .4, 0.], 'localized': True, 'docked': False, 'voltage': 15.8, 'twist': [0., 0.],
                   'inhibit': True})
    mind, acts = make_mind(world)
    mind.llm = OrderLLM(acts, 'не должно')
    spoken, decision = mind.converse('развернись', '')
    assert mind.llm.seen is None and spoken.startswith('Не выйдет') and 'двигаться' in spoken


def test_steps_the_model_really_wrote(world, clock):
    """11.10 on the phone the talking model wrote steps as tool calls, as words, or none at all."""
    mind, acts = make_mind(world)
    steps = mind.perform_steps({'steps': [{'tool': 'turn', 'args': {'degrees': 90}}, {'tool': 'wait', 'args': {'seconds': 1}},
                                          {'tool': 'turn', 'args': {'degrees': -180}}, {'tool': 'emote', 'args': {'kind': 'bow'}}]})
    assert [list(s)[0] for s in steps] == ['turn', 'pause', 'turn', 'emote'] and steps[2]['turn'] == -math.pi
    steps = mind.perform_steps({'kind': 'номер', 'steps': ['повернуться влево', 'качнуться вперёд-назад', 'повернуться вправо',
                                                         'вильнуть', 'замереть', 'что-то непонятное']})
    assert steps == [{'turn': math.pi / 2}, {'emote': 'bounce'}, {'turn': -math.pi / 2}, {'emote': 'wiggle'}, {'pause': 1.}]
    assert len(mind.perform_steps({'kind': 'dance'})) >= 3            # no steps at all: a ready number
    ok, _ = mind.do({'tool': 'perform', 'args': {'kind': 'dance'}}, owner=True)
    assert ok and [d for k, d in acts if k == 'body_do'][-1]['skill'] == 'perform'


def test_more_steps_the_model_really_wrote(world, clock):
    mind, acts = make_mind(world)
    steps = mind.perform_steps({'steps': ['turn 90', 'pause 1', 'turn -180', 'say Та-дам!']})
    assert steps[0] == {'turn': math.pi / 2} and steps[1] == {'pause': 1.} and steps[3]['say'] == 'Та-дам!'
    steps = mind.perform_steps({'steps': ['потереться вперёд-назад', 'мяукнуть', 'замурчать']})
    assert steps[1]['say'] == 'Мяу!' and steps[2]['emote'] == 'tremble'
    assert mind.perform_steps({'steps': [{'tool': 'wiggle'}]}) == [{'emote': 'wiggle'}]
    ok, why = mind.do({'tool': 'perform', 'args': {'steps': ['что-то', 'непонятное']}}, owner=True)
    assert ok and len([d for k, d in acts if k == 'body_do'][-1]['args']['steps']) >= 3     # a ready number
