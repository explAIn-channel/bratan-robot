"""Hearing with the body's ears (18.10): a speech detector instead of loudness gates, the end of a phrase, what goes to
Microsoft and when it is asked again. Vosk is replaced by a silent stand-in: these are the gates, not the words."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from bratan.voice import vad as vadmod
from bratan.voice.vad import Talk

RATE = 16000
ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = ROOT.parent / 'ushi' / 'recordings' / 'ab'


class SilentRecognizer:
    def __init__(self, *args, **kwargs):
        pass

    def SetWords(self, value):
        pass

    def AcceptWaveform(self, frame):
        return False

    def Result(self):
        return '{"text": "", "result": []}'

    def PartialResult(self):
        return '{"partial": ""}'

    def FinalResult(self):
        return '{"text": "", "result": []}'

    def Reset(self):
        pass


@pytest.fixture
def fake_vosk(monkeypatch):
    module = types.ModuleType('vosk')
    module.KaldiRecognizer = SilentRecognizer
    monkeypatch.setitem(sys.modules, 'vosk', module)
    return module


def noise_frame(rng, level):
    """20 ms of noise at an rms `level` (fraction of full scale) — the ears' room is ~.006."""
    return (rng.normal(0, level * 32768, 320)).astype('<i2').tobytes()


def test_talk_turns_on_at_half_and_off_below_a_third():
    talk = Talk()
    assert [talk.feed(p) for p in (.1, .6, .45, .4, .3, .45, .7)] == [False, True, True, True, False, False, True]
    assert talk.voiced == 4
    talk = Talk()
    for _ in range(10):
        talk.feed(.05)
    assert talk.silence == 10 and not talk.on


def test_the_name_gate_opens_on_a_voice_not_on_the_rooms_noise(fake_vosk):
    from bratan.voice.wake import Listener
    rng = np.random.default_rng(1)
    loud_room = Listener(model=None)
    by_voice = Listener(model=None)
    t = 0.
    for _ in range(500):                                  # 10 s of the ears' loud room, nobody speaking
        frame = noise_frame(rng, .006 * (1 + .5 * rng.random()))
        loud_room.feed(t, frame)
        by_voice.feed(t, frame, speech=.03)
        t += .02
    assert by_voice.frames_open == 0
    assert by_voice.noise > .0025                           # it still learns the room for the other gates
    for _ in range(25):                                     # then half a second of voice
        by_voice.feed(t, noise_frame(rng, .02), speech=.95)
        t += .02
    assert by_voice.frames_open == 25


class StopAt(SilentRecognizer):
    """A grammar decoder that makes «стой» of the phrase at its 10th frame."""

    def __init__(self, *args, **kwargs):
        self.frames = 0

    def AcceptWaveform(self, frame):
        self.frames += 1
        return self.frames == 10

    def Result(self):
        return '{"text": "стой", "result": [{"word": "стой", "conf": 0.95}]}'


@pytest.mark.parametrize('after,want', [(.05, True), (.9, False)])
def test_stop_counts_only_when_the_voice_stops_after_it(monkeypatch, after, want):
    """«Сколько сейчас времени» was made into «стой» (18.10 A/B at 2 m): the words went on after it — no stop. A real
    «стоп!» is followed by quiet."""
    module = types.ModuleType('vosk')
    module.KaldiRecognizer = StopAt
    monkeypatch.setitem(sys.modules, 'vosk', module)
    from bratan.voice.wake import Listener
    rng = np.random.default_rng(7)
    listener = Listener(model=None)
    results, t = [], 0.
    for k in range(60):
        speaking = k < 10 or (k * .02 - .2 < after)        # the phrase, and maybe more words after «стой»
        results.append(listener.feed(t, noise_frame(rng, .05 if speaking else .006), speech=.95 if speaking else .05))
        t += .02
    assert ('stop' in results) == want


def test_the_name_check_hears_the_end_of_the_phrase_through_the_ears(fake_vosk):
    """Until 18.10 its floor was a fixed .002 — under the ears' room: «Братан, привет» ended only at the 8 s limit."""
    from bratan.voice.wake import Confirmer
    rng = np.random.default_rng(2)
    start = [noise_frame(rng, .03) for _ in range(30)]      # the name, as the gate handed it over
    old = Confirmer(None, start)                            # the old fixed floor, loudness only
    by_voice = Confirmer(None, start, noise=.006)
    for _ in range(80):                                     # 1.6 s of the room after the phrase
        frame = noise_frame(rng, .006)
        old.feed(frame)
        by_voice.feed(frame, speech=.04)
    assert not old.ended                                    # what happened before
    assert by_voice.ended and by_voice.silence >= 60
    by_noise = Confirmer(None, start, noise=.006)           # without the detector: the learned noise works too
    for _ in range(80):
        by_noise.feed(noise_frame(rng, .006))
    assert by_noise.ended


def test_a_phrase_knows_where_its_speech_began(fake_vosk):
    from bratan.voice.local import Phrase
    rng = np.random.default_rng(3)
    frames = [noise_frame(rng, .006) for _ in range(40)]
    probs = [.02] * 25 + [.9] * 15
    phrase = Phrase(None, frames, noise=.006, max_s=20., probs=probs)
    assert phrase.speech and phrase.speech_frame == 25 and phrase.voiced == 15
    for _ in range(59):
        phrase.feed(noise_frame(rng, .006), .05)
    assert not phrase.ended                                 # a short opener waits 1.5 s
    for _ in range(20):
        phrase.feed(noise_frame(rng, .006), .05)
    assert phrase.ended


# ---------------------------------------------------------------- what goes to Microsoft
class FakeHearing:
    def __init__(self, answers=()):
        self.started = []
        self.answers = list(answers)
        self.stats = {'again': 0}

    def start(self, frames=()):
        utterance = FakeUtterance(self, list(frames), self.answers.pop(0) if self.answers else '')
        self.started.append(utterance)
        return utterance


class FakeUtterance:
    def __init__(self, hearing, frames, text):
        self.hearing, self.kept, self.text = hearing, frames, text

    def feed(self, frame):
        self.kept.append(frame)

    def result(self, wait=1.5):
        return self.text

    def again(self):
        utterance = self.hearing.start(self.kept)
        return utterance


def voice_stub(hearing):
    from bratan.voice.main import Voice
    stub = types.SimpleNamespace(hearing=hearing, events=[])
    stub.event = lambda kind, **data: stub.events.append((kind, data))
    stub.heard = lambda utterance, vosk: Voice.heard(stub, utterance, vosk)
    stub.start_hearing = lambda dialog, mono: Voice.start_hearing(stub, dialog, mono)
    return stub


@pytest.mark.parametrize('first,again,vosk,want', [
    ('как дела', None, 'как дела', 'как дела'),                         # heard: no second request
    ('', None, 'салют', ''),                                            # one word of Vosk's: a TV, a cough
    ('', 'расскажи что нибудь', 'расскажи мне', 'расскажи что нибудь'),  # asked again: the words came
    ('', '', 'расскажи мне', ''),                                       # twice nothing, two words: nothing
    ('', '', 'какая погода на улице', 'какая погода на улице'),         # twice nothing, four clear words: Vosk's
    (None, None, 'какая погода', 'какая погода'),                       # the service failed: Vosk's
])
def test_nothing_from_microsoft_is_heard_again(first, again, vosk, want):
    hearing = FakeHearing([again] if again is not None else [])
    stub = voice_stub(hearing)
    utterance = FakeUtterance(hearing, [b'x'], first)
    assert stub.heard(utterance, vosk) == want
    retried = first == '' and len(vosk.split()) >= 2
    assert len(hearing.started) == (1 if retried else 0)
    assert any(kind == 'heard_again' for kind, _ in stub.events) == retried


def test_a_follow_up_goes_to_microsoft_from_just_before_its_first_word():
    from collections import deque
    hearing = FakeHearing()
    stub = voice_stub(hearing)
    stub.recent_frames = deque(maxlen=150)
    for i in range(150):                                    # 3 s: quiet, then speech from 2.0 s
        stub.recent_frames.append((100. + i * .02, bytes([i % 256]) * 640, .9 if i >= 100 else .02))
    phrase = types.SimpleNamespace(speech=True, frames=150, speech_frame=100)
    dialog = {'phrase': phrase, 'utterance': None, 'listen_from': 100.5}
    stub.start_hearing(dialog, 100. + 149 * .02)
    sent = hearing.started[0].kept
    assert len(sent) == 50 + 40                             # 0.8 s before the first word, then the words
    assert sent[0] == bytes([60]) * 640
    dialog = {'phrase': phrase, 'utterance': None, 'listen_from': 101.9}     # our own line ended later
    stub.start_hearing(dialog, 100. + 149 * .02)
    assert len(hearing.started[1].kept) == 55
    quiet = types.SimpleNamespace(speech=False, frames=150, speech_frame=None)
    dialog = {'phrase': quiet, 'utterance': None, 'listen_from': 100.}
    stub.start_hearing(dialog, 103.)
    assert dialog['utterance'] is None and len(hearing.started) == 2     # no words, nothing sent


# ---------------------------------------------------------------- the dog's name, the room's hum
@pytest.mark.parametrize('heard,want', [
    ('дафа сегодня гулял два раза', 'таффа сегодня гулял два раза'),
    ('найди тафу', 'найди таффу'),
    ('погуляй с таффой', 'погуляй с таффой'),
    ('давай поиграем', 'давай поиграем'),                    # sounds alike, but not a noun's ending
    ('да фа', 'да фа'),
    ('тапа и туфли', 'тапа и туфли'),
])
def test_the_dogs_name_is_written_right(heard, want):
    from bratan.voice.hearing import fix_names
    assert fix_names(heard, ['Таффа']) == want
    assert fix_names(heard, []) == heard


def test_the_cleaner_passes_a_voice_and_takes_the_hum():
    from bratan.voice.denoise import DELAY, denoise_track
    rng = np.random.default_rng(4)
    t = np.arange(RATE * 6) / RATE
    hum = 600 * (np.sin(2 * np.pi * 93.7 * t) + .6 * np.sin(2 * np.pi * 375.1 * t) + .5 * np.sin(2 * np.pi * 844. * t))
    hum += rng.normal(0, 150, len(t))
    voice = np.zeros(len(t))
    for start in (3.0, 4.2):                                 # two «syllable» bursts after 3 s of hum alone
        n = int(.4 * RATE)
        i = int(start * RATE)
        voice[i:i + n] = rng.normal(0, 3000, n) * np.hanning(n)
    x = np.clip(hum + voice, -32768, 32767).astype(np.int16)
    y = denoise_track(x).astype(float)
    quiet = slice(int(2.0 * RATE), int(2.9 * RATE))
    burst = slice(int(3.1 * RATE), int(3.3 * RATE))
    assert 20 * np.log10(y[quiet].std() / x[quiet].std()) < -8           # the hum: well down
    assert 20 * np.log10(y[burst].std() / x[burst].std()) > -1.5         # the voice: kept
    assert np.array_equal(denoise_track(x, floor_db=0.)[:len(x) - DELAY], x[:len(x) - DELAY])  # gain 1: the input


def test_the_house_sounds_are_named_from_the_raw_ears():
    from bratan.voice.main import EarsFrame
    frame = EarsFrame(b'\x01\x00' * 320)
    frame.raw = b'\x02\x00' * 320
    assert bytes(frame) == b'\x01\x00' * 320 and frame.raw == b'\x02\x00' * 320
    assert (getattr(b'\x03\x00', 'raw', None) or b'\x03\x00') == b'\x03\x00'     # the phone's frames as they are


def test_three_microphones_summed_by_their_delays_hear_the_voice_better():
    from bratan.voice.beam import delay_and_sum, delays
    rng = np.random.default_rng(3)
    n = 32000
    src = np.convolve(rng.normal(0, 1, n + 64), np.ones(4) / 4, mode='same') * 3000      # a low-passed «voice»
    late = (0, 5, 12)
    x = np.zeros((n, 3))
    for ch, d in enumerate(late):
        x[:, ch] = src[32 - d:32 - d + n] + rng.normal(0, 900, n)          # equal voice, independent noise
    x = x.astype(np.int16)
    found, ref = delays(x)
    assert ref in (0, 1, 2) and np.allclose(found - found[0], np.array(late) - late[0], atol=.6)
    y = delay_and_sum(x, found).astype(float)
    target = src[32 - late[ref]:32 - late[ref] + n]
    best = max(np.corrcoef(x[:, ch].astype(float), src[32 - late[ch]:32 - late[ch] + n])[0, 1] for ch in range(3))
    assert np.corrcoef(y, target)[0, 1] > best + .005


# ---------------------------------------------------------------- the detector itself, on the owner's voice
def silero_path():
    spec = importlib.util.find_spec('faster_whisper')
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).parent / 'assets' / 'silero_vad_v6.onnx'
    return path if path.exists() else None


@pytest.mark.skipif(silero_path() is None or importlib.util.find_spec('onnxruntime') is None
                    or not (RECORDINGS / 'ab4m-array.wav').exists(), reason='no detector model or recording here')
def test_the_detector_hears_every_phrase_from_four_metres_and_not_the_knocks():
    """The A/B at 4 m (18.10): ten phrases read to the body's ears; a loudness split also finds three knocks."""
    from bratan.voice import ab
    x = ab.read(RECORDINGS / 'ab4m-array.wav')
    detector = vadmod.Silero(silero_path())
    probs = np.array([detector.feed(x[i * 320:(i + 1) * 320].tobytes()) for i in range(len(x) // 320)])
    spans = ab.segments([ab.read(RECORDINGS / 'ab4m-phone.wav'), x], pad_s=0.)
    found = [probs[a:b].max() for a, b in spans]
    assert sum(p >= .9 for p in found) == 10
    assert sum(p < .1 for p in found) == len(spans) - 10
