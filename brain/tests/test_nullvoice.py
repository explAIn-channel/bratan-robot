"""The ears take my own voice out while I speak (C52, voice/nullvoice.py): the owner's «подожди, а сколько времени?»
over my story was not heard by the phone at all."""
import json
import time
import types

import numpy as np
import pytest

from bratan.voice import array, nullvoice
from bratan.voice.audio import Audio
from bratan.voice.main import Voice
from bratan.voice.nullvoice import RobotNull
from tests.test_ears import adpcm_block, packet

RATE = 16000


def voice_like(seconds, seed, level=2000.):
    """A voice-band signal with syllables."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    spec = np.fft.rfft(rng.normal(0, 1, n))
    f = np.fft.rfftfreq(n, 1 / RATE)
    spec[(f < 200) | (f > 4000)] = 0
    x = np.fft.irfft(spec, n)
    envelope = np.repeat(rng.uniform(.3, 1., n // 2400 + 2), 2400)[:n]
    return x / x.std() * level * envelope


def at_three(source, delays, gains):
    """`source` reaching three microphones with fractional delays (samples) and gains: [n × 3]."""
    n = len(source)
    spec = np.fft.rfft(source, 2 * n)
    f = np.fft.rfftfreq(2 * n, 1 / RATE)
    return np.stack([g * np.fft.irfft(spec * np.exp(-2j * np.pi * f * d / RATE), 2 * n)[:n]
                     for d, g in zip(delays, gains)], 1)


ROBOT = ((0., 3.4, 2.1), (1.6, .8, 1.))        # the speaker in the body: М1 nearest
PERSON = ((4.2, 0., -2.7), (.7, 1.2, .9))      # someone on the right


def room(seconds, seed, robot=True, person=False, noise=40., echo=.25):
    """The ears' three microphones: the room's own noise, my voice from the body's speaker with the room's echo of it
    (different at every microphone: 12 dB under the direct sound, as measured — the filter takes ~15 dB out), a person."""
    rng = np.random.default_rng(seed + 100)
    x = rng.normal(0, noise, (int(seconds * RATE), 3))
    if robot:
        voice = voice_like(seconds, seed)
        x += at_three(voice, *ROBOT)
        if echo:
            x += np.stack([echo * np.convolve(voice, rng.normal(0, 1, 800) * np.exp(-np.arange(800) / 200.)
                                              / np.sqrt(100.), 'same') for _ in range(3)], 1)
    if person:
        x += at_three(voice_like(seconds, seed + 1, 1200.), *PERSON)
    return x.clip(-32768, 32767).astype(np.int16)


def run(null, x, first=0, chunk=512, **kw):
    """Feed in chunks; returns the output by the board's frame numbers (from, samples)."""
    out, start = [], None
    for i in range(0, len(x), chunk):
        at, y = null.feed(first + i, x[i:i + chunk], **kw)
        if at is not None:
            if start is None:
                start = at
            assert at == start + sum(len(o) for o in out)             # one unbroken stream
            out.append(y)
    return start, np.concatenate(out)


def level_db(y):
    return 10 * np.log10((np.asarray(y, float) ** 2).mean() + 1e-9)


def learned(seconds=8.):
    null = RobotNull()
    run(null, room(seconds, 1), learn=True, clean=True)
    return null


def test_while_i_am_silent_the_voice_is_the_picked_microphone_exactly_in_any_pieces():
    x = room(3., 7)
    null = RobotNull()
    rng = np.random.default_rng(4)
    pos, out, start = 0, [], None
    while pos < len(x):
        n = int(rng.integers(100, 1300))
        at, y = null.feed(1000 + pos, x[pos:pos + n], pick=2)
        if at is not None:
            start = at if start is None else start
            out.append(y)
        pos += n
    y = np.concatenate(out)
    assert start == 1000 and len(y) >= len(x) - 2 * nullvoice.HOP
    assert np.array_equal(y, x[:len(y), 2])                                # exact, on the same frame numbers


def test_a_gap_starts_the_stream_over_on_its_own_frames():
    x = room(1., 3)
    null = RobotNull()
    run(null, x[:5120], first=0)
    at, y = run(null, x[5120:], first=5120 + 9000)                        # 9000 frames lost
    assert at == 5120 + 9000 and np.array_equal(y, x[5120:5120 + len(y), 0])


def test_my_voice_is_learned_and_taken_out_and_a_person_is_kept():
    null = learned()
    assert null.ready and null.good and null.depth < -15
    robot = room(3., 21)
    _, mic = 0, robot[:, 0]
    _, out = run(null, robot, clean=True)
    assert level_db(out[RATE:]) < level_db(mic[RATE:len(out)]) - 15      # my voice: more than 15 dB down
    person = room(3., 31, robot=False, person=True)
    _, kept = run(null, person, clean=True)
    assert level_db(kept[RATE:]) > level_db(person[RATE:len(kept), 0]) - 6   # the person: within 6 dB


def test_not_clean_until_the_pattern_is_good():
    null = RobotNull()
    x = room(1., 5)
    _, y = run(null, x, learn=False, clean=True)
    assert not null.ready and np.array_equal(y, x[:len(y), 0])


@pytest.mark.parametrize('gate', [True, False])
def test_a_person_talking_over_me_does_not_become_my_pattern(monkeypatch, gate):
    """The very moment the filter is for: he talks over me for a few seconds while I go on learning my voice."""
    if not gate:
        monkeypatch.setattr(nullvoice, 'FIT_DB', 1e9)                     # every frequency learned: he is lost
    null = learned()
    person = room(3., 31, robot=False, person=True)
    before = level_db(run(null, person, first=10 ** 6, clean=True)[1][RATE:])
    run(null, room(3., 41, person=True), first=2 * 10 ** 6, learn=True, clean=True)   # he talks over me
    after = level_db(run(null, person, first=3 * 10 ** 6, clean=True)[1][RATE:])
    assert (after > before - 3.) if gate else (after < before - 10.)
    assert null.good


def test_a_stale_pattern_is_learned_anew():
    """Moved to another room (or the speaker shifted): my voice reaches the microphones another way."""
    null = learned()
    other = ((2.5, 0., 4.), (1., 1.4, .6))
    rng = np.random.default_rng(8)
    moved = (at_three(voice_like(30., 50), *other) + rng.normal(0, 40, (30 * RATE, 3))).astype(np.int16)
    _, out = run(null, moved, first=10 ** 6, learn=True, clean=True)
    late = slice(25 * RATE, len(out))
    assert null.good and level_db(out[late]) < level_db(moved[late, 0]) - 15


def test_the_pattern_is_kept_and_read_back(tmp_path):
    path = tmp_path / 'robot-voice-null.npz'
    null = learned()
    null.path = path
    null.save()
    again = RobotNull(path)
    assert again.learned == null.learned and again.good and abs(again.depth - null.depth) < 1e-9
    null._compute()                                                       # the filters of the pattern as saved
    assert np.allclose(again.W, null.W) and np.allclose(again.fit, null.fit)
    json.dumps(again.summary())                                           # into voice.state as it is
    kept = tmp_path / 'state' / 'robot-voice-null.npz'
    kept.parent.mkdir()
    started = RobotNull(kept, seed=path)                                  # nothing learned here yet: the release's seed
    assert started.good and started.learned == null.learned
    run(started, room(1., 12), learn=True, clean=True)
    started.save()
    assert kept.exists() and RobotNull(kept, seed=path).learned > null.learned     # its own pattern from now on
    path.write_bytes(b'not a pattern')
    assert not RobotNull(path).ready                                      # a bad file: learned afresh


def multi_packets(x, epoch=1):
    for k in range(len(x) // 512):
        block = x[k * 512:(k + 1) * 512]
        yield packet(b'BRM1', k, k * 512, [adpcm_block(block[:, c].copy()) for c in range(3)], 1, 0b1101, epoch)


def test_through_the_ears_the_voice_is_the_same_while_i_am_silent():
    x = room(2., 9)
    plain, filtered = [], []
    a = array.ArrayLink('127.0.0.1', voice='multi', on_voice=lambda m, f: plain.append((m, f)))
    b = array.ArrayLink('127.0.0.1', voice='multi', on_voice=lambda m, f: filtered.append((m, f)), null=RobotNull())
    b.robot = lambda: (False, False)
    for data in multi_packets(x):
        a._packet(data, ('192.0.2.85', 5771))
        b._packet(data, ('192.0.2.85', 5771))
    assert len(filtered) >= len(plain) - 1
    assert [f for _, f in filtered] == [f for _, f in plain][:len(filtered)]
    assert b.summary()['null']['learned'] == 0


def test_through_the_ears_my_voice_is_taken_out_while_i_speak():
    null = learned()
    frames = []
    link = array.ArrayLink('127.0.0.1', voice='multi', on_voice=lambda m, f: frames.append(f), null=null)
    link.robot = lambda: (True, True)
    assert link.null_good()
    x = room(2., 23)
    for data in multi_packets(x):
        link._packet(data, ('192.0.2.85', 5771))
    heard = np.frombuffer(b''.join(f for f in frames if f), np.int16)
    assert level_db(heard[RATE // 2:]) < level_db(x[RATE // 2:len(heard), link.pick[2]]) - 12


def test_a_broken_filter_never_takes_the_ears_down():
    frames = []
    broken = types.SimpleNamespace(reset=lambda: None, good=True, summary=lambda: {},
                                   feed=lambda *a, **kw: 1 / 0)
    link = array.ArrayLink('127.0.0.1', voice='multi', on_voice=lambda m, f: frames.append(f), null=broken)
    for data in multi_packets(room(.5, 2)):
        link._packet(data, ('192.0.2.85', 5771))
    assert link.null is None and 'ZeroDivisionError' in link.summary()['null'] and len(frames) >= 20


# ---------------------------------------------------------------- the voice
def voice_stub(speaking, good):
    stub = types.SimpleNamespace(config={'ears': {'asr': 'auto'}}, ears_source='array', ears_switches=0,
                                 ears_healthy_since=time.monotonic() - 60)
    stub.audio = types.SimpleNamespace(speaking=speaking, speaker_until=time.monotonic() + (1. if speaking else -30.))
    stub.ears = types.SimpleNamespace(healthy=lambda now=None: True, usable=lambda now=None: True,
                                      null_good=lambda: good)
    return stub


@pytest.mark.parametrize('speaking,good,want', [
    (False, False, 'array'), (True, False, 'phone'),     # no null: the phone hears while I speak, as before
    (True, True, 'array'), (False, True, 'array')])      # the null is good: the ears hear through my voice
def test_while_i_speak_the_ears_listen_when_they_can_take_my_voice_out(speaking, good, want):
    assert Voice.listen_with(voice_stub(speaking, good)) == want


def test_my_sound_is_in_the_room_while_i_speak_and_a_second_after_it_ends_or_is_cut():
    audio = Audio()
    stub = types.SimpleNamespace(audio=audio)
    assert Voice.robot_sound(stub) == (False, False)
    audio.speaker_until = time.monotonic() + 2.
    assert Voice.robot_sound(stub) == (True, True)
    audio.mode = None
    audio.playback_id = None
    audio.stop_playing()                                                  # cut off by the owner
    assert Voice.robot_sound(stub) == (False, True)
    audio.stopped_at -= 1.5
    assert Voice.robot_sound(stub) == (False, False)


def test_the_phone_grammar_rests_while_i_speak():
    import queue
    fed, resets = [], []
    stub = types.SimpleNamespace(
        phone_frames=queue.Queue(), ears_source='array', dialog=None, confirmer=None, main_ring=[],
        audio=types.SimpleNamespace(speaking=True), event=lambda kind, **d: None,
        phone_listener=types.SimpleNamespace(feed=lambda mono, frame, **kw: fed.append(mono), reset=lambda: resets.append(1),
                                             preroll=[b'old'], finals=[], phrase=[]))
    for k in range(5):
        stub.phone_frames.put((k * .02, b'\0\0' * 320))
    assert not Voice.phone_names(stub, None)
    assert fed == [] and resets == [1] and stub.phone_listener.preroll == []
    stub.audio.speaking = False
    stub.phone_frames.put((1., b'\0\0' * 320))
    Voice.phone_names(stub, None)
    assert fed == [1.]


def test_cut_off_i_am_quiet_at_once_though_the_app_still_says_playing():
    """19.10: the app's «playing» lagged seconds after a barge-in cut; the robot listened only after it, and the
    owner's question had gone from the kept frames."""
    audio = Audio()
    audio.mode, audio.sock = 'media', Socket()
    audio.status = {'playing': True}
    assert audio.speaking                                                 # the app plays something of mine
    audio.play(b'\0\0' * 320)
    audio.stop_playing()
    assert not audio.speaking                                             # cut: quiet, whatever the app says
    audio.play(b'\0\0' * 320)
    audio.speaker_until = 0.
    assert audio.speaking                                                 # something new: the app is believed again


# ---------------------------------------------------------------- the speaker's volume
class Socket:
    def __init__(self):
        self.sent = b''

    def sendall(self, data):
        self.sent += data[5:]                                             # [type][length] then PCM


def test_the_speaker_is_quieter_by_the_volume_and_odd_pieces_keep_whole_samples():
    audio = Audio()
    audio.volume = .7
    audio.mode, audio.sock = 'media', Socket()
    pcm = (np.arange(-500, 500) * 30).astype('<i2').tobytes()
    for piece in (pcm[:301], pcm[301:1000], pcm[1000:]):                   # cut mid-sample
        audio.play(piece)
    out = np.frombuffer(audio.sock.sent, '<i2')
    want = np.clip(np.frombuffer(pcm, '<i2') * np.float32(.7), -32768, 32767).astype(np.int16)
    assert np.array_equal(out, want)
    audio.play(b'\x01')                                                   # a broken stream's last byte…
    audio.stop_playing()
    audio.sock.sent = b''
    audio.play(pcm[:4])                                                   # …does not shift the next one
    assert np.array_equal(np.frombuffer(audio.sock.sent, '<i2'), want[:2])
