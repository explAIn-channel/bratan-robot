"""Ears in the body (C51, 17.10): the ESP32 array's packets, the voice frames for recognition, where a sound came from,
and what the mind does with it.

python -m pytest tests/test_ears.py
"""
import json
import math
import struct
import sys
import time
import types

import numpy as np
import pytest

from bratan.voice import array
from bratan.voice.main import Voice
from bratan.mind import life as lifemod
from tests.test_mind import clock, world  # noqa: F401  (fixtures)
from tests.test_senses import make_life, turns

RATE = array.RATE
STEP = [7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97,
        107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
        876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871,
        5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623,
        27086, 29794, 32767]
INDEX = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]


def encode(samples, pred=0, index=0):
    """The board's encoder (work/ushi/src/net.cpp, adpcm_code): first sample in the low nibble."""
    def code(sample):
        nonlocal pred, index
        step = STEP[index]
        diff = int(sample) - pred
        c = 0
        if diff < 0:
            c, diff = 8, -diff
        delta = step >> 3
        if diff >= step:
            c |= 4; diff -= step; delta += step
        step >>= 1
        if diff >= step:
            c |= 2; diff -= step; delta += step
        step >>= 1
        if diff >= step:
            c |= 1; delta += step
        pred = max(-32768, min(32767, pred - delta if c & 8 else pred + delta))
        index = max(0, min(88, index + INDEX[c]))
        return c
    start = (pred, index)
    out = bytearray()
    for i in range(0, len(samples), 2):
        lo, hi = code(samples[i]), code(samples[i + 1])
        out.append(lo | hi << 4)
    return start, bytes(out), (pred, index)


def packet(tag, seq, first, blocks, codec, slot_or_mask, epoch=1, n=512):
    head = struct.pack('<4sIIHBBHBB', tag, seq, first, n, codec, slot_or_mask, epoch, len(blocks), 0)
    return head + b''.join(blocks)


def adpcm_block(samples, state=(0, 0)):
    (pred, index), data, _ = encode(samples, *state)
    return struct.pack('<hBB', pred, index, 0) + data


# ---------------------------------------------------------------- packets
def test_adpcm_from_the_board_decodes_close_to_the_original():
    t = np.arange(512) / RATE
    x = (3000 * np.sin(2 * np.pi * 440 * t) + 800 * np.sin(2 * np.pi * 1700 * t)).astype(np.int16)
    (pred, index), data, _ = encode(x)
    y = array.adpcm_decode(data, pred, index)
    assert len(y) == 512
    err = np.sqrt(np.mean((y[64:].astype(float) - x[64:]) ** 2)) / np.sqrt(np.mean(x[64:].astype(float) ** 2))
    assert err < .15                                          # ADPCM: about −20 dB of error after the step settles


def test_voice_and_multi_packets_parse():
    pcm = (np.arange(512) % 100 - 50).astype('<i2')
    kind, seq, first, epoch, slots, block = array.parse(packet(b'BRV1', 7, 1024, [pcm.tobytes()], 0, 2))
    assert (kind, seq, first, epoch, slots) == ('voice', 7, 1024, 1, (2,)) and np.array_equal(block[:, 0], pcm)
    chans = [np.full(512, v, np.int16) for v in (100, -200, 300)]
    raw = packet(b'BRM1', 3, 512, [adpcm_block(c) for c in chans], 1, 0b1101)
    kind, seq, first, epoch, slots, block = array.parse(raw)
    assert kind == 'multi' and slots == (0, 2, 3) and block.shape == (512, 3)
    assert np.allclose(block[-100:].mean(axis=0), (100, -200, 300), atol=20)
    assert array.parse(raw[:-10]) is None and array.parse(b'xx') is None


# ---------------------------------------------------------------- voice frames for recognition
class Sink:
    def __init__(self):
        self.items = []

    def __call__(self, mono, frame):
        self.items.append((mono, frame))

    def frames(self):
        return [f for _, f in self.items if f is not None]

    def gaps(self):
        return sum(1 for _, f in self.items if f is None)


def feed(link, first, epoch=1, value=1, n=512):
    link._voice(epoch, first, np.full(n, value, np.int16), 100.)


def test_voice_is_cut_into_20_ms_frames_in_order():
    sink = Sink()
    link = array.ArrayLink('127.0.0.1', on_voice=sink)
    for k in range(5):
        feed(link, 512 * k, value=k + 1)
    frames = sink.frames()
    assert len(frames) == 512 * 5 // 320 and all(len(f) == 640 for f in frames)
    times = [m for m, f in sink.items]
    assert times == sorted(times) and abs(times[1] - times[0] - .02) < 1e-9


def test_a_lost_packet_is_silence_a_long_loss_or_a_restart_is_a_gap():
    sink = Sink()
    link = array.ArrayLink('127.0.0.1', on_voice=sink)
    feed(link, 0)
    feed(link, 1024)                                          # one 32 ms packet lost: filled with silence
    assert sink.gaps() == 0 and len(sink.frames()) == 1536 // 320
    feed(link, 1024 + 512 + 16000)                            # a second lost: the recognizer starts afresh
    assert sink.gaps() == 1
    feed(link, 0, epoch=2)                                    # the board restarted its microphones
    assert sink.gaps() == 2
    before = len(sink.frames())
    feed(link, 0, epoch=2)                                    # a repeat is dropped
    assert len(sink.frames()) == before


def test_both_streams_arrive_whole_through_real_sockets():
    """A fake board on this machine answers the subscriptions; nothing may be lost on the way in (17.10: waiting
    50 ms on each socket in turn dropped half of the three-microphone packets when the voice socket was idle)."""
    import socket
    import threading
    board = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    board.bind(('127.0.0.1', 0))
    board.settimeout(.05)
    subscribers = {}
    stop = threading.Event()

    def serve():
        seq = {'v': 0, 'm': 0}
        first = 0
        next_at = time.monotonic()
        while not stop.is_set():
            try:
                data, addr = board.recvfrom(256)
                if data.startswith(b'sub'):
                    subscribers[data[:4]] = addr
            except socket.timeout:
                pass
            while time.monotonic() >= next_at and subscribers:
                next_at += .032
                tone = (1000 * np.sin(np.arange(first, first + 512) / 5)).astype(np.int16)
                if b'subv' in subscribers:
                    board.sendto(packet(b'BRV1', seq['v'], first, [tone.astype('<i2').tobytes()], 0, 0),
                                 subscribers[b'subv'])
                    seq['v'] += 1
                if b'subm' in subscribers:
                    board.sendto(packet(b'BRM1', seq['m'], first, [adpcm_block(tone)] * 3, 1, 0b1101),
                                 subscribers[b'subm'])
                    seq['m'] += 1
                first += 512
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    sink = Sink()
    link = array.ArrayLink('127.0.0.1', port=board.getsockname()[1], on_voice=sink).start()
    try:
        time.sleep(2.5)
        assert link.loss('voice') == 0 and link.loss('multi') == 0
        assert link.healthy() and len(sink.frames()) > 40 and sink.gaps() == 0
        assert link.ring.latest(.5) is not None
    finally:
        link.stop()
        stop.set()
        thread.join(1)
        board.close()


def test_voice_from_the_three_microphones_picks_the_clearest_and_needs_no_voice_stream():
    """C52: Wi-Fi in the body fell to −85…−93 dBm; the voice now comes from the three microphones' stream (half the
    traffic). The microphone with the voice highest over its own noise is heard — not the loudest (М3 is loud in
    silence too)."""
    frames = []
    link = array.ArrayLink('127.0.0.1', voice='multi', on_voice=lambda mono, f: frames.append(f))
    assert link.voice_request is None and link.multi_request == b'subm 023'
    rng = np.random.default_rng(2)
    first = 0
    for k in range(80):
        n = rng.normal(0, 1, (512, 3)) * np.array([50., 50., 200.])      # М3 noisier
        if k >= 40:
            n[:, 1] += np.sin(np.arange(512) / 5.) * 2000.                # a voice on М2
        chans = [adpcm_block(np.clip(n[:, c], -32768, 32767).astype(np.int16)) for c in range(3)]
        link._packet(packet(b'BRM1', k, first, chans, 1, 0b1101), ('127.0.0.1', 5771))
        first += 512
    assert link.voice_slot == 2 and link.healthy()
    assert len(frames) == 80 * 512 // 320


def test_a_window_that_comes_both_by_bluetooth_and_wifi_counts_once():
    """The bridge (C52): the face app forwards the board's packets received over Bluetooth; while it is alive the Wi-Fi
    stream is not ordered, and a window arriving both ways (the switch-over) is taken once."""
    frames = []
    link = array.ArrayLink('127.0.0.1', voice='multi', on_voice=lambda mono, f: frames.append(f), local_port=5772)
    assert not link.via_bluetooth()
    tone = (np.sin(np.arange(512) / 4.) * 3000).astype(np.int16)
    for k in range(10):
        data = packet(b'BRM1', 100 + k, k * 512, [adpcm_block(tone)] * 3, 1, 0b1101)
        link._packet(data, ('127.0.0.1', 40000), local=True)
        link._packet(packet(b'BRM1', k, k * 512, [adpcm_block(tone)] * 3, 1, 0b1101), ('192.0.2.85', 5771))
    assert link.via_bluetooth() and link.summary()['via'] == 'both'
    assert len(frames) == 10 * 512 // 320 and link.loss('multi') == 0.


def test_a_board_restart_is_heard_at_once():
    """18.10 19:30: the board was reflashed over USB; its frame count began at 0 again under the same epoch, and every
    new window looked like one already had — the ears heard nothing until the voice was restarted."""
    frames = []
    link = array.ArrayLink('127.0.0.1', voice='multi', on_voice=lambda mono, f: frames.append(f))
    tone = [adpcm_block((np.sin(np.arange(512) / 4.) * 3000).astype(np.int16))] * 3
    for k in range(5):                                                   # long into its run
        link._packet(packet(b'BRM1', 1000 + k, 10_000_000 + k * 512, tone, 1, 0b1101), ('192.0.2.85', 5771))
    before = len([f for f in frames if f])
    for k in range(5):                                                   # restarted: from 0, epoch 1 again
        link._packet(packet(b'BRM1', k, k * 512, tone, 1, 0b1101), ('192.0.2.85', 5771))
    assert len([f for f in frames if f]) > before + 5 and link.loss('multi') < 1.


def test_bluetooth_goes_alone_when_wifi_is_much_worse_and_waits_longer_after_a_failure():
    """18.10 18:45: Wi-Fi at −89 dBm lost most windows, its retries took the radio from Bluetooth (both ~20 % lost)
    and the ears were «unhealthy»: Bluetooth alone gave every window. Wi-Fi far worse for 5 s → Bluetooth alone; it
    fails alone → both again, and the next try alone only after a growing pause."""
    link = array.ArrayLink('127.0.0.1', voice='multi', local_port=5772)
    t0 = time.monotonic()
    now = time.monotonic()
    link.seen['local'].extend((now, k) for k in range(0, 60) if k % 6)          # Bluetooth loses 1 in 6
    link.seen['wifi'].extend((now, k) for k in range(0, 60) if k % 2 == 0)      # Wi-Fi loses half
    link.last_local = t0
    assert not link.bluetooth_alone(t0)
    link.last_local = t0 + 5.
    assert link.bluetooth_alone(t0 + 5.)                                          # Wi-Fi far worse: alone
    link.last_local = t0 + 6.
    assert not link.bluetooth_alone(t0 + 6.)                                      # alone it loses > 5 %: both
    link.last_local = t0 + 12.
    assert not link.bluetooth_alone(t0 + 12.) and not link.bluetooth_alone(t0 + 15.9)   # waiting 20 s first


def test_both_paths_with_windows_half_a_window_apart_make_one_gapless_voice():
    """Wi-Fi and Bluetooth cut the stream into windows from different frames (the board's subscribers start apart);
    with every other Wi-Fi window lost the Bluetooth windows (256 frames later) fill the holes: every frame once,
    no silence patched in."""
    frames = []
    link = array.ArrayLink('127.0.0.1', voice='multi', on_voice=lambda mono, f: frames.append(f), local_port=5772)
    rng = np.random.default_rng(3)
    noise = lambda: [adpcm_block(rng.normal(0, 3000, 512).clip(-32768, 32767).astype(np.int16))] * 3
    for k in range(12):
        if k % 2 == 0:
            link._packet(packet(b'BRM1', k, k * 512, noise(), 1, 0b1101), ('192.0.2.85', 5771))
        link._packet(packet(b'BRM1', 100 + k, k * 512 + 256, noise(), 1, 0b1101), ('127.0.0.1', 40000), local=True)
    assert link.pending_first + len(link.pending) == 12 * 512 + 256
    voice = np.frombuffer(b''.join(frames), np.int16)
    assert len(frames) == (12 * 512 + 256) // 320 and None not in frames
    silent = np.convolve(voice == 0, np.ones(32), 'valid').max()
    assert silent < 32                                                  # no patched-in silence anywhere


def speech_at_three(seconds=6., delays=(0., 5.3, -3.2), noise=(300., 300., 450.), seed=5):
    """A voice-band signal reaching three microphones with fractional delays (samples) and their own noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    wide = rng.normal(0, 1, n + 400)
    spec = np.fft.rfft(wide)
    f = np.fft.rfftfreq(len(wide), 1 / RATE)
    spec[(f < 250) | (f > 3800)] = 0
    envelope = np.repeat(rng.uniform(.2, 1., n // 1600 + 2), 1600)[:len(wide)]       # syllables
    channels, clean = [], []
    for d, sigma in zip(delays, noise):
        voice = np.fft.irfft(spec * np.exp(-2j * np.pi * f * d / RATE), len(wide))
        voice = voice / voice.std() * 2000. * envelope
        clean.append(voice[200:200 + n])
        channels.append(voice[200:200 + n] + rng.normal(0, sigma, n))
    return np.stack(channels, 1).clip(-32768, 32767).astype(np.int16), np.stack(clean, 1)


def test_the_live_sum_finds_the_delays_and_hears_the_voice_over_more_noise():
    from bratan.voice.beam import Beam
    x, clean = speech_at_three()
    beam = Beam()
    out = np.concatenate([beam.feed(x[i:i + 512], 0) for i in range(0, len(x) - 511, 512)])
    assert beam.use[1] and beam.use[2]
    assert abs(beam.d[1] - 5.3) < .5 and abs(beam.d[2] + 3.2) < .5
    late = slice(3 * RATE, len(out))                                   # once learned
    truth = clean[late.start - Beam.LAG:len(out) - Beam.LAG, 0]
    def snr(track):
        gain = (track * truth).sum() / (truth * truth).sum()
        return 10 * np.log10((truth ** 2).sum() * gain ** 2 / ((track - gain * truth) ** 2).sum())
    best = snr(x[late.start - Beam.LAG:len(out) - Beam.LAG, 0].astype(float))
    summed = snr(out[late].astype(float))
    assert summed > best + 2.                                          # three ears hear more than one


def test_with_no_voice_the_live_sum_is_the_reference_microphone_itself():
    from bratan.voice.beam import Beam
    rng = np.random.default_rng(2)
    x = rng.normal(0, 300, (512 * 20, 3)).astype(np.int16)
    beam = Beam()
    out = np.concatenate([beam.feed(x[i:i + 512], 1) for i in range(0, len(x), 512)])
    assert not beam.use.any()
    assert np.array_equal(out[Beam.LAG:], x[:len(x) - Beam.LAG, 1])


def test_the_summed_voice_takes_each_frame_once_from_both_paths():
    frames = []
    link = array.ArrayLink('127.0.0.1', voice='multi', on_voice=lambda mono, f: frames.append(f), local_port=5772,
                           mix='sum')
    rng = np.random.default_rng(3)
    noise = lambda: [adpcm_block(rng.normal(0, 3000, 512).clip(-32768, 32767).astype(np.int16))] * 3
    for k in range(12):
        if k % 2 == 0:
            link._packet(packet(b'BRM1', k, k * 512, noise(), 1, 0b1101), ('192.0.2.85', 5771))
        link._packet(packet(b'BRM1', 100 + k, k * 512 + 256, noise(), 1, 0b1101), ('127.0.0.1', 40000), local=True)
    assert link.pending_first + len(link.pending) == 12 * 512 + 256 and None not in frames
    assert len(frames) == (12 * 512 + 256) // 320


def test_wifi_is_dropped_only_after_the_bridge_proves_itself_and_returns_when_it_loses():
    """Nothing gets worse with the bridge: Wi-Fi stays ordered until Bluetooth alone loses < 2 % for 5 s, and comes
    back as soon as Bluetooth loses > 5 % (from the PC the bridge gave 23 of 31 packets a second)."""
    link = array.ArrayLink('127.0.0.1', voice='multi', local_port=5772)
    tone = [adpcm_block((np.sin(np.arange(512) / 4.) * 3000).astype(np.int16))] * 3
    t0 = time.monotonic()
    link.last_local = t0
    link.seen['local'].extend((t0, k) for k in range(60))
    assert not link.bluetooth_alone(t0)
    link.last_local = t0 + 4.
    assert not link.bluetooth_alone(t0 + 4.)
    link.last_local = t0 + 5.
    assert link.bluetooth_alone(t0 + 5.) and link.summary()['via'] == 'bluetooth'
    link.seen['local'].extend((time.monotonic(), k) for k in range(60, 100, 4))     # three windows in four lost
    assert not link.bluetooth_alone(t0 + 5.1) and link.summary()['via'] == 'both'
    link.last_local = t0 - 10.                                                      # the bridge went quiet
    assert not link.bluetooth_alone(t0) and link.summary()['via'] == 'wifi'
    link._packet(packet(b'BRM1', 0, 0, tone, 1, 0b1101), ('127.0.0.1', 40000), local=True)
    assert link.via_bluetooth()


def test_loss_and_health():
    link = array.ArrayLink('127.0.0.1')
    now = time.monotonic()
    link.seen['voice'].extend((now, s) for s in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11))
    assert abs(link.loss('voice') - 1 / 11) < 1e-9
    link.last_voice = now
    assert link.healthy(now) and not link.healthy(now + 1.)


# ---------------------------------------------------------------- where it came from
def burst(gains_db, delays, seconds=1.5, seed=1, quiet=6.):
    """A ring with `quiet` s of faint noise, then a voice-band burst louder on some microphones (the body's shadow)
    and later on others (delays in samples)."""
    rng = np.random.default_rng(seed)
    n_quiet, n_loud = int(quiet * RATE), int(seconds * RATE)
    noise = rng.normal(0, 30, (n_quiet + n_loud, 3))
    src = rng.normal(0, 1, n_loud + 64)
    spec = np.fft.rfft(src)
    f = np.fft.rfftfreq(len(src), 1 / RATE)
    spec[(f < 300) | (f > 5000)] = 0
    src = np.fft.irfft(spec, len(src))
    src *= 2000 / src.std()
    x = noise.copy()
    for ch in range(3):
        d = int(delays[ch])
        x[n_quiet:, ch] += 10 ** (gains_db[ch] / 20) * src[32 - d:32 - d + n_loud]
    ring = array.Ring()
    ring.put(1, 0, np.clip(x, -32768, 32767).astype(np.int16), 0.)
    return ring


def model_delays(deg):
    """Arrival delays (samples, ≥0) of a sound from `deg` at М1, М2, М3 by the body model."""
    i = list(array.MODEL_DEG).index(int(round(array._wrap_deg(deg))))
    t = np.array([0., array.MODEL_TAU[i][0], array.MODEL_TAU[i][1]])
    return np.round(t - t.min()).astype(int)


@pytest.mark.parametrize('deg', [0, 45, 90, 135, 180, -45, -90, -135])
def test_the_side_comes_from_the_delays_at_syllable_starts(deg):
    found = array.Direction(burst((0, 0, 0), model_delays(deg))).estimate(2.)
    assert found is not None and found['method'] == 'model'
    assert abs(array._wrap_deg(found['deg'] - deg)) <= 20, found
    assert found['confidence'] >= .3


def test_loudness_alone_does_not_decide():
    """18.10: М3 is always the loudest (its sensitivity), wherever the voice comes from: a voice from the front,
    loud on М3, is still in front."""
    found = array.Direction(burst((-8, -4, 0), model_delays(0))).estimate(2.)
    assert abs(found['deg']) <= 25 and found['loudest'] == 'М3', found

def test_our_own_voice_does_not_count():
    """The burst (1.5 s to the ring's end) is the robot's own speech: its start is the only syllable start. With
    not_before after that start nothing is left; with not_before before it, it counts."""
    ring = burst((0, 0, 0), model_delays(0))
    end_mono = ring.end_mono
    assert array.Direction(ring).estimate(2.) is not None
    assert array.Direction(ring).estimate(2., not_before=end_mono - 1.) is None
    assert array.Direction(ring).estimate(2., not_before=end_mono - 1.8) is not None


def voice_band(rng, n):
    src = rng.normal(0, 1, n + 64)
    spec = np.fft.rfft(src)
    f = np.fft.rfftfreq(len(src), 1 / RATE)
    spec[(f < 300) | (f > 5000)] = 0
    src = np.fft.irfft(spec, len(src))
    return src / src.std()


def add_sound(x, start, n, deg, amplitude, rng, envelope=None):
    src = voice_band(rng, n) * amplitude
    if envelope is not None:
        src[:n] *= envelope
    for ch, d in enumerate(model_delays(deg)):
        x[start:start + n, ch] += src[32 - d:32 - d + n]


def test_the_tail_of_a_sound_keeps_its_side():
    """18.10, the owner: «when the sound dies away it shows the last direction wrong — the reflections». A two-syllable
    word from the right, then its echo from a wall dying away with one bump: a second later the window holds only the
    end of the word and the echo, and still says «справа» — the sound counts from its own start."""
    rng = np.random.default_rng(5)
    x = rng.normal(0, 30, (int(8 * RATE), 3))
    add_sound(x, int(6. * RATE), int(.25 * RATE), -90, 2000, rng)
    add_sound(x, int(6.3 * RATE), int(.25 * RATE), -90, 2000, rng)
    tail = int(.5 * RATE)
    t = np.arange(tail) / RATE
    envelope = 10 ** (np.linspace(-8, -30, tail) / 20) * (1 + np.exp(-((t - .15) / .03) ** 2))
    add_sound(x, int(6.55 * RATE), tail, 60, 2000, rng, envelope)
    ring = array.Ring()
    ring.put(1, 0, np.clip(x[:int(7.45 * RATE)], -32768, 32767).astype(np.int16), 0.)
    found = array.Direction(ring).estimate(1.)
    assert found is not None and abs(array._wrap_deg(found['deg'] + 90)) <= 25, found
    assert found['side'] == 'справа'


def test_an_earlier_sound_does_not_count_for_a_new_one():
    """Two sounds with a pause between: the window that only holds the second one is about the second alone."""
    rng = np.random.default_rng(6)
    x = rng.normal(0, 30, (int(9 * RATE), 3))
    add_sound(x, int(5 * RATE), int(.6 * RATE), 90, 2000, rng)
    add_sound(x, int(7.2 * RATE), int(.6 * RATE), -90, 2000, rng)
    ring = array.Ring()
    ring.put(1, 0, np.clip(x[:int(8.2 * RATE)], -32768, 32767).astype(np.int16), 0.)
    found = array.Direction(ring).estimate(1.)
    assert found is not None and abs(array._wrap_deg(found['deg'] + 90)) <= 25, found


def test_silence_has_no_direction():
    ring = array.Ring()
    ring.put(1, 0, np.random.default_rng(2).normal(0, 30, (RATE * 8, 3)).astype(np.int16), 0.)
    assert array.Direction(ring).estimate(2.) is None


def test_calibration_takes_away_the_models_lean(tmp_path):
    cal = array.Calibration(tmp_path / 'cal.json')
    for deg, fit in ((0, 19), (45, 78), (90, 106), (135, 145), (180, 160), (-135, -139), (-90, -74), (-45, -38)):
        cal.add(deg, fit, (0., 0., 0.))
    cal.save()
    again = array.Calibration(tmp_path / 'cal.json')
    assert again.ready()
    assert abs(again.correct(19) - 0) < 1 and abs(again.correct(78) - 45) < 1
    # Between two points the shift is between theirs: fit 160 → +20°, fit −139 → +4° (across ±180 too).
    for fit in (170, -170, -150):
        assert 4 - 1e-6 <= array._wrap_deg(again.correct(fit) - fit) <= 20 + 1e-6, fit


def test_old_calibration_entries_are_ignored(tmp_path):
    path = tmp_path / 'old.json'
    path.write_text('{"version": 1, "entries": [{"deg": 0, "ild": [1, 2, 3], "tau23": null}]}', encoding='utf-8')
    assert not array.Calibration(path).ready()

def test_side_words():
    assert [array.side_words(math.radians(d)) for d in (0, 40, 90, 140, 180, -40, -90, -140)] == [
        'спереди', 'слева-спереди', 'слева', 'слева-сзади', 'сзади', 'справа-спереди', 'справа', 'справа-сзади']


# ---------------------------------------------------------------- whose microphone hears words
def voice_stub(asr, speaking=False, healthy=True, speaker_ago=10.):
    stub = types.SimpleNamespace(config={'ears': {'asr': asr}}, ears_source='phone', ears_switches=0)
    stub.audio = types.SimpleNamespace(speaking=speaking, speaker_until=time.monotonic() - speaker_ago)
    stub.ears = types.SimpleNamespace(healthy=lambda now=None: healthy, usable=lambda now=None: healthy)
    return stub


@pytest.mark.parametrize('asr,speaking,healthy,ago,want', [
    ('phone', False, True, 10., 'phone'),
    ('auto', False, True, 10., 'array'),
    ('auto', True, True, 10., 'phone'),            # we speak: the phone cancels our echo
    ('auto', False, True, .3, 'phone'),            # our echo is still in the room
    ('auto', False, False, 10., 'phone'),          # the stream failed: the phone at once
    ('array', True, True, 10., 'array'),
    ('array', False, False, 10., 'phone'),
])
def test_listen_with(asr, speaking, healthy, ago, want):
    stub = voice_stub(asr, speaking, healthy, ago)
    assert Voice.listen_with(stub) == want
    assert stub.ears_switches == (want != 'phone')


def test_after_a_bad_stretch_the_ears_must_hold_before_words_go_back_to_them():
    stub = voice_stub('auto')
    now = time.monotonic()
    ears = lambda healthy, usable: types.SimpleNamespace(healthy=lambda now=None: healthy,
                                                          usable=lambda now=None: usable)
    stub.ears = ears(False, False)
    assert Voice.listen_with(stub, now) == 'phone'
    stub.ears = ears(True, True)
    assert Voice.listen_with(stub, now + 1.) == 'phone'          # healthy again, but only for a moment
    assert Voice.listen_with(stub, now + 11.5) == 'array'         # held for 10 s
    stub.ears = ears(False, True)                                # lossy (10–20 %) but usable: no swing mid-word
    assert Voice.listen_with(stub, now + 12.) == 'array'
    stub.ears = ears(False, False)
    assert Voice.listen_with(stub, now + 13.) == 'phone'          # really failing: away at once


@pytest.mark.parametrize('after, want', [(.05, 'stop'), (.95, None)])
def test_a_stop_with_my_name_counts_once_quiet_follows_it(monkeypatch, after, want):
    """18.10 18:33: «Братан, сколько сейчас времени» came out of the grammar as «братан стой» and stopped the body.
    With the name too, «стоп» counts once the voice stops after it; words going on are another phrase. (Over my own
    voice OverMe hears «стоп» through Microsoft instead.)"""
    class NameStop(StopRecognizer):
        def Result(self):
            return ('{"text": "братан стоп", "result": [{"word": "братан", "conf": 0.95}, '
                    '{"word": "стоп", "conf": 0.95}]}')
    module = types.ModuleType('vosk')
    module.KaldiRecognizer = NameStop
    monkeypatch.setitem(sys.modules, 'vosk', module)
    from bratan.voice.wake import Listener
    rng = np.random.default_rng(9)
    listener = Listener(model=None)
    loud = lambda: (rng.normal(0, .05 * 32768, 320)).astype('<i2').tobytes()
    results = [listener.feed(k * .02, loud(), speech=.95) for k in range(10)]           # «братан стоп» decoded
    results += [listener.feed(.2 + k * .02, loud(), speech=after) for k in range(30)]    # then quiet / more words
    assert (want in results) if want else ('stop' not in results)


def heard_while_speaking(monkeypatch, said):
    class Heard(StopRecognizer):
        def Result(self):
            return json.dumps({'text': said, 'result': [{'word': w, 'conf': .9} for w in said.split()]},
                              ensure_ascii=False)
    module = types.ModuleType('vosk')
    module.KaldiRecognizer = Heard
    monkeypatch.setitem(sys.modules, 'vosk', module)
    from bratan.voice.wake import Listener
    rng = np.random.default_rng(9)
    listener = Listener(model=None)
    return listener, [listener.feed(k * .02, (rng.normal(0, .05 * 32768, 320)).astype('<i2').tobytes(), speech=.95,
                                    strict=True) for k in range(12)]


def test_while_i_speak_only_the_name_itself_wakes(monkeypatch):
    listener, results = heard_while_speaking(monkeypatch, 'братан')
    assert 'wake' in results and listener.woke_by == 'братан'
    _, results = heard_while_speaking(monkeypatch, 'врата')        # sounds like it from afar: not over my voice
    assert 'wake' not in results
    _, results = heard_while_speaking(monkeypatch, 'братана')      # about me, not to me (the barge A/B, 18.10)
    assert 'wake' not in results


def barge_stub(dialog=None, spoken=()):
    import queue
    events, published = [], []
    stub = types.SimpleNamespace(
        listener=types.SimpleNamespace(woke_by='братан', phrase=[b'f%d' % k for k in range(100)]),
        spoken_log=[(time.monotonic() - 2., text) for text in spoken], dialog=dialog, say_queue=queue.Queue(),
        speech_generation=0, audio=types.SimpleNamespace(stopped=0), woken=[],
        event=lambda kind, **d: events.append(kind), heard_from=lambda s: {},
        bus=types.SimpleNamespace(publish=lambda topic, data: published.append((topic, data))))
    stub.audio.stop_playing = lambda: setattr(stub.audio, 'stopped', stub.audio.stopped + 1)
    stub.on_wake = lambda frames: stub.woken.append(frames)
    stub.cut_speech = lambda: Voice.cut_speech(stub)
    return stub, events, published


def test_the_name_while_i_speak_cuts_me_off_and_the_talk_hears_what_follows():
    """C52, the owner: «стоп» and corrections must be heard through his own speech. «Братан» stops the answer being
    said, cancels its rest and listens from the name on."""
    dialog = {'id': 'local-1', 'pending': True, 'waiting': True, 'first_part_at': 1., 'await_quiet': False,
              'heard': [], 'said': []}
    stub, events, published = barge_stub(dialog)
    stub.say_queue.put({'text': 'и ещё одно', 'local': True})
    assert Voice.barge_in(stub, 100.)
    assert stub.audio.stopped == 1 and stub.say_queue.empty() and stub.speech_generation == 1
    assert ('voice.cancel', {'id': 'local-1'}) in published and 'barge_in' in events
    assert dialog['id'] is None and dialog['await_quiet'] and not dialog['waiting']
    assert abs(dialog['barge_from'] - 99.7) < 1e-9


def test_the_name_while_i_speak_my_own_line_wakes_me_with_the_name_not_my_words():
    stub, events, _ = barge_stub()
    assert Voice.barge_in(stub, 100.)
    assert stub.woken and stub.woken[0] == stub.listener.phrase[-60:]


def test_my_own_name_in_what_i_say_does_not_cut_me_off():
    stub, events, _ = barge_stub(spoken=['Привет, я Братан!'])
    assert not Voice.barge_in(stub, 100.)
    assert events == ['barge_ignored'] and stub.audio.stopped == 0


def test_over_my_voice_only_a_persons_stop_word_counts():
    """C52: Microsoft over the microphone while I speak. Its own text of the A/B windows: the owner's «подожди» and
    «братан» count; my own words — even «стой» if I am saying it — do not."""
    from bratan.voice.overme import cues
    own = 'вода сама держит человека самое глубокое место'.split()
    assert cues('вода сама держит человека подожди', own) == ['подожди']
    assert cues('у рыбы и креветки братан киты', own) == ['братан']
    assert cues('Братан, стоп!', own) == ['братан', 'стоп']           # first, but his words follow it (19.10)
    assert cues('место стоп', own) == ['стоп']                        # first and my own after it: the window's edge
    assert cues('подожди', own) == ['подожди']                         # …unless it is all there is
    assert cues('марианская впадина почти 11 км', own) == []
    assert cues('стой тут не стой', ['стой', 'тут', 'не']) == []
    # 18.10 21:11: Microsoft wrote «слушай» at a window's start over my own «Осьминоги…»: not a cue any more
    assert cues('слушай могут попробовать', 'осьминоги видят мир кожей могут попробовать'.split()) == []
    assert cues('рюмки и стоп', ['рюмки', 'и', 'стопки']) == []        # my «стопки» cut to «стоп»
    # 19.10, the owner's test: I was saying «дожди», his «подожди» is not mine
    assert cues('подожди', 'на юпитере дожди идут из алмазов'.split()) == ['подожди']
    assert cues('идут подожди', 'на юпитере дожди идут из алмазов'.split()) == ['подожди']
    assert cues('и подожди', ['подождите']) == []                      # my «подождите» cut
    assert cues('и стой', ['стои']) == []                              # mine, misheard by a letter
    # 19.10 14:25: the window began with his «подожди» and his words followed — it counts now
    venus = 'на венере сутки длиннее года она вокруг себя оборачивается медленнее чем вокруг солнца'.split()
    assert cues('подожди а сколько сейчас', venus) == ['подожди']
    assert cues('подожди солнца', venus) == []                         # my word after it: the window's cut edge


def test_someone_elses_words_of_two_windows_together_cut_me_off(monkeypatch):
    """19.10 14:25: «…а сколько сейчас», «сколько сейчас времени» — one word of his in each window; together they cut
    in, and he is heard from the earlier window's start."""
    from bratan.voice import overme
    texts = iter(['солнца сколько сейчас', 'сколько сейчас времени'])
    got = []

    class Heard:
        def __init__(self, frames):
            pass

        def result(self, wait):
            return next(texts)
    watcher = overme.OverMe('k', 'westeurope', lambda: 'вокруг себя медленнее чем вокруг солнца'.split(),
                            lambda found, text, start: got.append((found, start)), lambda kind, **d: None)
    watcher.hearing.start = lambda frames: Heard(frames)
    monkeypatch.setattr(overme.threading, 'Thread',
                        lambda target, args, **kw: types.SimpleNamespace(start=lambda: target(*args)))
    for k in range(130):
        watcher.feed(k * .02, b'\0\0' * 320)
    assert len(got) == 1 and set(got[0][0]) == {'сколько', 'сейчас'}
    assert abs(got[0][1] - 0.) < .05                                   # from the first window's start


def test_over_me_sends_windows_while_i_speak_and_hands_the_cue_over(monkeypatch):
    from bratan.voice import overme
    sent, got, events = [], [], []

    class Heard:
        def __init__(self, frames):
            sent.append(len(frames))

        def result(self, wait):
            return 'и креветки братан киты'
    watcher = overme.OverMe('k', 'westeurope', lambda: ['и', 'креветки', 'киты'],
                            lambda found, text, start: got.append(found), lambda kind, **d: events.append(kind))
    watcher.hearing.start = lambda frames: Heard(frames)
    monkeypatch.setattr(overme.threading, 'Thread',
                        lambda target, args, **kw: types.SimpleNamespace(start=lambda: target(*args)))
    for k in range(250):                                     # 5 s of my speech
        watcher.feed(k * .02, b'\0\0' * 320)
    assert sent and all(n <= 100 for n in sent) and len(sent) == 4       # every 1.2 s, a 2 s window at most
    assert got == [['братан']] * 4 and events.count('over_me') == 4
    watcher.check_now(5.1)                                  # the grammar's word: Microsoft looks at once
    assert len(sent) == 5


def test_any_phrase_over_my_voice_cuts_me_off_and_my_own_words_do_not():
    """18.10, the owner: «хочу, чтобы можно было перебивать любой фразой». Windows Microsoft heard over my voice:
    the owner's «а сколько сейчас времени», «это понятно, давай дальше» interrupt; my own words misheard at the window's
    edges («после», «город», «сейчас минут» for «осьминог») do not; numbers it writes as digits are mine."""
    from bratan.voice.hearing import normalize
    from bratan.voice.overme import FOREIGN_MIN, foreign, strip_mine
    space = normalize('В космосе есть облако из спирта — размером с целую солнечную систему.').split()
    octo = normalize('Знаешь, у осьминога три сердца и голубая кровь. Два сердца гонят кровь через жабры, а третье '
                     'по всему телу. А ещё осьминог умеет менять цвет кожи за долю секунды.').split()
    assert len(foreign('солнечную систему а сколько сейчас времени', space)) >= FOREIGN_MIN
    assert len(foreign('облако из спирта это понятно давай дальше', space)) >= FOREIGN_MIN
    for mine in ('знаешь после', '2 сердца город', 'сейчас минут умеет менять цвет кожи за долю се',
                 'нет кровь через жабры а 3 по всему телу', 'осьминога 3 сердца и голубая кровь'):
        assert len(foreign(mine, octo)) < FOREIGN_MIN, mine
    assert strip_mine('солнечную систему а сколько сейчас времени', space) == 'а сколько сейчас времени'


def over_me_stub(dialog=None):
    stub, events, published = barge_stub(dialog)
    stub.bus = types.SimpleNamespace(publish=lambda topic, data: published.append((topic, data)))
    stub.session = None
    stub.ended = []
    stub.end_dialog = lambda why: stub.ended.append(why)
    stub.barge_in = lambda mono, word=None, start=None: events.append(('barge', word))
    return stub, events, published


def test_a_stop_word_over_my_voice_silences_me_and_ends_the_talk():
    stub, events, published = over_me_stub({'id': 'local-1'})
    Voice.over_me_cue(stub, ['стоп'], 'братан стоп', 48., 50., None)
    assert ('body.stop', {'reason': 'voice_stop'}) in published and stub.audio.stopped == 1
    assert 'stop' in events and stub.ended == ['stop']


def test_the_name_over_my_voice_silences_me_and_i_listen():
    stub, events, _ = over_me_stub({'id': 'local-1'})
    Voice.over_me_cue(stub, ['подожди'], 'подожди', 48., 50., None)
    assert ('barge', 'подожди') in events and not stub.ended


def test_the_name_heard_by_the_phone_wakes_me_while_the_ears_hear_words():
    """18.10, the owner: «далеко не с первого раза откликнулся». The ears' grammar missed «Братан»; the phone's
    microphone has its own grammar: it wakes me — and the talk is heard by the ears from the same moment (18.10 21:11:
    switching the talk to the phone restarted the speech detector and cut the phrase 1.5 s after the name)."""
    import queue
    from collections import deque
    woken, events = [], []
    heard = iter([None, None, 'wake'])
    ears = deque([(t / 50, b'e%d' % t) for t in range(-100, 5)])        # the ears' frames of the last 2 s
    stub = types.SimpleNamespace(
        phone_frames=queue.Queue(), ears_source='array', dialog=None, confirmer=None, main_ring=ears,
        audio=types.SimpleNamespace(speaking=False), event=lambda kind, **d: events.append(kind),
        phone_listener=types.SimpleNamespace(feed=lambda mono, frame, **kw: next(heard), woke_by='братан',
                                             phrase=[b'p'] * 25, finals=[]),         # 0.5 s of the phone's phrase
        on_wake=lambda frames: woken.append(frames))
    for k in range(3):
        stub.phone_frames.put((k * .02, b'\0\0' * 320))
    assert Voice.phone_names(stub, None)
    # woken at 0.04 s by a 0.5 s phrase: the ears from 0.04 - 0.5 - 0.3 = -0.76 s on, not the phone's frames
    assert woken[0][0] == b'e-38' and woken[0][-1] == b'e4' and events == ['wake_by_phone']
    stub.dialog = {'id': None}                                  # in a talk: the phone's grammar does not wake again
    stub.phone_frames.put((1., b'\0\0' * 320))
    heard = iter(['wake'])
    stub.phone_listener.feed = lambda mono, frame, **kw: next(heard)
    assert not Voice.phone_names(stub, None)


def test_in_quiet_hours_microsofts_misspelt_name_counts_after_vosks_clear_one():
    from bratan.voice.main import sounds_like_name
    assert sounds_like_name('брэдан сколько сейчас времени') and sounds_like_name('бретон какой сегодня день')
    assert sounds_like_name('бракенд сколько сейчас времени') and sounds_like_name('в браке сколько сейчас времени')
    assert not sounds_like_name('сколько сейчас времени братан')        # not in front: not a call
    assert not sounds_like_name('давай поговорим о погоде')


def test_barge_only_while_i_speak_on_my_own():
    stub = types.SimpleNamespace(audio=types.SimpleNamespace(speaking=True), confirmer=None, config={})
    assert Voice.can_barge(stub, None)
    assert Voice.can_barge(stub, types.SimpleNamespace(speak_only=True))
    assert not Voice.can_barge(stub, types.SimpleNamespace(speak_only=False))      # the live voice hears itself
    stub.config = {'ears': {'barge': False}}
    assert not Voice.can_barge(stub, None)
    stub.config, stub.audio.speaking = {}, False
    assert not Voice.can_barge(stub, None)


class StopRecognizer:
    def __init__(self, *args, **kwargs):
        self.frames = 0

    def SetWords(self, value):
        pass

    def AcceptWaveform(self, frame):
        self.frames += 1
        return self.frames == 10

    def PartialResult(self):
        return '{"partial": ""}'

    def FinalResult(self):
        return '{"text": ""}'

    def Reset(self):
        pass


def test_no_ears_means_the_phone():
    stub = voice_stub('array')
    stub.ears = None
    assert Voice.listen_with(stub) == 'phone'


# ---------------------------------------------------------------- the A/B recording
def test_ab_track_puts_frames_on_one_time_line():
    from bratan.voice.main import ab_track
    one, two = b'\x01\x00' * 320, b'\x02\x00' * 320
    track = ab_track([(10.00, one), (10.06, two), (11.5, one)], 10., 10.1)
    assert len(track) == 640 * 5
    assert track[:640] == one and track[640:1920] == bytes(1280) and track[1920:2560] == two


def test_ab_keeps_the_three_microphones_by_frame_number():
    from bratan.voice.main import ab_multi_track
    block = lambda v: np.full((512, 3), v, np.int16)
    blocks = [(7, 1000, block(1), 50.), (7, 1512, block(2), 50.032), (7, 2536, block(4), 50.096),   # one lost
              (3, 99999, block(9), 49.)]                                                          # an old epoch
    out = ab_multi_track(blocks, 49.5)
    assert out['x'].shape == (2048, 3) and out['first'] == 1000
    assert out['valid'][:1024].all() and not out['valid'][1024:1536].any() and out['valid'][1536:].all()
    assert out['x'][0, 0] == 1 and out['x'][1600, 2] == 4
    assert abs(out['t0'] - .5) < 1e-6
    assert ab_multi_track([], 0.) is None


def test_ab_finds_phrases_and_scores_words():
    from bratan.voice import ab
    rng = np.random.default_rng(3)
    quiet = rng.normal(0, 20, RATE).astype(np.int16)
    loud = rng.normal(0, 3000, RATE).astype(np.int16)
    track = np.concatenate([quiet, loud, quiet, loud[:8000], quiet])
    spans = ab.segments([track, track])
    assert len(spans) == 2
    assert abs(spans[0][0] * .02 - .75) < .1 and abs(spans[1][1] * .02 - 3.75) < .1
    assert ab.accuracy('Братан, привет, как дела?', 'братан привет как дела') == 1.
    assert ab.accuracy('братан привет', 'Братан, привет, как дела?') == .5
    assert ab.best_reference('поехали на кухню', '') == 'Поехали на кухню.'
    assert ab.best_reference('совсем другое', None) is None


# ---------------------------------------------------------------- what the mind does with it
def mind_with_ears(world):
    life = make_life(world)
    life.archive = types.SimpleNamespace(event=lambda *a, **k: None)
    life.wake_face = lambda: None
    life.face = lambda *a: None
    return life


def test_called_from_the_side_turns_to_the_voice_not_to_memory(world, clock):
    life = mind_with_ears(world)
    life.on_voice_event({'kind': 'wake', 'bearing': -1.4, 'bearing_conf': .8, 'side': 'справа'})
    life.turn_to_voice(None)
    assert [d['args']['angle'] for d in turns(life)] == [-1.4]
    assert life.gazes and life.gazes[-1][0] == -1.


def test_an_unsure_or_old_direction_is_not_used(world, clock):
    life = mind_with_ears(world)
    life.on_voice_event({'kind': 'wake', 'bearing': 1.5, 'bearing_conf': .1})
    life.turn_to_voice(None)
    assert not turns(life)
    life.on_voice_event({'kind': 'wake', 'bearing': 1.5, 'bearing_conf': .9})
    world.heard_bearing['t'] -= 30
    life.turn_to_voice(None)
    assert not turns(life)


def test_called_from_ahead_or_while_docked_does_not_turn_the_body(world, clock):
    life = mind_with_ears(world)
    life.on_voice_event({'kind': 'wake', 'bearing': .2, 'bearing_conf': .9})
    life.turn_to_voice(None)
    assert not turns(life)
    life = make_life(world, docked=True)
    life.archive = types.SimpleNamespace(event=lambda *a, **k: None)
    life.wake_face = life.face = lambda *a: None
    life.on_voice_event({'kind': 'wake', 'bearing': 1.6, 'bearing_conf': .9})
    life.turn_to_voice(None)
    assert not turns(life) and life.gazes                 # on the dock only the eyes go there


def test_a_bang_behind_turns_eyes_then_body(world, clock):
    life = mind_with_ears(world)
    life.on_voice_event({'kind': 'sound', 'sound': 'startle', 'words': 'резкий громкий звук', 'urgent': True,
                         'bearing': 2.8, 'bearing_conf': .7, 'side': 'сзади'})
    assert [d['args']['angle'] for d in turns(life)] == [2.8]
    assert any('вздрогнул: резкий громкий звук сзади' in e['text'] for e in world.events)


def test_a_bark_or_the_doorbell_beside_turns_eyes_then_body_but_music_does_not(world, clock):
    """C52: the ears point the eyes — the orienting reflex is not only for a bang now."""
    life = mind_with_ears(world)
    life.on_voice_event({'kind': 'sound', 'sound': 'dog', 'words': 'лает собака', 'urgent': True,
                         'bearing': 1.9, 'bearing_conf': .6, 'side': 'слева'})
    assert [d['args']['angle'] for d in turns(life)] == [1.9]
    life = mind_with_ears(world)
    life.on_voice_event({'kind': 'sound', 'sound': 'music', 'words': 'музыка', 'urgent': True,
                         'bearing': -1.9, 'bearing_conf': .9, 'side': 'справа'})
    life.on_voice_event({'kind': 'sound', 'sound': 'knock', 'words': 'стук', 'urgent': True,
                         'bearing': -1.9, 'bearing_conf': .1, 'side': 'справа'})      # unsure where: no turn
    assert not turns(life)


def test_a_sound_is_told_with_its_side_only_when_sure(world, clock):
    world.on_sound({'sound': 'dog', 'words': 'лай собаки', 'urgent': True, 'side': 'слева', 'bearing': 1.5,
                    'bearing_conf': .6})
    world.on_sound({'sound': 'music', 'words': 'музыка', 'side': 'справа', 'bearing': -1.5, 'bearing_conf': .1})
    texts = [e['text'] for e in world.events]
    assert 'слышу слева: лай собаки' in texts and 'слышу: музыка' in texts
