"""Уши в корпусе: три микрофона INMP441 на ESP32 (work/ushi, прошивка ushi-13+) — голос для распознавания и
направление на звук.

Плата (UDP 5771; записка bratan-ushi-mikrofony.md, разделы 16–17) шлёт подписчику:
  BRV1 — голос одним каналом («subv a»: плата сама берёт живой и самый громкий микрофон), PCM 16 бит, пакеты по
         512 кадров (32 мс), ~32 КБ/с;
  BRM1 — три слота сразу, сжатием IMA-ADPCM («subm 023», ~25 КБ/с) — для направления.
Подписку повторять раз в секунду: без неё плата перестаёт слать через 3 с. Wi-Fi в корпусе −71…−81 дБм: сырой
поток 128 КБ/с терял до 91 %, голос 32 КБ/с — меньше 1 % (17.10).

Слоты платы → микрофоны владельца (17.10): 0 — М1 спереди (линия D33), 2 — М2 справа, 3 — М3 слева (линия D32).
Угол — радианы, 0 — вперёд, + — влево (как bearing у тела и лидара).

Откуда звук. Корпус сильно затеняет: на 30 щелчках у проёмов «самый громкий микрофон» назвал сторону 30 из 30, а
время прихода (SRP по модели корпуса с огибанием) — 27 % (17.10; щелчки вплотную, звук обрезан, отражения внутри
корпуса). Поэтому основа — уровни трёх микрофонов в полосе 1–4 кГц над своим фоном; задержка М2–М3 (одна линия I2S,
без сдвига между линиями) решает «право/лево», когда уровни М2 и М3 близки. После калибровки (владелец говорит
«Братан» с 8 сторон, файл ears-calibration.json) — ближайшие отпечатки, это точнее грубого правила.
"""
from __future__ import annotations

import json
import math
import select
import socket
import struct
import threading
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    import audioop                       # быстрый расжиматель IMA-ADPCM на C (есть в Python 3.12 телефона)

PORT = 5771
RATE = 16000
FRAME = 320                              # кадр голоса для распознавания: 20 мс, как у микрофона телефона
SLOTS = (0, 2, 3)                        # М1 спереди, М2 справа, М3 слева
NAMES = ('М1', 'М2', 'М3')
SIDES = ('спереди', 'справа', 'слева')
SUB_EVERY_S = 1.
STALE_S = .4                             # голос без пакетов дольше — поток «нездоров», слушает телефон
LOCAL_FRESH_S = 1.5                      # мост Bluetooth молчит дольше — звук снова заказывается по Wi-Fi
# Wi-Fi отключается, только когда Bluetooth сам по себе не хуже: потерь < 2 % пять секунд подряд; вернуть Wi-Fi — при
# потерях > 5 %. Пока идут оба пути, окна складываются (C52: с ПК мост давал 23–31 пакет/с из 31 — не всегда хватает).
LOCAL_LOSS_OK, LOCAL_LOSS_BAD, LOCAL_GOOD_S = .02, .05, 5.
WIFI_WORSE = .1                          # Wi-Fi losing this much more than Bluetooth: Bluetooth alone is better
USABLE_LOSS = .2                         # listening by the ears already: kept up to this loss (healthy: under .1)
BOOT_JUMP_S = 2.                         # the frame count this far back under one epoch: the board restarted
ALONE_BACKOFF_S, ALONE_BACKOFF_MAX = 10., 160.
LOCAL_PORT = 5772                        # куда приложение лица шлёт пакеты платы, принятые по Bluetooth
HEADER = struct.Struct('<4sIIHBBHBB')     # метка, номер, первый кадр, кадров, кодек, слот/маска, эпоха, каналов, 0
_SWAP = np.array([((b & 0x0F) << 4) | (b >> 4) for b in range(256)], np.uint8)


def adpcm_decode(data: bytes, pred: int, index: int) -> np.ndarray:
    """Блок платы: первая тетрада — младшая (у audioop — старшая); состояние кодера — из начала блока."""
    swapped = _SWAP[np.frombuffer(data, np.uint8)].tobytes()
    pcm, _ = audioop.adpcm2lin(swapped, 2, (pred, index))
    return np.frombuffer(pcm, '<i2')


def parse(data: bytes):
    """Пакет платы → (вид, номер, первый кадр, эпоха, слоты, отсчёты [кадров × каналов] int16) или None."""
    if len(data) < HEADER.size:
        return None
    tag, seq, first, n, codec, slot_or_mask, epoch, channels, _ = HEADER.unpack_from(data)
    body = memoryview(data)[HEADER.size:]
    if tag == b'BRV1':
        slots = (slot_or_mask,)
    elif tag == b'BRM1':
        slots = tuple(ch for ch in range(4) if slot_or_mask >> ch & 1)
    else:
        return None
    out = np.zeros((n, len(slots)), np.int16)
    pos = 0
    for k in range(len(slots)):
        if codec == 0:
            need = 2 * n
            if len(body) < pos + need:
                return None
            out[:, k] = np.frombuffer(body[pos:pos + need], '<i2')
        else:
            need = 4 + n // 2
            if len(body) < pos + need:
                return None
            pred, index = struct.unpack_from('<hB', body, pos)
            out[:, k] = adpcm_decode(bytes(body[pos + 4:pos + need]), pred, index)
        pos += need
    return ('voice' if tag == b'BRV1' else 'multi'), seq, first, epoch, slots, out


class Clock:
    """Номер кадра платы → время телефона (time.monotonic): самая ранняя доставка за последние ~3 с —
    это почти чистая задержка сети; часы платы и телефона расходятся на миллионные доли, окно это догоняет."""

    def __init__(self):
        self.epoch = None
        self.recent = deque(maxlen=100)

    def offset(self, epoch, first_after, now):
        if epoch != self.epoch:
            self.epoch = epoch
            self.recent.clear()
        self.recent.append(now - first_after / RATE)
        return min(self.recent)


class Ring:
    """Последние секунды трёх микрофонов (слоты 0, 2, 3) с отметкой, где звук был, а где пакет потерялся."""

    def __init__(self, seconds=10.):
        self.size = int(seconds * RATE)
        self.data = np.zeros((self.size, 3), np.int16)
        self.valid = np.zeros(self.size, bool)
        self.end = None                  # номер кадра после последнего записанного (в текущей эпохе)
        self.epoch = None
        self.end_mono = 0.               # время телефона для кадра `end`
        self.lock = threading.Lock()

    def put(self, epoch, first, block, mono_end):
        with self.lock:
            n = len(block)
            if epoch != self.epoch or self.end is None or first + n <= self.end - self.size or first > self.end + self.size:
                self.valid[:] = False
                self.epoch, self.end = epoch, first
            if first > self.end:                                 # потерянные пакеты: дыра
                idx = np.arange(self.end, first) % self.size
                self.valid[idx] = False
            idx = np.arange(first, first + n) % self.size
            self.data[idx] = block
            self.valid[idx] = True
            if first + n >= self.end:
                self.end = first + n
                self.end_mono = mono_end

    def latest(self, seconds):
        """(отсчёты [N × 3] float, где был звук [N], время телефона конца) за последние `seconds`."""
        with self.lock:
            if self.end is None:
                return None
            n = min(self.size, int(seconds * RATE), self.end)
            idx = np.arange(self.end - n, self.end) % self.size
            return self.data[idx].astype(np.float32), self.valid[idx].copy(), self.end_mono


class ArrayLink:
    """Подписка на плату, приём, разбор. Голос — кадрами по 20 мс в `on_voice(mono, pcm_bytes | None)`
    (None — разрыв: распознаватель начинает заново); три микрофона — в кольцо `ring` для направления."""

    def __init__(self, host=None, port=PORT, voice='a', multi='023', on_voice=None, local_port=None, mix='best',
                 null=None):
        """voice=None — без голоса (панель на ПК: только направление, плата держит до 4 подписчиков);
        voice='multi' — голос из потока трёх микрофонов, отдельного голосового потока нет (18.10: Wi-Fi платы упал до
        −85…−93 дБм и терял 70 %; поток вдвое легче — 25 КБ/с вместо 57; по записи A/B лучший из трёх микрофонов
        распознаётся так же, как голос платы: 82 / 82 %).
        null — voice/nullvoice.py RobotNull: пока робот говорит, его голос вычитается из трёх микрофонов (C52), иначе
        голос — лучший микрофон как есть; `robot` — () → (учить, вычитать), ставит голос."""
        self.host, self.port = host, port
        self.voice_from_multi = voice == 'multi'
        self.voice_request = f'subv {voice}'.encode() if voice not in (None, 'multi') else None
        self.pick = None                 # выбор микрофона для голоса из трёх: [шум по каналам, сглаженное с/ш, выбранный]
        # mix='sum' — голос из трёх микрофонов, сложенных с задержками (voice/beam.py Beam); 'best' — лучший из трёх.
        self.beam = None
        if self.voice_from_multi and mix == 'sum':
            from .beam import Beam
            self.beam = Beam()
        self.null = null if self.voice_from_multi and self.beam is None else None
        self.null_error = None
        self.robot = None
        self.stream_end = (None, -1)     # (эпоха, кадр после последнего) непрерывного потока суммы или фильтра
        self.multi_request = f'subm {multi}'.encode()
        self.on_voice = on_voice
        self.on_multi = None             # (эпоха, первый кадр, блок [кадры × 3], время телефона первого кадра) — запись A/B
        self.ring = Ring()
        self.voice_clock, self.multi_clock = Clock(), Clock()
        self.pending = np.zeros(0, np.int16)       # голос, ещё не нарезанный на кадры по 20 мс
        self.pending_first = None
        self.voice_epoch = None
        self.last_voice = self.last_multi = 0.
        self.seen = {'voice': deque(maxlen=200), 'multi': deque(maxlen=200),     # (время, номер пакета)
                     'local': deque(maxlen=200),                                 # окна, пришедшие мостом
                     'wifi': deque(maxlen=200)}                                  # …и по Wi-Fi (сравнить пути)
        # Поток приёма дописывает, голос и мозг читают: без замка «deque mutated during iteration» ронял голос робота
        # (17.10, ночью: publish_state → summary → healthy → loss).
        self.seen_lock = threading.Lock()
        self.status = {}
        self.voice_slot = None
        self.stopping = threading.Event()
        self.sock_voice = self.sock_multi = self.sock_local = None
        # Мост по Bluetooth (18.10, владелец: «подружи телефон с платой напрямую, они всегда рядом»): приложение лица
        # принимает те же пакеты BRM1 с платы по BLE и шлёт их сюда, на 127.0.0.1:local_port. Пока мост живой, по Wi-Fi
        # звук не заказывается (плата перестаёт слать через 3 с) — Wi-Fi остаётся для пинга и как запасной путь.
        self.local_port = local_port
        self.last_local = 0.
        self.wifi_off = False            # Bluetooth справляется сам: звук по Wi-Fi не заказывается
        self.local_good_since = None
        self.multi_end = (None, -1)      # (эпоха, кадр после последнего принятого окна): одно окно двумя путями — один раз
        self.thread = threading.Thread(target=self._run, name='ears', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stopping.set()
        for sock in (self.sock_voice, self.sock_multi):
            if sock is not None and self.host:
                try:
                    sock.sendto(b'unsub', (self.host, self.port))
                except OSError:
                    pass

    # ------------------------------------------------------------ здоровье
    def loss(self, kind, within=3.):
        """Доля потерянных пакетов за последние `within` секунд (по номерам пакетов)."""
        now = time.monotonic()
        with self.seen_lock:
            seqs = [s for t, s in self.seen[kind] if now - t < within]
        if len(seqs) < 2:
            return 1.
        span = max(seqs) - min(seqs) + 1
        return max(0., 1 - len(set(seqs)) / span)

    def healthy(self, now=None):
        now = time.monotonic() if now is None else now
        return now - self.last_voice < STALE_S and self.loss('voice') < .1

    def usable(self, now=None):
        """Good enough to keep listening by (not to switch to): fresh and losing under USABLE_LOSS."""
        now = time.monotonic() if now is None else now
        return now - self.last_voice < STALE_S and self.loss('voice') < USABLE_LOSS

    def via_bluetooth(self, now=None):
        """Мост живой (окна приходят по Bluetooth); Wi-Fi при этом может ещё работать — см. bluetooth_alone."""
        now = time.monotonic() if now is None else now
        return bool(self.local_port) and now - self.last_local < LOCAL_FRESH_S

    def bluetooth_alone(self, now=None):
        """Можно не заказывать звук по Wi-Fi: мост живой и теряет мало. С запасом по времени и по потерям, чтобы
        не переключаться туда-сюда; хуже, чем по Wi-Fi, от этого не становится — пока мост не доказал себя, идут оба."""
        now = time.monotonic() if now is None else now
        if not self.via_bluetooth(now):
            self.wifi_off, self.local_good_since = False, None
            return False
        loss = self.loss('local')
        if self.wifi_off:
            if loss > LOCAL_LOSS_BAD:
                self.wifi_off, self.local_good_since = False, None
                # It did not hold: wait longer each time before trying alone again (no flapping).
                self.alone_backoff = min(ALONE_BACKOFF_MAX, getattr(self, 'alone_backoff', ALONE_BACKOFF_S) * 2)
                self.alone_retry = now + self.alone_backoff
        elif now >= getattr(self, 'alone_retry', 0.) and (
                loss < LOCAL_LOSS_OK or loss + WIFI_WORSE < self.loss('wifi')):
            # Bluetooth clean — or Wi-Fi much worse than it (18.10 18:45, Wi-Fi at −89 dBm: its retries took the
            # shared radio, both paths lost 10–31 % and the ears were «unhealthy», the robot swung between the ears
            # and the phone mid-word; Bluetooth alone gave every packet).
            self.local_good_since = self.local_good_since or now
            self.wifi_off = now - self.local_good_since >= LOCAL_GOOD_S
        else:
            self.local_good_since = None
        return self.wifi_off

    def summary(self):
        now = time.monotonic()
        return {'host': self.host, 'voice_age': round(now - self.last_voice, 2) if self.last_voice else None,
                'multi_age': round(now - self.last_multi, 2) if self.last_multi else None,
                'voice_loss': round(self.loss('voice'), 3), 'multi_loss': round(self.loss('multi'), 3),
                'voice_slot': self.voice_slot, 'healthy': self.healthy(now),
                'via': ('bluetooth' if self.wifi_off else 'both') if self.via_bluetooth(now) else 'wifi',
                'bluetooth_loss': round(self.loss('local'), 3) if self.via_bluetooth(now) else None,
                'null': self.null.summary() if self.null is not None else self.null_error}

    def null_good(self):
        """My voice can be taken out of the ears now: they can be listened to while I speak (voice/nullvoice.py)."""
        null = self.null
        return null is not None and null.good

    # ------------------------------------------------------------ приём
    def _discover(self, sock):
        """Адрес платы неизвестен или она молчит: широковещательный ping, плата отвечает JSON со своим адресом."""
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(b'ping', ('255.255.255.255', self.port))
        except OSError:
            pass

    def _run(self):
        self.sock_voice = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_multi = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socks = [self.sock_voice, self.sock_multi]
        for sock in socks:
            sock.bind(('', 0))
            sock.setblocking(False)
        if self.local_port:
            try:
                self.sock_local = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock_local.bind(('127.0.0.1', self.local_port))
                self.sock_local.setblocking(False)
                socks.append(self.sock_local)
            except OSError:
                self.sock_local = None              # порт занят: только Wi-Fi
        last_sub = last_seen_any = 0.
        while not self.stopping.is_set():
            now = time.monotonic()
            if now - last_sub >= SUB_EVERY_S:
                last_sub = now
                if self.host:
                    alone = self.bluetooth_alone(now)
                    for sock, request in ((self.sock_voice, self.voice_request), (self.sock_multi, self.multi_request)):
                        if request is None or alone:
                            continue
                        try:
                            sock.sendto(request, (self.host, self.port))
                        except OSError:
                            pass
                    try:
                        self.sock_multi.sendto(b'ping', (self.host, self.port))
                    except OSError:
                        pass
                if not self.host or now - max(last_seen_any, self.last_voice, self.last_multi) > 5.:
                    self._discover(self.sock_multi)
            # Ждать сразу оба сокета и вычитывать всё пришедшее (17.10: по очереди с ожиданием 50 мс на каждом
            # пустой голосовой сокет панели пропускал половину пакетов трёх микрофонов).
            try:
                readable, _, _ = select.select(socks, [], [], .05)
            except (OSError, ValueError):
                time.sleep(.2)
                continue
            for sock in readable:
                for _ in range(64):
                    try:
                        data, sender = sock.recvfrom(4096)
                    except (BlockingIOError, InterruptedError):
                        break
                    except OSError:
                        time.sleep(.05)
                        break
                    last_seen_any = time.monotonic()
                    self._packet(data, sender, local=sock is self.sock_local)

    def _packet(self, data, sender, local=False):
        now = time.monotonic()
        if data[:1] == b'{':                                        # ответ на ping: состояние платы, её адрес
            try:
                status = json.loads(data.decode('utf-8', 'replace'))
            except ValueError:
                return
            self.status = status
            if status.get('ip') and status.get('ip') != self.host:
                self.host = status['ip']
            return
        item = parse(data)
        if item is None:
            return
        kind, seq, first, epoch, slots, block = item
        # The board restarted (flashed, power): its frame count starts at 0 again under the same epoch number, and every
        # new window looked «already had» next to the old count — the ears heard nothing until the voice restarted
        # (18.10 19:30). A count more than BOOT_JUMP_S back is a new stream: a new epoch for everything here.
        last = getattr(self, 'last_first', None)
        if last is not None and epoch == last[0] and first + int(BOOT_JUMP_S * RATE) < last[1]:
            self.boots = getattr(self, 'boots', 0) + 1
        self.last_first = (epoch, max(first, last[1]) if last is not None and epoch == last[0]
                           and first + int(BOOT_JUMP_S * RATE) >= last[1] else first)
        epoch = epoch + 1000 * getattr(self, 'boots', 0)
        if kind == 'multi':
            if len(slots) != 3:
                return
            with self.seen_lock:
                self.seen['local' if local else 'wifi'].append((now, first // len(block)))
            if local:
                self.last_local = now
            # Окна по номеру кадра, а не по номеру пакета: у моста свой счёт пакетов, а окно одно и то же.
            end_epoch, end = self.multi_end
            if epoch == end_epoch and first + len(block) <= end:
                return                                              # это окно уже пришло другим путём
            self.multi_end = (epoch, max(first + len(block), end if epoch == end_epoch else 0))
            seq = first // len(block)
        with self.seen_lock:
            self.seen[kind].append((now, seq))
        if kind == 'multi':
            self.last_multi = now
            offset = self.multi_clock.offset(epoch, first + len(block), now)
            self.ring.put(epoch, first, block, offset + (first + len(block)) / RATE)
            if self.on_multi is not None:
                self.on_multi(epoch, first, block, offset + first / RATE)
            if self.voice_from_multi:
                k = self._pick(block)
                self.last_voice = now
                self.voice_slot = SLOTS[k]
                with self.seen_lock:
                    self.seen['voice'].append((now, seq))
                if self.beam is not None:
                    fresh = self._fresh(epoch, first, block, self.beam)
                    if fresh is not None:
                        self._voice(epoch, fresh[0], self.beam.feed(fresh[1], k), offset)
                elif self.null is not None:
                    self._null_voice(epoch, first, block, k, offset)
                else:
                    self._voice(epoch, first, block[:, k].copy(), offset)
            return
        self.last_voice = now
        self.voice_slot = slots[0]
        offset = self.voice_clock.offset(epoch, first + len(block), now)
        self._voice(epoch, first, block[:, 0], offset)

    def _fresh(self, epoch, first, block, stream):
        """The part of a window the summed (or filtered) voice has not had yet: (first, block) or None. The sum keeps
        the last samples of each microphone, so it must be fed one unbroken stream — a window of the other path half a
        window later gives its new half, a gap or a new epoch starts it afresh."""
        end_epoch, end = self.stream_end
        if epoch != end_epoch or first > end:
            stream.reset()
        elif first + len(block) <= end:
            return None
        elif first < end:
            block, first = block[end - first:], end
        self.stream_end = (epoch, first + len(block))
        return first, block

    def _null_voice(self, epoch, first, block, k, offset):
        """The voice through RobotNull: my own voice taken out while it may be in the room, microphone k as it is
        otherwise (16 ms later, on the same frame numbers). A failure turns the filter off, never the ears."""
        try:
            fresh = self._fresh(epoch, first, block, self.null)
            if fresh is None:
                return
            learn, clean = self.robot() if self.robot is not None else (False, False)
            at, pcm = self.null.feed(fresh[0], fresh[1], learn=learn, clean=clean, pick=k)
        except Exception as error:
            self.null, self.null_error = None, f'{type(error).__name__}: {error}'[:160]
            self._voice(epoch, first, block[:, k].copy(), offset)
            return
        if at is not None and len(pcm):
            self._voice(epoch, at, pcm, offset)

    def _pick(self, block):
        """Какой из трёх микрофонов слушать: у кого голос выше своего шума (сглажено ~0,5 с), смена — только если
        другой лучше на 3 дБ (как выбор слота на плате, но по с/ш, а не по громкости: М3 громче всех и в тишине)."""
        energy = (block.astype(np.float64) ** 2).mean(axis=0) + 1.
        if self.pick is None:
            self.pick = [energy.copy(), np.zeros(3), 0]
        noise, smooth, chosen = self.pick
        noise = np.where(energy < noise, energy, noise * 1.002)          # минимум, медленно вверх (~0,5 дБ/с)
        snr = 10 * np.log10(energy / noise)
        smooth = .94 * smooth + .06 * snr
        best = int(np.argmax(smooth))
        if smooth[best] > smooth[chosen] + 3.:
            chosen = best
        self.pick = [noise, smooth, chosen]
        return chosen

    def _voice(self, epoch, first, pcm, offset):
        """Голос → кадры по 20 мс. Потеря до 0,4 с — тишиной (распознавание не рвётся); больше или новая эпоха —
        разрыв (None)."""
        if self.voice_epoch != epoch or self.pending_first is None:
            restart = self.pending_first is not None
            self.voice_epoch, self.pending_first, self.pending = epoch, first, np.zeros(0, np.int16)
            if restart and self.on_voice:
                self.on_voice(time.monotonic(), None)
        expected = self.pending_first + len(self.pending)
        if first + len(pcm) <= expected:
            return                                                # повтор или опоздавший пакет: уже прошёл
        if first < expected:                                      # окно другого пути со сдвигом (Wi-Fi и Bluetooth
            pcm, first = pcm[expected - first:], expected         # режут поток на окна с разных кадров): новый хвост
        if first > expected:
            if first - expected <= int(.4 * RATE):
                self.pending = np.concatenate([self.pending, np.zeros(first - expected, np.int16)])
            else:
                self.pending_first, self.pending = first, np.zeros(0, np.int16)
                if self.on_voice:
                    self.on_voice(time.monotonic(), None)
        self.pending = np.concatenate([self.pending, pcm])
        count = len(self.pending) // FRAME
        for i in range(count):
            start = self.pending_first + i * FRAME
            if self.on_voice:
                self.on_voice(offset + start / RATE, self.pending[i * FRAME:(i + 1) * FRAME].tobytes())
        self.pending = self.pending[count * FRAME:]
        self.pending_first += count * FRAME


# ------------------------------------------------------------------ направление
# 18.10, калибровка владельцем с 8 сторон (записи work/ushi/recordings/calib-*.npz): громкость сторону не различает
# (М3 всегда громче М1 на ~5–10 дБ — чувствительность, не тень корпуса), а задержки по всем «звучащим» кадрам путают
# отражения (компас «рисовал рандомно»). Задержки только по НАЧАЛАМ слогов (прямой звук, отражения ещё не пришли) с
# весом по отношению сигнал/шум в каждой частоте (гул 50/95 Гц и тон 860 Гц не голосуют) совпали с моделью корпуса;
# угол — ближайший к модели по трём задержкам: 8 сторон из 8, ошибка 4–33°, в среднем ~16°. Калибровка убирает
# постоянный перекос (спереди модель тянет на +19…+33°).
BLOCK, HOP = 512, 128                          # окно 32 мс, шаг 8 мс
BAND = (300., 3500.)
ONSET_RISE_DB = 1.5                            # энергия выросла за шаг — начало слога
LOUD_DB = 12.                                  # и звук на столько выше своего фона
MAX_LAG = 30
# 18.10, владелец: «в конце, когда звук затихает, он неправильно показывает последнее направление — видимо из-за
# отражений». Окно в 1 с съезжало с начала слова, и последние слоги вместе с эхом комнаты голосовали за другую
# сторону (запись справа: +83° → +2° на хвосте). Как у людей (эффект предшествования: место звука задаёт его первый
# фронт) — звук берётся целиком, от своего начала, даже если окно ушло дальше; хвост своей оценки не даёт. На 8
# записях калибровки худшая ошибка на хвосте 88° → 27°. Вес частот по «новой, не эховой» энергии (Wilson & Darrell
# 2006) пробован: хвост тоже лечит, но в первые доли секунды звука ошибался до 130° — не взят (tools/dir_timeline.py).
EVENT_DB = 6.                                  # звук: энергия в полосе речи на столько выше фона
EVENT_GAP_S = .3                               # паузы короче — тот же звук (слоги одного слова)
# Модель корпуса в плоскости микрофонов (work/ushi/tools/layout_sim.py): прямоугольник x −155…+182, y ±116 мм
# (x — вперёд, y — влево); звук до затенённого микрофона огибает корпус по стенке от ближайшего угла.
BODY = (-155., 182., -116., 116.)
MIC_XY = ((182., 0.), (-84., -116.), (-84., 116.))     # М1 спереди, М2 справа, М3 слева
SOUND_MM_S = 343e3
PAIRS = ((0, 1), (0, 2), (1, 2))


def _arrival(p, u):
    """Время прихода плоской волны с направления u в точку p на стенке (с, относительно центра)."""
    x0, x1, y0, y1 = BODY
    corners = ((x1, y1), (x1, y0), (x0, y0), (x0, y1))
    perimeter = 2 * ((x1 - x0) + (y1 - y0))

    def along(q):
        x, y = q
        if abs(x - x1) < 1e-6:
            return y1 - y
        if abs(y - y0) < 1e-6:
            return (y1 - y0) + (x1 - x)
        if abs(x - x0) < 1e-6:
            return (y1 - y0) + (x1 - x0) + (y - y0)
        return 2 * (y1 - y0) + (x1 - x0) + (x - x0)

    x, y = p
    normal = (1., 0.) if abs(x - x1) < 1e-6 else (-1., 0.) if abs(x - x0) < 1e-6 else (0., 1.) if abs(y - y1) < 1e-6 \
        else (0., -1.)
    if normal[0] * u[0] + normal[1] * u[1] > 0:
        return -(x * u[0] + y * u[1]) / SOUND_MM_S
    best, s = math.inf, along(p)
    for q in corners:
        path = abs(along(q) - s)
        best = min(best, (-(q[0] * u[0] + q[1] * u[1]) + min(path, perimeter - path)) / SOUND_MM_S)
    return best


MODEL_DEG = np.arange(-180, 180)
_t = np.array([[_arrival(m, (math.cos(math.radians(d)), math.sin(math.radians(d)))) * RATE for m in MIC_XY]
               for d in MODEL_DEG])
MODEL_TAU = np.stack([_t[:, b] - _t[:, a] for a, b in PAIRS], axis=1)     # [360, 3] отсчётов, >0 — второй позже


def _wrap_deg(d):
    return (d + 180.) % 360. - 180.


def model_fit(taus):
    """Угол модели, ближайший по трём задержкам, и невязка (отсчёты)."""
    cost = np.sqrt(((MODEL_TAU - np.asarray(taus)[None, :]) ** 2).sum(axis=1))
    i = int(np.argmin(cost))
    return float(MODEL_DEG[i]), float(cost[i])


class Calibration:
    """Поправка к модели по калибровке: для известных сторон — какой угол дала модель (fit) и какой был на самом деле
    (deg). Новый угол модели правится сдвигом, интерполированным между соседними калибровочными точками по кругу.
    Записи старого вида (до 18.10: уровни и τ23, без fit) не используются."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.entries = []
        self.mtime = None
        self.refresh()

    def refresh(self):
        """Файл сменили (калибровка с панели на ПК, скопирована на телефон) — перечитать без перезапуска голоса."""
        if not self.path:
            return
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime == self.mtime:
            return
        self.mtime = mtime
        try:
            self.entries = json.loads(self.path.read_text(encoding='utf-8')).get('entries', [])
        except (OSError, ValueError):
            self.entries = []

    def add(self, deg, fit, taus):
        self.entries.append({'deg': float(deg), 'fit': round(float(fit), 1),
                             'taus': [round(float(v), 2) for v in taus], 't': time.time()})

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({'version': 2, 'entries': self.entries}, ensure_ascii=False, indent=1),
                                 encoding='utf-8')
            self.mtime = self.path.stat().st_mtime

    def points(self):
        return sorted((_wrap_deg(e['fit']), _wrap_deg(e['deg'] - e['fit'])) for e in self.entries if 'fit' in e)

    def ready(self):
        return len(self.points()) >= 4

    def correct(self, fit):
        """Угол модели → угол с поправкой (линейно по кругу между двумя соседними точками калибровки)."""
        pts = self.points()
        fit = _wrap_deg(fit)
        before = max((p for p in pts if p[0] <= fit), default=pts[-1])
        after = min((p for p in pts if p[0] > fit), default=pts[0])
        d_before = (fit - before[0]) % 360.
        d_after = (after[0] - fit) % 360.
        if d_before + d_after < 1e-6:
            return _wrap_deg(fit + before[1])
        offset = (before[1] * d_after + after[1] * d_before) / (d_before + d_after)
        return _wrap_deg(fit + offset)


class Direction:
    """Откуда звук за последние секунды: по кольцу ArrayLink. `estimate()` → словарь или None (нет начал слогов)."""

    def __init__(self, ring: Ring, calibration: Calibration | None = None):
        self.ring = ring
        self.calibration = calibration or Calibration()
        f = np.fft.rfftfreq(BLOCK, 1. / RATE)
        self.freq = f
        self.band = (f >= BAND[0]) & (f <= BAND[1])
        self.speech = (f >= 300.) & (f <= 3000.)

    def features(self, seconds=2., history=10., not_before=None):
        """Задержки трёх пар по началам слогов звуков, задевших последние `seconds` секунд, — каждый звук целиком, от
        своего начала (фон — 20-й перцентиль за `history` секунд). not_before (time.monotonic) — раньше не брать: там
        говорил сам робот (его динамик — спереди, у М1)."""
        got = self.ring.latest(history)
        if got is None:
            return None
        x, valid, end_mono = got
        x = x - x.mean(axis=0)
        count = (len(x) - BLOCK) // HOP + 1
        if count < 16:
            return None
        idx = np.arange(BLOCK)[None, :] + HOP * np.arange(count)[:, None]
        lost = np.r_[0, np.cumsum(~valid)]
        ok = (lost[idx[:, -1] + 1] - lost[idx[:, 0]]) == 0                            # окно без потерь
        if ok.sum() < 16:
            return None
        spec = np.fft.rfft(x[idx] * np.hanning(BLOCK)[None, :, None], axis=1)        # [кадры, частоты, 3]
        power = spec.real ** 2 + spec.imag ** 2
        noise = np.percentile(power[ok], 20, axis=0)                                  # фон: частота × микрофон
        level = 10 * np.log10(power[:, self.speech].sum(axis=1) + 1.)                 # [кадры, 3]
        energy = level.mean(axis=1)
        rise = np.r_[0., np.diff(energy)]
        floor = np.percentile(energy[ok], 20)
        starts = end_mono - (len(x) - HOP * np.arange(count)) / RATE                  # время начала каждого окна
        take = self._events(ok & (energy > floor + EVENT_DB), count - int(seconds * RATE / HOP)) & ok
        if not_before is not None:
            take &= starts >= not_before
        onset = take & (rise > ONSET_RISE_DB) & (energy > floor + LOUD_DB)
        if onset.sum() < 2:
            return None
        taus = []
        for a, b in PAIRS:
            cross = spec[onset][:, :, b] * np.conj(spec[onset][:, :, a])
            snr = np.minimum(power[onset][:, :, a] / (noise[None, :, a] + 1e-9),
                             power[onset][:, :, b] / (noise[None, :, b] + 1e-9))
            cross = cross / (np.abs(cross) + 1e-9) * (snr / (1. + snr))               # шумная частота не голосует
            cross[:, ~self.band] = 0
            corr = np.fft.irfft(cross.sum(axis=0), BLOCK * 8)
            lim = MAX_LAG * 8
            window = np.concatenate([corr[-lim:], corr[:lim + 1]])
            taus.append((int(np.argmax(window)) - lim) / 8.)
        floor_mic = np.percentile(level[ok], 20, axis=0)
        excess = 10 * np.log10((10 ** ((level[onset] - floor_mic) / 10.)).mean(axis=0))
        used = np.flatnonzero(take)
        return {'taus': np.array(taus), 'onsets': int(onset.sum()), 'excess': excess, 'end_mono': end_mono,
                'since': float(starts[used[0]]), 'until': float(starts[used[-1]] + BLOCK / RATE)}

    @staticmethod
    def _events(active, recent_from):
        """Кадры звуков, задевших окно с кадра `recent_from`: каждый звук от его начала (паузы короче EVENT_GAP_S —
        внутри одного звука)."""
        count = len(active)
        gap = int(EVENT_GAP_S * RATE / HOP)
        event = np.zeros(count, bool)
        last = None
        for t in np.flatnonzero(active):
            if last is not None and t - last <= gap:
                event[last:t + 1] = True
            event[t] = True
            last = t
        take = np.zeros(count, bool)
        t = count - 1
        while t >= max(0, recent_from):
            if not event[t]:
                t -= 1
                continue
            first = t
            while first > 0 and event[first - 1]:
                first -= 1
            take[first:t + 1] = True
            t = first - 1
        return take

    @staticmethod
    def fingerprint(feat):
        """Для калибровки: угол модели по этому звуку и сами задержки."""
        return model_fit(feat['taus'])[0], feat['taus']

    def estimate(self, seconds=2., not_before=None):
        feat = self.features(seconds, not_before=not_before)
        if feat is None:
            return None
        self.calibration.refresh()
        fit, residual = model_fit(feat['taus'])
        if self.calibration.ready():
            deg, method = self.calibration.correct(fit), 'calibrated'
        else:
            deg, method = fit, 'model'
        deg = _wrap_deg(deg)
        conf = max(0., min(1., (18. - residual) / 12.)) * min(1., feat['onsets'] / 4.)
        e = feat['excess']
        return {'bearing': math.radians(deg), 'deg': round(deg, 1), 'confidence': round(float(conf), 2),
                'method': method, 'loudest': NAMES[int(np.argmax(e))], 'excess': [round(float(v), 1) for v in e],
                'tau23': round(float(feat['taus'][2]), 2), 'taus': [round(float(v), 2) for v in feat['taus']],
                'residual': round(residual, 1), 'onsets': feat['onsets'], 'side': side_words(math.radians(deg))}


def side_words(bearing):
    """Радианы → «спереди», «слева-спереди», «слева», «сзади»…"""
    deg = _wrap_deg(math.degrees(bearing))
    names = ((22.5, 'спереди'), (67.5, 'слева-спереди'), (112.5, 'слева'), (157.5, 'слева-сзади'))
    a = abs(deg)
    for limit, word in names:
        if a <= limit:
            return word if deg >= 0 or word == 'спереди' else word.replace('слева', 'справа')
    return 'сзади'
