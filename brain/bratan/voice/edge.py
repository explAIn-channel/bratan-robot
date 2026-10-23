"""Backup voice when OpenAI is down: Microsoft Edge neural TTS (ru-RU-DmitryNeural), no key needed.

This is the unofficial Edge «Read aloud» endpoint: Microsoft throttles it per address
(empty answers, slow answers) and more so for many connections or parallel requests
(measured 06.10: after a burst of tests DmitryNeural returned no audio for most phrases,
minutes later it answered in ~2 s; before the burst ~0.5 s fresh, ~0.1 s warm). So it is
used gently:
- one channel, opened when speech is needed and kept while a conversation goes on
  (a warm channel answers in ~0.1 s); closed after MAX_IDLE_S (the service drops idle
  channels after 30–60 s anyway, pings do not help) — never reconnected just to keep warm;
- no parallel duplicate requests;
- no sound within FIRST_SOUND_S: the caller speaks with the offline voice.
The MP3 stream is decoded frame by frame as it arrives; ready lines play instantly.
"""
from __future__ import annotations

import asyncio
import ssl
import threading
import time
import uuid
from collections import deque
from xml.sax.saxutils import escape

import numpy as np

from .local import READY_LINES
from .tts import Resampler

VOICE = 'ru-RU-DmitryNeural'
RATE_UP = '+8%'
CHANNELS = 1
HEDGE_S = 99.             # no duplicate requests: they only make the throttling worse
FIRST_SOUND_S = 2.5
GAIN = .8       # no sound by then: the offline voice speaks instead
MAX_IDLE_S = 20.          # the service silently drops a channel idle 30-60 s (pings do not help)
MAX_AGE_S = 300.
CONFIG = ('X-Timestamp:{ts}\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n'
          '{{"context":{{"synthesis":{{"audio":{{"metadataoptions":{{"sentenceBoundaryEnabled":"false",'
          '"wordBoundaryEnabled":"false"}},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}}}}}\r\n')


def _retrieved(task):
    """A finished attempt's error is seen (its refusal is in the stats and the trace already)."""
    if task.done() and not task.cancelled():
        task.exception()


class Channel:
    """One open WebSocket to the service, one phrase at a time."""

    def __init__(self, session):
        self.session = session
        self.ws = None
        self.busy = False
        self.opened = self.used = 0.

    async def open(self):
        from edge_tts.communicate import date_to_string
        from edge_tts.constants import SEC_MS_GEC_VERSION, WSS_HEADERS, WSS_URL
        from edge_tts.drm import DRM
        self.ws = await self.session.ws_connect(
            f'{WSS_URL}&ConnectionId={uuid.uuid4().hex}&Sec-MS-GEC={DRM.generate_sec_ms_gec()}'
            f'&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}', compress=15, headers=DRM.headers_with_muid(WSS_HEADERS),
            ssl=ssl.create_default_context(), heartbeat=None, timeout=5.)
        await self.ws.send_str(CONFIG.format(ts=date_to_string()))
        self.opened = self.used = time.monotonic()

    @property
    def alive(self):
        return self.ws is not None and not self.ws.closed

    @property
    def stale(self):
        now = time.monotonic()
        return now - self.used > MAX_IDLE_S or now - self.opened > MAX_AGE_S

    async def close(self):
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        self.ws = None

    async def phrase(self, text, voice, rate):
        """MP3 bytes of one phrase as they arrive."""
        import aiohttp
        from edge_tts.communicate import date_to_string
        self.busy = True
        self.used = time.monotonic()
        try:
            await self.ws.send_str(
                f'X-RequestId:{uuid.uuid4().hex}\r\nContent-Type:application/ssml+xml\r\n'
                f'X-Timestamp:{date_to_string()}Z\r\nPath:ssml\r\n\r\n'
                "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='ru-RU'>"
                f"<voice name='{voice}'><prosody pitch='+0Hz' rate='{rate}' volume='+0%'>{escape(text)}"
                '</prosody></voice></speak>')
            got = False
            async for message in self.ws:
                if message.type == aiohttp.WSMsgType.BINARY:
                    head = int.from_bytes(message.data[:2], 'big')
                    data = message.data[head + 2:]
                    if data:
                        got = True
                        yield data
                elif message.type == aiohttp.WSMsgType.TEXT:
                    if 'Path:turn.end' in message.data:
                        if not got:
                            raise RuntimeError('no_audio')
                        self.used = time.monotonic()
                        return
                else:
                    raise RuntimeError(f'ws_{message.type.name}')
            raise RuntimeError('ws_closed')
        finally:
            self.busy = False


class EdgeSpeech:
    def __init__(self, audio, voice=VOICE, rate=RATE_UP):
        self.audio, self.voice, self.rate = audio, voice, rate
        self.cache = {}
        self.channels: list[Channel] = []
        self.trace = deque(maxlen=50)
        self.stats = {'phrases': 0, 'hedged': 0, 'hedge_won': 0, 'failed': 0, 'reconnects': 0, 'last_first_s': None}
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name='edge-tts', daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._keep(), self.loop)

    # ------------------------------------------------------------ channels
    async def _keep(self):
        import aiohttp
        self.session = aiohttp.ClientSession()
        self.channels = [Channel(self.session) for _ in range(CHANNELS)]
        while True:
            for channel in self.channels:
                if channel.busy:
                    continue
                if channel.alive and channel.stale:
                    await channel.close()          # idle: let it go, open again when needed
            if len(self.cache) < len(READY_LINES) and time.monotonic() >= getattr(self, 'cache_retry_at', 0.):
                for line in READY_LINES:
                    if line not in self.cache:
                        try:
                            self.cache[line] = await asyncio.wait_for(self._collect(line), 10)
                            self.cache_fails = 0
                        except Exception:
                            # 18.10: the free endpoint answers 403 now; every 2 s it filled the voice's log with
                            # tracebacks. Try again later and later (up to every 10 min); Azure speaks meanwhile.
                            self.cache_fails = getattr(self, 'cache_fails', 0) + 1
                            self.cache_retry_at = time.monotonic() + min(600., 2. * 2 ** self.cache_fails)
                            break
            await asyncio.sleep(2.)

    def _free(self, exclude=()):
        for channel in self.channels:
            if channel not in exclude and channel.alive and not channel.busy and not channel.stale:
                return channel
        return None

    async def _pooled(self, exclude=()):
        """A warm channel, or open the pool's closed one now (it then stays for the conversation)."""
        channel = self._free(exclude)
        if channel is not None:
            return channel, False
        for channel in self.channels:
            if channel not in exclude and not channel.busy and not channel.alive:
                await channel.open()
                self.stats['reconnects'] += 1
                return channel, False
        return await self._fresh(), True

    async def _fresh(self):
        channel = Channel(self.session)
        await channel.open()
        return channel

    # ------------------------------------------------------------ hedged phrase
    async def _mp3(self, text):
        """MP3 bytes of the phrase from whichever channel answers first (hedged, retried)."""
        queue: asyncio.Queue = asyncio.Queue()
        first = asyncio.Event()
        state = {'winner': None, 'used': []}

        async def attempt(tag):
            t0 = time.monotonic()
            warm = self._free(exclude=state['used'])
            note = {'tag': tag, 'warm': warm is not None, 'idle': None if warm is None else round(t0 - warm.used, 1)}
            self.trace.append(note)
            channel, temporary = await self._pooled(exclude=state['used'])
            if warm is None:
                note['connect'] = round(time.monotonic() - t0, 3)
            state['used'].append(channel)
            try:
                async for data in channel.phrase(text, self.voice, self.rate):
                    if state['winner'] is None:
                        state['winner'] = tag
                        note['first'] = round(time.monotonic() - t0, 3)
                        first.set()
                    if state['winner'] != tag:
                        await channel.close()        # the other channel was faster: drop this stream
                        return
                    queue.put_nowait(data)
                if state['winner'] == tag:
                    queue.put_nowait(None)
                if temporary:
                    await channel.close()
            except BaseException as error:
                note['error'] = type(error).__name__ + ':' + str(error)[:60]
                await channel.close()                # a stream broken mid-turn cannot be reused
                if state['winner'] == tag:
                    queue.put_nowait(None)           # what was said stays said
                raise

        self.stats['phrases'] += 1
        started = time.monotonic()
        tasks = [asyncio.ensure_future(attempt('a'))]
        signal = asyncio.ensure_future(first.wait())
        hedged = False
        try:
            while not first.is_set():
                elapsed = time.monotonic() - started
                if elapsed >= FIRST_SOUND_S:
                    self.stats['failed'] += 1
                    return
                limit = FIRST_SOUND_S if hedged else HEDGE_S
                await asyncio.wait([signal, *[t for t in tasks if not t.done()]],
                                   timeout=max(.01, min(limit, FIRST_SOUND_S) - elapsed),
                                   return_when=asyncio.FIRST_COMPLETED)
                if first.is_set():
                    break
                all_failed = all(t.done() for t in tasks)
                if not hedged and (time.monotonic() - started >= HEDGE_S or all_failed):
                    hedged = True                    # slow or failed: ask another channel as well
                    self.stats['hedged'] += 1
                    tasks.append(asyncio.ensure_future(attempt('b')))
                elif hedged and all_failed:
                    self.stats['failed'] += 1
                    return
            if state['winner'] == 'b':
                self.stats['hedge_won'] += 1
            while True:
                data = await queue.get()
                if data is None:
                    return
                yield data
        finally:
            signal.cancel()
            for task in tasks:
                if not task.done():
                    task.cancel()
                    task.add_done_callback(_retrieved)
                else:
                    _retrieved(task)     # its refusal is counted above; not «Task exception was never retrieved»

    async def _pcm(self, text):
        import av
        codec = av.CodecContext.create('mp3', 'r')
        resample = Resampler()
        async for data in self._mp3(text):
            for packet in codec.parse(data):
                for frame in codec.decode(packet):
                    x = frame.to_ndarray()
                    if x.ndim > 1:
                        x = x[0]
                    if x.dtype.kind == 'f':
                        x = np.clip(x, -1, 1) * 32767
                    x = x.astype(np.float32) * GAIN          # the owner: this voice is loud
                    pcm = resample(x.astype('<i2').tobytes())
                    if pcm:
                        yield pcm

    async def _collect(self, text):
        pcm = b''.join([pcm async for pcm in self._pcm(text)])
        if not pcm:
            raise RuntimeError('edge_no_audio')     # refused: not an empty line to keep in the cache
        return pcm

    async def _speak(self, text, is_current, started):
        begun = time.monotonic()
        async for pcm in self._pcm(text):
            if not is_current():
                return
            if not started.is_set():
                self.stats['last_first_s'] = round(time.monotonic() - begun, 3)
            self.audio.play(pcm)
            started.set()

    # ------------------------------------------------------------ API
    def say(self, text, is_current=lambda: True):
        """Speak; returns seconds to the first sound, or None if Microsoft did not answer in time."""
        begun = time.monotonic()
        text = text.strip()
        if text in self.cache:
            self.audio.play(self.cache[text])
            return time.monotonic() - begun
        started = threading.Event()
        future = asyncio.run_coroutine_threadsafe(self._speak(text, is_current, started), self.loop)
        if not started.wait(FIRST_SOUND_S + .3):
            future.cancel()
            return None
        first = time.monotonic() - begun
        try:
            future.result(60)
        except Exception:
            pass
        return first
