"""Backup voice when OpenAI is down: official Azure Speech, ru-RU-DmitryNeural (the voice the old robot had).

Steady where the free Edge endpoint is not (06.10: 10/10 phrases, first audio 0.15–0.7 s).
- two HTTPS connections stay open (keep-alive), reopened quietly when idle too long;
- raw 16 kHz PCM comes straight from the service: no decoding, no resampling;
- the first clause is requested separately and in parallel with the rest, so the first
  sound waits only for a short piece; the rest is ready by the time the first ends;
- volume 80 % (the owner: this voice is loud).
"""
from __future__ import annotations

import http.client
import queue
import re
import ssl
import threading
import time
from xml.sax.saxutils import escape

import numpy as np

from .local import READY_LINES

VOICE = 'ru-RU-DmitryNeural'
RATE_UP = '+8%'
GAIN = .8
IDLE_REOPEN_S = 180.      # the cloud's load balancer drops idle keep-alive connections after a few minutes
FIRST_SOUND_S = 2.
HEAD = re.compile(r'^(\S+(?:\s+\S+){1,}?[,.!?…;:—])\s+(.+)$', re.S)     # first clause of 2+ words


class Unavailable(RuntimeError):
    pass


class AzureSpeech:
    def __init__(self, audio, key, region, voice=VOICE, rate=RATE_UP):
        self.audio, self.key, self.voice, self.rate = audio, key, voice, rate
        self.host = f'{region}.tts.speech.microsoft.com'
        self.context = ssl.create_default_context()
        self.pool: queue.Queue = queue.Queue()
        self.cache = {}
        self.stats = {'phrases': 0, 'failed': 0, 'last_first_s': None, 'last_error': None}
        self.down_until = 0.
        threading.Thread(target=self._warm, name='azure-warm', daemon=True).start()

    # ------------------------------------------------------------ connections
    def _open(self):
        conn = http.client.HTTPSConnection(self.host, 443, timeout=5, context=self.context)
        conn.connect()
        return [conn, time.monotonic()]

    def _take(self):
        while True:
            try:
                item = self.pool.get_nowait()
            except queue.Empty:
                return self._open()
            if time.monotonic() - item[1] < IDLE_REOPEN_S:
                return item
            item[0].close()

    def _give(self, item):
        item[1] = time.monotonic()
        if self.pool.qsize() < 2:
            self.pool.put(item)
        else:
            item[0].close()

    def _warm(self):
        for _ in range(2):
            try:
                self.pool.put(self._open())
            except OSError:
                pass
        for line in READY_LINES:
            try:
                self.cache[line] = b''.join(self._stream(line))
            except Exception:
                pass

    # ------------------------------------------------------------ one request
    def _stream(self, text):
        """16 kHz PCM pieces of one text as they arrive (volume applied)."""
        ssml = (f"<speak version='1.0' xml:lang='ru-RU'><voice name='{self.voice}'>"
                f"<prosody rate='{self.rate}'>{escape(text)}</prosody></voice></speak>").encode()
        for attempt in range(2):
            item = self._take()
            conn = item[0]
            try:
                conn.request('POST', '/cognitiveservices/v1', ssml, {
                    'Ocp-Apim-Subscription-Key': self.key, 'Content-Type': 'application/ssml+xml',
                    'X-Microsoft-OutputFormat': 'raw-16khz-16bit-mono-pcm', 'User-Agent': 'bratan'})
                response = conn.getresponse()
                if response.status != 200:
                    response.read()
                    self._give(item)
                    raise Unavailable(f'http_{response.status}')
                carry = b''
                while True:
                    chunk = response.read1(6400) if hasattr(response, 'read1') else response.read(6400)
                    if not chunk:
                        break
                    chunk = carry + chunk
                    cut = len(chunk) & ~1
                    carry = chunk[cut:]
                    x = np.frombuffer(chunk[:cut], dtype='<i2').astype(np.float32) * GAIN
                    yield x.astype('<i2').tobytes()
                self._give(item)
                return
            except Unavailable:
                raise
            except (OSError, http.client.HTTPException):
                conn.close()                     # a dropped keep-alive: one fresh retry
                if attempt:
                    raise

    # ------------------------------------------------------------ API
    def say(self, text, is_current=lambda: True):
        """Speak; returns seconds to the first sound, or None if Azure is unavailable (caller falls back)."""
        begun = time.monotonic()
        text = text.strip()
        if begun < self.down_until:
            return None
        if text in self.cache:
            self.audio.play(self.cache[text])
            return time.monotonic() - begun
        self.stats['phrases'] += 1
        match = HEAD.match(text)
        parts = [match.group(1), match.group(2)] if match else [text]
        # Every part streams into its own queue at once; they play in order.
        queues = [queue.Queue() for _ in parts]

        def fetch(part, out):
            try:
                for pcm in self._stream(part):
                    out.put(pcm)
                out.put(None)
            except Exception as error:
                out.put(error)

        for part, out in zip(parts, queues):
            threading.Thread(target=fetch, args=(part, out), name='azure-fetch', daemon=True).start()
        first = None
        for index, out in enumerate(queues):
            last = index == len(queues) - 1
            lead = True                          # the service pads every request with silence: cut it at the joins
            tail = b''
            while True:
                try:
                    item = out.get(timeout=FIRST_SOUND_S if first is None else 10.)
                except queue.Empty:
                    item = Unavailable('timeout')
                if isinstance(item, Exception):
                    self.stats['last_error'] = str(item)[:80]
                    if first is None:
                        self.stats['failed'] += 1
                        # Refused (quota, key): do not ask again for a while.
                        if isinstance(item, Unavailable) and str(item).startswith('http_4'):
                            self.down_until = time.monotonic() + 600
                        return None
                    return first                  # broke off mid-way: what was said stays said
                if item is None:
                    if tail:
                        self.audio.play(tail if last else trim_end(tail))
                    break
                if not is_current():
                    return first
                if lead:
                    item = trim_start(item)
                    if not item:
                        continue
                    lead = False
                pcm, tail = (tail + item)[:-TAIL_BYTES], (tail + item)[-TAIL_BYTES:]
                if pcm:
                    if first is None:
                        first = time.monotonic() - begun
                        self.stats['last_first_s'] = round(first, 3)
                    self.audio.play(pcm)
        return first


TAIL_BYTES = 19200            # 0.6 s held back so the silence at a join can be cut
QUIET = 250


def trim_start(pcm):
    x = np.frombuffer(pcm, dtype='<i2')
    loud = np.flatnonzero(np.abs(x) > QUIET)
    return b'' if not len(loud) else x[max(0, loud[0] - 160):].tobytes()     # keep 10 ms before the voice


def trim_end(pcm):
    x = np.frombuffer(pcm, dtype='<i2')
    loud = np.flatnonzero(np.abs(x) > QUIET)
    return b'' if not len(loud) else x[:loud[-1] + 1600].tobytes()          # keep 0.1 s: a natural comma pause
