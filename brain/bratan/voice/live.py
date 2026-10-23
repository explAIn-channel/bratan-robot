"""GPT-Live conversation session: the voice and ears; the mind owns deeds.

Lifecycle: `connect()` early (on the first hint of «братан»), `begin()` with the
audio already heard, `release()` once the name is confirmed (output is held until
then), `abort()` if it was not an address. One session lives until 20 s of
silence, a goodbye, or `close()`.
"""
from __future__ import annotations

import base64
import json
import queue
import threading
import time
from collections import deque

URL = 'wss://api.openai.com/v1/live/sessions'
IDLE_CLOSE_S = 20.
PROXY_PORTS = (19408, 19406)


class LiveSession:
    def __init__(self, key, audio, *, instructions, voice='coral', on_event=None, proxy_url=None):
        self.key, self.audio = key, audio
        self.instructions, self.voice = instructions, voice
        self.on_event = on_event or (lambda kind, **data: None)
        self.proxy_url = proxy_url
        self.ws = None
        self.state = 'idle'              # idle -> connecting -> connected -> live -> closed
        self.hold = True
        self.held = []
        self.early = []
        self.lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.activity = time.monotonic()
        self.input_text = []
        self.output_text = ''
        self.input_consumed = 0
        self.pending = deque()           # delegation ids waiting for transcript
        self.answers: queue.Queue = queue.Queue()
        self.ask = None                  # callable(delegation_id, utterance) -> None (answer via answer())
        self.unknown = set()
        self.started_at = None
        self.first_audio_at = None
        self.idle_close_s = IDLE_CLOSE_S      # a speak-only session closes soon after its line
        self.deadline = None                  # hard end (monotonic) for a speak-only session
        self.thread = None

    # ------------------------------------------------------------ setup
    def connect(self):
        """Open the socket in the background (TLS + proxy ~0.8 s)."""
        with self.lock:
            if self.state != 'idle':
                return
            self.state = 'connecting'
        threading.Thread(target=self._connect, name='live-connect', daemon=True).start()

    def _connect(self):
        from websockets.sync.client import connect
        error = None
        for port in PROXY_PORTS:
            try:
                ws = connect(URL, proxy=self.proxy_url(port) if self.proxy_url else None,
                             additional_headers={'Authorization': 'Bearer ' + self.key},
                             open_timeout=4, close_timeout=1, ping_interval=15, ping_timeout=10,
                             max_size=1 << 21)
                with self.lock:
                    if self.state == 'closed':
                        ws.close()
                        return
                    self.ws = ws
                    self.state = 'connected'
                return
            except Exception as exc:
                error = exc
        with self.lock:
            self.state = 'closed'
        self.on_event('failed', error=f'{type(error).__name__}:{error}'[:160])

    def begin(self, frames=(), first_text=None):
        """Start the session (waits for the socket) and stream the audio heard so far."""
        deadline = time.monotonic() + 5.
        while self.state == 'connecting' and time.monotonic() < deadline:
            time.sleep(.01)
        if self.state != 'connected':
            return False
        start = {'type': 'session.start', 'session': {
            'model': 'gpt-live-1', 'instructions': self.instructions(),
            'audio': {'format': {'type': 'audio/pcm', 'rate': 16000}, 'output': {'voice': self.voice}},
            'delegation': {'type': 'client'}, 'store': False}}
        if first_text:
            start['session']['input'] = [{'role': 'user', 'content': [{'type': 'input_text', 'text': first_text}]}]
        deadline = time.monotonic() + 6.
        try:
            self._send(start)
            while time.monotonic() < deadline:
                event = json.loads(self.ws.recv(timeout=max(.05, deadline - time.monotonic())))
                if event.get('type') == 'session.started':
                    break
                if event.get('type') == 'error':
                    detail = event.get('error') or {}
                    self.close()
                    self.on_event('failed', error='start_rejected', code=str(detail.get('code') or ''),
                                  message=str(detail.get('message') or '')[:160])
                    return False
            else:
                self.close()
                self.on_event('failed', error='start_timeout')
                return False
        except Exception as error:              # aborted meanwhile, or the socket dropped
            self.close()
            self.on_event('failed', error=type(error).__name__)
            return False
        with self.lock:
            audio = b''.join(list(frames) + self.early)
            self.early = []
            self.state = 'live'
        self.started_at = time.monotonic()
        self.activity = time.monotonic()
        if audio:
            self.feed(audio)
        self.thread = threading.Thread(target=self._loop, name='live-session', daemon=True)
        self.thread.start()
        self.on_event('started')
        return True

    # ------------------------------------------------------------ streaming
    def feed(self, pcm):
        if not pcm:
            return
        with self.lock:
            if self.state in ('connecting', 'connected', 'starting'):
                self.early.append(pcm)          # heard while the session was opening
                return
            if self.state != 'live':
                return
        self._send({'type': 'session.input_audio.append', 'audio': base64.b64encode(pcm).decode('ascii')})

    def release(self):
        """The name was confirmed: play what was held and everything after."""
        with self.lock:
            self.hold = False
            held, self.held = self.held, []
        for pcm in held:
            self.audio.play(pcm)

    def abort(self):
        with self.lock:
            self.held = []
        self.close(quiet=True)

    def instruct(self, text):
        if self.state == 'live':
            self._send({'type': 'session.instructions.append', 'delegation_id': None, 'content': text})

    def answer(self, delegation_id, text, *, silent=False):
        self.answers.put((delegation_id, text, silent))

    def close(self, quiet=False):
        with self.lock:
            was = self.state
            self.state = 'closed'
        ws = self.ws
        if ws is not None and was in ('live', 'connected', 'closing'):
            try:
                self._send({'type': 'session.close'})
            except Exception:
                pass
            try:
                ws.close()
            except Exception:
                pass
        if was == 'connected' and ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if was in ('live', 'closing') and not quiet:
            self.on_event('closed', heard=''.join(self.input_text).strip(), said=self.output_text.strip())
        elif was in ('live', 'closing'):
            self.on_event('aborted')

    @property
    def active(self):
        return self.state in ('connecting', 'connected', 'live')

    def _send(self, event):
        with self.send_lock:
            self.ws.send(json.dumps(event, ensure_ascii=False))

    # ------------------------------------------------------------ events
    def _loop(self):
        try:
            while self.state == 'live':
                self._deliver_answers()
                self._serve_delegations()
                if time.monotonic() - self.activity > self.idle_close_s and not self.pending and not self.audio.speaking:
                    break
                if self.deadline is not None and time.monotonic() > self.deadline:
                    self.audio.stop_playing()
                    break
                try:
                    event = json.loads(self.ws.recv(timeout=.05))
                except TimeoutError:
                    continue
                self._handle(event)
        except Exception as error:
            if self.state == 'live':
                self.on_event('failed', error=f'{type(error).__name__}:{error}'[:160])
        finally:
            self.close()

    def _handle(self, event):
        kind = event.get('type')
        now = time.monotonic()
        if kind == 'session.output_audio.delta':
            pcm = base64.b64decode(event.get('delta', ''))
            if self.first_audio_at is None:
                self.first_audio_at = now
                self.on_event('first_audio', after_start=round(now - (self.started_at or now), 3))
            with self.lock:
                if self.hold:
                    self.held.append(pcm)
                    return
            self.audio.play(pcm)
            self.activity = now
        elif kind == 'session.output_transcript.delta':
            self.output_text += event.get('delta', '')
            self.activity = now
            self.on_event('said', delta=event.get('delta', ''))
        elif kind == 'session.input_transcript.delta':
            self.input_text.append(event.get('delta', ''))
            self.activity = now
            self.on_event('heard', delta=event.get('delta', ''))
        elif kind == 'session.delegation.created':
            item = event.get('delegation', {})
            if item.get('target') == 'client' and item.get('id'):
                self.pending.append((item['id'], now))
                self.activity = now
        elif kind == 'session.closed':
            self.state = 'closing'
        elif kind == 'error':
            self.on_event('api_error', code=(event.get('error') or {}).get('code'))
        elif kind not in ('session.usage.updated', 'session.commentary.appended', 'session.thinking.appended',
                          'session.instructions.appended', 'info', 'response') and kind not in self.unknown:
            self.unknown.add(kind)
            self.on_event('unknown_event', kind=kind)

    def _serve_delegations(self):
        """Hand a delegated request to the mind with the words heard since the last one."""
        if not self.pending or self.ask is None:
            return
        delegation_id, since = self.pending[0]
        text = ''.join(self.input_text[self.input_consumed:]).strip()
        waited = time.monotonic() - since
        if not text and waited < 1.5:
            return                       # transcripts can trail the delegation slightly
        self.pending.popleft()
        self.input_consumed = len(self.input_text)
        if not text:
            # The words never arrived as text: ask the person rather than invent a deed.
            self._send({'type': 'session.thinking.append', 'delegation_id': delegation_id,
                        'content': 'Слова просьбы не расслышаны. Коротко попроси повторить, что сделать.'})
            return
        self.ask(delegation_id, text, self.output_text[-300:])

    def _deliver_answers(self):
        while True:
            try:
                delegation_id, text, silent = self.answers.get_nowait()
            except queue.Empty:
                return
            if silent:
                content = 'Это было сказано не тебе. Ничего не отвечай.'
                self._send({'type': 'session.thinking.append', 'delegation_id': delegation_id, 'content': content})
            else:
                self._send({'type': 'session.commentary.append', 'delegation_id': delegation_id,
                            'content': text[:1200]})
            self.activity = time.monotonic()
