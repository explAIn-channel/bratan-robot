"""Chat model client with persistent HTTPS connections (no TLS handshake per call).

OpenRouter / DeepSeek V4 Flash, reasoning disabled, JSON or plain text,
optional streaming with a callback on accumulated text.
"""
from __future__ import annotations

import http.client
import json
import queue
import ssl
import threading
import time
from urllib.parse import urlsplit


class LLMError(RuntimeError):
    pass


STALL_S = 8.          # no new bytes for this long: stuck (a big prompt may take ~6 s to its first token)


class LLM:
    """Answers always stream internally: a stuck request shows itself after STALL_S of silence
    instead of eating the whole timeout (07.10: ~3 % of thoughts were lost to 10 s timeouts).
    OpenRouter falls back to `fallback` by itself when the main model fails."""

    def __init__(self, base_url, key, model, *, pool=3, fallback=None, providers=(), direct=None):
        # direct: (LLM on DeepSeek's own API, {OpenRouter model id: DeepSeek's model name}) — those models go there
        # first, OpenRouter (and its fallback) only if DeepSeek fails (18.10, the owner: «апи именно дипсика, юзай»).
        self.direct = direct
        parts = urlsplit(base_url)
        self.host, self.path = parts.hostname, parts.path.rstrip('/') + '/chat/completions'
        self.key, self.model, self.fallback = key, model, fallback
        self.providers = list(providers)       # preferred OpenRouter providers, fastest first
        self.openrouter = 'openrouter' in self.host
        self.context = ssl.create_default_context()
        self.idle: queue.LifoQueue = queue.LifoQueue()
        self.pool = pool
        self.lock = threading.Lock()
        self.stats = {'calls': 0, 'errors': 0, 'last_s': None, 'last_first_token_s': None}

    def warm(self):
        """Open the connections now so the first real call pays no handshake."""
        if self.direct is not None:
            self.direct[0].warm()
        for _ in range(self.pool):
            try:
                conn = self._new()
                conn.connect()
                self.idle.put((time.monotonic(), conn))
            except OSError:
                pass

    def _new(self):
        return http.client.HTTPSConnection(self.host, 443, timeout=20, context=self.context)

    def _take(self):
        while True:
            try:
                stamp, conn = self.idle.get_nowait()
            except queue.Empty:
                return self._new()
            if time.monotonic() - stamp < 25:          # the server drops idle keep-alive connections early
                return conn
            conn.close()

    def _give(self, conn):
        if self.idle.qsize() < self.pool:
            self.idle.put((time.monotonic(), conn))
        else:
            conn.close()

    def chat(self, system, user, *, max_tokens=400, temperature=.8, json_mode=False, timeout=8.,
             on_text=None, model=None, stall=None, fallback=None, reasoning=None, plugins=None):
        """`reasoning` — 'low' | 'medium' | 'high': the model thinks before it answers (the night's dream); by default
        it does not (a thought must come in seconds)."""
        started = time.monotonic()
        name = model or self.model
        if self.direct is not None and name in self.direct[1]:
            try:
                text = self.direct[0].chat(system, user, max_tokens=max_tokens, temperature=temperature,
                                           json_mode=json_mode, timeout=timeout, on_text=on_text,
                                           model=self.direct[1][name], stall=stall, reasoning=reasoning)
                self.stats.update(calls=self.stats['calls'] + 1, via='deepseek',
                                  last_s=self.direct[0].stats.get('last_s'),
                                  last_first_token_s=self.direct[0].stats.get('last_first_token_s'))
                return text
            except Exception as error:                     # anything at all on that path: OpenRouter answers instead
                self.stats['direct_errors'] = self.stats.get('direct_errors', 0) + 1
                self.stats['last_direct_error'] = str(error)[:120]
                timeout = timeout - (time.monotonic() - started)
                if timeout < 1.5:
                    raise LLMError(f'deepseek: {type(error).__name__}: {error}'[:200]) from error
            self.stats['via'] = 'openrouter'
        body = {'model': model or self.model, 'max_tokens': max_tokens, 'temperature': temperature,
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
        if self.openrouter and self.providers:
            # 07.10: the auto-router spread calls over providers (1.9–16 s); StreamLake/Novita answer in ~2 s,
            # and staying with one provider also keeps its cache of the unchanged prompt beginning.
            body['provider'] = {'order': self.providers, 'allow_fallbacks': not getattr(self, 'only', False)}
        fallback = self.fallback if fallback is None else fallback
        if self.openrouter and fallback and (model or self.model) != fallback:
            body['models'] = [model or self.model, fallback]            # OpenRouter's own failover
        if reasoning:
            # DeepSeek's own API knows low | high | max (its default is high).
            body.update({'reasoning': {'effort': reasoning, 'exclude': True}} if self.openrouter
                        else {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low' if reasoning == 'low' else 'high'})
        else:
            body.update({'reasoning': {'enabled': False}} if self.openrouter else {'thinking': {'type': 'disabled'}})
        if json_mode:
            body['response_format'] = {'type': 'json_object'}
        if plugins and self.openrouter:
            body['plugins'] = plugins                  # OpenRouter's own (web search for the robot's «web», C52)
        body['stream'] = True                  # always: stalls are seen early (see STALL_S)
        body['stream_options' if not self.openrouter else 'usage'] = (
            {'include_usage': True} if not self.openrouter else {'include': True})    # tokens, cache hits, price
        payload = json.dumps(body, ensure_ascii=False).encode()
        headers = {'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'}
        started = time.monotonic()
        for attempt in range(3):
            conn = self._take()
            left = max(1., timeout - (time.monotonic() - started))
            conn.timeout = min(stall or STALL_S, left)
            try:
                if conn.sock is not None:
                    conn.sock.settimeout(min(stall or STALL_S, left))
                conn.request('POST', self.path, payload, headers)
                response = conn.getresponse()
                if response.status != 200:
                    detail = response.read()[:300]
                    conn.close()
                    raise LLMError(f'http_{response.status}:{detail[:120]!r}')
                text = self._stream(response, on_text or (lambda _text: None), started)
                try:
                    response.read()                # drain the end of the chunked body, or the next request on
                except Exception:                  # this connection breaks (07.10: a retry on every call)
                    conn.close()
                    conn = None
                if conn is not None:
                    self._give(conn)
                self.stats.update(calls=self.stats['calls'] + 1, last_s=round(time.monotonic() - started, 3))
                return text
            except (OSError, http.client.HTTPException) as error:
                self.stats['last_retry'] = f'{type(error).__name__}:{error}'[:80]
                conn.close()
                while not self.idle.empty():          # the others idled as long: start fresh
                    self.idle.get_nowait()[1].close()
                self.stats['retries'] = self.stats.get('retries', 0) + 1
                if attempt == 2 or time.monotonic() - started > timeout - 1.:
                    self.stats['errors'] += 1
                    raise LLMError(f'{type(error).__name__}:{error}') from error
            except LLMError:
                self.stats['errors'] += 1
                raise

    def _whole(self, response):
        value = json.loads(response.read())
        choice = (value.get('choices') or [{}])[0]
        if choice.get('finish_reason') == 'length':
            raise LLMError('truncated')
        return (choice.get('message') or {}).get('content') or ''

    def _stream(self, response, on_text, started):
        text = ''
        buffer = b''
        first = None
        while True:
            chunk = response.read1(4096) if hasattr(response, 'read1') else response.read(4096)
            if not chunk:
                break
            buffer += chunk
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                line = line.strip()
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    return text
                try:
                    value = json.loads(data)
                    delta = ((value.get('choices') or [{}])[0].get('delta') or {}).get('content') or ''
                except (ValueError, AttributeError, IndexError):
                    continue
                if value.get('usage'):
                    self.stats['last_usage'] = value['usage']
                if delta:
                    if first is None:
                        first = time.monotonic() - started
                        self.stats['last_first_token_s'] = round(first, 3)
                    text += delta
                    try:
                        on_text(text)
                    except Exception:
                        pass
        return text


def direct_llm(config):
    """pool -> (LLM on DeepSeek's own API, model map) for LLM(direct=…), or None without its key (config llm_direct)."""
    direct = config.get('llm_direct') or {}

    def make(pool):
        if not direct.get('key') or not direct.get('models'):
            return None
        from ..common.config import secret
        try:
            key = secret(direct['key']).lstrip('﻿').strip()    # a key written from Windows may carry a BOM
        except OSError:
            return None
        if not key.isascii():
            return None
        return LLM(direct.get('base_url', 'https://api.deepseek.com'), key, next(iter(direct['models'].values())),
                   pool=pool), dict(direct['models'])
    return make


def parse_json(raw):
    """First JSON object in a model reply (tolerates code fences and prose)."""
    raw = (raw or '').strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except ValueError:
        pass
    start, end = raw.find('{'), raw.rfind('}')
    if start >= 0 and end > start:
        try:
            value = json.loads(raw[start:end + 1])
            return value if isinstance(value, dict) else {}
        except ValueError:
            return {}
    return {}
