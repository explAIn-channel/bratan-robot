"""Own speech (when no live session): OpenAI TTS streamed as PCM, played while it arrives."""
from __future__ import annotations

import base64
import http.client
import json
import ssl
import time

import numpy as np

from ..common.persona import TTS_STYLE

HOST = 'api.openai.com'


def _tunnel(proxy_port, credential):
    conn = http.client.HTTPSConnection('127.0.0.1', proxy_port, timeout=10, context=ssl.create_default_context())
    token = base64.b64encode(f'robotbrain:{credential}'.encode()).decode()
    conn.set_tunnel(HOST, 443, headers={'Proxy-Authorization': 'Basic ' + token})
    return conn


class Resampler:
    """24 kHz -> 16 kHz (up 2, low-pass, down 3), continuous across chunks."""

    def __init__(self):
        from scipy.signal import firwin
        self.taps = firwin(96, 7600, fs=48000) * 2          # cut below the new 8 kHz Nyquist
        self.history = np.zeros(len(self.taps) - 1, dtype=np.float64)
        self.phase = 0

    def __call__(self, pcm24: bytes) -> bytes:
        x = np.frombuffer(pcm24, dtype='<i2').astype(np.float64)
        if not len(x):
            return b''
        up = np.zeros(len(x) * 2)
        up[::2] = x
        joined = np.concatenate((self.history, up))
        filtered = np.convolve(joined, self.taps, mode='valid')
        self.history = joined[-(len(self.taps) - 1):]
        out = filtered[self.phase::3]
        self.phase = (self.phase - len(filtered)) % 3
        return np.clip(out, -32768, 32767).astype('<i2').tobytes()


class Speech:
    def __init__(self, key, audio, proxy_credential, voice='coral', model='gpt-4o-mini-tts'):
        self.key, self.audio, self.voice, self.model = key, audio, voice, model
        self.proxy_credential = proxy_credential

    def say(self, text, is_current=lambda: True):
        """Stream speech to the speaker; returns seconds until the first sound or None."""
        started = time.monotonic()
        body = json.dumps({'model': self.model, 'voice': self.voice, 'input': text[:600],
                           'instructions': TTS_STYLE, 'response_format': 'pcm'}).encode()
        for port in (19408, 19406):
            try:
                conn = _tunnel(port, self.proxy_credential())
                conn.request('POST', '/v1/audio/speech', body, {'Authorization': 'Bearer ' + self.key,
                                                                'Content-Type': 'application/json'})
                response = conn.getresponse()
                if response.status != 200:
                    response.read()
                    continue
                resample = Resampler()
                first = None
                while is_current():
                    chunk = response.read1(9600) if hasattr(response, 'read1') else response.read(9600)
                    if not chunk:
                        break
                    if len(chunk) % 2:
                        chunk += response.read(1)
                    pcm = resample(chunk)
                    if pcm:
                        if first is None:
                            first = time.monotonic() - started
                        self.audio.play(pcm)
                conn.close()
                return first
            except OSError:
                continue
        return None
