"""Android face app API (127.0.0.1:8766): camera, microphone, speaker, face, light, proxy.

Every scope has its own capability token in the secrets directory.
"""
from __future__ import annotations

import json
import os
import urllib.request

from ..common.config import SECRETS

BASE = 'http://127.0.0.1:8766'
SCOPES = {
    'status.read': 'face-status-key', 'camera.read': 'face-camera-read-key',
    'microphone.read': 'face-microphone-read-key', 'speaker.write': 'face-speaker-write-key',
    'face.write': 'face-state-key', 'proxy.read': 'face-proxy-read-key',
    'proxy.connect': 'face-proxy-connect-key',
}
_tokens: dict[str, tuple[float, str]] = {}


def token(scope: str) -> str:
    path = SECRETS / SCOPES[scope]
    mtime = os.stat(path).st_mtime
    cached = _tokens.get(scope)
    if cached and cached[0] == mtime:
        return cached[1]
    value = path.read_text().strip()
    _tokens[scope] = (mtime, value)
    return value


def request(path, scope, *, data=None, method=None, timeout=3., headers=None):
    request_headers = {'Authorization': 'Bearer ' + token(scope), **(headers or {})}
    return urllib.request.urlopen(urllib.request.Request(BASE + path, data=data, method=method,
                                                         headers=request_headers), timeout=timeout)


def get_json(path, scope, timeout=3.):
    with request(path, scope, timeout=timeout) as response:
        return json.load(response)


def post(path, scope, data=b'', timeout=3., content_type='application/octet-stream'):
    with request(path, scope, data=data, method='POST', timeout=timeout,
                 headers={'Content-Type': content_type}) as response:
        body = response.read()
    try:
        return json.loads(body)
    except ValueError:
        return {}


def proxy_url(port):
    return f'http://robotbrain:{token("proxy.connect")}@127.0.0.1:{port}'
