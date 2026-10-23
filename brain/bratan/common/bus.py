"""Local publish/subscribe bus: one hub, newline-delimited JSON over a socket.

Wire format, one JSON object per line:
  {"t": topic, "d": data}            publish (hub forwards to matching subscribers)
  {"sub": ["body.", "voice.heard"]}  subscribe to topic prefixes (client -> hub)

Topics ending in ".state" are retained: a new subscriber immediately receives the
latest value. Delivery is best effort and never blocks a publisher: a subscriber
that falls far behind is disconnected and reconnects.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from collections import deque

DEFAULT_ADDRESS = os.environ.get('BRATAN_BUS', '/run/bratan/bus.sock')
MAX_LINE = 1 << 20
MAX_BACKLOG = 4 << 20
PING_S = 5.            # the hub pings every client; a client hearing nothing for
SILENT_S = 15.         # this long treats the connection as dead and reconnects
PING = b'{"t":"_ping"}\n'


def _plain(value):
    """numpy scalars and arrays as plain JSON (a numpy bool in body.state broke a docking in the simulator)."""
    if getattr(value, 'shape', None) == () and callable(getattr(value, 'item', None)):
        return value.item()
    if callable(getattr(value, 'tolist', None)):
        return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def _encode(topic, data) -> bytes:
    return (json.dumps({'t': topic, 'd': data}, ensure_ascii=False, separators=(',', ':'),
                       allow_nan=False, default=_plain) + '\n').encode()


def _tcp(address):
    if address.startswith('tcp:'):
        host, port = address[4:].rsplit(':', 1)
        return host, int(port)
    return None


class Hub:
    """Asyncio hub; runs inside the supervisor's event loop."""

    def __init__(self, address=DEFAULT_ADDRESS):
        self.address = address
        self.clients: dict[asyncio.StreamWriter, tuple[str, ...]] = {}
        self.retained: dict[str, bytes] = {}
        self.local = []          # in-process callbacks: (prefixes, fn(topic, data))
        self.server = None

    async def start(self):
        tcp = _tcp(self.address)
        if tcp:
            self.server = await asyncio.start_server(self._serve, *tcp, limit=MAX_LINE)
        else:
            os.makedirs(os.path.dirname(self.address), exist_ok=True)
            try:
                os.unlink(self.address)
            except FileNotFoundError:
                pass
            self.server = await asyncio.start_unix_server(self._serve, self.address, limit=MAX_LINE)
            os.chmod(self.address, 0o600)
        self.pinger = asyncio.get_running_loop().create_task(self._ping())

    async def _ping(self):
        """A peer that vanished without a close (seen after restarts) must not keep a client deaf forever."""
        while True:
            await asyncio.sleep(PING_S)
            for writer in list(self.clients):
                try:
                    writer.write(PING)
                except Exception:
                    self.clients.pop(writer, None)

    def subscribe(self, prefixes, callback):
        """In-process subscriber (called on the hub's loop, must not block)."""
        self.local.append((tuple(prefixes), callback))
        for topic, line in self.retained.items():
            if topic.startswith(tuple(prefixes)):
                callback(topic, json.loads(line)['d'])

    def publish(self, topic, data):
        self._route(topic, _encode(topic, data), data)

    def _route(self, topic, line, data=None):
        if topic.endswith('.state'):
            self.retained[topic] = line
        for writer, prefixes in list(self.clients.items()):
            if prefixes and topic.startswith(prefixes):
                transport = writer.transport
                if transport.get_write_buffer_size() > MAX_BACKLOG:
                    writer.close()
                    self.clients.pop(writer, None)
                    continue
                writer.write(line)
        if self.local:
            if data is None:
                data = json.loads(line)['d']
            for prefixes, callback in self.local:
                if topic.startswith(prefixes):
                    try:
                        callback(topic, data)
                    except Exception as error:  # a subscriber bug must not stop routing
                        print('bus_local_callback_error', topic, type(error).__name__, error, flush=True)

    async def _serve(self, reader, writer):
        self.clients[writer] = ()
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if 'sub' in message:
                    prefixes = tuple(str(p) for p in message['sub'])
                    self.clients[writer] = prefixes
                    for topic, retained in self.retained.items():
                        if topic.startswith(prefixes):
                            writer.write(retained)
                elif 't' in message:
                    self._route(str(message['t']), line if line.endswith(b'\n') else line + b'\n',
                                message.get('d'))
        except (ConnectionError, asyncio.LimitOverrunError, ValueError):
            pass
        finally:
            self.clients.pop(writer, None)
            writer.close()


class Client:
    """Thread-based client for worker processes. Reconnects forever.

    `on_message(topic, data)` runs on the receive thread; keep it short.
    `publish` never blocks for long: while disconnected, messages are dropped
    except the latest retained-state messages, which are replayed on connect.
    """

    def __init__(self, name, prefixes=(), on_message=None, address=DEFAULT_ADDRESS):
        self.name, self.prefixes, self.on_message = name, tuple(prefixes), on_message
        self.address = address
        self.sock = None
        self.lock = threading.Lock()
        self.pending_state: dict[str, bytes] = {}
        self.stopping = threading.Event()
        self.connected = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f'bus-{name}', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def wait_connected(self, timeout=5.):
        return self.connected.wait(timeout)

    def publish(self, topic, data):
        line = _encode(topic, data)
        with self.lock:
            if topic.endswith('.state'):
                self.pending_state[topic] = line
            sock = self.sock
            if sock is None:
                return False
            try:
                sock.sendall(line)
                if topic.endswith('.state'):
                    self.pending_state.pop(topic, None)
                return True
            except OSError:
                self._drop()
                return False

    def _drop(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.connected.clear()

    def _connect(self):
        tcp = _tcp(self.address)
        if tcp:
            sock = socket.create_connection(tcp, timeout=2.)
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2.)
            sock.connect(self.address)
        sock.settimeout(SILENT_S)
        if self.prefixes:
            sock.sendall((json.dumps({'sub': list(self.prefixes)}) + '\n').encode())
        with self.lock:
            for line in self.pending_state.values():
                sock.sendall(line)
            self.pending_state.clear()
            self.sock = sock
        self.connected.set()
        return sock

    def _run(self):
        while not self.stopping.is_set():
            try:
                sock = self._connect()
            except OSError:
                self.stopping.wait(.5)
                continue
            buffer = b''
            try:
                while not self.stopping.is_set():
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buffer += chunk
                    if len(buffer) > MAX_LINE * 4:
                        break
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        if not line or self.on_message is None or line == PING[:-1]:
                            continue
                        try:
                            message = json.loads(line)
                            self.on_message(message['t'], message.get('d'))
                        except Exception as error:
                            print(f'bus_{self.name}_handler_error', type(error).__name__, error, flush=True)
            except OSError:             # includes the silence timeout: reconnect
                pass
            with self.lock:
                if self.sock is sock:
                    self._drop()
            self.stopping.wait(.2)

    def close(self):
        self.stopping.set()
        with self.lock:
            self._drop()


class Inbox:
    """Thread-safe latest-value + queued-event store fed by a Client callback."""

    def __init__(self, maxlen=256):
        self.lock = threading.Lock()
        self.latest: dict[str, tuple[float, object]] = {}
        self.events = deque(maxlen=maxlen)
        self.signal = threading.Event()

    def __call__(self, topic, data):
        now = time.monotonic()
        with self.lock:
            self.latest[topic] = (now, data)
            if not topic.endswith('.state'):
                self.events.append((now, topic, data))
        self.signal.set()

    def get(self, topic, default=None, max_age=None):
        with self.lock:
            item = self.latest.get(topic)
        if item is None or (max_age is not None and time.monotonic() - item[0] > max_age):
            return default
        return item[1]

    def drain(self):
        with self.lock:
            items = list(self.events)
            self.events.clear()
            self.signal.clear()
        return items
