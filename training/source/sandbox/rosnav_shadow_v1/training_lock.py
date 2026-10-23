"""Exclusive output lock for offline training only."""

from __future__ import annotations
import os, hashlib
from pathlib import Path
from typing import Iterable
_HELD_GUARDS = {}

class HandoffError(RuntimeError):
    """The handoff cannot continue without weakening its identity contract."""

def _guard_key(lock: Path) -> str:
    return os.path.normcase(str(Path(lock).resolve()))

class CoordinatorGuard:
    """One OS-held lease serialises both first launch and stale recovery."""

    def __init__(self, lock: Path):
        self.lock = Path(lock)
        self.path = self.lock.with_name(f"{self.lock.name}.guard")
        self.handle = None

    def __enter__(self) -> "CoordinatorGuard":
        key = _guard_key(self.lock)
        if key in _HELD_GUARDS:
            raise HandoffError(f"handoff guard is already held: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            handle.close()
            raise HandoffError(
                f"another coordinator holds handoff guard {self.path}") from error
        self.handle = handle
        _HELD_GUARDS[key] = self
        return self

    def __exit__(self, _error_type, _error, _traceback) -> None:
        key = _guard_key(self.lock)
        handle = self.handle
        try:
            if handle is not None:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            if handle is not None:
                handle.close()
            self.handle = None
            _HELD_GUARDS.pop(key, None)

def exclusive_guard(lock: Path) -> CoordinatorGuard:
    return CoordinatorGuard(lock)

def canonical_path(path: Path | str, cwd: Path | str | None = None) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate if cwd is None else Path(cwd) / candidate
    rendered = os.path.normcase(os.path.realpath(os.path.abspath(candidate)))
    return rendered.casefold() if os.name == "nt" else rendered

def parsed_option(command: Iterable[str], option: str) -> str | None:
    arguments = [str(item) for item in command]
    found: str | None = None
    prefix = f"{option}="
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments):
                raise HandoffError(f"{option} has no value in process command")
            value = arguments[index + 1]
        elif argument.startswith(prefix):
            value = argument[len(prefix):]
        else:
            continue
        if found is not None:
            raise HandoffError(f"process command repeats {option}")
        found = value
    return found

def output_guard_scope(output: Path | str) -> Path:
    resolved = Path(output).resolve()
    identity = hashlib.sha256(
        canonical_path(resolved).encode("utf-8")).hexdigest()[:24]
    return resolved.parent / f".robotbrain-training-output-{identity}"
