"""Step the training environments on more than one core.

Measured 2026-08-31 on this machine (i7-12700K, 12 cores): one environment
step costs about 2.4 ms and a whole PPO update spends 5.4 seconds inside the
environments against 0.1 in the network -- 98 % against 2 %.  The rollout loop
stepped its eight environments one after another in a Python loop, so eleven
cores watched one work, and a graphics card would have attacked the 2 %.

So the environments are dealt to worker processes.  Each worker builds its own
map and its own environments and is handed only the actions for the ones it
owns; nothing is shared, so nothing needs locking.

Determinism is preserved exactly.  Environment ``i`` is seeded ``seed + i`` and
receives the same actions in the same order whether it is stepped here or in a
worker, so a parallel run reproduces a serial one step for step -- which is
asserted in ``test_parallel_envs.py`` rather than assumed.
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np


def _unwrap(message):
    """Raise what a worker reported instead of returning its error tuple."""
    if (isinstance(message, tuple) and len(message) == 2 and
            isinstance(message[0], str) and message[0] == "error"):
        raise RuntimeError("environment worker failed:" + chr(10) + message[1])
    return message


def _build(map_path, seed, indices, scenarios, encoding):
    """Construct the environments one worker owns.  Import inside, because on
    spawn the child starts from a bare interpreter."""
    # Each worker is one core's worth of work.  Left alone, numpy and torch
    # each open a thread pool the size of the machine inside every worker,
    # and eight of those fight each other for twelve cores.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    from rl_env import DriveEnv, FastFlat
    flat = FastFlat(Path(map_path))
    return [DriveEnv(flat, np.random.default_rng(seed + i),
                     scenario=scenarios[i % len(scenarios)], encoding=encoding)
            for i in indices]


def _checkpoint_boundary_envs(envs):
    """Reset a group at a reproducible checkpoint boundary.

    A DriveEnv and its World intentionally share one Generator.  Capturing the
    generator immediately before reset is therefore enough to reproduce the
    complete freshly sampled episode without serialising the much larger
    environment object.
    """
    states = [copy.deepcopy(env.rng.bit_generator.state) for env in envs]
    observations = np.stack([env.reset() for env in envs])
    return observations, states


def _restore_checkpoint_boundary_envs(envs, states):
    if len(states) != len(envs):
        raise ValueError(
            f"checkpoint has {len(states)} environment RNG states, "
            f"but rollout owns {len(envs)} environments")
    for env, state in zip(envs, states):
        env.rng.bit_generator.state = copy.deepcopy(state)
    return np.stack([env.reset() for env in envs])


def _worker(connection, map_path, seed, indices, scenarios, encoding):
    # A worker that dies quietly leaves the parent blocked on a pipe that will
    # never answer, and the traceback -- the one thing that would explain it --
    # goes wherever the child's stderr went.  So failures are sent home.
    import traceback
    try:
        envs = _build(map_path, seed, indices, scenarios, encoding)
    except BaseException:                                       # noqa: BLE001
        connection.send(("error", traceback.format_exc()))
        connection.close()
        return
    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                return
            if command == "reset":
                connection.send(np.stack([env.reset() for env in envs]))
                continue
            if command == "checkpoint_boundary":
                connection.send(_checkpoint_boundary_envs(envs))
                continue
            if command == "restore_checkpoint_boundary":
                connection.send(_restore_checkpoint_boundary_envs(envs,
                                                                    payload))
                continue
            if command == "step":
                try:
                    observations, rewards, dones = [], [], []
                    for env, twist in zip(envs, payload):
                        nxt, reward, done, _outcome = env.step(twist)
                        observations.append(env.reset() if done else nxt)
                        rewards.append(reward)
                        dones.append(float(done))
                except BaseException:                           # noqa: BLE001
                    connection.send(("error", traceback.format_exc()))
                    return
                connection.send((np.stack(observations),
                                 np.asarray(rewards, dtype=np.float32),
                                 np.asarray(dones, dtype=np.float32)))
                continue
            raise ValueError(f"unknown command: {command}")
    finally:
        connection.close()


class SerialRollout:
    """The original behaviour, kept so the parallel one has something to be
    checked against."""

    def __init__(self, map_path, seed, count, scenarios, encoding=None):
        self.envs = _build(map_path, seed, range(count), scenarios, encoding)
        self.count = count

    def reset_all(self):
        return np.stack([env.reset() for env in self.envs])

    def step_all(self, twists):
        observations, rewards, dones = [], [], []
        for env, twist in zip(self.envs, twists):
            nxt, reward, done, _outcome = env.step(twist)
            observations.append(env.reset() if done else nxt)
            rewards.append(reward)
            dones.append(float(done))
        return (np.stack(observations),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=np.float32))

    def checkpoint_boundary(self):
        return _checkpoint_boundary_envs(self.envs)

    def restore_checkpoint_boundary(self, states):
        return _restore_checkpoint_boundary_envs(self.envs, states)

    def close(self):
        pass


class ThreadedRollout:
    """The same environments, stepped by a pool of threads in this process.

    Worth trying before processes because the expensive part of a step is
    numpy -- the raycast alone is a 360 by 470 array -- and numpy drops the
    GIL for that.  What it buys over `ParallelRollout` is the absence of the
    pipes: measured on this machine the process version spent about 0.8 ms per
    worker per step just getting the answer home, against roughly 1.9 ms of
    actual work, so most of the machine went into the postage.
    """

    def __init__(self, map_path, seed, count, scenarios, encoding=None,
                 workers=4):
        from concurrent.futures import ThreadPoolExecutor
        self.envs = _build(map_path, seed, range(count), scenarios, encoding)
        self.count = count
        self.pool = ThreadPoolExecutor(max_workers=max(1, min(workers, count)))

    def reset_all(self):
        return np.stack(list(self.pool.map(lambda e: e.reset(), self.envs)))

    def step_all(self, twists):
        def one(pair):
            env, twist = pair
            nxt, reward, done, _outcome = env.step(twist)
            return (env.reset() if done else nxt), reward, float(done)

        results = list(self.pool.map(one, zip(self.envs, twists)))
        return (np.stack([r[0] for r in results]),
                np.asarray([r[1] for r in results], dtype=np.float32),
                np.asarray([r[2] for r in results], dtype=np.float32))

    def checkpoint_boundary(self):
        # State capture happens before any reset task is submitted.  That
        # ordering is the checkpoint contract, independent of thread timing.
        states = [copy.deepcopy(env.rng.bit_generator.state)
                  for env in self.envs]
        observations = np.stack(list(
            self.pool.map(lambda env: env.reset(), self.envs)))
        return observations, states

    def restore_checkpoint_boundary(self, states):
        if len(states) != len(self.envs):
            raise ValueError(
                f"checkpoint has {len(states)} environment RNG states, "
                f"but rollout owns {len(self.envs)} environments")
        for env, state in zip(self.envs, states):
            env.rng.bit_generator.state = copy.deepcopy(state)
        return np.stack(list(self.pool.map(lambda env: env.reset(), self.envs)))

    def close(self):
        self.pool.shutdown(wait=True)


class ParallelRollout:
    """The same, dealt across processes."""

    def __init__(self, map_path, seed, count, scenarios, encoding=None,
                 workers=4):
        # Two hard ceilings, because the cost of getting this wrong is not a
        # slow run -- it is the owner's machine.  A runaway spawn on
        # 2026-08-31 took the desktop down and forced a reboot, so the number
        # of workers can never exceed the environments they would own, and
        # never take the whole machine: at least two cores are left for
        # whoever is actually sitting at it.
        spare = max(1, (os.cpu_count() or 4) - 2)
        workers = max(1, min(int(workers), count, spare))
        # Contiguous blocks, so environment i keeps its seed and its scenario.
        edges = np.linspace(0, count, workers + 1).astype(int)
        self.slices = [range(edges[w], edges[w + 1]) for w in range(workers)
                       if edges[w + 1] > edges[w]]
        context = mp.get_context("spawn")
        self.pipes, self.processes = [], []
        for indices in self.slices:
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(child, str(map_path), seed, list(indices), scenarios,
                      encoding),
                daemon=True)
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)
        self.count = count

    def reset_all(self):
        for pipe in self.pipes:
            pipe.send(("reset", None))
        return np.concatenate([_unwrap(pipe.recv()) for pipe in self.pipes])

    def step_all(self, twists):
        # Send every worker its slice before reading any reply, or the workers
        # run one at a time and the whole exercise buys nothing.
        for pipe, indices in zip(self.pipes, self.slices):
            pipe.send(("step", [twists[i] for i in indices]))
        observations, rewards, dones = [], [], []
        for pipe in self.pipes:
            chunk = _unwrap(pipe.recv())
            observations.append(chunk[0])
            rewards.append(chunk[1])
            dones.append(chunk[2])
        return (np.concatenate(observations), np.concatenate(rewards),
                np.concatenate(dones))

    def checkpoint_boundary(self):
        for pipe in self.pipes:
            pipe.send(("checkpoint_boundary", None))
        observations, states = [], []
        for pipe in self.pipes:
            chunk_observations, chunk_states = _unwrap(pipe.recv())
            observations.append(chunk_observations)
            states.extend(chunk_states)
        return np.concatenate(observations), states

    def restore_checkpoint_boundary(self, states):
        if len(states) != self.count:
            raise ValueError(
                f"checkpoint has {len(states)} environment RNG states, "
                f"but rollout owns {self.count} environments")
        for pipe, indices in zip(self.pipes, self.slices):
            pipe.send(("restore_checkpoint_boundary",
                       [states[i] for i in indices]))
        return np.concatenate([_unwrap(pipe.recv()) for pipe in self.pipes])

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            if process.is_alive():
                raise RuntimeError(
                    f"environment worker {process.pid} did not stop")
            process.close()
        for pipe in self.pipes:
            pipe.close()
        # Make close idempotent and release Windows process/pipe handles before
        # the temporary evaluation pool is allowed to start.
        self.pipes.clear()
        self.processes.clear()


def make_rollout(map_path, seed, count, scenarios, encoding=None,
                 workers=0, mode="process"):
    """``workers`` of 0 or 1 keeps everything in this process.

    The default is processes because threads were measured and lost.  On this
    machine, 8 environments over 128 steps: serial 538 environment-steps a
    second, processes 970, threads 438 -- threads are *slower than doing it in
    one place*, because the expensive part of a step turns out to be Python
    rather than the numpy the GIL would have been released for.  The threaded
    rollout is kept because that is worth being able to re-check, but it must
    never be what a caller gets by accident: shipping it as the default cost a
    training run 200 updates at 0.7 times serial speed before the missing
    worker processes were noticed.
    """
    if mode == "thread" and workers and workers > 1:
        return ThreadedRollout(map_path, seed, count, scenarios, encoding,
                               workers)
    if mode == "process" and workers and workers > 1:
        return ParallelRollout(map_path, seed, count, scenarios, encoding,
                               workers)
    return SerialRollout(map_path, seed, count, scenarios, encoding)
