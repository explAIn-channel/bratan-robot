#!/usr/bin/env python3
# ruff: noqa: E402
"""Learn to drive from consequence, and prove it against the teacher.

Behaviour cloning gave us a policy that a one-line baseline beat, because the
target it was fitted to -- the next command, given the last one -- is almost
entirely inertia.  This trains on outcomes instead: reach the goal, stay off
the walls, move smoothly.  The ceiling is the task, so beating the eight-line
geometric teacher is possible rather than definitionally impossible.

PPO, on the CPU, in one file.  The environment costs about a sixth of a
millisecond a step, which is what makes that reasonable: a million steps of
experience is a few minutes rather than a few days.

The exported model keeps the feature/twist contract explicit and digest-bound,
so nothing downstream has to infer what geometry or observation produced it.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import json
import math
import multiprocessing as mp
import os
import platform
import random
import secrets
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# This module is also a supported direct entry point.  Set native-pool limits
# here, before importing NumPy/Torch, so direct runs cannot write artifacts that
# falsely claim the lightweight launcher contract.
NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _native_thread_variable in NATIVE_THREAD_ENV_VARS:
    os.environ[_native_thread_variable] = "1"
del _native_thread_variable

import numpy as np
import psutil
import torch
import yaml
from torch import nn

from features import FEATURE_DIM
from dynamic_world import SCENARIOS
from features import RANGE_ENCODING, RANGE_ENCODINGS
from parallel_envs import make_rollout
from policy_artifacts import (ArtifactContractError, CHECKPOINT_SCHEMA,
                              LEGACY_CHECKPOINT_SCHEMA_V3, METRICS_SCHEMA,
                              PROMOTION_ARRIVAL_FLOORS,
                              PROMOTION_SCENARIOS,
                              CHECKPOINT_SELECTION_RULE, scenario_gate_progress,
                              build_trajectory_identity, contract_digest,
                              current_policy_contract, file_digest,
                              promotion_quality_gate,
                              validate_checkpoint_contract,
                              validate_promotable_artifact,
                              validate_trajectory_identity,
                              write_promotable_manifest, write_run_manifest)
from rl_env import MAX_ANGULAR, MAX_LINEAR, DriveEnv, FastFlat
from symmetrize import SymmetricActor

GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
EPOCHS_PER_UPDATE = 4
MINIBATCH = 256
ENTROPY_COEFFICIENT = 0.003
MAX_GRADIENT_NORM = 0.5
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0

# Evaluation is the only phase able to use six independent cores without
# changing the experience stream. Its workers are deliberately temporary.
# Measurements with the lightweight Windows launcher bound a worker below the
# 512 MiB admission budget; the budget deliberately remains much larger than
# the observed steady state to cover imports and allocator variation.
TRAINER_TORCH_INTEROP_THREADS = 1
EVAL_WORKER_TORCH_THREADS = 1
EVAL_WORKER_TORCH_INTEROP_THREADS = 1
EVAL_COMMIT_RESERVE_BYTES = 4 * 1024 ** 3
EVAL_WORKER_COMMIT_BUDGET_BYTES = 512 * 1024 ** 2
EVAL_MEMORY_WAIT_SECONDS = 300.0
EVAL_MEMORY_POLL_SECONDS = 5.0
EVAL_MEMORY_LOG_SECONDS = 30.0

# Rollout workers do not load the policy. Measurements with the lightweight
# launcher bound them below the 384 MiB admission budget. The same 4 GiB host
# reserve prevents training from consuming the owner's last usable commit.
ROLLOUT_COMMIT_RESERVE_BYTES = 4 * 1024 ** 3
ROLLOUT_WORKER_COMMIT_BUDGET_BYTES = 384 * 1024 ** 2
ROLLOUT_MEMORY_WAIT_SECONDS = 300.0
ROLLOUT_MEMORY_POLL_SECONDS = 5.0

SOURCE_FILES = (
    "training_lock.py",
    "train_rl_launcher.py",
    "train_rl.py",
    "policy_artifacts.py",
    "parallel_envs.py",
    "rl_env.py",
    "dynamic_world.py",
    "features.py",
    "simulate_expert_dataset.py",
    "closed_loop_eval.py",
    "symmetrize.py",
)


def map_provenance(map_path: Path) -> dict:
    """Bind a run to both halves of the occupancy-map bundle."""
    yaml_path = Path(map_path).resolve()
    if not yaml_path.is_file():
        raise ArtifactContractError(f"map YAML does not exist: {yaml_path}")
    try:
        description = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ArtifactContractError(f"cannot read map YAML: {error}") from error
    image_reference = (description.get("image")
                       if isinstance(description, dict) else None)
    if not isinstance(image_reference, str) or not image_reference:
        raise ArtifactContractError("map YAML has no image path")
    image_path = Path(image_reference)
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise ArtifactContractError(f"map image does not exist: {image_path}")
    yaml_sha256 = file_digest(yaml_path)
    image_sha256 = file_digest(image_path)
    identity = {
        "yaml_name": yaml_path.name,
        "yaml_sha256": yaml_sha256,
        "image_reference": image_reference,
        "image_sha256": image_sha256,
    }
    return {
        "yaml_path": str(yaml_path),
        "yaml_name": yaml_path.name,
        "yaml_sha256": yaml_sha256,
        "image_path": str(image_path),
        "image_reference": image_reference,
        "image_sha256": image_sha256,
        "bundle_sha256": contract_digest(identity),
    }


def entrypoint_provenance(path: Path | str | None = None) -> dict[str, str]:
    supplied = Path(__file__) if path is None else Path(path)
    resolved = supplied.resolve()
    if not resolved.is_file():
        raise ArtifactContractError(
            f"trainer entrypoint does not exist: {resolved}")
    return {"path": canonical_local_path(resolved),
            "sha256": file_digest(resolved)}


def canonical_local_path(path: Path | str) -> str:
    rendered = os.path.normcase(os.path.realpath(os.path.abspath(Path(path))))
    return rendered.casefold() if os.name == "nt" else rendered


def source_provenance(trainer_entrypoint: Path | str | None = None) -> dict:
    """Identify the executable training/simulation implementation."""
    here = Path(__file__).resolve().parent
    paths = {name: here / name for name in SOURCE_FILES}
    entrypoint = entrypoint_provenance(trainer_entrypoint)
    paths["actual_trainer_entrypoint"] = Path(entrypoint["path"])
    paths["sim_geometry.py"] = (
        here.parents[1] / "sim_geometry.py")
    paths["sim_map.py"] = (
        here.parents[1] / "sim_map.py")
    files = {}
    identity = {}
    for name, path in paths.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise ArtifactContractError(
                f"training source does not exist: {resolved}")
        digest = file_digest(resolved)
        files[name] = {"path": str(resolved), "sha256": digest}
        identity[name] = digest
    return {"files": files, "bundle_sha256": contract_digest(identity)}


def runtime_provenance() -> dict[str, str]:
    """Stable software/ABI identity that can change numerical trajectories."""
    import onnx
    import onnxruntime
    import scipy
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable_architecture": platform.architecture()[0],
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "pyyaml": yaml.__version__,
        "psutil": psutil.__version__,
        "byteorder": sys.byteorder,
    }


def native_thread_limit_provenance() -> dict:
    """Record the pre-native-import launcher contract and observed values."""
    return {
        "enforced_by": "entrypoint_before_native_imports",
        "required_value": "1",
        "observed": {
            name: os.environ.get(name) for name in NATIVE_THREAD_ENV_VARS
        },
    }


def configure_torch_thread_pools(intra_op_threads: int,
                                 inter_op_threads: int) -> dict[str, int]:
    """Pin both PyTorch CPU pools before any model or worker work begins.

    Environment variables constrain OpenMP/BLAS, but they do not constrain
    PyTorch's inter-op executor.  On the 12-core training host that otherwise
    leaves another 12-thread pool in every spawned evaluator.  Treat an
    already-started, differently-sized pool as a hard provenance failure.
    """
    intra = int(intra_op_threads)
    inter = int(inter_op_threads)
    if intra <= 0 or inter <= 0:
        raise ArtifactContractError(
            "PyTorch intra-op and inter-op thread limits must be positive")
    # Inter-op must be configured first.  ``set_num_threads`` may initialise
    # the CPU executor, after which PyTorch permanently rejects changing the
    # inter-op pool in this process.
    if torch.get_num_interop_threads() != inter:
        try:
            torch.set_num_interop_threads(inter)
        except RuntimeError as error:
            raise ArtifactContractError(
                "PyTorch inter-op pool was initialised before its pinned "
                f"limit could be applied: required={inter}, "
                f"observed={torch.get_num_interop_threads()}") from error
    torch.set_num_threads(intra)
    observed = {
        "intra_op_threads": int(torch.get_num_threads()),
        "inter_op_threads": int(torch.get_num_interop_threads()),
    }
    expected = {
        "intra_op_threads": intra,
        "inter_op_threads": inter,
    }
    if observed != expected:
        raise ArtifactContractError(
            f"PyTorch thread-pool limits did not take effect: "
            f"required={expected}, observed={observed}")
    return observed


def torch_thread_pool_provenance(trainer_intra_op_threads: int) -> dict:
    """Bind the live trainer pools and the evaluator spawn contract."""
    observed = {
        "intra_op_threads": int(torch.get_num_threads()),
        "inter_op_threads": int(torch.get_num_interop_threads()),
    }
    required = {
        "intra_op_threads": int(trainer_intra_op_threads),
        "inter_op_threads": TRAINER_TORCH_INTEROP_THREADS,
    }
    if observed != required:
        raise ArtifactContractError(
            f"trainer PyTorch pools differ from the pinned run: "
            f"required={required}, observed={observed}")
    return {
        "trainer": {"required": required, "observed": observed},
        "evaluation_worker": {
            "required": {
                "intra_op_threads": EVAL_WORKER_TORCH_THREADS,
                "inter_op_threads": EVAL_WORKER_TORCH_INTEROP_THREADS,
            },
            "enforced_by": "spawn_initializer_before_model_construction",
        },
        "rollout_worker": {
            "torch_import_allowed": False,
            "enforced_by": "lightweight_launcher_spawn_contract",
        },
    }


def build_run_config(args, scenarios: list[str]) -> dict:
    entrypoint = entrypoint_provenance(
        getattr(args, "trainer_entrypoint", None))
    return {
        "map": str(args.map.resolve()),
        "seed": args.seed,
        "updates": args.updates,
        "envs": args.envs,
        "horizon": args.horizon,
        "scenarios": list(scenarios),
        "encoding": args.encoding,
        "lr": args.lr,
        "bootstrap_rounds": getattr(args, "bootstrap_rounds", 0),
        "check_every": args.check_every,
        "check_episodes": args.check_episodes,
        "eval_episodes": args.eval_episodes,
        "evaluation_episode_rng": "numpy_seed_sequence_seed_episode_v1",
        "workers": args.workers,
        "rollout_mode": args.rollout_mode,
        "torch_threads": args.torch_threads,
        "torch_interop_threads": TRAINER_TORCH_INTEROP_THREADS,
        "rollout_memory": {
            "worker_strategy": "adaptive_available_commit_v1",
            "budget_basis": "measured_lightweight_windows_launcher_v1",
            "requested_workers": args.workers,
            "commit_reserve_bytes": ROLLOUT_COMMIT_RESERVE_BYTES,
            "worker_commit_budget_bytes":
                ROLLOUT_WORKER_COMMIT_BUDGET_BYTES,
            "minimum_workers": 1,
            "wait_seconds": ROLLOUT_MEMORY_WAIT_SECONDS,
            "start_method": "spawn",
            "commit_probe":
                "GlobalMemoryStatusEx.ullAvailPageFile_on_windows",
            "selection_points": [
                "initial_pool", "post_evaluation_replacement"],
            "effective_workers_recorded_by": "rollout_pool_admitted",
            "evaluation_pool_overlap": False,
        },
        "eval_workers": args.eval_workers,
        "evaluation_memory": {
            "worker_strategy": "adaptive_available_commit_v1",
            "budget_basis": "measured_lightweight_windows_launcher_v1",
            "commit_reserve_bytes": EVAL_COMMIT_RESERVE_BYTES,
            "worker_commit_budget_bytes": EVAL_WORKER_COMMIT_BUDGET_BYTES,
            "minimum_workers": 1,
            "wait_seconds": EVAL_MEMORY_WAIT_SECONDS,
            "worker_torch_threads": EVAL_WORKER_TORCH_THREADS,
            "worker_torch_interop_threads":
                EVAL_WORKER_TORCH_INTEROP_THREADS,
            "start_method": "spawn",
            "rollout_pool_overlap": False,
        },
        "trainer": {
            "gamma": GAMMA,
            "gae_lambda": LAMBDA,
            "ppo_clip": CLIP,
            "epochs_per_update": EPOCHS_PER_UPDATE,
            "minibatch": MINIBATCH,
            "entropy_coefficient": ENTROPY_COEFFICIENT,
            "entropy_regularizer":
                "tanh_squashed_action_entropy_monte_carlo_v1",
            "entropy_samples_per_observation": 1,
            "max_gradient_norm": MAX_GRADIENT_NORM,
            "action_distribution":
                "symmetric_tanh_squashed_diagonal_normal",
            "deterministic_mean":
                "reflection_ensemble_mean_v1_deployment_action",
            "checkpoint_selection":
                CHECKPOINT_SELECTION_RULE,
            "log_std_bounds": [LOG_STD_MIN, LOG_STD_MAX],
            "learning_rate_schedule": "linear_to_5_percent",
        },
        "provenance": {
            "map": map_provenance(args.map),
            "source": source_provenance(entrypoint["path"]),
            "runtime": runtime_provenance(),
            "native_thread_limits": native_thread_limit_provenance(),
            "torch_thread_pools":
                torch_thread_pool_provenance(args.torch_threads),
            "trainer_entrypoint": entrypoint,
        },
    }


def _first_difference(left, right, path="run_config") -> str:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                return child
            difference = _first_difference(left[key], right[key], child)
            if difference:
                return difference
        return ""
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}.length"
        for index, (old, new) in enumerate(zip(left, right)):
            difference = _first_difference(old, new, f"{path}[{index}]")
            if difference:
                return difference
        return ""
    return "" if left == right else path


def selection_score(by_scenario: dict[str, dict]) -> tuple[float, ...]:
    """Lexicographic progress toward the per-scenario promotion gate.

    Absolute arrival rates are not comparable when ``clear`` must reach 75 %
    and ``crowded`` only 15 %.  Cap each scenario's progress at its gate so an
    already-solved easy case cannot hide an unsolved hard one. Collision-free
    progress also caps every scenario, before arrival and collision tiebreaks.
    """
    if set(by_scenario) != set(PROMOTION_SCENARIOS):
        raise ValueError("selection score requires all six promotion scenarios")
    arrivals = [float(by_scenario[name]["arrived"])
                for name in PROMOTION_SCENARIOS]
    gate_progress = [
        scenario_gate_progress(name, arrival,
                               float(by_scenario[name]["collision"]))
        for name, arrival in zip(PROMOTION_SCENARIOS, arrivals)
    ]
    collisions = [float(by_scenario[name]["collision"])
                  for name in PROMOTION_SCENARIOS]
    return (min(gate_progress), float(np.mean(gate_progress)),
            float(np.mean(arrivals)),
            -max(collisions), -float(np.mean(collisions)))


def score_document(score: tuple[float, ...]) -> dict[str, float]:
    return {
        "worst_gate_progress": score[0],
        "mean_gate_progress": score[1],
        "mean_arrival": score[2],
        "negative_max_collision": score[3],
        "negative_mean_collision": score[4],
    }


def should_evaluate(update: int, total_updates: int, check_every: int) -> bool:
    """Select the first, periodic, and unconditionally the final update."""
    return ((update + 1) % check_every == 0 or update == 0 or
            update + 1 == total_updates)


def capture_rng_state(generator: np.random.Generator) -> dict:
    """Capture every random stream the trainer itself can consume."""
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_generator": copy.deepcopy(generator.bit_generator.state),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else []),
    }


def restore_rng_state(state: dict, generator: np.random.Generator) -> None:
    """Restore the streams after model/optimizer construction consumed RNG."""
    required = {"python", "numpy_global", "numpy_generator", "torch_cpu",
                "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise ArtifactContractError(
            f"trainer checkpoint is missing RNG state: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    generator.bit_generator.state = copy.deepcopy(state["numpy_generator"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ArtifactContractError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def trainer_checkpoint(actor, critic, log_std, optimizer, *,
                       completed_update: int, best: dict,
                       best_actor_state: dict,
                       best_metrics: dict | None, history: list[dict],
                       latest_metrics: dict | None,
                        contract: dict, generator: np.random.Generator,
                        run_config: dict,
                        rollout_rng_states: list[dict],
                        trajectory: dict,
                        trajectory_digest: str) -> dict:
    """Build one complete, resumable training checkpoint."""
    return {
        "schema": CHECKPOINT_SCHEMA,
        "completed_update": int(completed_update),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "log_std": log_std.detach().clone(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(generator),
        "rollout_boundary_rng_states": copy.deepcopy(rollout_rng_states),
        "best": dict(best),
        "best_actor": best_actor_state,
        "best_metrics": (None if best_metrics is None else dict(best_metrics)),
        "latest_metrics": (None if latest_metrics is None
                           else dict(latest_metrics)),
        "history": list(history),
        "contract": contract,
        "contract_digest": contract_digest(contract),
        "run_config": dict(run_config),
        "run_config_digest": contract_digest(run_config),
        "trajectory": dict(trajectory),
        "trajectory_digest": trajectory_digest,
    }


def restore_trainer_checkpoint(checkpoint: dict, actor, critic, log_std,
                               optimizer, generator: np.random.Generator,
                               *, expected_contract: dict,
                               run_config: dict,
                               expected_trajectory: dict,
                               expected_trajectory_digest: str) -> dict:
    """Restore a full checkpoint and return its loop/selection state."""
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        if checkpoint.get("schema") == LEGACY_CHECKPOINT_SCHEMA_V3:
            raise ArtifactContractError(
                "legacy v3 trainer checkpoints cannot resume the r11 policy; "
                "start r11 from scratch or use a separately selected best.pt "
                "only for explicit actor warm-start analysis")
        raise ArtifactContractError(
            "initialisation file is not a complete v4 trainer checkpoint")
    validate_checkpoint_contract(checkpoint, expected_contract)
    recorded_config = checkpoint.get("run_config")
    if not isinstance(recorded_config, dict):
        raise ArtifactContractError("trainer checkpoint has no run configuration")
    recorded_config_digest = checkpoint.get("run_config_digest")
    if (not isinstance(recorded_config_digest, str) or
            recorded_config_digest != contract_digest(recorded_config)):
        raise ArtifactContractError(
            "trainer checkpoint run configuration digest is missing or corrupt")
    difference = _first_difference(recorded_config, run_config)
    if difference:
        raise ArtifactContractError(
            f"resume configuration mismatch at {difference}: checkpoint and "
            "current run are not identical")
    validate_trajectory_identity(
        checkpoint.get("trajectory"), checkpoint.get("trajectory_digest"),
        "trainer checkpoint", expected=expected_trajectory,
        expected_digest=expected_trajectory_digest,
        required_origin=expected_trajectory.get("origin"),
        expected_contract_digest=contract_digest(expected_contract),
        expected_run_config_digest=contract_digest(run_config))
    rollout_rng_states = checkpoint.get("rollout_boundary_rng_states")
    if (not isinstance(rollout_rng_states, list) or
            len(rollout_rng_states) != int(run_config["envs"]) or
            not all(isinstance(state, dict) for state in rollout_rng_states)):
        raise ArtifactContractError(
            "trainer checkpoint has no complete rollout-boundary RNG state")
    completed_update = checkpoint.get("completed_update")
    if (isinstance(completed_update, bool) or
            not isinstance(completed_update, int) or completed_update < 0):
        raise ArtifactContractError(
            "trainer checkpoint has an invalid completed update")
    saved_log_std = checkpoint.get("log_std")
    if (not isinstance(saved_log_std, torch.Tensor) or
            saved_log_std.shape != log_std.shape or
            not bool(torch.isfinite(saved_log_std).all()) or
            not bool((saved_log_std >= LOG_STD_MIN).all()) or
            not bool((saved_log_std <= LOG_STD_MAX).all())):
        raise ArtifactContractError(
            "trainer checkpoint has invalid bounded log_std state")
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    with torch.no_grad():
        log_std.copy_(saved_log_std)
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng_state(checkpoint["rng"], generator)
    return {
        "completed_update": completed_update,
        "best": dict(checkpoint["best"]),
        "best_actor": checkpoint["best_actor"],
        "best_metrics": checkpoint.get("best_metrics"),
        "latest_metrics": checkpoint.get("latest_metrics"),
        "history": list(checkpoint.get("history", [])),
        "rollout_rng_states": copy.deepcopy(rollout_rng_states),
    }


def save_torch_atomic(value, path: Path) -> None:
    """Never leave a half-written checkpoint with a trustworthy filename."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_json_atomic(value: dict, path: Path) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def restore_best_artifacts_from_resume(checkpoint: dict, output: Path) -> None:
    """Repair the atomic-save crash window without changing the trajectory."""
    best = checkpoint.get("best")
    completed = checkpoint.get("completed_update")
    if not isinstance(best, dict) or checkpoint.get("best_actor") is None:
        raise ArtifactContractError("resume checkpoint has no selected best actor")
    save_torch_atomic(checkpoint["best_actor"], Path(output) / "best.pt")
    if completed == best.get("update"):
        # trainer.pt is written before best_trainer.pt. If power was lost in
        # between, this exact payload is the best checkpoint and is the only
        # byte-for-byte honest reconstruction, including at update 4000.
        save_torch_atomic(checkpoint, Path(output) / "best_trainer.pt")


def cloned_state_dict(module: nn.Module) -> dict:
    """A high-water mark must not share storage with the actor that moves on."""
    return {name: tensor.detach().cpu().clone()
            for name, tensor in module.state_dict().items()}


def normalize_advantages(values: torch.Tensor) -> torch.Tensor:
    """Normalise even a one-sample smoke rollout without producing NaNs."""
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-8)


class Actor(nn.Module):
    """The deployable half: features in, a twist in real units out.

    Same shape as the cloned policy it replaces, so the ONNX file drops into
    the existing runtime unchanged.  The squashing and the unit scaling live
    inside the module rather than in the caller, because a scale factor kept
    on the outside is a scale factor that eventually gets applied twice.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("feature_mean",
                             torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("feature_std",
                             torch.from_numpy(std.astype(np.float32)))
        self.trunk = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 2),
        )

    def normalized(self, features):
        return (features - self.feature_mean) / self.feature_std

    def mean_action(self, features):
        """The action in [-1, 1], before it is given units."""
        return torch.tanh(self.raw_action(features))

    def raw_action(self, features):
        """The unsquashed Normal mean used by PPO exploration."""
        return self.trunk(self.normalized(features))

    def forward(self, features):
        action = self.mean_action(features)
        linear = (action[..., :1] + 1.0) * (0.5 * MAX_LINEAR)
        angular = action[..., 1:] * MAX_ANGULAR
        return torch.cat([linear, angular], dim=-1)


class ActionMean(nn.Module):
    """Expose an actor's bounded action before chassis-unit scaling.

    ``SymmetricActor(Actor)`` is the deployed twist policy.  Because chassis
    scaling is affine, reflecting and averaging this bounded action is exactly
    equivalent to reflecting and averaging the deployed twists.  PPO can then
    centre its squashed Normal on the same deterministic command without
    duplicating the mirror implementation.
    """

    def __init__(self, actor: Actor):
        super().__init__()
        self.actor = actor

    def forward(self, features):
        return self.actor.mean_action(features)


def symmetric_latent_mean(policy: nn.Module,
                          features: torch.Tensor) -> torch.Tensor:
    """Normal-space mean whose squashed value is the deployed action mean."""
    bounded = policy(features)
    epsilon = 4.0 * torch.finfo(bounded.dtype).eps
    return torch.atanh(bounded.clamp(-1.0 + epsilon, 1.0 - epsilon))


def tanh_log_abs_det_jacobian(latent: torch.Tensor) -> torch.Tensor:
    """Stable elementwise ``log(1 - tanh(latent) ** 2)``.

    Computing this from the already-squashed action loses all precision near
    either boundary.  This softplus identity remains finite for large positive
    and negative latents and is the Jacobian needed by the executed policy's
    density and differential entropy.
    """
    return 2.0 * (
        math.log(2.0) - latent - nn.functional.softplus(-2.0 * latent))


def squashed_normal_entropy(
        distribution: torch.distributions.Normal) -> torch.Tensor:
    """One-sample pathwise estimate of entropy after the tanh boundary.

    For ``action = tanh(latent)``, ``H(action)`` is ``H(latent)`` plus the
    expected log absolute Jacobian.  The chassis-unit affine transform adds
    only a parameter-independent constant, so normalized action entropy has
    exactly the gradient the PPO regularizer needs.
    """
    latent = distribution.rsample()
    transformed = (distribution.entropy() +
                   tanh_log_abs_det_jacobian(latent))
    return transformed.sum(dim=-1).mean()


class Critic(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("feature_mean",
                             torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("feature_std",
                             torch.from_numpy(std.astype(np.float32)))
        self.trunk = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.trunk((features - self.feature_mean)
                          / self.feature_std).squeeze(-1)


def to_twist(action: np.ndarray) -> np.ndarray:
    """Map an action in [-1, 1] onto the twist the chassis is asked for."""
    return np.array([(action[0] + 1.0) * 0.5 * MAX_LINEAR,
                     action[1] * MAX_ANGULAR], dtype=np.float32)


def expert_action(env: DriveEnv) -> np.ndarray:
    """Track a near waypoint without driving past a slow chassis turn."""
    point = env.local[min(3, len(env.local) - 1)]
    bearing = math.atan2(point[1], point[0])
    reach = float(math.hypot(point[0], point[1]))
    angular = float(np.clip(bearing - .5 * env.angular, -MAX_ANGULAR, MAX_ANGULAR))
    linear = min(.15, reach * .7) * max(0.0, 1.0 - abs(bearing) / .35)
    return np.array([linear, angular], dtype=np.float32)


def bootstrap_actor(actor, flat, rounds: int, seed: int, encoding: str, *,
                    samples_per_round: int = 24000, fit_updates: int = 1000):
    """Distill basic steering, then label the policy's own visited states.

    This teacher follows the shared scan-conditioned route. Dynamic avoidance
    must still be learned and pass the independent deployment evaluation.
    """
    rng = np.random.default_rng(seed)
    env = DriveEnv(flat, rng, scenario="clear", encoding=encoding)
    observation = env.reset()
    policy = SymmetricActor(actor)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    features, targets, history = [], [], []
    scale = torch.tensor([MAX_LINEAR, MAX_ANGULAR])
    for round_id in range(rounds):
        for _ in range(samples_per_round):
            target = expert_action(env)
            features.append(observation.copy()); targets.append(target)
            action = target
            if round_id and rng.random() < .6:
                with torch.no_grad():
                    action = policy(torch.from_numpy(observation[None]))[0].numpy()
            observation, _, done, _ = env.step(action)
            if done:
                observation = env.reset()
        x = torch.from_numpy(np.stack(features))
        y = torch.from_numpy(np.stack(targets))
        if not round_id:
            actor.feature_mean.copy_(x.mean(dim=0))
            actor.feature_std.copy_(x.std(dim=0, unbiased=False).clamp(min=.02))
        for _ in range(fit_updates):
            indices = torch.randint(len(x), (512,))
            loss = (((policy(x[indices]) - y[indices]) / scale) ** 2).mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRADIENT_NORM)
            optimizer.step()
        row = {"bootstrap_round": round_id + 1, "samples": len(x),
               "imitation_loss": float(loss.detach()), "motion_authority": False}
        history.append(row); print(json.dumps(row), flush=True)
    return history


def evaluate(flat, actor, episodes: int, seed: int, *, expert: bool = False,
             scenario: str = "clear", encoding: str | None = None):
    """Run whole drives and count what happened, on identical start/goal pairs.

    The encoding changes only how the drive is *observed*: the seed still
    decides the start, the goal and where everybody is standing, so two
    policies fitted to different encodings are still scored on the same
    episodes.
    """
    outcomes = {"arrived": 0, "collision": 0, "timeout": 0}
    steps, jerk = [], []
    initial_rejections = []
    for episode in range(episodes):
        # Sensor/world draws from a longer drive must not change the next
        # start/goal/occupants when comparing candidates or the expert.
        env = DriveEnv(flat, np.random.default_rng([seed, episode]),
                       scenario=scenario, encoding=encoding)
        observation = env.reset()
        initial_rejections.extend(dict(episode=episode, **certificate)
            for certificate in getattr(env, 'initial_world_rejections', ()))
        previous = np.zeros(2, dtype=np.float32)
        total_jerk, count = 0.0, 0
        while True:
            if expert:
                twist = expert_action(env)
            else:
                with torch.inference_mode():
                    twist = actor(torch.from_numpy(observation[None, :])
                                  )[0].numpy()
            total_jerk += float(abs(twist[0] - previous[0])
                                + 0.3 * abs(twist[1] - previous[1]))
            previous, count = twist, count + 1
            observation, _reward, done, outcome = env.step(twist)
            if done:
                outcomes[outcome] += 1
                steps.append(env.steps)
                jerk.append(total_jerk / max(1, count))
                break
    return {"episodes": episodes,
            "initial_world_rejections": initial_rejections,
            "arrived": outcomes["arrived"] / episodes,
            "collision": outcomes["collision"] / episodes,
            "timeout": outcomes["timeout"] / episodes,
            "mean_steps": float(np.mean(steps)),
            "mean_jerk": float(np.mean(jerk))}


def available_commit_bytes() -> int:
    """Return memory the host can still commit, not merely unused RAM."""
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPageFile)
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return int(memory.available + swap.free)


def _wait_for_worker_memory(
        requested_workers: int, *, event_prefix: str,
        probe=None, clock=None, sleeper=None, emit=None,
        reserve_bytes: int, worker_budget_bytes: int,
        timeout_seconds: float, poll_seconds: float) -> dict:
    """Wait a bounded time for enough commit to spawn a worker pool safely."""
    if requested_workers <= 0:
        raise ValueError("worker count must be positive")
    probe = available_commit_bytes if probe is None else probe
    clock = time.monotonic if clock is None else clock
    sleeper = time.sleep if sleeper is None else sleeper
    if emit is None:
        def default_emit(document):
            print(json.dumps(document), flush=True)
        emitter = default_emit
    else:
        emitter = emit
    minimum_required = int(reserve_bytes + worker_budget_bytes)
    requested_required = int(
        reserve_bytes + requested_workers * worker_budget_bytes)
    started = clock()
    deadline = started + timeout_seconds
    last_log = float("-inf")
    while True:
        available = int(probe())
        affordable = max(0, (available - reserve_bytes) // worker_budget_bytes)
        effective_workers = min(int(requested_workers), int(affordable))
        now = clock()
        waited = max(0.0, now - started)
        details = {
            "available_commit_bytes": available,
            "minimum_required_commit_bytes": minimum_required,
            "requested_required_commit_bytes": requested_required,
            "reserve_bytes": int(reserve_bytes),
            "worker_budget_bytes": int(worker_budget_bytes),
            "requested_workers": int(requested_workers),
            "effective_workers": effective_workers,
            "waited_seconds": round(waited, 3),
        }
        if effective_workers >= 1:
            document = {"event": f"{event_prefix}_memory_ready", **details}
            emitter(document)
            return document
        if now >= deadline:
            emitter({"event": f"{event_prefix}_memory_timeout", **details})
            raise MemoryError(
                f"insufficient commit for {event_prefix} workers after "
                f"{waited:.1f}s: available={available}, "
                f"minimum_required={minimum_required}")
        if now - last_log >= EVAL_MEMORY_LOG_SECONDS:
            emitter({"event": f"{event_prefix}_memory_wait", **details})
            last_log = now
        sleeper(min(poll_seconds, max(0.0, deadline - now)))


def wait_for_evaluation_memory(
        requested_workers: int, *, probe=None, clock=None, sleeper=None,
        emit=None,
        reserve_bytes: int = EVAL_COMMIT_RESERVE_BYTES,
        worker_budget_bytes: int = EVAL_WORKER_COMMIT_BUDGET_BYTES,
        timeout_seconds: float = EVAL_MEMORY_WAIT_SECONDS,
        poll_seconds: float = EVAL_MEMORY_POLL_SECONDS) -> dict:
    """Wait a bounded time for enough commit to spawn evaluation safely."""
    return _wait_for_worker_memory(
        requested_workers, event_prefix="evaluation", probe=probe,
        clock=clock, sleeper=sleeper, emit=emit, reserve_bytes=reserve_bytes,
        worker_budget_bytes=worker_budget_bytes,
        timeout_seconds=timeout_seconds, poll_seconds=poll_seconds)


def wait_for_rollout_memory(
        requested_workers: int, *, probe=None, clock=None, sleeper=None,
        emit=None,
        reserve_bytes: int = ROLLOUT_COMMIT_RESERVE_BYTES,
        worker_budget_bytes: int = ROLLOUT_WORKER_COMMIT_BUDGET_BYTES,
        timeout_seconds: float = ROLLOUT_MEMORY_WAIT_SECONDS,
        poll_seconds: float = ROLLOUT_MEMORY_POLL_SECONDS) -> dict:
    """Admit only the rollout workers current Windows commit can sustain."""
    return _wait_for_worker_memory(
        requested_workers, event_prefix="rollout", probe=probe,
        clock=clock, sleeper=sleeper, emit=emit, reserve_bytes=reserve_bytes,
        worker_budget_bytes=worker_budget_bytes,
        timeout_seconds=timeout_seconds, poll_seconds=poll_seconds)


def admitted_rollout_workers(requested_workers: int, mode: str, *,
                             selection_point: str = "unspecified",
                             memory_guard=wait_for_rollout_memory,
                             emit=None) -> int:
    """Resolve a requested cap immediately before constructing its pool."""
    requested = int(requested_workers)
    if requested <= 0:
        raise ValueError("rollout worker count must be positive")
    if mode != "process":
        return requested
    memory = memory_guard(requested)
    effective = int(memory.get("effective_workers", 0))
    if effective < 1 or effective > requested:
        raise MemoryError(
            "rollout memory guard returned an invalid effective worker count: "
            f"requested={requested}, effective={effective}")
    document = {
        "event": "rollout_pool_admitted",
        "selection_point": selection_point,
        "requested_workers": requested,
        "effective_workers": effective,
        "available_commit_bytes": memory.get("available_commit_bytes"),
        "reserve_bytes": memory.get("reserve_bytes"),
        "worker_budget_bytes": memory.get("worker_budget_bytes"),
    }
    if emit is None:
        print(json.dumps(document), flush=True)
    else:
        emit(document)
    return effective


_EVALUATION_FLAT = None
_EVALUATION_POLICY = None


def _initialise_evaluation_worker(map_path: str,
                                  actor_state: dict) -> None:
    """Build one read-only map and policy in each spawned evaluator."""
    global _EVALUATION_FLAT, _EVALUATION_POLICY
    configure_torch_thread_pools(
        EVAL_WORKER_TORCH_THREADS, EVAL_WORKER_TORCH_INTEROP_THREADS)
    _EVALUATION_FLAT = FastFlat(Path(map_path))
    actor = Actor(actor_state["feature_mean"].numpy(),
                  actor_state["feature_std"].numpy())
    actor.load_state_dict(actor_state)
    actor.eval()
    _EVALUATION_POLICY = SymmetricActor(actor).eval()


def _evaluation_worker(job: tuple) -> tuple[str, str, dict]:
    label, expert, episodes, seed, scenario, encoding = job
    if _EVALUATION_FLAT is None or _EVALUATION_POLICY is None:
        raise RuntimeError("evaluation worker was not initialised")
    result = evaluate(
        _EVALUATION_FLAT, _EVALUATION_POLICY, episodes, seed,
        expert=expert, scenario=scenario, encoding=encoding)
    return label, scenario, result


def parallel_evaluate_scenarios(
        map_path: Path, actor_state: dict, episodes: int, seed: int,
        scenarios: tuple[str, ...] | list[str], encoding: str | None,
        eval_workers: int, *,
        variants: tuple[tuple[str, bool], ...] = (("learned", False),),
        executor_factory=None) -> dict[str, dict[str, dict]]:
    """Evaluate independent scenarios in a temporary, fail-closed pool."""
    scenario_order = tuple(scenarios)
    if not scenario_order or not variants:
        raise ValueError("parallel evaluation needs scenarios and variants")
    worker_count = max(1, min(int(eval_workers), len(scenario_order)))
    jobs = [
        (label, expert, episodes, seed, scenario, encoding)
        for label, expert in variants
        for scenario in scenario_order
    ]
    factory = ProcessPoolExecutor if executor_factory is None else executor_factory
    pool = factory(
        max_workers=worker_count,
        mp_context=mp.get_context("spawn"),
        initializer=_initialise_evaluation_worker,
        initargs=(str(Path(map_path).resolve()), actor_state),
    )
    futures = []
    failed = True
    try:
        futures = [pool.submit(_evaluation_worker, job) for job in jobs]
        returned = [future.result() for future in futures]
        results = {(label, scenario): result
                   for label, scenario, result in returned}
        expected = {(label, scenario)
                    for label, _expert in variants
                    for scenario in scenario_order}
        if set(results) != expected:
            raise RuntimeError("evaluation pool returned an incomplete result set")
        failed = False
    finally:
        if failed:
            for future in futures:
                future.cancel()
        pool.shutdown(wait=True, cancel_futures=failed)
    return {
        label: {scenario: results[(label, scenario)]
                for scenario in scenario_order}
        for label, _expert in variants
    }


def parallel_evaluate_preserving_rng(
        generator: np.random.Generator, *args, **kwargs
        ) -> dict[str, dict[str, dict]]:
    """A scoring pass must consume none of the trainer's random streams."""
    state = capture_rng_state(generator)
    try:
        return parallel_evaluate_scenarios(*args, **kwargs)
    finally:
        restore_rng_state(state, generator)


def _checkpoint_parallel_evaluation_impl(
        rollout, *, map_path: Path, rollout_seed: int, rollout_count: int,
        rollout_scenarios: list[str], encoding: str | None,
        rollout_workers: int, rollout_mode: str, actor_state: dict,
        episodes: int, eval_workers: int, generator: np.random.Generator,
        scenarios: tuple[str, ...] = PROMOTION_SCENARIOS,
        variants: tuple[tuple[str, bool], ...] = (("learned", False),),
        memory_guard=wait_for_evaluation_memory,
        rollout_memory_guard=wait_for_rollout_memory,
        rollout_admission_emit=None,
        evaluator=parallel_evaluate_preserving_rng,
        rollout_factory=make_rollout):
    """Swap rollout and evaluation pools without ever overlapping them."""
    boundary_observations, rollout_rng_states = rollout.checkpoint_boundary()
    rollout.close()
    gc.collect()

    evaluation_result = None
    evaluation_error = None
    evaluation_traceback = None
    worker_count = max(1, min(int(eval_workers), len(scenarios)))
    try:
        memory = memory_guard(worker_count)
        effective_workers = int(memory["effective_workers"])
        evaluation_result = evaluator(
            generator, map_path, actor_state, episodes, 4242, scenarios,
            encoding, effective_workers, variants=variants)
    except BaseException as error:  # preserve the original worker failure
        evaluation_error = error
        evaluation_traceback = error.__traceback__

    if evaluation_error is not None:
        raise evaluation_error.with_traceback(evaluation_traceback)

    replacement_workers = admitted_rollout_workers(
        rollout_workers, rollout_mode,
        selection_point="post_evaluation_replacement",
        memory_guard=rollout_memory_guard, emit=rollout_admission_emit)
    replacement = rollout_factory(
        map_path, rollout_seed, rollout_count, rollout_scenarios,
        encoding=encoding, workers=replacement_workers, mode=rollout_mode)
    try:
        restored = replacement.restore_checkpoint_boundary(rollout_rng_states)
        if not np.array_equal(restored, boundary_observations):
            raise RuntimeError(
                "rollout checkpoint boundary changed across evaluation")
    except BaseException:
        replacement.close()
        raise
    return (replacement, restored, rollout_rng_states, evaluation_result)


def checkpoint_parallel_evaluation(
        rollout, *, generator: np.random.Generator, **kwargs):
    """Preserve trainer RNG across pool teardown, scoring and reconstruction."""
    state = capture_rng_state(generator)
    try:
        return _checkpoint_parallel_evaluation_impl(
            rollout, generator=generator, **kwargs)
    finally:
        restore_rng_state(state, generator)


def _main_with_output_lease(
        coordinator_context: dict | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=2,
                        help="the policy is 228-128-128-64-2 and gains nothing "
                             "from more; left alone torch opens a pool the "
                             "size of the machine and fights the workers for it")
    parser.add_argument("--rollout-mode", choices=("process", "thread"),
                        default="process",
                        help="threads were measured slower than serial; the "
                             "choice exists to re-check that, not to use")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes to deal the environments across; "
                             "1 steps them all in this one, which is "
                             "what every run before 2026-08-31 did")
    parser.add_argument("--eval-workers", type=int, default=6,
                        help="temporary spawned workers used only while the "
                             "rollout pool is closed")
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--check-every", type=int, default=50,
                        help="updates between the checks that pick best.pt")
    parser.add_argument("--check-episodes", type=int, default=120,
                        help="episodes per scenario in those checks; 40 -- "
                             "what every run before 2026-08-31 used -- has a "
                             "standard error near 0.079 on an arrival rate "
                             "around a half, so best.pt was largely picking "
                             "the luckiest check rather than the best policy")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bootstrap-rounds", type=int, default=0,
                        help="initial static steering imitation rounds before PPO")
    parser.add_argument("--encoding", choices=RANGE_ENCODINGS,
                        default=RANGE_ENCODING)
    parser.add_argument("--initialise-from", type=Path,
                        help="resume from trainer.pt, or warm-start from a "
                             "manifest-selected best.pt with the current "
                             "policy contract")
    parser.add_argument("--scenarios", default=",".join(PROMOTION_SCENARIOS),
                        help="comma-separated; the environments are dealt "
                             "these in turn, and each is scored separately")
    args = parser.parse_args()
    if not 0 <= args.bootstrap_rounds <= 12:
        parser.error("--bootstrap-rounds must be between 0 and 12")
    args.trainer_entrypoint = Path(
        coordinator_context["trainer_entrypoint"]["path"]
        if coordinator_context is not None else __file__).resolve()
    scenarios = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    unknown = sorted(set(scenarios) - set(SCENARIOS))
    if unknown:
        parser.error(f"unknown scenario(s): {', '.join(unknown)}")
    for name in ("updates", "envs", "horizon", "eval_episodes", "check_every",
                 "check_episodes", "workers", "eval_workers"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    configure_torch_thread_pools(
        args.torch_threads, TRAINER_TORCH_INTEROP_THREADS)
    flat = FastFlat(args.map)
    rng = np.random.default_rng(args.seed)
    contract = current_policy_contract(args.encoding)
    run_config = build_run_config(args, scenarios)
    run_config_digest = contract_digest(run_config)
    output_identity = canonical_local_path(args.output)
    trajectory = None
    trajectory_digest = None
    if coordinator_context is not None:
        actual_entrypoint = entrypoint_provenance(args.trainer_entrypoint)
        if coordinator_context.get("trainer_entrypoint") != actual_entrypoint:
            raise ArtifactContractError(
                "coordinator lease trainer entrypoint changed before training")
        trajectory, trajectory_digest = validate_trajectory_identity(
            coordinator_context.get("trajectory"),
            coordinator_context.get("trajectory_digest"),
            "coordinator lease", required_origin="fresh",
            expected_contract_digest=contract_digest(contract),
            expected_run_config_digest=run_config_digest,
            expected_output=output_identity)
    resume = None
    if args.initialise_from is not None:
        # The statistics belong to the weights.  Recomputing them here would
        # hand the loaded network a differently-scaled input than the one it
        # was fitted against, which does not fail -- it just quietly discards
        # the run being continued.
        try:
            loaded = torch.load(args.initialise_from, map_location="cpu",
                                weights_only=False)
            if (isinstance(loaded, dict) and loaded.get("schema") ==
                    LEGACY_CHECKPOINT_SCHEMA_V3):
                raise ArtifactContractError(
                    "legacy v3 trainer checkpoints cannot resume r11; "
                    "start from scratch (a legacy best.pt may be inspected "
                    "only through its selected v1 evidence)")
            if (isinstance(loaded, dict) and
                    loaded.get("schema") == CHECKPOINT_SCHEMA):
                validate_checkpoint_contract(loaded, contract)
                resume, state = loaded, loaded["actor"]
                if trajectory is None:
                    trajectory, trajectory_digest = (
                        validate_trajectory_identity(
                            loaded.get("trajectory"),
                            loaded.get("trajectory_digest"),
                            "direct resume checkpoint",
                            expected_contract_digest=contract_digest(contract),
                            expected_run_config_digest=run_config_digest,
                            expected_output=output_identity))
            else:
                # Actor-only warm starts are allowed only from the artifact the
                # completed run explicitly selected.  checkpoint.pt and a
                # copied state dict are latest/unknown weights, not evidence.
                if coordinator_context is not None:
                    raise ArtifactContractError(
                        "fresh coordinator permits only its own full checkpoint "
                        "resume, never an actor warm-start")
                parent_manifest = validate_promotable_artifact(
                    args.initialise_from, contract)
                manifest_path = args.initialise_from.parent / "selected-model.json"
                trajectory, trajectory_digest = build_trajectory_identity(
                    trajectory_id=secrets.token_hex(32),
                    origin="actor_warm_start",
                    policy_contract_digest=contract_digest(contract),
                    run_config_digest=run_config_digest,
                    output=output_identity,
                    parent={
                        "selected_artifact_sha256":
                            file_digest(args.initialise_from),
                        "selected_manifest_sha256": file_digest(manifest_path),
                        "trajectory_digest": parent_manifest["trajectory_digest"],
                    })
                state = loaded
            mean = state["feature_mean"].numpy()
            std = state["feature_std"].numpy()
        except Exception as error:  # malformed local artifact: quarantine it
            details = {}
            if trajectory is not None and trajectory_digest is not None:
                details.update(trajectory=trajectory,
                               trajectory_digest=trajectory_digest)
            write_run_manifest(
                args.output, "quarantined", contract,
                reason=str(error),
                source_checkpoint=str(args.initialise_from),
                run_config=run_config,
                run_config_digest=run_config_digest, **details)
            parser.error(str(error))
    else:
        if trajectory is None:
            trajectory, trajectory_digest = build_trajectory_identity(
                trajectory_id=secrets.token_hex(32), origin="fresh",
                policy_contract_digest=contract_digest(contract),
                run_config_digest=run_config_digest, output=output_identity)
        # Feature statistics from a random walk, so normalisation does not
        # depend on a policy that does not exist yet.
        sample = []
        probe = DriveEnv(flat, np.random.default_rng(args.seed + 999),
                         scenario=scenarios[0], encoding=args.encoding)
        observation = probe.reset()
        for _ in range(4000):
            sample.append(observation)
            observation, _r, done, _o = probe.step(
                to_twist(rng.uniform(-1, 1, size=2)))
            if done:
                observation = probe.reset()
        sample = np.stack(sample)
        mean = sample.mean(axis=0)
        std = np.maximum(sample.std(axis=0), 1e-3)

    # Validate a selected warm-start before replacing a manifest when input and
    # output deliberately name the same run directory.
    write_run_manifest(
        args.output, "training", contract,
        source_checkpoint=(str(args.initialise_from)
                           if args.initialise_from else None),
        run_config=run_config,
        run_config_digest=run_config_digest,
        trajectory=trajectory,
        trajectory_digest=trajectory_digest)

    actor = Actor(mean, std)
    if args.bootstrap_rounds and args.initialise_from is None:
        bootstrap = bootstrap_actor(actor, flat, args.bootstrap_rounds,
                                    args.seed + 10000, args.encoding)
        (args.output / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2))
        mean = actor.feature_mean.numpy(); std = actor.feature_std.numpy()
    action_policy = SymmetricActor(ActionMean(actor))
    critic = Critic(mean, std)
    log_std = nn.Parameter(torch.full((2,), -1.0))
    if args.initialise_from is not None and resume is None:
        actor.load_state_dict(state)
        print(json.dumps({
            "continued_from": str(args.initialise_from),
            "mode": "contract_verified_warm_start",
            "critic_restored": False}), flush=True)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()) + [log_std],
        lr=args.lr)

    best = {"score": None, "scored_on": [], "update": 0}
    best_actor_state = None
    best_metrics = None
    latest_metrics = None
    history = []
    start_update = 0
    rollout_rng_states = None
    if resume is not None:
        try:
            restored = restore_trainer_checkpoint(
                resume, actor, critic, log_std, optimizer, rng,
                expected_contract=contract, run_config=run_config,
                expected_trajectory=trajectory,
                expected_trajectory_digest=trajectory_digest)
        except Exception as error:  # incomplete/corrupt trainer checkpoint
            write_run_manifest(
                args.output, "quarantined", contract,
                reason=str(error), source_checkpoint=str(args.initialise_from),
                run_config=run_config,
                run_config_digest=run_config_digest,
                trajectory=trajectory,
                trajectory_digest=trajectory_digest)
            parser.error(str(error))
        start_update = restored["completed_update"]
        best = restored["best"]
        best_actor_state = restored["best_actor"]
        best_metrics = restored["best_metrics"]
        latest_metrics = restored["latest_metrics"]
        history = restored["history"]
        rollout_rng_states = restored["rollout_rng_states"]
        if start_update > args.updates:
            parser.error(
                f"checkpoint completed update {start_update}, beyond "
                f"--updates {args.updates}")
        restore_best_artifacts_from_resume(resume, args.output)
        print(json.dumps({
            "continued_from": str(args.initialise_from),
            "mode": "full_resume",
            "completed_update": start_update,
            "optimizer_restored": True,
            "rng_restored": True,
            "best_restored": best}), flush=True)

    # Dealt round-robin rather than sampled per episode, so every update
    # carries experience of every scenario.  A policy that met a walking
    # person only in the updates where the dice said so would spend the rest
    # of training quietly forgetting.  Environment state is deliberately
    # restarted at an update boundary; model, optimizer and all trainer RNG
    # streams resume exactly, while no half-rollout is ever checkpointed.
    initial_workers = admitted_rollout_workers(
        args.workers, args.rollout_mode, selection_point="initial_pool")
    rollout = make_rollout(args.map, args.seed, args.envs, scenarios,
                           encoding=args.encoding, workers=initial_workers,
                           mode=args.rollout_mode)
    if rollout_rng_states is None:
        observations = rollout.reset_all()
    else:
        observations = rollout.restore_checkpoint_boundary(rollout_rng_states)

    def anneal(update: int) -> float:
        """Linear decay to a five-percent floor, the usual PPO shape.

        Held constant, the run does not converge -- it circles.  On the empty
        flat it reached 67 % arrivals with 25 % collisions at update 75 and was
        back to 52 % and 42 % by 125, taking full-sized steps in a region where
        the remaining gains are small.  Decaying the step size is what turns
        the last third of a run into refinement instead of a random walk.
        """
        remaining = 1.0 - update / max(1, args.updates)
        rate = args.lr * max(remaining, 0.05)
        for group in optimizer.param_groups:
            group["lr"] = rate
        return rate

    started = time.perf_counter()
    for update in range(start_update, args.updates):
        anneal(update)
        obs_buffer = np.zeros((args.horizon, args.envs, FEATURE_DIM), np.float32)
        latent_buffer = np.zeros((args.horizon, args.envs, 2), np.float32)
        logp_buffer = np.zeros((args.horizon, args.envs), np.float32)
        rew_buffer = np.zeros((args.horizon, args.envs), np.float32)
        done_buffer = np.zeros((args.horizon, args.envs), np.float32)
        val_buffer = np.zeros((args.horizon + 1, args.envs), np.float32)

        for t in range(args.horizon):
            tensor = torch.from_numpy(observations)
            # Collecting experience must not build a graph: the standard
            # deviation is a learned parameter, so a distribution assembled
            # outside this guard would carry gradients into the rollout.
            with torch.no_grad():
                raw_mean = symmetric_latent_mean(action_policy, tensor)
                value = critic(tensor)
                distribution = torch.distributions.Normal(raw_mean,
                                                           log_std.exp())
                latent = distribution.sample()
                action = torch.tanh(latent)
                # The tanh Jacobian cancels from the PPO likelihood ratio for
                # the same sampled latent.  Retaining the latent therefore
                # gives the exact ratio without an unstable inverse tanh, while
                # the environment receives the bounded action it was scored on.
                logp = distribution.log_prob(latent).sum(-1)

            obs_buffer[t] = observations
            latent_buffer[t] = latent.numpy()
            logp_buffer[t] = logp.numpy()
            val_buffer[t] = value.numpy()

            nxt, rewards, dones = rollout.step_all(
                [to_twist(action[i].numpy()) for i in range(args.envs)])
            rew_buffer[t] = rewards
            done_buffer[t] = dones
            observations = nxt

        with torch.inference_mode():
            val_buffer[args.horizon] = critic(
                torch.from_numpy(observations)).numpy()

        advantages = np.zeros_like(rew_buffer)
        running = np.zeros(args.envs, np.float32)
        for t in reversed(range(args.horizon)):
            mask = 1.0 - done_buffer[t]
            delta = (rew_buffer[t] + GAMMA * val_buffer[t + 1] * mask
                     - val_buffer[t])
            running = delta + GAMMA * LAMBDA * mask * running
            advantages[t] = running
        returns = advantages + val_buffer[:args.horizon]

        flat_obs = torch.from_numpy(obs_buffer.reshape(-1, FEATURE_DIM))
        flat_latent = torch.from_numpy(latent_buffer.reshape(-1, 2))
        flat_logp = torch.from_numpy(logp_buffer.reshape(-1))
        flat_ret = torch.from_numpy(returns.reshape(-1))
        flat_adv = torch.from_numpy(advantages.reshape(-1))
        flat_adv = normalize_advantages(flat_adv)

        count = flat_obs.shape[0]
        for _ in range(EPOCHS_PER_UPDATE):
            order = torch.randperm(count)
            for start in range(0, count, MINIBATCH):
                index = order[start:start + MINIBATCH]
                distribution = torch.distributions.Normal(
                    symmetric_latent_mean(action_policy, flat_obs[index]),
                    log_std.exp())
                logp = distribution.log_prob(flat_latent[index]).sum(-1)
                ratio = (logp - flat_logp[index]).exp()
                surrogate = torch.min(
                    ratio * flat_adv[index],
                    torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * flat_adv[index])
                value_loss = nn.functional.mse_loss(
                    critic(flat_obs[index]), flat_ret[index])
                entropy = squashed_normal_entropy(distribution)
                loss = (-surrogate.mean() + 0.5 * value_loss -
                        ENTROPY_COEFFICIENT * entropy)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()) +
                    [log_std],
                    MAX_GRADIENT_NORM)
                optimizer.step()
                with torch.no_grad():
                    log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)

        if should_evaluate(update, args.updates, args.check_every):
            # A candidate is judged on the complete deployment suite every
            # time.  Otherwise an improvement on the scenarios omitted from a
            # spot-check can never influence which weights become best.pt.
            candidate_state = cloned_state_dict(actor)
            (rollout, observations, rollout_rng_states,
             evaluated) = checkpoint_parallel_evaluation(
                 rollout,
                 map_path=args.map,
                 rollout_seed=args.seed,
                 rollout_count=args.envs,
                 rollout_scenarios=scenarios,
                 encoding=args.encoding,
                 rollout_workers=args.workers,
                 rollout_mode=args.rollout_mode,
                 actor_state=candidate_state,
                 episodes=args.check_episodes,
                 eval_workers=args.eval_workers,
                 generator=rng)
            checked = evaluated["learned"]
            candidate_score = selection_score(checked)
            report = {
                "update": update + 1,
                "steps": (update + 1) * args.horizon * args.envs,
                "reward": float(rew_buffer.mean()),
                "elapsed_s": round(time.perf_counter() - started, 1),
                "inference_wrapper": contract["inference"]["wrapper"],
                "by_scenario": checked,
                "selection_score": score_document(candidate_score),
            }
            history.append(report)
            latest_metrics = report
            print(json.dumps(report), flush=True)
            recorded_score = best.get("score")
            best_tuple = (None if recorded_score is None else (
                float(recorded_score["worst_gate_progress"]),
                float(recorded_score["mean_gate_progress"]),
                float(recorded_score["mean_arrival"]),
                float(recorded_score["negative_max_collision"]),
                float(recorded_score["negative_mean_collision"])))
            better = best_tuple is None or candidate_score > best_tuple
            if better:
                best = {"score": score_document(candidate_score),
                        "scored_on": list(PROMOTION_SCENARIOS),
                        "episodes_each": args.check_episodes,
                        "update": update + 1}
                best_metrics = dict(report)
                best_actor_state = candidate_state
                save_torch_atomic(best_actor_state, args.output / "best.pt")

            # Save only after updating the high-water mark: a process killed
            # here resumes the same update, optimizer, random streams and best
            # candidate, rather than forgetting the evaluation it just ran.
            if best_actor_state is None:
                raise RuntimeError("evaluation did not select an initial best actor")
            payload = trainer_checkpoint(
                actor, critic, log_std, optimizer,
                completed_update=update + 1,
                best=best,
                best_actor_state=best_actor_state,
                best_metrics=best_metrics,
                history=history,
                latest_metrics=latest_metrics,
                contract=contract,
                generator=rng,
                run_config=run_config,
                rollout_rng_states=rollout_rng_states,
                trajectory=trajectory,
                trajectory_digest=trajectory_digest)
            save_torch_atomic(cloned_state_dict(actor),
                              args.output / "checkpoint.pt")
            save_torch_atomic(payload, args.output / "trainer.pt")
            if better:
                save_torch_atomic(payload, args.output / "best_trainer.pt")

    rollout.close()
    gc.collect()
    best_path = args.output / "best.pt"
    if not best_path.is_file():
        reason = "training completed without a selected best.pt"
        write_run_manifest(args.output, "quarantined", contract,
                           reason=reason, run_config=run_config,
                           run_config_digest=run_config_digest,
                           trajectory=trajectory,
                           trajectory_digest=trajectory_digest)
        raise RuntimeError(reason)
    best_state = torch.load(best_path, map_location="cpu", weights_only=True)
    selected_actor = Actor(best_state["feature_mean"].numpy(),
                           best_state["feature_std"].numpy())
    selected_actor.load_state_dict(best_state)
    selected_actor.eval()
    selected_policy = SymmetricActor(selected_actor).eval()

    # Scored one scenario at a time.  A single blended number hides the case
    # that matters: a policy can carry an excellent average while being the
    # one that walks into the dog.
    final_scenarios = tuple(PROMOTION_SCENARIOS)
    final_memory = wait_for_evaluation_memory(
        max(1, min(args.eval_workers, len(final_scenarios))))
    final_evaluated = parallel_evaluate_preserving_rng(
        rng, args.map, best_state, args.eval_episodes, 99,
        final_scenarios, args.encoding,
        int(final_memory["effective_workers"]),
        variants=(("learned", False), ("expert", True)))
    by_scenario = {
        scenario: {
            "learned": final_evaluated["learned"][scenario],
            "expert": final_evaluated["expert"][scenario],
        }
        for scenario in final_scenarios
    }
    learned = by_scenario[scenarios[0]]["learned"]
    teacher = by_scenario[scenarios[0]]["expert"]
    verdict = {
        "schema": METRICS_SCHEMA,
        "environment_steps": args.updates * args.horizon * args.envs,
        "trained_on": scenarios,
        "range_encoding": args.encoding,
        "contract": contract,
        "contract_digest": contract_digest(contract),
        "inference_wrapper": dict(contract["inference"]),
        "continued_from": (str(args.initialise_from)
                           if args.initialise_from else None),
        "best_checkpoint": best,
        "best_metrics": best_metrics,
        "selected_artifact": "best.pt",
        "selected_artifact_sha256": file_digest(best_path),
        "learned": learned,
        "expert": teacher,
        "beats_expert_on_arrival": learned["arrived"] > teacher["arrived"],
        "by_scenario": by_scenario,
        "history": history,
        "run_config": run_config,
        "run_config_digest": run_config_digest,
        "trajectory": trajectory,
        "trajectory_digest": trajectory_digest,
    }
    verdict["quality_gate"] = promotion_quality_gate(by_scenario)
    save_json_atomic(verdict, args.output / "metrics.json")
    if verdict["quality_gate"]["passed"] is not True:
        reason = ("selected best.pt failed final quality gate: " +
                  "; ".join(verdict["quality_gate"]["failures"]))
        retained_names = (
            "best.pt", "trainer.pt", "best_trainer.pt", "checkpoint.pt",
            "metrics.json")
        retained_artifacts = [
            {"path": name, "sha256": file_digest(args.output / name),
             "bytes": (args.output / name).stat().st_size}
            for name in retained_names
        ]
        retained_bundle = {
            "files": retained_artifacts,
            "bundle_sha256": contract_digest({
                item["path"]: item["sha256"] for item in retained_artifacts}),
        }
        write_run_manifest(
            args.output, "quarantined", contract,
            reason=reason,
            disposition="retain for analysis; do not deploy or warm-start",
            source_checkpoint=str(best_path),
            metrics="metrics.json",
            quality_gate=verdict["quality_gate"],
            run_config=run_config,
            run_config_digest=run_config_digest,
            completed_update=args.updates,
            retained_bundle=retained_bundle,
            trajectory=trajectory,
            trajectory_digest=trajectory_digest)
        print(json.dumps({"status": "quarantined", "reason": reason}),
              flush=True)
        return 2

    onnx_path = args.output / "model.onnx"
    temporary_onnx = onnx_path.with_name(f".{onnx_path.stem}.tmp.onnx")
    torch.onnx.export(
        selected_policy, torch.zeros(1, FEATURE_DIM),
        str(temporary_onnx),
        input_names=["features"], output_names=["twist"],
        dynamic_axes={"features": {0: "batch"}, "twist": {0: "batch"}},
        opset_version=17, dynamo=False)
    temporary_onnx.replace(onnx_path)
    # Kept only for older analysis scripts.  Byte identity makes it impossible
    # for the compatibility name to drift away from the selected best.pt.
    alias_path = args.output / "model.pt"
    temporary_alias = alias_path.with_name(f".{alias_path.name}.tmp")
    shutil.copyfile(best_path, temporary_alias)
    temporary_alias.replace(alias_path)
    write_promotable_manifest(
        args.output, contract,
        completed_update=args.updates)

    print(json.dumps({"by_scenario": by_scenario}, indent=2))
    return 0


def main() -> int:
    """Run offline training with an exclusive lock on its output directory."""
    from training_lock import exclusive_guard, output_guard_scope, parsed_option

    supplied = parsed_option(sys.argv[1:], "--output")
    if supplied is None:
        # Let argparse produce its normal required-argument diagnostic.
        return _main_with_output_lease()
    output = Path(supplied).resolve()
    with exclusive_guard(output_guard_scope(output)):
        return _main_with_output_lease(None)


if __name__ == "__main__":
    raise SystemExit(main())
