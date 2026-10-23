"""The immutable contract and promotion gate for learned driving policies.

A state dict says how many numbers a network contains.  It does not say what
those numbers mean.  In particular, an actor trained against the old circular
body, the old LiDAR acceptance window, or a one-frame observation can still
have a shape that loads cleanly and then drive the wrong robot.  This module
turns all of those otherwise implicit assumptions into one canonical document
and makes its digest travel with every resumable or deployable artifact.

``selected-model.json`` is deliberately written last.  A run directory with a
``best.pt`` but no promotable manifest is an interrupted experiment, not a
production candidate.  Exporters call :func:`validate_promotable_artifact`, so
neither a latest checkpoint nor a contract-mismatched run can be selected by
mistake.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from features import (ANGULAR_SCALE_RADPS, FEATURE_DIM, HISTORY_DELAYS_S,
                      HISTORY_MAXIMUM_GAP_S, LASER_YAW_RAD, LIDAR_BINS,
                      LINEAR_SCALE_MPS, PATH_POINTS, PATH_SCALE_M, PATH_LOOKAHEAD_M,
                      RANGE_ENCODING, RANGE_ENCODINGS)
from rl_env import MAX_ANGULAR, MAX_LINEAR
from sim_geometry import TURN_COAST_S
from simulate_expert_dataset import (BEAMS, BODY_RADIUS_M, COLLISION_FOOTPRINT_M, DT,
                                     FOOTPRINT_FRONT_M,
                                     FOOTPRINT_HALF_WIDTH_M,
                                     FOOTPRINT_REAR_M, RANGE_MAX, RANGE_MIN,
                                     SENSOR_RANGE_MAX, SENSOR_RANGE_MIN)


CONTRACT_SCHEMA = "robotbrain.rosnav-shadow.policy-contract.v1"
LEGACY_CHECKPOINT_SCHEMA_V3 = (
    "robotbrain.rosnav-shadow.trainer-checkpoint.v3")
CHECKPOINT_SCHEMA = "robotbrain.rosnav-shadow.trainer-checkpoint.v4"
LEGACY_MANIFEST_SCHEMA_V1 = "robotbrain.rosnav-shadow.selected-model.v1"
MANIFEST_SCHEMA = "robotbrain.rosnav-shadow.selected-model.v2"
LEGACY_METRICS_SCHEMA_V1 = "robotbrain.rosnav-shadow.rl-metrics.v1"
METRICS_SCHEMA = "robotbrain.rosnav-shadow.rl-metrics.v2"
QUALITY_GATE_SCHEMA = "robotbrain.rosnav-shadow.quality-gate.v2"
TRAJECTORY_SCHEMA = "robotbrain.rosnav-shadow.training-trajectory.v1"
TRAJECTORY_ORIGINS = {"fresh", "actor_warm_start"}
NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
MANIFEST_NAME = "selected-model.json"
PROMOTABLE_ARTIFACT = "best.pt"
INFERENCE_WRAPPER = "reflection_ensemble_mean_v1"
# Owner scope, 2026-09-06: sensor stop and replan; no predictive moving-actor
# avoidance. Physical stopping remains a separate required acceptance check.
PROMOTION_SCENARIOS = ("clear", "static", "standing")
PROMOTION_MIN_EPISODES = 120
CHECKPOINT_SELECTION_RULE = "promotion_arrival_and_collision_progress_v2"
LEGACY_SELECTION_RULE = "promotion_gate_normalized_lexicographic_v1"
PROMOTION_ARRIVAL_FLOORS = {
    "clear": 0.75,
    "static": 0.50,
    "standing": 0.55,
}


class ArtifactContractError(RuntimeError):
    """An artifact is incomplete, unselected, corrupt, or from another robot."""


def scenario_gate_progress(scenario, arrival, collision,
                           rule=CHECKPOINT_SELECTION_RULE):
    if rule not in (CHECKPOINT_SELECTION_RULE, LEGACY_SELECTION_RULE):
        raise ArtifactContractError(f"unknown checkpoint selection rule: {rule}")
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0
               for v in (arrival, collision)):
        raise ArtifactContractError("selection rates must be finite probabilities")
    # Progress can reach 1 only when BOTH acceptance conditions hold. Otherwise
    # a fast model with collisions could outrank an entirely passing model.
    ceiling = 1.0 if rule == LEGACY_SELECTION_RULE else 1.0 - collision
    return min(arrival / PROMOTION_ARRIVAL_FLOORS[scenario], ceiling)


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Stable UTF-8 representation used as the policy-contract identity."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def contract_digest(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(contract)).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _calibration() -> dict[str, Any]:
    v5 = Path(__file__).resolve().parents[2]
    path = v5 / "navigation" / "calibration.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _same(name: str, left: float, right: float) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
        raise ArtifactContractError(
            f"policy geometry drift: {name} is {left!r} in the simulator "
            f"and {right!r} in navigation/calibration.json")


def current_policy_contract(range_encoding: str | None = None) -> dict[str, Any]:
    """Return every assumption that can change what learned weights mean."""
    range_encoding = RANGE_ENCODING if range_encoding is None else range_encoding
    if range_encoding not in RANGE_ENCODINGS:
        raise ArtifactContractError(f"unknown policy range encoding: {range_encoding}")
    calibration = _calibration()
    _same("body_front_m", FOOTPRINT_FRONT_M, calibration["body_front_m"])
    _same("body_rear_m", FOOTPRINT_REAR_M, calibration["body_rear_m"])
    _same("body_half_width_m", FOOTPRINT_HALF_WIDTH_M,
          calibration["body_half_width_m"])
    _same("body_sweep_radius_m", BODY_RADIUS_M,
          calibration["body_sweep_radius_m"])

    if calibration["guard_footprint_m"] != COLLISION_FOOTPRINT_M:
        raise ArtifactContractError("policy collision contour drift")

    layout: list[dict[str, Any]] = []
    offset = 0
    for index, delay in enumerate((0.0, *HISTORY_DELAYS_S)):
        layout.append({
            "name": "lidar_current" if index == 0 else f"lidar_history_{index}",
            "offset": offset,
            "size": LIDAR_BINS,
            "temporal_offset_s": float(delay),
        })
        offset += LIDAR_BINS
    layout.append({"name": "local_path_xy", "offset": offset,
                   "size": PATH_POINTS * 2})
    offset += PATH_POINTS * 2
    layout.append({"name": "measured_twist", "offset": offset, "size": 2})
    offset += 2
    if offset != FEATURE_DIM:
        raise ArtifactContractError(
            f"feature layout describes {offset} values, FEATURE_DIM is {FEATURE_DIM}")

    footprint = [
        [FOOTPRINT_FRONT_M, FOOTPRINT_HALF_WIDTH_M],
        [FOOTPRINT_FRONT_M, -FOOTPRINT_HALF_WIDTH_M],
        [-FOOTPRINT_REAR_M, -FOOTPRINT_HALF_WIDTH_M],
        [-FOOTPRINT_REAR_M, FOOTPRINT_HALF_WIDTH_M],
    ]
    return {
        "schema": CONTRACT_SCHEMA,
        "features": {
            "dtype": "float32",
            "dimension": FEATURE_DIM,
            "layout": layout,
            "lidar_bins_per_view": LIDAR_BINS,
            "path_points": PATH_POINTS,
            "view_offsets_s": [0.0, *map(float, HISTORY_DELAYS_S)],
            "history_maximum_gap_s": HISTORY_MAXIMUM_GAP_S,
            "history_selection": "latest_sample_not_after_target",
            "history_warmup_s": (max(HISTORY_DELAYS_S)
                                   if HISTORY_DELAYS_S else 0.0),
            "laser_yaw_rad": LASER_YAW_RAD,
            "range_encoding": range_encoding,
            "path_scale_m": PATH_SCALE_M,
            "path_sampling": "uniform_near_arc_length_v1",
            "path_lookahead_m": PATH_LOOKAHEAD_M,
            "path_source": "causal_local_map_replan_v1",
            "linear_velocity_scale_mps": LINEAR_SCALE_MPS,
            "angular_velocity_scale_radps": ANGULAR_SCALE_RADPS,
        },
        "geometry": {
            "frame": "base_link",
            "footprint_m": footprint,
            "guard_footprint_m": [list(point) for point in COLLISION_FOOTPRINT_M],
            "body_sweep_radius_m": BODY_RADIUS_M,
            "wheel_radius_m": calibration["wheel_radius_m"],
            "wheel_base_m": calibration["wheel_base_m"],
            "encoder_ticks_per_revolution":
                calibration["encoder_ticks_per_revolution"],
            "left_distance_scale": calibration["left_distance_scale"],
            "right_distance_scale": calibration["right_distance_scale"],
        },
        "sensor": {
            "beams_per_revolution": BEAMS,
            "angle_binning": "nearest_bin_center_ties_even_v1",
            "physical_window_m": {
                "minimum": SENSOR_RANGE_MIN,
                "maximum": SENSOR_RANGE_MAX,
            },
            "policy_range_m": {"minimum": RANGE_MIN, "maximum": RANGE_MAX},
            "missing_return": "positive_infinity_encoded_as_policy_maximum",
        },
        "simulation": {"control_period_s": DT,
                       "collision_geometry": "convex_guard_polygon_cell_circumcircles_v1",
                       "angular_response_time_s": TURN_COAST_S,
                       "linear_response_fraction_per_step": 0.5},
        "actions": {
            "outputs": ["linear_x_mps", "angular_z_radps"],
            "linear_x_mps": {"minimum": 0.0, "maximum": MAX_LINEAR},
            "angular_z_radps": {
                "minimum": -MAX_ANGULAR,
                "maximum": MAX_ANGULAR,
            },
            "sample_space": [-1.0, 1.0],
        },
        "inference": {
            "wrapper": INFERENCE_WRAPPER,
            "base_model": "actor_mlp_228_128_128_64_2_v1",
            "passes": 2,
            "reflection_axis": "base_link_x",
            "feature_transform": "mirror_every_lidar_view_and_path_lateral_and_angular_velocity",
            "output_transform": "preserve_linear_and_negate_angular",
            "aggregation": "arithmetic_mean_of_direct_and_reflected_outputs",
            "formula": "0.5*(actor(x)+mirror_twist(actor(mirror_features(x))))",
        },
    }


def validate_checkpoint_contract(checkpoint: Mapping[str, Any],
                                 expected: Mapping[str, Any] | None = None) -> str:
    """Verify both the checkpoint's self-consistency and current compatibility."""
    expected = current_policy_contract() if expected is None else dict(expected)
    recorded = checkpoint.get("contract")
    recorded_digest = checkpoint.get("contract_digest")
    if not isinstance(recorded, Mapping) or not isinstance(recorded_digest, str):
        raise ArtifactContractError(
            "checkpoint has no policy contract; it is benchmark-only and cannot "
            "be resumed into a production candidate")
    actual_recorded_digest = contract_digest(recorded)
    if actual_recorded_digest != recorded_digest:
        raise ArtifactContractError(
            "checkpoint policy contract is corrupt: its recorded digest does "
            "not describe its embedded metadata")
    expected_digest = contract_digest(expected)
    if recorded_digest != expected_digest:
        raise ArtifactContractError(
            f"checkpoint contract mismatch: checkpoint={recorded_digest}, "
            f"current={expected_digest}; quarantine and retrain")
    return recorded_digest


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def promotion_quality_gate(by_scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Require task completion and no observed collision in the held-out suite.

    Passing this finite sample permits hardware evaluation; it does not prove
    collision-free behavior outside the tested scenarios.
    """
    if not isinstance(by_scenario, Mapping):
        raise ArtifactContractError("metrics by_scenario is not an object")
    missing = sorted(set(PROMOTION_SCENARIOS) - set(by_scenario))
    extra = sorted(set(by_scenario) - set(PROMOTION_SCENARIOS))
    if missing or extra:
        raise ArtifactContractError(
            f"metrics scenarios differ from the promotion suite: "
            f"missing={missing}, extra={extra}")

    results: dict[str, Any] = {}
    failures: list[str] = []
    for scenario in PROMOTION_SCENARIOS:
        entry = by_scenario[scenario]
        learned = entry.get("learned") if isinstance(entry, Mapping) else None
        if not isinstance(learned, Mapping):
            raise ArtifactContractError(
                f"metrics for {scenario!r} have no learned result")
        episodes = learned.get("episodes")
        arrived = learned.get("arrived")
        collision = learned.get("collision")
        timeout = learned.get("timeout")
        if (isinstance(episodes, bool) or not isinstance(episodes, int) or
                episodes < 0):
            raise ArtifactContractError(
                f"metrics for {scenario!r} have invalid episode count")
        for name, value in (("arrived", arrived), ("collision", collision),
                            ("timeout", timeout)):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or
                    not 0.0 <= float(value) <= 1.0):
                raise ArtifactContractError(
                    f"metrics for {scenario!r} have invalid {name} rate")
        outcome_sum = float(arrived) + float(collision) + float(timeout)
        if not math.isclose(outcome_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ArtifactContractError(
                f"metrics for {scenario!r} have outcome rates summing to "
                f"{outcome_sum!r}, not 1.0")
        floor = PROMOTION_ARRIVAL_FLOORS[scenario]
        enough_episodes = episodes >= PROMOTION_MIN_EPISODES
        enough_arrivals = float(arrived) >= floor
        collision_free = float(collision) == 0.0
        passed = enough_episodes and enough_arrivals and collision_free
        if not collision_free:
            failures.append(f"{scenario}: observed collision rate {float(collision):.6f} > 0")
        if not enough_episodes:
            failures.append(
                f"{scenario}: {episodes} episodes < {PROMOTION_MIN_EPISODES}")
        if not enough_arrivals:
            failures.append(
                f"{scenario}: arrival {float(arrived):.6f} < {floor:.6f}")
        results[scenario] = {
            "episodes": episodes,
            "arrived": float(arrived),
            "collision": float(collision),
            "timeout": float(timeout),
            "minimum_arrival": floor,
            "passed": passed,
        }
    return {
        "schema": QUALITY_GATE_SCHEMA,
        "passed": not failures,
        "minimum_episodes_per_scenario": PROMOTION_MIN_EPISODES,
        "arrival_floors": dict(PROMOTION_ARRIVAL_FLOORS),
        "maximum_observed_collision_rate": 0.0,
        "scenarios": results,
        "failures": failures,
    }


def _sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def build_trajectory_identity(
        *, trajectory_id: str, origin: str, policy_contract_digest: str,
        run_config_digest: str, output: str,
        parent: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    """Build the immutable identity of one stochastic training trajectory."""
    document: dict[str, Any] = {
        "schema": TRAJECTORY_SCHEMA,
        "trajectory_id": str(trajectory_id),
        "origin": str(origin),
        "contract_digest": str(policy_contract_digest),
        "run_config_digest": str(run_config_digest),
        "output": str(output),
        "parent": None if parent is None else dict(parent),
    }
    digest = contract_digest(document)
    validate_trajectory_identity(
        document, digest, "trajectory", expected_contract_digest=
        policy_contract_digest, expected_run_config_digest=run_config_digest,
        expected_output=output)
    return document, digest


def validate_trajectory_identity(
        trajectory: Any, trajectory_digest: Any, label: str, *,
        expected: Mapping[str, Any] | None = None,
        expected_digest: str | None = None,
        required_origin: str | None = None,
        expected_contract_digest: str | None = None,
        expected_run_config_digest: str | None = None,
        expected_output: str | None = None) -> tuple[dict[str, Any], str]:
    """Validate both the trajectory document and its cryptographic digest."""
    if not isinstance(trajectory, Mapping) or not _sha256(trajectory_digest):
        raise ArtifactContractError(f"{label} has no valid trajectory identity")
    document = dict(trajectory)
    expected_fields = {
        "schema", "trajectory_id", "origin", "contract_digest",
        "run_config_digest", "output", "parent",
    }
    if set(document) != expected_fields:
        raise ArtifactContractError(
            f"{label} trajectory has the wrong schema fields")
    if document.get("schema") != TRAJECTORY_SCHEMA:
        raise ArtifactContractError(f"{label} trajectory schema is invalid")
    if not _sha256(document.get("trajectory_id")):
        raise ArtifactContractError(f"{label} trajectory_id is invalid")
    origin = document.get("origin")
    if origin not in TRAJECTORY_ORIGINS:
        raise ArtifactContractError(f"{label} trajectory origin is invalid")
    if required_origin is not None and origin != required_origin:
        raise ArtifactContractError(
            f"{label} trajectory origin is {origin!r}, expected "
            f"{required_origin!r}")
    if (not _sha256(document.get("contract_digest")) or
            not _sha256(document.get("run_config_digest")) or
            not isinstance(document.get("output"), str) or
            not document["output"] or not Path(document["output"]).is_absolute()):
        raise ArtifactContractError(
            f"{label} trajectory binding fields are invalid")
    parent = document.get("parent")
    if origin == "fresh":
        if parent is not None:
            raise ArtifactContractError(
                f"{label} fresh trajectory must not have a parent")
    else:
        parent_fields = {
            "selected_artifact_sha256", "selected_manifest_sha256",
            "trajectory_digest",
        }
        if (not isinstance(parent, Mapping) or set(parent) != parent_fields or
                not all(_sha256(parent.get(field)) for field in parent_fields)):
            raise ArtifactContractError(
                f"{label} actor warm-start trajectory has invalid parent evidence")
    actual_digest = contract_digest(document)
    if trajectory_digest != actual_digest:
        raise ArtifactContractError(f"{label} trajectory digest is corrupt")
    if expected is not None and document != dict(expected):
        raise ArtifactContractError(f"{label} belongs to another trajectory")
    if expected_digest is not None and trajectory_digest != expected_digest:
        raise ArtifactContractError(f"{label} trajectory digest is not pinned")
    expected_values = (
        ("contract_digest", expected_contract_digest),
        ("run_config_digest", expected_run_config_digest),
        ("output", expected_output),
    )
    for field, value in expected_values:
        if value is not None and document.get(field) != value:
            raise ArtifactContractError(
                f"{label} trajectory {field} is not pinned")
    return document, trajectory_digest


def _validate_run_identity(document: Mapping[str, Any], label: str, *,
                           allow_legacy_runtime: bool = False
                           ) -> tuple[dict[str, Any], str]:
    run_config = document.get("run_config")
    run_config_digest = document.get("run_config_digest")
    if not isinstance(run_config, Mapping) or not _sha256(run_config_digest):
        raise ArtifactContractError(
            f"{label} has no valid run configuration identity")
    if contract_digest(run_config) != run_config_digest:
        raise ArtifactContractError(
            f"{label} run configuration digest is corrupt")
    provenance = run_config.get("provenance")
    map_identity = (provenance.get("map")
                    if isinstance(provenance, Mapping) else None)
    source_identity = (provenance.get("source")
                       if isinstance(provenance, Mapping) else None)
    runtime_identity = (provenance.get("runtime")
                         if isinstance(provenance, Mapping) else None)
    native_limits = (provenance.get("native_thread_limits")
                     if isinstance(provenance, Mapping) else None)
    trainer_entrypoint = (provenance.get("trainer_entrypoint")
                          if isinstance(provenance, Mapping) else None)
    if not isinstance(map_identity, Mapping):
        raise ArtifactContractError(f"{label} has no map provenance")
    for field in ("yaml_sha256", "image_sha256", "bundle_sha256"):
        if not _sha256(map_identity.get(field)):
            raise ArtifactContractError(
                f"{label} has invalid map provenance field {field}")
    for field in ("yaml_path", "yaml_name", "image_path", "image_reference"):
        if not isinstance(map_identity.get(field), str):
            raise ArtifactContractError(
                f"{label} has invalid map provenance field {field}")
    expected_map_bundle = contract_digest({
        "yaml_name": map_identity["yaml_name"],
        "yaml_sha256": map_identity["yaml_sha256"],
        "image_reference": map_identity["image_reference"],
        "image_sha256": map_identity["image_sha256"],
    })
    if map_identity.get("bundle_sha256") != expected_map_bundle:
        raise ArtifactContractError(f"{label} map bundle digest is corrupt")
    if not isinstance(source_identity, Mapping) or not _sha256(
            source_identity.get("bundle_sha256")):
        raise ArtifactContractError(f"{label} has invalid source provenance")
    source_files = source_identity.get("files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise ArtifactContractError(f"{label} has no source-file provenance")
    for name, identity in source_files.items():
        if (not isinstance(name, str) or not isinstance(identity, Mapping) or
                not isinstance(identity.get("path"), str) or
                not _sha256(identity.get("sha256"))):
            raise ArtifactContractError(
                f"{label} contains invalid source-file provenance")
    expected_source_bundle = contract_digest({
        name: identity["sha256"] for name, identity in source_files.items()})
    if source_identity.get("bundle_sha256") != expected_source_bundle:
        raise ArtifactContractError(f"{label} source bundle digest is corrupt")
    runtime_fields = {
        "python_implementation", "python_version",
        "python_executable_architecture", "os", "os_release", "machine",
        "torch", "numpy", "scipy", "onnx", "onnxruntime", "pyyaml",
        "psutil", "byteorder",
    }
    recorded_runtime_fields = (set(runtime_identity)
                               if isinstance(runtime_identity, Mapping)
                               else set())
    permitted_runtime_fields = {frozenset(runtime_fields)}
    if allow_legacy_runtime:
        # r10's producer already recorded psutil, but earlier v1 fixtures and
        # archived evidence predate that field.  Legacy inspection accepts
        # both shapes; every newly written v2 artifact requires the complete
        # current identity above.
        permitted_runtime_fields.add(frozenset(runtime_fields - {"psutil"}))
    if (not isinstance(runtime_identity, Mapping) or
            frozenset(recorded_runtime_fields) not in permitted_runtime_fields or
            not all(isinstance(value, str) and value
                    for value in runtime_identity.values())):
        raise ArtifactContractError(f"{label} has invalid runtime provenance")
    if not allow_legacy_runtime:
        observed = (native_limits.get("observed")
                    if isinstance(native_limits, Mapping) else None)
        if (not isinstance(native_limits, Mapping) or
                set(native_limits) != {"enforced_by", "required_value",
                                       "observed"} or
                native_limits.get("enforced_by") !=
                "entrypoint_before_native_imports" or
                native_limits.get("required_value") != "1" or
                observed != {name: "1" for name in NATIVE_THREAD_ENV_VARS}):
            raise ArtifactContractError(
                f"{label} has invalid native-thread-limit provenance")
        if (not isinstance(trainer_entrypoint, Mapping) or
                set(trainer_entrypoint) != {"path", "sha256"} or
                not isinstance(trainer_entrypoint.get("path"), str) or
                not Path(trainer_entrypoint["path"]).is_absolute() or
                not _sha256(trainer_entrypoint.get("sha256"))):
            raise ArtifactContractError(
                f"{label} has invalid trainer-entrypoint provenance")
    return dict(run_config), run_config_digest


def _validate_best_evidence(document: Mapping[str, Any],
                            metrics_schema: str) -> None:
    best = document.get("best_checkpoint")
    metrics = document.get("best_metrics")
    history = document.get("history")
    if not isinstance(best, Mapping) or not isinstance(metrics, Mapping):
        raise ArtifactContractError(
            "metrics.json has no selected-checkpoint evidence")
    if (isinstance(best.get("update"), bool) or
            not isinstance(best.get("update"), int) or
            metrics.get("update") != best.get("update")):
        raise ArtifactContractError(
            "metrics.json best checkpoint and best metrics updates disagree")
    if metrics.get("inference_wrapper") != INFERENCE_WRAPPER:
        raise ArtifactContractError(
            "metrics.json best checkpoint was not scored with the deploy wrapper")
    if not isinstance(history, list) or not any(item == metrics for item in history):
        raise ArtifactContractError(
            "metrics.json best metrics are absent from evaluation history")
    scored_on = best.get("scored_on")
    episodes_each = best.get("episodes_each")
    by_scenario = metrics.get("by_scenario")
    if scored_on != list(PROMOTION_SCENARIOS) or not isinstance(
            by_scenario, Mapping):
        raise ArtifactContractError(
            "metrics.json best checkpoint was not scored on all scenarios")
    if (isinstance(episodes_each, bool) or not isinstance(episodes_each, int) or
            episodes_each <= 0):
        raise ArtifactContractError(
            "metrics.json best checkpoint has an invalid episode count")
    arrivals, collisions, gate_progress = [], [], []
    for scenario in PROMOTION_SCENARIOS:
        result = by_scenario.get(scenario)
        if (not isinstance(result, Mapping) or
                result.get("episodes") != episodes_each):
            raise ArtifactContractError(
                "metrics.json best checkpoint episode counts disagree")
        try:
            arrival = float(result["arrived"])
            collision = float(result["collision"])
            if (not math.isfinite(arrival) or not 0.0 <= arrival <= 1.0 or
                    not math.isfinite(collision) or
                    not 0.0 <= collision <= 1.0):
                raise ValueError("rates must be finite probabilities")
            arrivals.append(arrival)
            collisions.append(collision)
            rule = document.get("run_config", {}).get("trainer", {}).get(
                "checkpoint_selection", LEGACY_SELECTION_RULE)
            gate_progress.append(scenario_gate_progress(
                scenario, arrival, collision, rule))
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactContractError(
                "metrics.json best checkpoint rates are invalid") from error
    if metrics_schema == METRICS_SCHEMA:
        expected_score = {
            "worst_gate_progress": min(gate_progress),
            "mean_gate_progress": sum(gate_progress) / len(gate_progress),
            "mean_arrival": sum(arrivals) / len(arrivals),
            "negative_max_collision": -max(collisions),
            "negative_mean_collision": -sum(collisions) / len(collisions),
        }
    elif metrics_schema == LEGACY_METRICS_SCHEMA_V1:
        expected_score = {
            "worst_arrival": min(arrivals),
            "mean_arrival": sum(arrivals) / len(arrivals),
            "negative_max_collision": -max(collisions),
            "negative_mean_collision": -sum(collisions) / len(collisions),
        }
    else:  # guarded by _load_metrics; defensive for direct internal calls
        raise ArtifactContractError(
            f"unsupported metrics schema {metrics_schema!r}")
    recorded_score = best.get("score")
    metric_score = metrics.get("selection_score")
    if not isinstance(recorded_score, Mapping) or not isinstance(
            metric_score, Mapping):
        raise ArtifactContractError("metrics.json has no selection score")
    if (set(recorded_score) != set(expected_score) or
            set(metric_score) != set(expected_score)):
        raise ArtifactContractError(
            "metrics.json selection score has the wrong schema fields")
    for name, expected_value in expected_score.items():
        for label, score in (("best checkpoint", recorded_score),
                             ("best metrics", metric_score)):
            value = score.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isclose(float(value), expected_value,
                                     rel_tol=0.0, abs_tol=1e-12)):
                raise ArtifactContractError(
                    f"metrics.json {label} selection score disagrees with "
                    "scenario results")


def _load_metrics(path: Path, expected: Mapping[str, Any],
                  selected_digest: str, *,
                  expected_schema: str = METRICS_SCHEMA,
                  required_quality_gate: bool = True,
                  expected_trajectory: Mapping[str, Any] | None = None,
                  expected_trajectory_digest: str | None = None
                  ) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactContractError(f"cannot read metrics.json: {error}") from error
    if expected_schema not in {METRICS_SCHEMA, LEGACY_METRICS_SCHEMA_V1}:
        raise ValueError(f"unsupported expected metrics schema {expected_schema!r}")
    if not isinstance(document, dict) or document.get("schema") != expected_schema:
        raise ArtifactContractError("metrics.json has an invalid schema")
    if document.get("selected_artifact") != PROMOTABLE_ARTIFACT:
        raise ArtifactContractError("metrics.json does not evaluate best.pt")
    if document.get("selected_artifact_sha256") != selected_digest:
        raise ArtifactContractError(
            "metrics.json does not bind the selected best.pt bytes")
    if document.get("inference_wrapper") != expected.get("inference"):
        raise ArtifactContractError(
            "metrics.json inference wrapper disagrees with the policy contract")
    validate_checkpoint_contract(document, expected)
    _validate_run_identity(
        document, "metrics.json",
        allow_legacy_runtime=expected_schema == LEGACY_METRICS_SCHEMA_V1)
    if expected_schema == METRICS_SCHEMA:
        validate_trajectory_identity(
            document.get("trajectory"), document.get("trajectory_digest"),
            "metrics.json", expected=expected_trajectory,
            expected_digest=expected_trajectory_digest,
            expected_contract_digest=contract_digest(expected),
            expected_run_config_digest=document.get("run_config_digest"))
    _validate_best_evidence(document, expected_schema)
    recomputed = promotion_quality_gate(document.get("by_scenario"))
    recorded = document.get("quality_gate")
    if not isinstance(recorded, Mapping) or dict(recorded) != recomputed:
        raise ArtifactContractError(
            "metrics.json quality gate is missing or disagrees with its results")
    if recomputed["passed"] is not required_quality_gate:
        expected_label = "passed" if required_quality_gate else "failed"
        raise ArtifactContractError(
            f"selected policy quality gate is not {expected_label}: " +
            "; ".join(recomputed["failures"]))
    return document


def validate_evaluated_metrics(
        path: Path, expected: Mapping[str, Any], selected_digest: str, *,
        required_quality_gate: bool,
        expected_trajectory: Mapping[str, Any] | None = None,
        expected_trajectory_digest: str | None = None) -> dict[str, Any]:
    """Validate terminal evaluation evidence for either gate outcome."""
    return _load_metrics(
        path, expected, selected_digest,
        expected_schema=METRICS_SCHEMA,
        required_quality_gate=required_quality_gate,
        expected_trajectory=expected_trajectory,
        expected_trajectory_digest=expected_trajectory_digest)


def _load_selected_policy(path: Path, expected: Mapping[str, Any]):
    """Strictly load the base weights and build the policy Rust executes."""
    import torch

    from symmetrize import SymmetricActor
    from train_rl import Actor

    inference = expected.get("inference")
    if (not isinstance(inference, Mapping) or
            inference.get("wrapper") != INFERENCE_WRAPPER or
            inference.get("passes") != 2):
        raise ArtifactContractError(
            "policy contract does not describe the supported symmetric wrapper")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ArtifactContractError(f"best.pt cannot be loaded: {error}") from error
    if not isinstance(state, Mapping):
        raise ArtifactContractError("best.pt is not an Actor state dictionary")

    dimension = int(expected["features"]["dimension"])
    mean = state.get("feature_mean")
    std = state.get("feature_std")
    if (not isinstance(mean, torch.Tensor) or mean.shape != (dimension,) or
            not isinstance(std, torch.Tensor) or std.shape != (dimension,)):
        raise ArtifactContractError(
            "best.pt has invalid feature normalisation tensors")
    if mean.dtype != torch.float32 or std.dtype != torch.float32:
        raise ArtifactContractError(
            "best.pt feature normalisation must be float32")

    actor = Actor(mean.detach().cpu().numpy(), std.detach().cpu().numpy())
    expected_state = actor.state_dict()
    if set(state) != set(expected_state):
        missing = sorted(set(expected_state) - set(state))
        extra = sorted(set(state) - set(expected_state))
        raise ArtifactContractError(
            f"best.pt Actor keys disagree: missing={missing}, extra={extra}")
    for name, reference in expected_state.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ArtifactContractError(f"best.pt {name} is not a tensor")
        if value.shape != reference.shape or value.dtype != torch.float32:
            raise ArtifactContractError(
                f"best.pt {name} has shape/dtype {tuple(value.shape)}/{value.dtype}, "
                f"expected {tuple(reference.shape)}/torch.float32")
        if not bool(torch.isfinite(value).all()):
            raise ArtifactContractError(f"best.pt {name} contains non-finite values")
    if not bool((std > 0.0).all()):
        raise ArtifactContractError("best.pt feature_std must be strictly positive")
    try:
        actor.load_state_dict(dict(state), strict=True)
    except Exception as error:
        raise ArtifactContractError(
            f"best.pt cannot initialise the canonical Actor: {error}") from error
    actor.eval()
    return SymmetricActor(actor).eval()


def _validate_onnx(path: Path, expected: Mapping[str, Any], policy) -> None:
    """Prove the hashed ONNX is the same symmetric policy as ``best.pt``."""
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from onnx import TensorProto
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model)
    except Exception as error:
        raise ArtifactContractError(f"model.onnx is invalid: {error}") from error

    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    if len(inputs) != 1 or inputs[0].name != "features":
        raise ArtifactContractError(
            "model.onnx must have exactly one input named 'features'")
    if len(outputs) != 1 or outputs[0].name != "twist":
        raise ArtifactContractError(
            "model.onnx must have exactly one output named 'twist'")

    def tensor_shape(value, *, name: str, width: int) -> None:
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type != TensorProto.FLOAT:
            raise ArtifactContractError(f"model.onnx {name} must be float32")
        dimensions = tensor_type.shape.dim
        if len(dimensions) != 2:
            raise ArtifactContractError(f"model.onnx {name} must be rank two")
        if dimensions[1].dim_value != width:
            raise ArtifactContractError(
                f"model.onnx {name} width is {dimensions[1].dim_value}, "
                f"expected {width}")
        if not dimensions[0].dim_param:
            raise ArtifactContractError(
                f"model.onnx {name} must have a dynamic batch dimension")

    tensor_shape(inputs[0], name="features",
                 width=int(expected["features"]["dimension"]))
    tensor_shape(outputs[0], name="twist",
                 width=len(expected["actions"]["outputs"]))

    # Shape alone accepts an unrelated 228 -> 2 graph.  Probe around the
    # selected actor's own normalisation scale so neither implementation is
    # hidden behind saturated tanh outputs, then compare the actual functions.
    try:
        mean = policy.actor.feature_mean.detach().cpu().numpy()
        std = policy.actor.feature_std.detach().cpu().numpy()
        rng = np.random.default_rng(20260831)
        z = rng.normal(0.0, 0.65, size=(16, mean.size)).astype(np.float32)
        probes = mean[None, :] + z * std[None, :]
        probes[0] = mean
        probes[1] = mean + np.linspace(-0.5, 0.5, mean.size,
                                      dtype=np.float32) * std
        with torch.inference_mode():
            reference = policy(torch.from_numpy(probes)).cpu().numpy()
        session = ort.InferenceSession(
            path.read_bytes(), providers=["CPUExecutionProvider"])
        actual = session.run(["twist"], {"features": probes})[0]
    except Exception as error:
        raise ArtifactContractError(
            f"model.onnx cannot execute as the selected policy: {error}") from error
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ArtifactContractError(
            "model.onnx produced an invalid output shape or non-finite values")
    if not np.allclose(actual, reference, rtol=1e-5, atol=2e-6):
        error = float(np.max(np.abs(actual - reference)))
        raise ArtifactContractError(
            "model.onnx is not numerically equivalent to the symmetric "
            f"best.pt policy (maximum absolute error {error:.8g})")


def write_run_manifest(output: Path, status: str, contract: Mapping[str, Any],
                       **details: Any) -> dict[str, Any]:
    """Record a non-promotable run state (``training`` or ``quarantined``)."""
    if status not in {"training", "quarantined"}:
        raise ValueError(f"invalid non-promotable status: {status}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": status,
        "promotable": False,
        "contract": dict(contract),
        "contract_digest": contract_digest(contract),
        "selected_artifact": None,
    }
    reserved = set(document) & set(details)
    if reserved:
        raise ValueError(f"manifest details cannot replace {sorted(reserved)}")
    document.update(details)
    _write_json_atomic(output / MANIFEST_NAME, document)
    return document


def write_legacy_quarantine_manifest(
        output: Path, *, reason: str, disposition: str,
        source_artifacts: Mapping[str, Path],
        source_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Describe pre-contract weights without laundering them as current.

    This deliberately never calls :func:`current_policy_contract`: an old
    tensor shape is not evidence of the geometry, sensor window, history, or
    action semantics used to train it.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    discovered = []
    for name, supplied_path in sorted(source_artifacts.items()):
        path = Path(supplied_path)
        if path.is_file():
            discovered.append({
                "name": str(name),
                "path": str(path.resolve()),
                "sha256": file_digest(path),
            })
    document: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "quarantined",
        "promotable": False,
        "contract": None,
        "contract_digest": "legacy-unrecorded",
        "selected_artifact": None,
        "reason": str(reason),
        "disposition": str(disposition),
        "source_artifacts": discovered,
        "source_identity": (None if source_identity is None
                            else dict(source_identity)),
    }
    _write_json_atomic(output / MANIFEST_NAME, document)
    return document


def write_promotable_manifest(output: Path, contract: Mapping[str, Any],
                              **details: Any) -> dict[str, Any]:
    """Select exactly ``best.pt`` after all final artifacts have succeeded."""
    output = Path(output)
    selected = output / PROMOTABLE_ARTIFACT
    alias = output / "model.pt"
    if not selected.is_file():
        raise ArtifactContractError(f"cannot promote missing {selected}")
    selected_digest = file_digest(selected)
    if not alias.is_file() or file_digest(alias) != selected_digest:
        raise ArtifactContractError(
            "model.pt must be a byte-identical compatibility alias of best.pt")
    onnx = output / "model.onnx"
    metrics = output / "metrics.json"
    if not onnx.is_file():
        raise ArtifactContractError(f"cannot promote missing {onnx}")
    if not metrics.is_file():
        raise ArtifactContractError(f"cannot promote missing {metrics}")
    policy = _load_selected_policy(selected, contract)
    _validate_onnx(onnx, contract, policy)
    metrics_document = _load_metrics(metrics, contract, selected_digest)
    document: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "promotable",
        "promotable": True,
        "contract": dict(contract),
        "contract_digest": contract_digest(contract),
        "run_config": metrics_document["run_config"],
        "run_config_digest": metrics_document["run_config_digest"],
        "trajectory": metrics_document["trajectory"],
        "trajectory_digest": metrics_document["trajectory_digest"],
        "inference_wrapper": metrics_document["inference_wrapper"],
        "best_checkpoint": metrics_document["best_checkpoint"],
        "best_metrics": metrics_document["best_metrics"],
        "selected_artifact": {
            "path": PROMOTABLE_ARTIFACT,
            "sha256": selected_digest,
            "selection": "best_evaluation_not_last_training_update",
        },
        "compatibility_aliases": [
            {"path": "model.pt", "sha256": selected_digest,
             "source": PROMOTABLE_ARTIFACT},
        ],
        "derived_artifacts": [
            {"path": onnx.name, "sha256": file_digest(onnx),
             "source": PROMOTABLE_ARTIFACT, "role": "deployment"},
            {"path": metrics.name, "sha256": file_digest(metrics),
             "source": PROMOTABLE_ARTIFACT, "role": "evaluation"},
        ],
    }
    reserved = set(document) & set(details)
    if reserved:
        raise ValueError(f"manifest details cannot replace {sorted(reserved)}")
    document.update(details)
    _write_json_atomic(output / MANIFEST_NAME, document)
    return document


def validate_promotable_artifact(
        weights: Path, expected: Mapping[str, Any] | None = None, *,
        manifest_schema: str = MANIFEST_SCHEMA,
        metrics_schema: str = METRICS_SCHEMA) -> dict[str, Any]:
    """Refuse latest, interrupted, quarantined, mismatched, or altered weights."""
    schema_pair = (manifest_schema, metrics_schema)
    supported_pairs = {
        (MANIFEST_SCHEMA, METRICS_SCHEMA),
        (LEGACY_MANIFEST_SCHEMA_V1, LEGACY_METRICS_SCHEMA_V1),
    }
    if schema_pair not in supported_pairs:
        raise ValueError(f"unsupported artifact schema pair {schema_pair!r}")
    weights = Path(weights)
    manifest_path = weights.parent / MANIFEST_NAME
    if weights.name != PROMOTABLE_ARTIFACT:
        raise ArtifactContractError(
            f"{weights.name} is not selectable; production must use "
            f"{PROMOTABLE_ARTIFACT}")
    if not manifest_path.is_file():
        raise ArtifactContractError(
            f"{weights} has no {MANIFEST_NAME}; the run is not promotable")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactContractError(
            f"cannot read selected-model manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ArtifactContractError("selected-model manifest is not an object")
    if manifest.get("schema") != manifest_schema:
        raise ArtifactContractError(
            f"selected-model manifest schema is {manifest.get('schema')!r}, "
            f"expected {manifest_schema!r}")
    if (manifest.get("status") != "promotable" or
            manifest.get("promotable") is not True):
        raise ArtifactContractError(
            f"{weights.parent} is {manifest.get('status', 'invalid')!r}, not "
            "promotable")
    selected = manifest.get("selected_artifact")
    if not isinstance(selected, Mapping) or selected.get("path") != weights.name:
        raise ArtifactContractError("manifest does not select these weights")
    if not weights.is_file() or selected.get("sha256") != file_digest(weights):
        raise ArtifactContractError("selected weights are missing or were altered")
    aliases = manifest.get("compatibility_aliases", [])
    derived = manifest.get("derived_artifacts", [])
    if not isinstance(aliases, list) or not isinstance(derived, list):
        raise ArtifactContractError("manifest linked-artifact lists are invalid")
    for linked in [*aliases, *derived]:
        if (not isinstance(linked, Mapping) or
                not isinstance(linked.get("path"), str) or
                not isinstance(linked.get("sha256"), str)):
            raise ArtifactContractError("manifest contains an invalid linked artifact")
        path = weights.parent / linked["path"]
        if not path.is_file() or file_digest(path) != linked.get("sha256"):
            raise ArtifactContractError(
                f"linked artifact {linked.get('path')!r} is missing or altered")
    alias_paths = {entry.get("path") for entry in aliases
                   if isinstance(entry, Mapping)}
    derived_paths = {entry.get("path") for entry in derived
                     if isinstance(entry, Mapping)}
    if alias_paths != {"model.pt"}:
        raise ArtifactContractError("manifest must bind the model.pt alias")
    if derived_paths != {"model.onnx", "metrics.json"}:
        raise ArtifactContractError(
            "manifest must bind model.onnx and metrics.json")
    validate_checkpoint_contract(manifest, expected)
    manifest_run_config, manifest_run_digest = _validate_run_identity(
        manifest, "selected-model.json",
        allow_legacy_runtime=(
            manifest_schema == LEGACY_MANIFEST_SCHEMA_V1))
    manifest_trajectory = None
    manifest_trajectory_digest = None
    if manifest_schema == MANIFEST_SCHEMA:
        manifest_trajectory, manifest_trajectory_digest = (
            validate_trajectory_identity(
                manifest.get("trajectory"), manifest.get("trajectory_digest"),
                "selected-model.json",
                expected_contract_digest=contract_digest(
                    current_policy_contract() if expected is None else expected),
                expected_run_config_digest=manifest_run_digest))
    expected_contract = (current_policy_contract() if expected is None
                          else dict(expected))
    policy = _load_selected_policy(weights, expected_contract)
    _validate_onnx(weights.parent / "model.onnx", expected_contract, policy)
    metrics_document = _load_metrics(
        weights.parent / "metrics.json", expected_contract,
        selected.get("sha256"), expected_schema=metrics_schema,
        expected_trajectory=manifest_trajectory,
        expected_trajectory_digest=manifest_trajectory_digest)
    if (manifest_run_config != metrics_document.get("run_config") or
            manifest_run_digest != metrics_document.get("run_config_digest")):
        raise ArtifactContractError(
            "selected-model.json and metrics.json run identities disagree")
    for field in ("best_checkpoint", "best_metrics"):
        if manifest.get(field) != metrics_document.get(field):
            raise ArtifactContractError(
                f"selected-model.json and metrics.json {field} disagree")
    if manifest.get("inference_wrapper") != metrics_document.get(
            "inference_wrapper"):
        raise ArtifactContractError(
            "selected-model.json and metrics.json inference wrappers disagree")
    return manifest


def validate_legacy_promotable_artifact_evidence(
        weights: Path, expected: Mapping[str, Any] | None = None
        ) -> dict[str, Any]:
    """Read and verify a v1 artifact without treating it as an r11 release.

    Current exporters intentionally call :func:`validate_promotable_artifact`
    with its v2 defaults.  This explicit audit-only entry point keeps archived
    r10 evidence readable while preventing a legacy score payload from being
    silently relabelled as the current format.
    """
    return validate_promotable_artifact(
        weights, expected,
        manifest_schema=LEGACY_MANIFEST_SCHEMA_V1,
        metrics_schema=LEGACY_METRICS_SCHEMA_V1)
