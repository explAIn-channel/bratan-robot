#!/usr/bin/env python3
"""Drive the trained policy around the map and count what it actually does.

Per-sample error says how closely the network copies its teacher on frames
the teacher chose.  It says nothing about what happens once the network picks
the next frame itself, where small errors compound.  So this closes the loop:
the policy steers the same differential-drive model over the same map, and we
count arrivals, collisions and stalls against the expert on identical starts
and goals.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features import LASER_YAW_RAD, build_features                # noqa: E402
import sim_geometry as core                             # noqa: E402
from simulate_expert_dataset import (BEAMS, DT, Flat, RANGE_MAX,   # noqa: E402
                                     RANGE_MIN, descend,
                                     expert_command, to_base_link, wrap)

ARRIVED_M = 0.30
COLLISION_M = 0.12


def rollout(flat: Flat, session, start, goal, *, expert: bool,
            max_steps: int = 500):
    x, y, yaw = start
    route = core.Wavefront(flat.field, clearance=flat.clearance)
    if not route.plan(goal):
        return None
    if not math.isfinite(route.cost_at((x, y))):
        return None
    linear = angular = 0.0
    travelled = 0.0
    for step in range(max_steps):
        remaining = math.hypot(goal[0] - x, goal[1] - y)
        if remaining <= ARRIVED_M:
            return {"outcome": "arrived", "steps": step,
                    "travelled_m": round(travelled, 2)}
        scan = flat.raycast(x, y, yaw + LASER_YAW_RAD)
        if float(np.min(scan)) < COLLISION_M:
            return {"outcome": "collision", "steps": step,
                    "travelled_m": round(travelled, 2)}
        path = descend(route, flat, x, y)
        if len(path) < 2:
            return {"outcome": "no_path", "steps": step,
                    "travelled_m": round(travelled, 2)}
        local = to_base_link(path, x, y, yaw)
        if expert:
            index = min(6, len(local) - 1)
            bearing = math.atan2(local[index][1], local[index][0])
            reach = math.hypot(local[index][0], local[index][1])
            target_linear, target_angular = expert_command(
                bearing, scan, remaining, reach)
        else:
            vector = build_features(scan, RANGE_MIN, RANGE_MAX, local,
                                    linear, angular, angle_min=0.0,
                                    angle_increment=2.0 * math.pi / BEAMS,
                                    laser_yaw=LASER_YAW_RAD)[None, :]
            prediction = session.run(["twist"], {"features": vector})[0][0]
            target_linear = float(np.clip(prediction[0], 0.0, 0.25))
            target_angular = float(np.clip(prediction[1], -0.25, 0.25))
        linear += (target_linear - linear) * 0.5
        angular += (target_angular - angular) * 0.6
        x += linear * math.cos(yaw) * DT
        y += linear * math.sin(yaw) * DT
        yaw = wrap(yaw + angular * DT)
        travelled += abs(linear) * DT
    return {"outcome": "timeout", "steps": max_steps,
            "travelled_m": round(travelled, 2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=404)
    args = parser.parse_args()

    flat = Flat(args.map)
    session = ort.InferenceSession(str(args.model),
                                   providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(args.seed)
    standable = np.argwhere(flat.standable)

    tally = {"policy": {}, "expert": {}}
    pairs = 0
    while pairs < args.trials:
        start_row, start_column = standable[rng.integers(len(standable))]
        goal_row, goal_column = standable[rng.integers(len(standable))]
        x, y = flat.metres(int(start_row), int(start_column))
        goal = flat.metres(int(goal_row), int(goal_column))
        if math.hypot(goal[0] - x, goal[1] - y) < 1.0:
            continue
        start = (x, y, float(rng.uniform(-math.pi, math.pi)))
        expert_run = rollout(flat, session, start, goal, expert=True)
        if expert_run is None:
            continue
        policy_run = rollout(flat, session, start, goal, expert=False)
        if policy_run is None:
            continue
        pairs += 1
        for name, run in (("expert", expert_run), ("policy", policy_run)):
            tally[name][run["outcome"]] = tally[name].get(run["outcome"], 0) + 1

    summary = {"trials": pairs}
    for name in ("expert", "policy"):
        arrived = tally[name].get("arrived", 0)
        summary[name] = {"outcomes": tally[name],
                         "arrival_rate": round(arrived / max(1, pairs), 3)}
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
