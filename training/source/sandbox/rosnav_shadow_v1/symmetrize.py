#!/usr/bin/env python3
"""Remove the policy's handedness by construction rather than by training.

The first policy drove a standing right-hand arc.  The one trained by
reinforcement leans the other way -- told to follow a straight path it turns
left at about 3.5 degrees a second.  That is not surprising: it was rewarded
for reaching goals in one particular flat, never for being even-handed, and
the flat is not symmetric.

Averaging the network with its own reflection fixes it exactly.  Write M for
mirroring an observation and L for mirroring a command -- both are their own
inverse -- and define

    g(x) = (f(x) + L f(Mx)) / 2

Then g(Mx) = L g(x) identically, for every input, whatever f happens to have
learned.  Not a smaller bias: no bias, as an algebraic property of the wrapper
rather than a number that came out well on the day.

It costs a second forward pass, which on this network is 0.03 ms, and it is
folded into every deployable representation so the runtime sees the canonical
temporal feature vector in and one twist out.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from features import FEATURE_DIM, LIDAR_BINS, PATH_POINTS, SCAN_VIEWS


def mirror_plan() -> tuple:
    """The permutation and signs that reflect an observation about forward.

    Sector zero straddles straight ahead, so it is its own reflection and the
    others reverse around it.  A path point reflects by negating its lateral
    offset, and a rotation reflects by changing sign.  This mirrors exactly
    what ``features.mirror_features`` does, in a form the ONNX exporter can
    follow.
    """
    order = np.arange(FEATURE_DIM)
    one_scan = np.concatenate(
        ([0], np.arange(LIDAR_BINS - 1, 0, -1)))
    for view in range(SCAN_VIEWS):
        start = view * LIDAR_BINS
        order[start:start + LIDAR_BINS] = start + one_scan
    sign = np.ones(FEATURE_DIM, dtype=np.float32)
    start = LIDAR_BINS * SCAN_VIEWS
    for point in range(PATH_POINTS):
        sign[start + 2 * point + 1] = -1.0        # lateral offset
    sign[-1] = -1.0                               # angular velocity
    return order.astype(np.int64), sign


class SymmetricActor(nn.Module):
    """The trained policy, made exactly even-handed."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor
        order, sign = mirror_plan()
        self.register_buffer("order", torch.from_numpy(order))
        self.register_buffer("sign", torch.from_numpy(sign))
        self.register_buffer("flip", torch.tensor([1.0, -1.0]))

    def forward(self, features):
        mirrored = features.index_select(-1, self.order) * self.sign
        return 0.5 * (self.actor(features)
                      + self.actor(mirrored) * self.flip)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Lazy by design: train_rl imports SymmetricActor so selection and final
    # promotion exercise the exact wrapper deployed by Rust.  Keeping the
    # concrete Actor import in the CLI path avoids a module cycle.
    from train_rl import Actor

    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    actor = Actor(state["feature_mean"].numpy(), state["feature_std"].numpy())
    actor.load_state_dict(state)
    actor.eval()
    model = SymmetricActor(actor).eval()

    # Prove the property on the way out rather than asserting it in prose.
    probe = torch.from_numpy(
        np.random.default_rng(0).standard_normal((512, FEATURE_DIM)
                                                 ).astype(np.float32))
    order, sign = mirror_plan()
    reflected = probe[:, order] * torch.from_numpy(sign)
    with torch.inference_mode():
        direct = model(probe)
        through_mirror = model(reflected) * torch.tensor([1.0, -1.0])
    error = float((direct - through_mirror).abs().max())
    if error > 1e-5:
        raise SystemExit(f"the wrapper is not symmetric: {error}")

    torch.onnx.export(
        model, torch.zeros(1, FEATURE_DIM), str(args.output),
        input_names=["features"], output_names=["twist"],
        dynamic_axes={"features": {0: "batch"}, "twist": {0: "batch"}},
        opset_version=17, dynamo=False)
    print(f"symmetry error {error:.2e}, written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
