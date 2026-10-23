"""The apartment map: structural walls, live occupancy evidence, clearance.

Map frame: metres, x right, y up; cell (row, col) with row 0 at the top (PGM).
Evidence per cell: structural walls start at +5, known floor at -5. A lidar ray
lowers every cell it crosses (-2, floor -8) and raises its end cell (+3, max 8),
so moved furniture disappears and new furniture appears within a few scans.
Localization uses only the structural walls (stable when furniture moves).
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from ..common.geometry import BODY_RADIUS, transform

VISIT_HALF_LIFE_S = 1800.
FIELD_CAP = .5


def read_map_yaml(path):
    values = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            values[key.strip()] = value.strip()
    origin = [float(v) for v in values['origin'].strip('[]').split(',')]
    return Path(path).parent / values['image'], float(values['resolution']), origin


class Grid:
    def __init__(self, map_yaml):
        image, self.resolution, origin = read_map_yaml(map_yaml)
        self.pixels = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if self.pixels is None:
            raise ValueError(f'map_unreadable:{image}')
        self.origin = (origin[0], origin[1])
        self.height, self.width = self.pixels.shape
        self.frame_id = hashlib.sha256(self.pixels.tobytes() + repr(
            (self.pixels.shape, self.resolution, self.origin)).encode()).hexdigest()[:16]
        self.structural = self.pixels <= 65
        # Lidar returns hug the walls they hit: evidence this close to a wall is the wall itself
        # and must not narrow doorways (06.10: the 0.51 m kitchen door "grew shut").
        from scipy.ndimage import binary_dilation
        self.near_wall = binary_dilation(self.structural, iterations=2) & ~self.structural
        known = (self.pixels >= 250) | self.structural
        self.evidence = np.where(self.structural, 5, np.where(known, -5, 0)).astype(np.int8)
        self.known = known
        self.visual = np.zeros(self.pixels.shape, dtype=bool)
        self.visited = np.zeros(self.pixels.shape, dtype=np.uint16)
        self.visit_decay_at = time.time()
        self.revision = 0
        # Localization field (metres to the nearest structural wall) and its gradient.
        self.field = np.minimum(distance_transform_edt(~self.structural) * self.resolution,
                                FIELD_CAP).astype(np.float32)
        gr, gc = np.gradient(self.field)
        self.field_gx = (gc / self.resolution).astype(np.float32)
        self.field_gy = (-gr / self.resolution).astype(np.float32)
        self._blocked = None
        self._clearance = None

    # ------------------------------------------------------------------ cells
    def cell(self, xy):
        col = int(math.floor((xy[0] - self.origin[0]) / self.resolution))
        row = self.height - 1 - int(math.floor((xy[1] - self.origin[1]) / self.resolution))
        return row, col

    def cells(self, xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        cols = np.floor((xy[:, 0] - self.origin[0]) / self.resolution).astype(int)
        rows = self.height - 1 - np.floor((xy[:, 1] - self.origin[1]) / self.resolution).astype(int)
        return rows, cols

    def xy(self, cell):
        return (self.origin[0] + (cell[1] + .5) * self.resolution,
                self.origin[1] + (self.height - cell[0] - .5) * self.resolution)

    def inside(self, cell):
        return 0 <= cell[0] < self.height and 0 <= cell[1] < self.width

    # ------------------------------------------------------- field sampling
    def sample(self, world):
        """Bilinear field value and gradient at (N, 2) world points."""
        fc = (world[:, 0] - self.origin[0]) / self.resolution - .5
        fr = self.height - .5 - (world[:, 1] - self.origin[1]) / self.resolution
        c0 = np.floor(fc).astype(int)
        r0 = np.floor(fr).astype(int)
        inside = (r0 >= 0) & (c0 >= 0) & (r0 < self.height - 1) & (c0 < self.width - 1)
        r0c = np.clip(r0, 0, self.height - 2)
        c0c = np.clip(c0, 0, self.width - 2)
        a = fc - c0c
        b = fr - r0c

        def bilinear(grid):
            v00 = grid[r0c, c0c]
            v01 = grid[r0c, c0c + 1]
            v10 = grid[r0c + 1, c0c]
            v11 = grid[r0c + 1, c0c + 1]
            return (v00 * (1 - a) + v01 * a) * (1 - b) + (v10 * (1 - a) + v11 * a) * b

        value = np.where(inside, bilinear(self.field), FIELD_CAP)
        gx = np.where(inside, bilinear(self.field_gx), 0.)
        gy = np.where(inside, bilinear(self.field_gy), 0.)
        return value, gx, gy

    # ------------------------------------------------------------- evidence
    def observe(self, pose, scan_points):
        """Ray-clear free space and mark hits from one localized scan."""
        start = self.cell(pose)
        if not self.inside(start) or not len(scan_points):
            return
        hits = transform(pose, scan_points)
        rows, cols = self.cells(hits)
        dr, dc = rows - start[0], cols - start[1]
        steps = np.maximum(np.abs(dr), np.abs(dc))
        keep = steps >= 1
        rows, cols, dr, dc, steps = rows[keep], cols[keep], dr[keep], dc[keep], steps[keep]
        if not len(steps):
            return
        ray = np.repeat(np.arange(len(steps)), steps)          # each ray, excluding its end cell
        offset = np.arange(len(ray)) - np.repeat(np.cumsum(steps) - steps, steps)
        fraction = offset / steps[ray]
        fr = np.rint(start[0] + fraction * dr[ray]).astype(int)
        fcol = np.rint(start[1] + fraction * dc[ray]).astype(int)
        valid = (fr >= 0) & (fr < self.height) & (fcol >= 0) & (fcol < self.width)
        free = np.unique(fr[valid] * self.width + fcol[valid])
        hit_ok = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        occupied = np.unique(rows[hit_ok] * self.width + cols[hit_ok])
        free = np.setdiff1d(free, occupied, assume_unique=True)
        flat = self.evidence.reshape(-1)
        before = (flat[free] > 0).any() or (flat[occupied] <= 0).any() or \
            (~self.known.reshape(-1)[free]).any() or (~self.known.reshape(-1)[occupied]).any()
        flat[free] = np.maximum(-8, flat[free] - 2)
        flat[occupied] = np.minimum(8, flat[occupied] + 3)
        known = self.known.reshape(-1)
        known[free] = True
        known[occupied] = True
        self.visited[start] = min(65535, int(self.visited[start]) + 1)
        if before:
            self._invalidate()

    def set_visual(self, discs):
        """Camera-triangulated obstacles as (x, y, radius) map discs."""
        mask = np.zeros_like(self.visual)
        for x, y, radius in discs:
            r, c = self.cell((x, y))
            k = max(1, int(math.ceil(radius / self.resolution)))
            if self.inside((r, c)):
                cv2.circle(mask.view(np.uint8), (c, r), k, 1, -1)
        if not np.array_equal(mask, self.visual):
            self.visual = mask
            self._invalidate()

    def _invalidate(self):
        self._blocked = None
        self._clearance = None
        self.revision += 1

    @property
    def blocked(self):
        if self._blocked is None:
            self._blocked = self.structural | ((self.evidence > 0) & ~self.near_wall) | ~self.known | self.visual
        return self._blocked

    @property
    def clearance(self):
        """Metres from each cell to the nearest blocked cell."""
        if self._clearance is None:
            self._clearance = (distance_transform_edt(~self.blocked) * self.resolution).astype(np.float32)
        return self._clearance

    def body_fits(self, xy, margin=0.):
        cell = self.cell(xy)
        return self.inside(cell) and self.clearance[cell] >= BODY_RADIUS + margin

    def decay_visits(self, now=None):
        now = time.time() if now is None else now
        steps = int((now - self.visit_decay_at) // VISIT_HALF_LIFE_S)
        if steps > 0:
            self.visited >>= min(16, steps)
            self.visit_decay_at += steps * VISIT_HALF_LIFE_S
