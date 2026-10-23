"""Routes on the clearance grid: A* with a soft wall cost, then shortcut smoothing.

A cell is drivable when its clearance (distance to the nearest blocked cell)
admits the body: BODY_RADIUS for normal driving, PASSAGE for a straight squeeze
through a doorway (half-width + margin). Doorways found on a route are marked so
the driver can cross them straight.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from ..common.geometry import BODY_RADIUS, HALF_WIDTH

PASSAGE = HALF_WIDTH + .027           # .17 m: the body's width, not its turning circle
DOOR_CLEARANCE_M = .46                 # a pinch narrower than ~0.9 m is an opening to square up for
DOOR_MAX_RUN_M = 1.0                   # longer narrow stretches are corridors, driven normally
DOOR_WIDTH_M = (.42, 1.15)
_STEPS = ((-1, 0, 1.), (1, 0, 1.), (0, -1, 1.), (0, 1, 1.),
          (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142))


class NoRoute(ValueError):
    pass


class Route(list):
    """Planned map points; `doors` are the openings on it: {center, axis, width}."""
    doors: tuple = ()


def plan(grid, start, goal, clearance=None, goal_tolerance=0.):
    clearance = grid.clearance if clearance is None else clearance
    src, dst = grid.cell(start), grid.cell(goal)
    if not grid.inside(src) or not grid.inside(dst):
        raise NoRoute('goal_outside_map')
    passable = clearance >= PASSAGE
    if clearance[dst] < BODY_RADIUS:
        if goal_tolerance <= 0:
            raise NoRoute('goal_has_no_body_clearance')
        dst = _nearest_clear(grid, clearance, dst, goal_tolerance)
        if dst is None:
            raise NoRoute('goal_has_no_body_clearance')
    # The body is physically where it is: let it leave its own cells.
    r0, c0 = src
    k = int(math.ceil(PASSAGE / grid.resolution)) + 1
    window = (slice(max(0, r0 - k), r0 + k + 1), slice(max(0, c0 - k), c0 + k + 1))
    passable = passable.copy()
    passable[window] |= ~grid.blocked[window]
    passable[src] = True
    # Soft cost: tight cells are expensive, so routes keep to the middle of free space.
    tight = np.clip((BODY_RADIUS + .10 - clearance) / (BODY_RADIUS + .10), 0., 1.)
    penalty = 1. + 8. * tight * tight
    width = grid.width
    flat_pass = passable.ravel()
    flat_pen = penalty.ravel()
    start_i, goal_i = src[0] * width + src[1], dst[0] * width + dst[1]
    gr, gc = dst
    cost = {start_i: 0.}
    parent = {}
    heap = [(0., 0., start_i)]
    while heap:
        _, g, node = heapq.heappop(heap)
        if node == goal_i:
            break
        if g > cost.get(node, math.inf):
            continue
        r, c = divmod(node, width)
        for dr, dc, step in _STEPS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= grid.height or nc >= width:
                continue
            nxt = nr * width + nc
            if not flat_pass[nxt]:
                continue
            if dr and dc and not (flat_pass[r * width + nc] and flat_pass[nr * width + c]):
                continue
            ng = g + step * flat_pen[nxt]
            if ng < cost.get(nxt, math.inf):
                cost[nxt] = ng
                parent[nxt] = node
                heapq.heappush(heap, (ng + math.hypot(gr - nr, gc - nc), ng, nxt))
    if goal_i not in cost:
        raise NoRoute('no_route')
    chain = [goal_i]
    while chain[-1] != start_i:
        chain.append(parent[chain[-1]])
    chain.reverse()
    cells = [divmod(i, width) for i in chain]
    keep = _shortcut(grid, clearance, cells, set(_doors(grid, clearance, cells)))
    end = grid.xy(dst) if goal_tolerance and dst != grid.cell(goal) else tuple(goal[:2])
    points = Route([tuple(start[:2])] + [grid.xy(cells[i]) for i in keep[1:-1]] + [end])
    points.doors = tuple(openings(grid, points, clearance))
    return points


def _nearest_clear(grid, clearance, cell, tolerance):
    k = int(math.ceil(tolerance / grid.resolution))
    r, c = cell
    window = clearance[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1]
    rows, cols = np.nonzero(window >= BODY_RADIUS)
    if not len(rows):
        return None
    rows = rows + max(0, r - k)
    cols = cols + max(0, c - k)
    i = int(np.argmin((rows - r) ** 2 + (cols - c) ** 2))
    return int(rows[i]), int(cols[i])


def _doors(grid, clearance, cells):
    """Tightest cell of every short narrow run (a doorway, not a corridor)."""
    doors, run = [], []
    for index, cell in enumerate(cells):
        tight = 0 < index < len(cells) - 1 and clearance[cell] < BODY_RADIUS
        if tight:
            run.append(index)
        if run and (not tight or index == len(cells) - 1):
            if len(run) * grid.resolution <= DOOR_MAX_RUN_M:
                doors.append(min(run, key=lambda i: clearance[cells[i]]))
            run = []
    return doors


def _line_clear(grid, clearance, a, b, need):
    n = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    if n == 0:
        return True
    rows = np.rint(np.linspace(a[0], b[0], n + 1)).astype(int)
    cols = np.rint(np.linspace(a[1], b[1], n + 1)).astype(int)
    return bool(np.all(clearance[rows, cols] >= need))


def _shortcut(grid, clearance, cells, doors):
    """Drop intermediate cells while the straight segment keeps the same clearance."""
    keep = [0]
    i = 0
    n = len(cells)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            if any(i < d < j for d in doors):
                j = min(d for d in doors if i < d < j)
                continue
            need = min(BODY_RADIUS + .05, min(clearance[cells[i]], clearance[cells[j]]))
            if j - i <= 40 and _line_clear(grid, clearance, cells[i], cells[j], need):
                break
            j -= 1
        keep.append(j)
        i = j
    return keep


def route_clear(grid, route, clearance=None):
    clearance = grid.clearance if clearance is None else clearance
    for a, b in zip(route, route[1:]):
        if not _line_clear(grid, clearance, grid.cell(a), grid.cell(b), PASSAGE):
            return False
    return True


def ring_goal(grid, start, target, distance, clearance=None):
    """Reachable point `distance` from target, nearest side first (people, objects)."""
    clearance = grid.clearance if clearance is None else clearance
    bearing = math.atan2(start[1] - target[1], start[0] - target[0])
    best = None
    for ring in (distance, distance + .15, distance + .3, distance + .5):
        for offset in (0, -.5, .5, -1., 1., -1.5, 1.5, math.pi):
            goal = (target[0] + ring * math.cos(bearing + offset), target[1] + ring * math.sin(bearing + offset))
            cell = grid.cell(goal)
            if not grid.inside(cell) or clearance[cell] < BODY_RADIUS:
                continue
            try:
                route = plan(grid, start, goal, clearance)
            except NoRoute:
                continue
            length = sum(math.dist(a, b) for a, b in zip(route, route[1:])) + .2 * abs(offset)
            if best is None or length < best[0]:
                best = (length, route)
        if best is not None:
            return best[1]
    raise NoRoute('no_reachable_viewpoint')


def openings(grid, route, clearance=None):
    """Doorways and arches on a route: centre, axis (direction of travel) and width.

    Found as short pinches of the clearance along the route; the two jambs are the
    nearest blocked cells on either side, so the centre is the true middle of the
    opening and the axis is perpendicular to the line between the jambs.
    """
    clearance = grid.clearance if clearance is None else clearance
    dense = []
    for a, b in zip(route, route[1:]):
        n = max(1, int(math.dist(a, b) / grid.resolution))
        for k in range(n):
            f = k / n
            dense.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
    dense.append(tuple(route[-1]))
    values = []
    for p in dense:
        c = grid.cell(p)
        values.append(float(clearance[c]) if grid.inside(c) else 0.)
    # Pinch points: the narrowest cell within +-0.5 m, inside a narrow stretch.
    found = []
    for i, v in enumerate(values):
        if v < DOOR_CLEARANCE_M and 3 < i < len(values) - 3 and v <= min(values[max(0, i - 10):i + 11]):
            if not found or i - found[-1] > 10:
                found.append(i)
    doors = []
    blocked = grid.blocked
    for i in found:
        point = dense[i]
        a, b = dense[max(0, i - 8)], dense[min(len(dense) - 1, i + 8)]
        travel = math.atan2(b[1] - a[1], b[0] - a[0])
        reach = values[i] + .15
        r0, c0 = grid.cell(point)
        k = int(math.ceil(reach / grid.resolution))
        window = blocked[max(0, r0 - k):r0 + k + 1, max(0, c0 - k):c0 + k + 1]
        rows, cols = np.nonzero(window)
        if not len(rows):
            continue
        xy = np.array([grid.xy((int(r) + max(0, r0 - k), int(c) + max(0, c0 - k))) for r, c in zip(rows, cols)])
        rel = xy - point
        side = math.cos(travel) * rel[:, 1] - math.sin(travel) * rel[:, 0]
        along = math.cos(travel) * rel[:, 0] + math.sin(travel) * rel[:, 1]
        distance = np.hypot(rel[:, 0], rel[:, 1])
        near = np.abs(along) < .35
        left, right = (side > 0) & near, (side < 0) & near
        if not left.any() or not right.any():
            continue
        jl = xy[left][np.argmin(distance[left])]
        jr = xy[right][np.argmin(distance[right])]
        width = float(np.hypot(*(jl - jr))) - grid.resolution
        if not DOOR_WIDTH_M[0] <= width <= DOOR_WIDTH_M[1]:
            continue
        center = tuple(((jl + jr) / 2).tolist())
        axis = math.atan2(jl[1] - jr[1], jl[0] - jr[0]) - math.pi / 2
        if math.cos(axis - travel) < 0:
            axis += math.pi
        axis = math.atan2(math.sin(axis), math.cos(axis))
        # A real doorway: thin jambs, open floor on both sides of it along the axis.
        def opened(sign):
            best = 0.
            for d in (.45, .6, .75):
                c = grid.cell((center[0] + sign * d * math.cos(axis), center[1] + sign * d * math.sin(axis)))
                if grid.inside(c):
                    best = max(best, float(clearance[c]))
            return best
        pinch = float(clearance[grid.cell(center)]) if grid.inside(grid.cell(center)) else values[i]
        # Open floor before the opening (where we square up); beyond it may turn at once.
        if opened(-1) < max(pinch * 1.35, pinch + .1) or opened(1) < pinch:
            continue
        if any(math.dist(center, d['center']) < .4 for d in doors):
            continue
        doors.append({'center': center, 'axis': axis, 'width': round(width, 2)})
    return doors
