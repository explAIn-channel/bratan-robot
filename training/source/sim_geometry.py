"""Offline training helpers only; extracted from the original training snapshot."""

from __future__ import annotations
import math
import time
TURN_COAST_S = 0.55

def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))

def guard_polygon_edges(points):
    """Validate a convex CCW base_link contour and return outward unit planes."""
    if (not isinstance(points, list) or not 3 <= len(points) <= 16 or
            any(not isinstance(p, list) or len(p) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) or
                    abs(v) > .5 for v in p) for p in points)):
        raise ValueError("invalid_guard_footprint")
    edges = []
    for a, b in zip(points, points[1:] + points[:1]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if length < .001:
            raise ValueError("invalid_guard_footprint")
        nx, ny = dy/length, -dx/length
        offset = nx*a[0] + ny*a[1]
        if offset < .05 or any(nx*p[0]+ny*p[1] > offset+1e-9 for p in points):
            raise ValueError("invalid_guard_footprint")
        edges.append((nx, ny, offset))
    return edges

class LocalMap:

    def __init__(self, robot, merge=0.12, forget_s=25.0, touched_forget_s=20.0, max_points=60):
        self.robot = robot
        self.merge = merge
        self.forget_s = forget_s
        self.touched_forget_s = touched_forget_s
        self.points = []
        self.overflow = False
        self.max_points = max_points

    def mark(self, x, y, reach, now, touched=False, source='lidar', radius=0.0, height_m=None):
        source = 'contact' if touched else source
        if source not in {'lidar', 'contact', 'camera'}:
            raise ValueError('invalid_obstacle_source')
        for known in self.points:
            known_source = known.get('source', 'contact' if known.get('touched') else 'lidar')
            if known_source == source and math.hypot(x - known['xy'][0], y - known['xy'][1]) <= self.merge:
                (old_x, old_y) = known['xy']
                distance = math.hypot(x - old_x, y - old_y)
                old_radius = known.get('radius', 0.0)
                if source != 'camera':
                    if reach < known['range']:
                        known['xy'] = (x, y)
                    known['radius'] = max(radius, old_radius)
                elif radius >= distance + old_radius:
                    (known['xy'], known['radius']) = ((x, y), radius)
                elif old_radius < distance + radius:
                    combined = (distance + old_radius + radius) / 2
                    shift = (combined - old_radius) / distance
                    known['xy'] = (old_x + shift * (x - old_x), old_y + shift * (y - old_y))
                    known['radius'] = combined
                known['range'] = min(reach, known['range'])
                known['seen'] = now
                if height_m is not None:
                    old_height = known.get('height_m', height_m)
                    known['height_m'] = [min(old_height[0], height_m[0]), max(old_height[1], height_m[1])]
                known['touched'] = known.get('touched', False) or touched
                return
        if source == 'camera' and len(self.points) >= self.max_points:
            self.overflow = True
            return
        self.points.append({'xy': (x, y), 'range': reach, 'seen': now, 'touched': touched, 'source': source, 'radius': radius})
        if height_m is not None:
            self.points[-1]['height_m'] = list(height_m)
        while len(self.points) > self.max_points:
            disposable = next((p for p in self.points if p.get('source') != 'camera'), None)
            if disposable is None:
                self.overflow = True
                break
            self.points.remove(disposable)

    def _clear(self, pose, now, geometry=None):
        robot = self.robot
        numpy = robot.numpy
        if geometry is None:
            geometry = getattr(robot, 'scan_points_base', lambda : None)()
        if geometry is not None:
            (scan_x, scan_y, radii) = geometry[:3]
            (angles, ranges) = (numpy.arctan2(scan_y, scan_x), numpy.hypot(scan_x, scan_y))
        kept = []
        for known in self.points:
            if known.get('source') == 'camera':
                kept.append(known)
                continue
            if known.get('touched'):
                if now - known['seen'] <= self.touched_forget_s:
                    kept.append(known)
                continue
            if now - known['seen'] > self.forget_s:
                continue
            if geometry is None or numpy.any(radii > 0.0):
                kept.append(known)
                continue
            dx = known['xy'][0] - pose[0]
            dy = known['xy'][1] - pose[1]
            span = math.hypot(dx, dy)
            bearing = wrap(math.atan2(dy, dx) - pose[2])
            offset = numpy.abs(numpy.arctan2(numpy.sin(angles - bearing), numpy.cos(angles - bearing)))
            near = offset <= math.radians(1.5)
            if near.any() and float(ranges[near].min()) > span + 0.2:
                continue
            kept.append(known)
        self.points = kept

class Wavefront:
    """Distance-to-goal over the whole map, respecting walls.

    Local steering alone is greedy: it picks the most open heading that
    points at the goal, which is right in an open room and wrong in every
    doorway and concave corner, where the way out briefly leads away.  The
    stock firmware answers this with A* plus D* -- plan, then replan when
    blocked.

    Flooding outward
    from the goal labels every reachable cell with its travel cost, so any
    cell already knows which way to go, and when local avoidance shoves the
    robot off the line nothing has to be recomputed. New or expired local
    obstacles change the traversable cells and trigger a new flood.
    """

    def __init__(self, distance_map, clearance=0.20):
        self.map = distance_map
        self.clearance = clearance
        self.free = (distance_map.field >= clearance) & distance_map.free
        self.obstacle_cells = set()
        self.cost = None
        self.goal = None

    def update_obstacles(self, points):
        """Overlay temporary observations without changing the saved map."""
        import numpy
        from scipy.ndimage import distance_transform_edt

        cells = {(*self._cell(point["xy"]),
                  int(math.ceil(point.get("radius", 0.) / self.map.resolution)))
                 for point in points}
        cells = {(r, c, radius) for r, c, radius in cells
                 if 0 <= r < self.map.height and 0 <= c < self.map.width}
        if cells == self.obstacle_cells:
            return False
        free = (self.map.field >= self.clearance) & self.map.free
        if cells:
            observed_free = numpy.ones_like(free)
            for row, column, radius in cells:
                y0, y1 = max(0, row-radius), min(self.map.height, row+radius+1)
                x0, x1 = max(0, column-radius), min(self.map.width, column+radius+1)
                yy, xx = numpy.ogrid[y0:y1, x0:x1]
                observed_free[y0:y1, x0:x1] &= ((yy-row)**2 + (xx-column)**2 > radius**2)
            margin = distance_transform_edt(observed_free) * self.map.resolution
            free = free & (margin >= self.clearance) & observed_free
        self.free, self.obstacle_cells = free, cells
        self.cost = None
        return True

    def _cell(self, xy):
        column = int(round((xy[0] - self.map.origin[0]) / self.map.resolution))
        row = int(round(self.map.height - 1.0
                        - (xy[1] - self.map.origin[1]) / self.map.resolution))
        return row, column

    def plan(self, goal_xy, limit=4000, goal_snap_max_m=0.50):
        import numpy
        from scipy.ndimage import binary_dilation

        row, column = self._cell(goal_xy)
        if not (0 <= row < self.map.height and 0 <= column < self.map.width):
            return False
        if not self.free[row, column]:
            # A goal inside the inflated band is normal -- people ask for
            # the dock, which sits against a wall.  Accept the nearest cell
            # the body actually fits in.
            rows, columns = numpy.nonzero(self.free)
            if not len(rows):
                return False
            index = int(numpy.argmin((rows - row) ** 2 + (columns - column) ** 2))
            snapped_row, snapped_column = int(rows[index]), int(columns[index])
            snap_m = math.hypot(snapped_row - row,
                                snapped_column - column) * self.map.resolution
            if snap_m > goal_snap_max_m:
                return False
            row, column = snapped_row, snapped_column
        reached = numpy.zeros_like(self.free)
        reached[row, column] = True
        cost = numpy.full(self.free.shape, numpy.inf, dtype=numpy.float32)
        cost[row, column] = 0.0
        straight = numpy.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        diagonal = numpy.ones((3, 3), dtype=bool)
        step = 0
        while step < limit:
            structure = straight if step % 2 else diagonal
            grown = binary_dilation(reached, structure=structure) & self.free
            fresh = grown & ~reached
            if not fresh.any():
                break
            step += 1
            cost[fresh] = step * self.map.resolution * 1.2
            reached = grown
        self.cost = cost
        self.goal = goal_xy
        return True

    def path(self, x, y, points=30):
        """The same neighbouring-cell descent used for training observations."""
        if self.cost is None:
            return []
        row, column = self._cell((x, y))
        path = []
        for _ in range(points):
            best = None
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r, c = row + dr, column + dc
                    if (dr == dc == 0 or not 0 <= r < self.map.height
                            or not 0 <= c < self.map.width):
                        continue
                    value = float(self.cost[r, c])
                    if math.isfinite(value) and (best is None or value < best[0]):
                        best = value, r, c
            if best is None:
                break
            value, row, column = best
            path.append((self.map.origin[0] + column*self.map.resolution,
                         self.map.origin[1] + (self.map.height-1-row)*self.map.resolution))
            if value <= 0.:
                break
        return path

    def cost_at(self, xy, look=7):
        """Distance to the goal from ``xy``, tolerating a blocked own cell.

        The clearance dilation is about where the robot may *travel*, not
        about where it happens to be standing.  Parked on its dock the robot
        sits inside that margin -- measured, the dock cell only comes free
        below 0.14 m -- so a literal lookup answers "unreachable" and every
        route out of the dock is refused before a wheel turns.  ``to_goal``
        already steers by the best cell on a ring around the robot; this
        reports the same thing, so a plan and the drive that follows it
        agree.
        """
        if self.cost is None:
            return math.inf
        row, column = self._cell(xy)
        if not (0 <= row < self.map.height and 0 <= column < self.map.width):
            return math.inf
        here = float(self.cost[row, column])
        if math.isfinite(here) or look <= 0:
            return here
        low, high = max(0, row - look), min(self.map.height, row + look + 1)
        left, right = max(0, column - look), min(self.map.width, column + look + 1)
        window = self.cost[low:high, left:right]
        best = float(window.min()) if window.size else math.inf
        return best

    def to_goal(self, pose, look=6):
        """Descend the field: the map-frame heading that gets closer.

        ``look`` is in cells; sampling a ring rather than the immediate
        neighbours keeps the heading steady instead of chattering between
        two equal-cost cells.
        """
        if self.cost is None:
            return None
        row, column = self._cell(pose)
        inside = (0 <= row < self.map.height and 0 <= column < self.map.width)
        here = self.cost[row, column] if inside else math.inf
        best, best_cost = None, None
        for degrees in range(0, 360, 10):
            angle = math.radians(degrees)
            probe_row = int(round(row - look * math.sin(angle)))
            probe_column = int(round(column + look * math.cos(angle)))
            if not (0 <= probe_row < self.map.height
                    and 0 <= probe_column < self.map.width):
                continue
            value = float(self.cost[probe_row, probe_column])
            if not math.isfinite(value):
                continue
            if best_cost is None or value < best_cost:
                best, best_cost = angle, value
        if best is None or (math.isfinite(here) and best_cost >= here):
            return None
        return best
