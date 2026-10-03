"""Where the base must stand, how it gets there, and which camera can see from where - ROS-free.

THE GOAL IS INHERITED FROM THE STAND-UP, NOT INVENTED HERE
---------------------------------------------------------
The shared stand-up (flip_task.stand_up) is verified with the object REACH = 0.30 m dead ahead
of the base and its axis ACROSS the reach - tangential - within +/-20 deg. A base that can
only rotate (perceive) can put an object dead ahead but can never change the angle between the
object's axis and the line to it. Driving can. For an object at P with axis a, the base stands
at P - R (cos t, sin t) facing t = a +/- 90 deg: square to the object's side, on either side of
it. That is :func:`approach_poses`, and at R = REACH it is the park pose.

WHICH CAMERA SEES IT, FROM WHERE (measured: test_the_camera_ranges_are_what_the_approach_uses)
------------------------------------------------------------------------------------------
  mast camera       forward, 15 deg down. With the arms in the look pose the bottom of its
                    image is arm below row 330, so it sees a lying object WHOLE only from
                    FAR_MIN = 1.0 m out; it trusts what it sees to FAR_MAX = 2.4 m, past which
                    the lying/standing margin gets thin.
  workspace camera  71 deg down over the reach. It sees a lying object whole only inside
                    NEAR_MAX = 0.45 m, and a standing one inside 0.35 m.
Between 0.45 and 1.0 m NEITHER camera sees the object whole. So the approach is:

    LOOK   stop LOOK_REACH = 1.15 m out, square to its side: the mast camera's last clear view
    BLIND  drive to STAGE_REACH = 0.42 m on the wall fix alone - 0.73 m without seeing it
    STAGE  the workspace camera measures it precisely
    PARK   the last 0.12 m to REACH, then measure and correct until it is in the window

THE FOOTPRINT
-------------
Against OBJECTS the robot is its base, radius BASE_RADIUS: every part of the parked arms below
0.30 m - above the tallest object - is inside it (test_the_parked_arms_are_inside_the_base_
below_object_height). Against WALLS, which are taller than the hands, it is the parked arms'
reach, ARM_RADIUS. An object is a capsule: its axis as a segment (a point, standing) and its
half width as the radius.
"""
from __future__ import annotations

import heapq
import math

from . import collision as col

REACH = 0.30                 # the stand-up's verified station
STAGE_REACH = 0.42           # the workspace camera sees a lying object whole from here
LOOK_REACH = 1.15            # the mast camera's last whole view, with margin
FAR_MIN = 1.00               # mast camera: sees an object whole only beyond this ...
FAR_MAX = 2.40               # ... and trusts it only inside this
NEAR_MAX = 0.45              # workspace camera: a lying object is whole only inside this
NEAR_STANDING_MAX = 0.35     # ... and a standing one inside this

BASE_RADIUS = col.BASE_RADIUS     # 0.1775: the footprint against objects
ARM_RADIUS = 0.26                 # the parked arms' reach (0.251): the footprint against walls
OBJECT_CLEAR = 0.10          # in transit, base edge to any object's surface
WALL_CLEAR = 0.12            # in transit, the parked arms to any wall
STAND_CLEAR = 0.50           # an object this close to a wall has no room for the stand-up


def norm(a):
    return math.remainder(a, 2 * math.pi)


# ------------------------------------------------------------------ geometry
def seg_point(p, a, b):
    """Distance from point p to segment ab, in the plane."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    L2 = ex * ex + ey * ey
    t = 0.0 if L2 < 1e-18 else max(0.0, min(1.0, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / L2))
    return math.hypot(p[0] - a[0] - t * ex, p[1] - a[1] - t * ey)


def seg_seg(a, b, c, d):
    """Distance between segments ab and cd, in the plane."""
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1, d2 = cross(a, b, c), cross(a, b, d)
    d3, d4 = cross(c, d, a), cross(c, d, b)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 * d2 != 0 and d3 * d4 != 0:
        return 0.0
    return min(seg_point(a, c, d), seg_point(b, c, d), seg_point(c, a, b), seg_point(d, a, b))


def footprint(x, y, axis, upright, length, half_width):
    """An object on the floor as ``(end_a, end_b, radius)``: its axis, or a point standing."""
    if upright:
        return ((x, y), (x, y), half_width)
    c, s = 0.5 * length * math.cos(axis), 0.5 * length * math.sin(axis)
    return ((x - c, y - s), (x + c, y + s), half_width)


def approach_poses(obj_xy, axis, reach):
    """The two base poses that put the object ``reach`` dead ahead, its axis across the reach.

    ``[(x, y, heading), (x, y, heading)]`` - one either side of it, each facing it.
    """
    out = []
    for sgn in (+1.0, -1.0):
        t = norm(axis + sgn * math.pi / 2.0)
        out.append((obj_xy[0] - reach * math.cos(t), obj_xy[1] - reach * math.sin(t), t))
    return out


def point_clearance(p, obstacles, room, skip=()):
    """``(object gap, wall gap, nearest object)`` for the base centred at p.

    Object gap: base edge to the nearest object's surface. Wall gap: the parked arms' reach to
    the nearest wall face.
    """
    g, who = float('inf'), None
    for name, (a, b, r) in obstacles.items():
        if name in skip:
            continue
        d = seg_point(p, a, b) - r - BASE_RADIUS
        if d < g:
            g, who = d, name
    return g, room.wall_distance(p) - ARM_RADIUS, who


def segment_clearance(p, q, obstacles, room, skip=()):
    """point_clearance() over the whole straight drive from p to q (walls: a half-plane's
    distance is linear along a segment, so its minimum is at an end)."""
    g, who = float('inf'), None
    for name, (a, b, r) in obstacles.items():
        if name in skip:
            continue
        d = seg_seg(p, q, a, b) - r - BASE_RADIUS
        if d < g:
            g, who = d, name
    return g, min(room.wall_distance(p), room.wall_distance(q)) - ARM_RADIUS, who


def _ring(obstacles, room, skip, inflate, n=16):
    """Candidate waypoints: a ring round every obstacle, just outside its inflated capsule."""
    pts = []
    for name, (a, b, r) in obstacles.items():
        if name in skip:
            continue
        R = r + BASE_RADIUS + inflate
        for end in (a, b):
            for k in range(n):
                t = 2 * math.pi * k / n
                p = (end[0] + R * math.cos(t), end[1] + R * math.sin(t))
                if seg_point(p, a, b) >= R - 1e-9:
                    pts.append(p)
    return pts


def plan_route(start, goal, obstacles, room, skip=(), clear=OBJECT_CLEAR, wall_clear=WALL_CLEAR):
    """The shortest route of straight legs from start to goal, every leg clear of every object
    (base edge by ``clear``) and wall (parked arms by ``wall_clear``). A list of points from
    ``start`` to ``goal``, or None.

    A visibility graph over rings of points round each obstacle - there are three objects and
    four walls, so this is exact enough and costs nothing. A START inside an obstacle's margin
    (the robot has just backed up from one) may leave by any leg that does not get closer.
    """
    s_gap, s_wall, _ = point_clearance(start, obstacles, room, skip)
    g_gap, g_wall, g_who = point_clearance(goal, obstacles, room, skip)
    if g_gap < clear or g_wall < wall_clear:
        return None
    nodes = [tuple(start), tuple(goal)] + [
        p for p in _ring(obstacles, room, skip, clear + 0.03)
        if point_clearance(p, obstacles, room, skip)[0] >= clear
        and point_clearance(p, obstacles, room, skip)[1] >= wall_clear]

    def ok(i, j):
        gap, wall, _ = segment_clearance(nodes[i], nodes[j], obstacles, room, skip)
        if i == 0 or j == 0:
            return gap >= min(clear, s_gap) - 1e-9 and wall >= min(wall_clear, s_wall) - 1e-9
        return gap >= clear and wall >= wall_clear
    dist = {0: 0.0}
    prev = {}
    heap = [(0.0, 0)]
    done = set()
    while heap:
        d, i = heapq.heappop(heap)
        if i in done:
            continue
        done.add(i)
        if i == 1:
            break
        for j in range(len(nodes)):
            if j in done or j == i:
                continue
            nd = d + math.dist(nodes[i], nodes[j])
            if nd < dist.get(j, float('inf')) and ok(i, j):
                dist[j] = nd
                prev[j] = i
                heapq.heappush(heap, (nd, j))
    if 1 not in done:
        return None
    idx = [1]
    while idx[-1] != 0:
        idx.append(prev[idx[-1]])
    return [nodes[i] for i in reversed(idx)]


def route_length(route):
    return sum(math.dist(a, b) for a, b in zip(route, route[1:]))


def route_clearance(route, obstacles, room, skip=()):
    """The tightest object and wall gaps along a route, and the object that sets the first."""
    g, w, who = float('inf'), float('inf'), None
    for a, b in zip(route, route[1:]):
        gg, ww, wh = segment_clearance(a, b, obstacles, room, skip)
        if gg < g:
            g, who = gg, wh
        w = min(w, ww)
    return g, w, who


def plan_approach(name, obj, robot_xy, obstacles, room):
    """Choose the side to stand up ``name`` from, and the route there.

    ``obj`` is ``(x, y, axis)``. For each side: the park pose must leave the base clear of every
    OTHER object and the walls; the staging pose likewise, and reachable by a route that keeps
    clear of ``name`` too; the look pose is used if it is clear and reachable, and skipped
    otherwise. The object itself must be STAND_CLEAR from every wall. Returns the cheaper side
    as a dict - ``side``, ``look`` (or None), ``stage``, ``park``, ``route`` (to the look pose,
    else to the stage) and ``length`` - or ``(None, reason)``.
    """
    x, y, axis = obj
    if room.wall_distance((x, y)) < STAND_CLEAR:
        return None, (f'it lies {room.wall_distance((x, y)):.2f} m from a wall - no room to '
                      f'stand it up')
    others = tuple(n for n in obstacles if n != name)
    best, why = None, []
    parks = approach_poses((x, y), axis, REACH)
    stages = approach_poses((x, y), axis, STAGE_REACH)
    looks = approach_poses((x, y), axis, LOOK_REACH)
    for k in range(2):
        park, stage, look = parks[k], stages[k], looks[k]
        gap, wall, who = point_clearance(park[:2], obstacles, room, skip=(name,))
        if gap < OBJECT_CLEAR or wall < WALL_CLEAR:
            why.append(f'side {k + 1}: the park pose is '
                       + (f'{gap * 1000:.0f} mm from the {who}' if gap < OBJECT_CLEAR
                          else f'{wall * 1000:.0f} mm from a wall'))
            continue
        gap, wall, who = point_clearance(stage[:2], obstacles, room)
        if gap < OBJECT_CLEAR or wall < WALL_CLEAR:
            why.append(f'side {k + 1}: the staging pose is not clear')
            continue
        lk = look
        route = None
        if lk is not None:
            g2, w2, _ = point_clearance(lk[:2], obstacles, room)
            leg = segment_clearance(lk[:2], stage[:2], obstacles, room)
            if g2 >= OBJECT_CLEAR and w2 >= WALL_CLEAR and leg[0] >= OBJECT_CLEAR \
                    and leg[1] >= WALL_CLEAR:
                route = plan_route(robot_xy, lk[:2], obstacles, room)
            if route is None:
                lk = None
        if route is None:
            route = plan_route(robot_xy, stage[:2], obstacles, room)
        if route is None:
            why.append(f'side {k + 1}: no clear route')
            continue
        length = route_length(route) + (math.dist(lk[:2], stage[:2]) if lk else 0.0)
        cand = {'side': k + 1, 'look': lk, 'stage': stage, 'park': park, 'route': route,
                'length': length, 'others': others}
        if best is None or length < best['length']:
            best = cand
    if best is None:
        return None, '; '.join(why) or 'no approach'
    return best, ''


# ------------------------------------------------------------------ driving
def drive_command(pose, goal, v_max, v_min=0.02, w_max=0.6, k_v=0.8, k_w=2.5,
                  reverse=False, tol=0.0015):
    """One step of driving to a point: ``(v, w, along, lateral)``; ``v == 0`` means arrived.

    ``along`` is what is left in the direction of travel, ``lateral`` the goal's offset across
    it. The speed falls with ``along`` to ``v_min``, so the stop needs no braking distance; the
    turn rate steers onto the goal while there is still distance to correct over, and not in the
    last few centimetres, where the bearing to a point is noise.
    """
    x, y, h = pose
    if reverse:
        h = norm(h + math.pi)
    dx, dy = goal[0] - x, goal[1] - y
    along = dx * math.cos(h) + dy * math.sin(h)
    lateral = -dx * math.sin(h) + dy * math.cos(h)
    if along <= tol:
        return 0.0, 0.0, along, lateral
    v = max(v_min, min(v_max, k_v * along))
    w = 0.0
    if math.hypot(dx, dy) > 0.05:
        w = max(-w_max, min(w_max, k_w * math.atan2(lateral, along)))
    return (-v if reverse else v), w, along, lateral


def park_error(det_x, det_y, det_axis, reach=REACH, lateral_axis=math.pi / 2.0):
    """How far a MEASURED object (base frame) is from the park: (reach error, bearing, axis dev)."""
    r = math.hypot(det_x, det_y)
    return (r - reach, math.atan2(det_y, det_x),
            math.remainder(det_axis - lateral_axis, math.pi))
