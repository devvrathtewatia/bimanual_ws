"""Link-level collision checking — the geometry the earlier tests never modelled.

WHY THIS EXISTS
---------------
Every offline test in this project used to treat a gripper as a *point*: it checked
that the two grasp points were reachable and more than 50 mm apart. But a hand is a
body 140 mm across with fingers 75 mm long, and an arm is two 160 mm links with real
thickness. A point model cannot see:

  * the two arms' forearms passing through each other on the way to two grasp points
    that are themselves comfortably separated,
  * the fingers reaching below the floor when grasping something lying on the ground,
  * either arm sweeping through the robot's own base.

All three happened in simulation while 54 point-based checks passed. This module
replaces the point model with a **conservative body model** and is now what the
choreography is validated against.

REPRESENTATION
--------------
Every link is one or more **capsules**: a line segment plus a radius. A capsule that
encloses the real box or cylinder is used, so the check is conservative — it can
report a collision that the true geometry would just avoid, but it can never miss a
real one. That is the correct direction to err.

Distances are reported as **signed clearance**: positive means a gap, negative means
interpenetration by that amount.
"""
from __future__ import annotations

import math

from . import ik

# ----------------------------------------------------------------- robot dims
# These mirror robot.urdf.xacro / arm.xacro / base.xacro and MUST be kept in step
# with them. The test suite asserts the grasp point derived from this chain agrees
# with ik.fk(), which catches any drift in the kinematic values.
BASE_RADIUS = 0.1775
DECK_TOP_Z = 0.140
BASE_BODY_TOP = 0.125
SHOULDER_X = ik.SHOULDER_X
SHOULDER_Y = ik.SHOULDER_Y
SHOULDER_Z = ik.SHOULDER_Z
L1 = ik.L1
L2 = ik.L2
LG = ik.LG

MAST_X = -0.100
MAST_R = 0.030
# The mast is TWO rods at y = +/-0.099, not one on the centreline. Modelling a
# single central cylinder left the real posts invisible to this checker.
MAST_Y = 0.099
MAST_TOP = 0.695          # follows lidar_z = 0.68


def _half_diag(a: float, b: float) -> float:
    """Radius of a capsule enclosing a rectangular cross-section a x b."""
    return math.hypot(a / 2.0, b / 2.0)


# Per-link capsules, expressed in that link's own frame as
# (name, p0, p1, radius). Each encloses the collision geometry in arm.xacro.
LINK_CAPSULES = {
    'shoulder': [('riser', (0.0, 0.0, 0.0), (0.0, 0.0, 0.10), 0.020)],
    'upper_arm': [('upper', (0.0, 0.0, 0.0), (L1, 0.0, 0.0),
                   _half_diag(0.045, 0.032))],
    'forearm': [('fore', (0.0, 0.0, 0.0), (L2, 0.0, 0.0),
                 _half_diag(0.038, 0.020))],
    'wrist_roll': [('wroll', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.020)],
    'wrist_pitch': [('wpitch', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.020)],
    # The palm plate is WIDE across the finger-travel axis, so its capsule runs
    # along local y - not along the approach axis. Getting this axis wrong is what
    # made an earlier hand analysis of the same geometry come out wrong.
    'palm': [('palm_hub', (0.0, 0.0, 0.0), (0.030, 0.0, 0.0), 0.018),
             ('palm_plate', (0.032, -0.070, 0.0), (0.032, 0.070, 0.0),
              _half_diag(0.022, 0.040))],
}
FINGER_CAPSULE = ('finger', (0.030, 0.0, 0.0), (0.110, 0.0, 0.0),
                  _half_diag(0.012, 0.030))
FINGERTIP_X = 0.110          # how far the fingertips reach past the wrist centre

# THE DISTAL PHALANX EXTENDS THE FINGER, and the floor checks have to know. FINGERTIP_X
# is the end of the PROXIMAL plate; the hinged segment adds up to DISTAL_L beyond it. When
# the hand is OPEN the segment is straight, so the finger is at its LONGEST exactly when it
# is descending onto a floor-lying object - which is the pose that matters for ground
# clearance. Sizing the segment without this gave a tip sitting exactly at floor level
# while the suite still reported 13.8 mm, because it was measuring the plate alone.
DISTAL_L = 0.022


def finger_reach(curl=0.0):
    """How far the fingertip reaches past the wrist, for a given distal curl (rad)."""
    import math as _m
    return FINGERTIP_X + DISTAL_L * _m.cos(curl)
# The fingertips therefore reach FINGERTIP_X - LG = 0.040 m past the grasp point,
# which with a grasp 0.025 m above the object's axis puts them 0.015 m BELOW that
# axis - i.e. the fingers straddle the widest part of the body instead of pinching
# its shoulder. See the finger note in arm.xacro.
# That single number sets the minimum height an object may lie at: below about
# 0.037 m the fingers are driven into the ground on every grasp, which is exactly
# what happened with the original 0.119 m fingers and 0.030 m objects.


# ------------------------------------------------------------------ 3D helpers
def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _apply(T, p):
    """Apply a (R, t) transform to a point."""
    R, t = T
    return (R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0],
            R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1],
            R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2])


def _compose(A, B):
    """Compose transforms: result = A then B applied in A's frame."""
    RA, tA = A
    RB, tB = B
    R = ik.mat_mul(RA, RB)
    t = _apply(A, tB)
    return (R, t)


def segment_distance(p0, p1, q0, q1):
    """Shortest distance between two 3D line segments.

    Handles the degenerate cases (either segment a point, the two parallel) because
    several of the arm's capsules are zero-length points at the wrist.
    """
    u = _sub(p1, p0)
    v = _sub(q1, q0)
    w = _sub(p0, q0)
    a, b, c = _dot(u, u), _dot(u, v), _dot(v, v)
    d, e = _dot(u, w), _dot(v, w)
    denom = a * c - b * b

    if a <= 1e-16 and c <= 1e-16:            # both points
        return _norm(w)
    if a <= 1e-16:                            # first is a point
        t = min(1.0, max(0.0, e / c))
        return _norm(_sub(p0, _add(q0, _scale(v, t))))
    if c <= 1e-16:                            # second is a point
        s = min(1.0, max(0.0, -d / a))
        return _norm(_sub(_add(p0, _scale(u, s)), q0))

    if denom <= 1e-16:                        # parallel
        s = min(1.0, max(0.0, -d / a))
        t = min(1.0, max(0.0, (b * s + e) / c))
    else:
        s = min(1.0, max(0.0, (b * e - c * d) / denom))
        t = min(1.0, max(0.0, (a * e - b * d) / denom))
        # one clamped iteration is enough for the accuracy needed here
        s = min(1.0, max(0.0, (b * t - d) / a))
    closest_p = _add(p0, _scale(u, s))
    closest_q = _add(q0, _scale(v, t))
    return _norm(_sub(closest_p, closest_q))


def _seg_dist(p0, p1, q0, q1):
    """segment_distance() with the vector helpers inlined - the planner calls this millions
    of times, and the helper calls were most of its cost. Same algorithm, same result."""
    ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    vx, vy, vz = q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]
    wx, wy, wz = p0[0] - q0[0], p0[1] - q0[1], p0[2] - q0[2]
    a = ux * ux + uy * uy + uz * uz
    b = ux * vx + uy * vy + uz * vz
    c = vx * vx + vy * vy + vz * vz
    d = ux * wx + uy * wy + uz * wz
    e = vx * wx + vy * wy + vz * wz
    if a <= 1e-16 and c <= 1e-16:
        return math.sqrt(wx * wx + wy * wy + wz * wz)
    if a <= 1e-16:
        s, t = 0.0, min(1.0, max(0.0, e / c))
    elif c <= 1e-16:
        s, t = min(1.0, max(0.0, -d / a)), 0.0
    else:
        denom = a * c - b * b
        if denom <= 1e-16:
            s = min(1.0, max(0.0, -d / a))
            t = min(1.0, max(0.0, (b * s + e) / c))
        else:
            s = min(1.0, max(0.0, (b * e - c * d) / denom))
            t = min(1.0, max(0.0, (a * e - b * d) / denom))
            s = min(1.0, max(0.0, (b * t - d) / a))
    dx = wx + s * ux - t * vx
    dy = wy + s * uy - t * vy
    dz = wz + s * uz - t * vz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _sphere(cap_):
    """Bounding sphere of a capsule: (centre, radius)."""
    _n, a, b, r = cap_
    m = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)
    h = 0.5 * math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)
    return m, h + r


def _min_gap(caps_a, caps_b, cap, skip=None):
    """Smallest capsule-to-capsule gap between two lists, EXACT below ``cap``.

    Pairs whose bounding spheres are already ``cap`` apart - or further apart than the best
    gap found so far - cannot change a minimum that is below ``cap``, so they are skipped.
    ``skip(i, j)`` excludes pairs (adjacent bodies, bolted shoulders).
    """
    sa = [_sphere(c) for c in caps_a]
    sb = sa if caps_b is caps_a else [_sphere(c) for c in caps_b]
    best = cap
    for i, (na, a0, a1, ra) in enumerate(caps_a):
        (ma, Ra) = sa[i]
        for j, (nb, b0, b1, rb) in enumerate(caps_b):
            if skip is not None and skip(i, j):
                continue
            mb, Rb = sb[j]
            dm = math.sqrt((ma[0] - mb[0]) ** 2 + (ma[1] - mb[1]) ** 2 + (ma[2] - mb[2]) ** 2)
            if dm - Ra - Rb >= best:
                continue
            g = _seg_dist(a0, a1, b0, b1) - ra - rb
            if g < best:
                best = g
    return best


_SELF_PAIRS = {}


def _self_min(real, cap):
    """self_clearance() (fine model, capsule radii) exact below ``cap``, far pairs skipped.

    Which pairs count - non-adjacent bodies of the one arm - depends only on the capsule
    names, which are the same for every arm state, so the list is worked out once.
    """
    key = tuple(c[0] for c in real)
    pairs = _SELF_PAIRS.get(key)
    if pairs is None:
        ranks = [_body_rank(n) for n in key]
        pairs = [(i, j) for i in range(len(key)) for j in range(i + 1, len(key))
                 if ranks[i] >= 0 and ranks[j] >= 0 and abs(ranks[i] - ranks[j]) >= 2]
        _SELF_PAIRS[key] = pairs
    sph = [_sphere(c) for c in real]
    best = cap
    for i, j in pairs:
        (ma, Ra), (mb, Rb) = sph[i], sph[j]
        dm = math.sqrt((ma[0] - mb[0]) ** 2 + (ma[1] - mb[1]) ** 2 + (ma[2] - mb[2]) ** 2)
        if dm - Ra - Rb >= best:
            continue
        a, b = real[i], real[j]
        g = _seg_dist(a[1], a[2], b[1], b[2]) - a[3] - b[3]
        if g < best:
            best = g
    return best


def _pose_clearance_capped(caps, other, floor_z, obstacles, cap):
    """pose_clearance() for the PATH SEARCH: exact below ``cap``, and ``cap`` otherwise.

    The search's penalty is zero above CLEAR_TARGET, so values above it are all the same to
    it - which lets every pair of bodies that is plainly far apart be skipped. Reporting and
    the tests use the exact pose_clearance().
    """
    real = [c for c in caps
            if (c[1][0] - c[2][0]) ** 2 + (c[1][1] - c[2][1]) ** 2
            + (c[1][2] - c[2][2]) ** 2 > 1e-18]
    worst = min(cap, _self_min(real, cap),
                floor_clearance(caps, floor_z)[0], base_clearance(caps)[0],
                mast_clearance(caps)[0])
    if other is not None:
        def shoulders(i, j):
            return 'shoulder' in caps[i][0] and 'shoulder' in other[j][0]
        worst = min(worst, _min_gap(caps, other, cap, shoulders))
    for obs, links, margin in _entries(obstacles):
        sub = [c for c in caps if links is None or c[0].startswith(links)]
        if not sub:
            continue
        if is_solid(obs):
            worst = min(worst, _solid_gap(sub, obs, cap + margin) - margin)
        else:
            worst = min(worst, _min_gap(sub, [('obstacle',) + tuple(obs)],
                                        cap + margin) - margin)
    return worst


# ---------------------------------------------------------------------------
# SOLID BOXES - shelves, tables, pedestals, walls.
#
# The floor is a plane and the objects are capsules, and until 2026-09-28 that was all the
# world had in it. A shelf is neither: an infinite plane at its height would cut through the
# robot's own shoulders (0.24 m) and a capsule round it overstates it by a whole radius. So a
# fixture is a box - centre, yaw about the vertical, full size, as an SDF <box> is written -
# and the arm's capsules are measured against it exactly.
#
# An obstacle entry is ``(shape, links)`` or ``(shape, links, margin)``; a shape is either a
# capsule ``(p0, p1, radius)`` or a box from box_obstacle(). Objects keep OBJECT_MARGIN; a
# fixture is normally given 0 - like the floor, it is where the hands are MEANT to go close.
# ---------------------------------------------------------------------------
def box_obstacle(centre, yaw, size):
    """A solid box obstacle. ``size`` is the FULL extent along its own x, y, z."""
    return ('box', tuple(float(v) for v in centre), (math.cos(yaw), math.sin(yaw)),
            tuple(0.5 * float(v) for v in size))


def cylinder_obstacle(centre, axis, length, radius):
    """A solid cylinder obstacle - a bottle, a can - with flat ends, EXACT."""
    n = math.sqrt(sum(a * a for a in axis))
    return ('cyl', tuple(float(v) for v in centre), tuple(a / n for a in axis),
            0.5 * float(length), float(radius))


def is_box(obs):
    return len(obs) == 4 and obs[0] == 'box'


def is_cylinder(obs):
    return len(obs) == 5 and obs[0] == 'cyl'


def is_solid(obs):
    """A box or a cylinder, as opposed to a capsule obstacle ``(p0, p1, radius)``."""
    return is_box(obs) or is_cylinder(obs)


def _entries(obstacles):
    """Normalise obstacle entries to (shape, links, margin)."""
    for ent in obstacles:
        yield ent[0], ent[1], (ent[2] if len(ent) > 2 else OBJECT_MARGIN)


def _in_box(p, box):
    _t, c, (co, si), _h = box
    dx, dy, dz = p[0] - c[0], p[1] - c[1], p[2] - c[2]
    return (co * dx + si * dy, -si * dx + co * dy, dz)


def _box_sdf(u, h):
    """Signed distance from a point (box frame) to the box: negative inside."""
    qx, qy, qz = abs(u[0]) - h[0], abs(u[1]) - h[1], abs(u[2]) - h[2]
    out = math.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2 + max(qz, 0.0) ** 2)
    return out + min(max(qx, qy, qz), 0.0)


def segment_box_distance(p0, p1, box):
    """Signed distance between a line segment and a solid box, negative inside it.

    EXACT. Outside the box the squared distance along the segment is a sum of three
    piecewise quadratics, one per axis, whose pieces change only where the segment crosses
    a face plane - so between those crossings it is one quadratic, minimised in closed form.
    If the segment reaches the box, the signed distance is still convex along it (the box
    is convex), so a ternary search finds how deep it goes.
    """
    h = box[3]
    a, b = _in_box(p0, box), _in_box(p1, box)
    d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ts = [0.0, 1.0]
    for k in range(3):
        if abs(d[k]) > 1e-15:
            for s in (-h[k], h[k]):
                t = (s - a[k]) / d[k]
                if 0.0 < t < 1.0:
                    ts.append(t)
    ts.sort()
    best = _INF
    for t0, t1 in zip(ts, ts[1:]):
        tm = 0.5 * (t0 + t1)
        A = B = C = 0.0
        for k in range(3):
            u = a[k] + tm * d[k]
            if u > h[k]:
                off = a[k] - h[k]
            elif u < -h[k]:
                off = a[k] + h[k]
            else:
                continue
            A += d[k] * d[k]
            B += 2.0 * off * d[k]
            C += off * off
        t = min(t1, max(t0, -B / (2.0 * A))) if A > 1e-18 else t0
        best = min(best, A * t * t + B * t + C)
    if best > 1e-18:
        return math.sqrt(best)
    lo, hi = 0.0, 1.0

    def sd(t):
        return _box_sdf((a[0] + t * d[0], a[1] + t * d[1], a[2] + t * d[2]), h)
    for _ in range(40):
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if sd(m1) < sd(m2):
            hi = m2
        else:
            lo = m1
    return min(sd(0.0), sd(1.0), sd(0.5 * (lo + hi)))


def _cyl_sdf(p, cyl):
    """Signed distance from a point to a solid cylinder with flat ends: negative inside."""
    _t, c, u, h, r = cyl
    d = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
    a = d[0] * u[0] + d[1] * u[1] + d[2] * u[2]
    rad = math.sqrt(max(0.0, d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - a * a))
    dr, da = rad - r, abs(a) - h
    if dr > 0.0 and da > 0.0:
        return math.sqrt(dr * dr + da * da)
    return max(dr, da)


_GOLD = (math.sqrt(5.0) - 1.0) / 2.0


def segment_cylinder_distance(p0, p1, cyl):
    """Signed distance between a line segment and a solid cylinder, negative inside it.

    The cylinder is convex, so its signed distance is a convex function along the segment
    and a golden-section search finds the minimum - to well under a micron in 30 steps.
    """
    d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])

    def f(t):
        return _cyl_sdf((p0[0] + t * d[0], p0[1] + t * d[1], p0[2] + t * d[2]), cyl)
    lo, hi = 0.0, 1.0
    x1, x2 = hi - _GOLD * (hi - lo), lo + _GOLD * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(30):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - _GOLD * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + _GOLD * (hi - lo)
            f2 = f(x2)
    return min(f(0.0), f(1.0), f1, f2)


def solid_sdf(p, obs):
    """Signed distance from a point to a box or cylinder obstacle."""
    if is_box(obs):
        return _box_sdf(_in_box(p, obs), obs[3])
    return _cyl_sdf(p, obs)


def segment_solid_distance(p0, p1, obs):
    """Signed distance between a segment and a box or cylinder obstacle."""
    if is_box(obs):
        return segment_box_distance(p0, p1, obs)
    return segment_cylinder_distance(p0, p1, obs)


def _solid_gap(caps, obs, cap):
    """Smallest capsule-to-solid gap, exact below ``cap``; far capsules skipped by sphere."""
    best = cap
    for c in caps:
        m, R = _sphere(c)
        if solid_sdf(m, obs) - R >= best:
            continue
        g = segment_solid_distance(c[1], c[2], obs) - c[3]
        if g < best:
            best = g
    return best


# ---------------------------------------------------------------- arm geometry
def link_frames(q, side):
    """World transforms of every arm link frame, following arm.xacro exactly.

    Returns a dict of ``name -> (R, t)``. The palm frame's x axis is the approach
    direction and its y axis is the finger-travel direction, matching
    ``ik.frame(approach, finger)``.
    """
    q1, q2, q2b, q3, q4, q5, q6 = q
    ident = ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0.0, 0.0, 0.0))

    mount = (ik.rot_z(q1), (SHOULDER_X, side * SHOULDER_Y, DECK_TOP_Z))
    shoulder = _compose(ident, mount)
    # J2b, THE SEVENTH AXIS, sits between the pitch yoke and the upper arm and rolls
    # about the upper arm's own x. It does not move the elbow - the elbow is at
    # yoke + L1 along x either way - it rotates the plane the forearm swings in.
    # The yoke carries no capsule: it is a 22 mm cylinder wholly inside the shoulder
    # riser's envelope, so giving it one would only add a pair that adjacency skips.
    yoke = _compose(shoulder,
                    (ik.rot_y(q2), (0.0, 0.0, SHOULDER_Z - DECK_TOP_Z)))
    upper = _compose(yoke, (ik.rot_x(q2b), (0.0, 0.0, 0.0)))
    fore = _compose(upper, (ik.rot_y(q3), (L1, 0.0, 0.0)))
    wroll = _compose(fore, (ik.rot_x(q4), (L2, 0.0, 0.0)))
    wpitch = _compose(wroll, (ik.rot_y(q5), (0.0, 0.0, 0.0)))
    palm = _compose(wpitch, (ik.rot_x(q6), (0.0, 0.0, 0.0)))
    return {'shoulder': shoulder, 'upper_yoke': yoke, 'upper_arm': upper,
            'forearm': fore, 'wrist_roll': wroll, 'wrist_pitch': wpitch,
            'palm': palm}


def arm_capsules(q, side, finger_pos=0.0):
    """World-frame capsules for one arm: ``[(name, p0, p1, radius), ...]``."""
    frames = link_frames(q, side)
    out = []
    for link, caps in LINK_CAPSULES.items():
        T = frames[link]
        for name, a, b, r in caps:
            out.append((f'{link}/{name}', _apply(T, a), _apply(T, b), r))
    T = frames['palm']
    name, a, b, r = FINGER_CAPSULE
    for sgn in (+1, -1):
        aa = (a[0], sgn * finger_pos, a[2])
        bb = (b[0], sgn * finger_pos, b[2])
        out.append((f'finger{"L" if sgn > 0 else "R"}',
                    _apply(T, aa), _apply(T, bb), r))
    return out


def object_capsule(centre, axis_angle, length, radius, theta=0.0):
    """Capsule for a held object of given length along its axis.

    ``theta`` rotates the axis about the horizontal perpendicular, so an upright
    object (after a flip) is described by ``theta = pi/2``.
    """
    ca, sa = math.cos(axis_angle), math.sin(axis_angle)
    ct, st = math.cos(theta), math.sin(theta)
    # axis direction: horizontal at theta=0, vertical at theta=pi/2
    d = (ca * ct, sa * ct, st)
    half = _scale(d, length / 2.0)
    return ('object', _sub(centre, half), _add(centre, half), radius)


# -------------------------------------------------------------------- checks
# ---------------------------------------------------------------------------
# FINE GEOMETRY - flat links represented as flat.
#
# WHY THIS EXISTS. Every link here is a BOX, and a single circular capsule enclosing a box
# over-reports badly in the thin axis: the finger plate is 12 mm thick and its enclosing
# capsule is 42 mm across. That 20 mm of slop is what made every self-collision and
# arm-arm figure a BRACKET rather than a number, and several dead ends this project chased
# were the model rather than the robot.
#
# A box of cross-section (thin x wide) is decomposed into a ROW of capsules laid along its
# length, each of radius thin/2, spaced across the wide axis. The union follows the box to
# within a fraction of a millimetre in the thin direction - which is the direction that
# matters, because it is the one the old model got wrong by 20 mm.
#
# BOX_LINKS: link -> (thin, wide, which local axis is 'wide')
BOX_LINKS = {
    'upper_arm': (0.032, 0.045, 'y'),
    'forearm':   (0.020, 0.038, 'y'),      # 20 mm thick - see the note in arm.xacro
}
FINGER_BOX = (0.012, 0.040, 'z')      # thin across the closing axis, wide along the object
# THE AXIS HERE IS z, NOT y, AND THAT WAS GOT WRONG ONCE. The palm plate's capsule
# already runs along y (-0.070 to +0.070), so its length is covered; what the row has
# to fill is the OTHER cross-section direction, z. Offsetting along y instead widened
# the plate from 140 mm to 258 mm and made it, not the real geometry, the nearest body
# to the base on the descent - a phantom -1.8 mm breach.
PALM_BOX = (0.022, 0.040, 'z')


def _row(a, b, thin, wide, axis, n=3):
    """Capsules along a..b filling a (thin x wide) cross-section."""
    r = thin / 2.0
    out = []
    span = wide - thin                      # centres must stay inside the box
    for i in range(n):
        f = -0.5 + (i / (n - 1) if n > 1 else 0.5)
        d = f * span
        off = (0.0, d, 0.0) if axis == 'y' else (0.0, 0.0, d)
        out.append(([a[k] + off[k] for k in range(3)],
                    [b[k] + off[k] for k in range(3)], r))
    return out


def arm_capsules_fine(q, side, finger_pos=0.0):
    """Like arm_capsules, but flat links are decomposed so the thin axis is honest."""
    frames = link_frames(q, side)
    out = []
    for link, caps in LINK_CAPSULES.items():
        T = frames[link]
        for name, a, b, r in caps:
            box = BOX_LINKS.get(link) if not name.startswith('palm_plate') else PALM_BOX
            if box is None and name.startswith('palm_plate'):
                box = PALM_BOX
            if box is None:
                out.append((f'{link}/{name}', _apply(T, a), _apply(T, b), r))
                continue
            thin, wide, axis = box
            for aa, bb, rr in _row(a, b, thin, wide, axis):
                out.append((f'{link}/{name}', _apply(T, aa), _apply(T, bb), rr))
    T = frames['palm']
    name, a, b, r = FINGER_CAPSULE
    thin, wide, axis = FINGER_BOX
    for sgn in (+1, -1):
        aa = (a[0], sgn * finger_pos, a[2])
        bb = (b[0], sgn * finger_pos, b[2])
        for p0, p1, rr in _row(aa, bb, thin, wide, axis):
            out.append((f'finger{"L" if sgn > 0 else "R"}', _apply(T, p0),
                        _apply(T, p1), rr))
    return out


# ---------------------------------------------------------------------------
# SELF-COLLISION - one arm against itself.
#
# THIS EXISTS BECAUSE IT WAS GOT WRONG FOUR TIMES. There is no self-collision check
# anywhere else in this module: arm_arm_clearance compares two arms, and floor/base/mast
# compare an arm with the world. An arm folding onto itself was therefore invisible, and
# when it was finally noticed the calculation was rewritten inline each time it was needed
# - four ad-hoc copies with different conventions, three of them wrong:
#
#   1. included the ZERO-LENGTH wrist capsules. Those are joint markers sitting exactly at
#      the forearm's end, so they overlap it by construction -> a constant -43 mm in every
#      pose, including the parked one.
#   2. ranked bodies by a list that included those markers, so forearm and palm came out
#      NON-adjacent -> a constant -41.6 mm, again in every pose.
#   3. looked up the finger radius as THIN['fingerL'], but the key is 'finger', so it
#      silently used the 0.02 default instead of 0.006 -> every figure 14 mm too pessimistic.
#   4. compared palm_hub against the forearm. The hub is bolted to the forearm through the
#      zero-length wrist joints, so it can never be separated -> every waypoint "blocked".
#
# The rules that make it correct:
#   * drop zero-length capsules - they are joints, not bodies
#   * rank by the chain of REAL BODIES, in which the palm attaches directly to the forearm
#   * skip pairs adjacent in that chain
#   * bracket the radii, because a circular capsule around a flat plate over-reports by up
#     to 20 mm: the finger plate is 12 mm thick and its capsule is 42 mm across
# ---------------------------------------------------------------------------
BODY_CHAIN = ('shoulder', 'upper_arm', 'forearm', 'palm', 'finger')

# real half-thickness in the THIN axis - the best a flat plate can present
THIN_HALF = {'shoulder': 0.018, 'upper_arm': 0.016, 'forearm': 0.010,
             'palm': 0.011, 'finger': 0.006}


def _body_rank(name):
    base = name.split('/')[0]
    for i, k in enumerate(BODY_CHAIN):
        if base.startswith(k):
            return i
    return -1


def self_clearance(caps, optimistic=True):
    """Worst clearance between NON-ADJACENT bodies of ONE arm, and the pair responsible.

    optimistic=True  uses the real thin half-thickness  - a LOWER bound on the true gap
    optimistic=False uses the enclosing capsule radius   - an UPPER bound on interference

    A collision is only CERTAIN where the optimistic bound is negative.
    """
    real = [(n, a, b, r) for n, a, b, r in caps
            if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 > 1e-18]
    worst, who = 1e9, None
    for i, (n1, a0, a1, r1) in enumerate(real):
        for (n2, b0, b1, r2) in real[i + 1:]:
            k1, k2 = _body_rank(n1), _body_rank(n2)
            if k1 < 0 or k2 < 0 or abs(k1 - k2) < 2:
                continue
            if optimistic:
                r1 = THIN_HALF.get(n1.split('/')[0].rstrip('LR'), r1)
                r2 = THIN_HALF.get(n2.split('/')[0].rstrip('LR'), r2)
            gap = segment_distance(a0, a1, b0, b1) - r1 - r2
            if gap < worst:
                worst, who = gap, (n1, n2)
    return worst, who


def arm_arm_clearance(caps_left, caps_right, ignore_shoulders=True):
    """Worst signed clearance between the two arms, and which pair caused it."""
    worst, who = 1e9, None
    for nl, a0, a1, ra in caps_left:
        for nr, b0, b1, rb in caps_right:
            if ignore_shoulders and 'shoulder' in nl and 'shoulder' in nr:
                continue      # rigidly mounted 240 mm apart; cannot collide
            gap = segment_distance(a0, a1, b0, b1) - ra - rb
            if gap < worst:
                worst, who = gap, (nl, nr)
    return worst, who


def floor_clearance(caps, floor_z=0.0):
    """Worst signed clearance between any capsule and the floor plane."""
    worst, who = 1e9, None
    for name, a, b, r in caps:
        gap = min(a[2], b[2]) - r - floor_z
        if gap < worst:
            worst, who = gap, name
    return worst, who


def base_clearance(caps, radius=BASE_RADIUS, top_z=DECK_TOP_Z):
    """Worst signed clearance between any capsule and the base cylinder.

    The base is treated as an upright cylinder of ``radius`` from the floor to
    ``top_z``. A capsule above the deck cannot hit it, which is why the arms are
    mounted on a riser in the first place.
    """
    worst, who = 1e9, None
    for name, a, b, r in caps:
        if name.startswith('shoulder'):
            continue                       # bolted to the deck by construction
        for p in (a, b):
            if p[2] - r > top_z:
                continue                   # clear above the deck
            gap = math.hypot(p[0], p[1]) - r - radius
            if gap < worst:
                worst, who = gap, name
    return worst, who


def mast_clearance(caps):
    """Worst signed clearance against the sensor mast."""
    worst, who = 1e9, None
    for name, a, b, r in caps:
        gap = min(
            segment_distance(a, b, (MAST_X, sy * MAST_Y, DECK_TOP_Z),
                             (MAST_X, sy * MAST_Y, MAST_TOP)) - r - MAST_R
            for sy in (-1.0, 1.0))
        if gap < worst:
            worst, who = gap, name
    return worst, who


# ---------------------------------------------------------------------------
# CHOOSING THE ELBOW SWIVEL
#
# The seventh axis makes the arm redundant: every hand pose has a CIRCLE of elbow
# positions. ik.solve() deliberately takes psi as an input so it stays closed form, which
# leaves choosing psi as an explicit, testable concern - here.
#
# Two things are wanted at once and they conflict:
#   * CLEARANCE  - pick the roomiest elbow
#   * CONTINUITY - do not jump between waypoints
# Taking the clearance maximum alone produces exactly the fault the seventh axis was added
# to remove: the optimum hops around the circle and the joint snaps, which is what makes
# the 6-DOF elbow-up/elbow-down switch unusable at 207 deg.
#
# So: take every psi that clears PSI_MARGIN, and of those pick the one NEAREST THE SEED.
# Clearance becomes a constraint and continuity the objective. Only if nothing clears the
# margin does it fall back to the roomiest, so a genuinely tight pose still gets the best
# available arm rather than an exception.
# ---------------------------------------------------------------------------
PSI_MARGIN = 0.008          # 8 mm, the same bar the validation sweep used
# 10 deg, was 5. Measured on both cycles: identical object and floor clearances, self
# within 0.7 mm, and half the planning time - which on the test laptop, sharing its CPU with
# Gazebo, was 43 s per object at 5 deg.
PSI_STEP_DEG = 10
# HOW FAR ANY ONE JOINT MAY MOVE BETWEEN WAYPOINTS.
# The cap is on the JOINT VECTOR, not on psi, because psi is only a parameter - what the
# hardware has to track is the joints, and a small change in psi near a stretched arm can
# still swing j2b a long way. Capping psi instead let a 30 deg swivel step through as a
# 160 deg lurch in j2b.
JOINT_MAX_STEP = math.radians(45)
# 45 deg, CALIBRATED AGAINST THE BASELINE RATHER THAN CHOSEN. The confirmed-working 6-DOF
# run already moves the left arm by up to 14.5 deg between waypoints and the right by 81 deg
# (the old handover withdrawal, now gone). 45 is comfortably above what the smooth phases
# need and below the 81 deg lurch that phase used to contain, so it admits every motion the
# working robot already made while still refusing a branch flip, which is 180 deg.


# The links that are an ARM rather than a HAND. A hand that is about to take hold of an
# object, or has just let go of it, is next to it by design and its path is fixed by the
# choreography; what the swivel choice decides is where the upper arm and forearm go.
ARM_LINKS = ('shoulder', 'upper_arm', 'forearm')
OBJECT_MARGIN = 0.020


def obstacle_clearance(caps, obstacle, links=None):
    """Worst gap between an arm's capsules and one obstacle: a capsule ``(p0, p1, radius)``,
    a box from box_obstacle() or a cylinder from cylinder_obstacle().

    ``links`` restricts the check to capsules whose name starts with one of those body
    names - ARM_LINKS for an object a hand is deliberately next to - or None for all.
    """
    sub = [(p0, p1, r) for n, p0, p1, r in caps if links is None or n.startswith(links)]
    if not sub:
        return 1e9
    if is_solid(obstacle):
        return min(segment_solid_distance(p0, p1, obstacle) - r for p0, p1, r in sub)
    a0, a1, rad = obstacle
    return min(segment_distance(p0, p1, a0, a1) - r - rad for p0, p1, r in sub)


def object_capsules(centre, axis, length, half_width, round_=True):
    """An object as obstacle shapes, EXACT: a solid cylinder, or a box for a cuboid.

    ``axis`` is the long axis as a 3D direction - horizontal for a lying object, vertical for
    a standing one.

    FLAT ENDS ARE FLAT. A cylinder used to be one CAPSULE over its whole length - exact on the
    curved side but 45 mm too long at each end, its round end bulging a whole radius past the
    flat face - and a cuboid a core capsule plus edge lines. Nothing noticed on the floor. On a
    0.10 m table the retreat is capped by the arm's reach, and the wrist rotation on the way
    home put a finger through that phantom 45 mm above a bottle it really cleared, so the model
    reported -12 mm where the truth was a miss. Solids with a signed distance fix it outright.
    A cuboid's box needs its axis horizontal or vertical (lying or standing, the only ways an
    object is left); at any other angle it falls back to the enclosing cylinder.
    """
    n = math.sqrt(sum(a * a for a in axis))
    u = [a / n for a in axis]
    if round_:
        return [cylinder_obstacle(centre, u, length, half_width)]
    level = abs(u[2]) < 1e-6
    if level:
        return [box_obstacle(centre, math.atan2(u[1], u[0]),
                             (length, 2 * half_width, 2 * half_width))]
    if abs(abs(u[2]) - 1.0) < 1e-6:
        return [box_obstacle(centre, 0.0, (2 * half_width, 2 * half_width, length))]
    return [cylinder_obstacle(centre, u, length, half_width * math.sqrt(2.0))]

def pose_clearance(caps, other=None, floor_z=0.0, obstacles=()):
    """Worst of every clearance that applies to one arm. Fine model in, so flat links
    are flat - a capsule around the 12 mm finger plate over-reports by 20 mm.

    ``obstacles`` are the OBJECTS and FIXTURES in the world, as ``[(shape, links), ...]`` or
    ``[(shape, links, margin), ...]`` - see obstacle_clearance and box_obstacle. Until
    2026-09-27 there were none: the planner knew the arms, the floor, the base and the mast,
    and nothing else, so it backed the right hand straight into the box lying next to the
    bottle and swung an arm through the bottle it had just stood up.
    """
    terms = [self_clearance(caps, optimistic=False), floor_clearance(caps, floor_z),
             base_clearance(caps), mast_clearance(caps)]
    if other is not None:
        terms.append(arm_arm_clearance(caps, other))
    # OBJECTS KEEP A LARGER MARGIN than the robot's own bodies. The path search maximises
    # the WORST clearance along the path, and that worst is set by the arm's own geometry -
    # +2 mm of hand against forearm at the top of the flip - so everything else only has to
    # beat that. Two millimetres from a bottle is a knock, not a miss, once tracking lag is
    # added; so an object counts as touched OBJECT_MARGIN before it really is.
    for obs, links, margin in _entries(obstacles):
        terms.append(obstacle_clearance(caps, obs, links) - margin)
    # these helpers are a mix of bare floats and (gap, who) tuples - normalise
    return min(t[0] if isinstance(t, tuple) else t for t in terms)


def _joint_step(q, seed):
    """The largest distance any joint must TRAVEL between two arm states.

    RAW DIFFERENCE, NOT AN ANGLE MODULO 360. Every arm joint here has hard limits inside
    +/-180 deg, so a joint cannot pass through +/-180 - it has to go the long way round.
    This used math.remainder(), which scores -3.02 -> +2.58 rad as a 39 deg step; the joint
    actually travels 321 deg. In Gazebo that was the right upper arm spinning almost a full
    turn on the last retreat waypoint (`right_j2b off 4.865`), sweeping the bottle it had
    just stood up onto the floor.
    """
    return max(abs(q[i] - seed[i]) for i in range(len(q)))


# WHICH JOINTS CARRY THE CONFIGURATION, as opposed to the position.
# j1, j2 and j3 place the hand: between the parked pose and the first waypoint of a leg
# they legitimately move a long way, because the hand does. j2b, j4, j5 and j6 are the
# REDUNDANT ones - the elbow swivel and the wrist triple - and those are what must not
# jump, because the two wrist branches give the identical hand pose 180 deg apart.
# Capping all seven at the first waypoint rejected the parked-to-approach move outright;
# capping only these keeps the arm in the same configuration while letting it reach.
CONFIG_JOINTS = (2, 4, 5, 6)

# HOW FAR THE CONFIGURATION MAY DIFFER FROM THE SEED AT A LEG'S FIRST WAYPOINT.
# Wider than JOINT_MAX_STEP on purpose, and the width is measured rather than guessed:
#     largest LEGITIMATE configuration change, parked pose -> a leg's first waypoint   80 deg
#     what the planner picks if it chases clearance with no seed at all               145 deg
#     a wrist BRANCH FLIP, by construction (q4+pi, -q5, q6+pi)                        180 deg
# 120 deg sits above everything the robot genuinely needs and below the flip, so the arm can
# reach out from parked while still being refused the 180 deg wrist inversion that showed up
# in Gazebo as right_j4 off 87.7 deg with the hand spinning through the bottle.
SEED_MAX_STEP = math.radians(120)


def _config_step(q, seed):
    """Raw travel on the configuration joints - see _joint_step for why not modulo 360."""
    return max(abs(q[i] - seed[i]) for i in CONFIG_JOINTS)


def _arm_options(p, R, side, other, finger_pos, floor_z, obstacles=(), both_shoulders=False):
    """Every arm available at one hand pose, as (q, clearance), over psi x branch."""
    out = []
    for i in range(0, 360, PSI_STEP_DEG):
        for q in ik.solve_branches(p, R, side, math.radians(i), both_shoulders):
            out.append((q, _pose_clearance_capped(arm_capsules_fine(q, side, finger_pos),
                                                  other, floor_z, obstacles, CLEAR_TARGET)))
    return out


# ---------------------------------------------------------------------------
# THE MOVE HOME - the one motion the planner never used to see.
#
# After the last planned waypoint the arms go to the parked pose in ONE joint-space move:
# the trajectory controller interpolates joint angles, and a straight line in joint space
# knows nothing about the object that was just stood up. Which elbow swivel and wrist
# branch the arm ENDS the cycle in decides where that line goes, and measured on the
# whole-cycle plan the default choice (the roomiest end state) sent the right hand's
# fingers up to 50 mm INTO the standing object on the way home.
#
# So the end state is chosen with the move home in view: among the end states the path
# search reaches, take the best one whose straight joint line home keeps HOME_MARGIN from
# the standing object, the floor, the base, the mast and the other arm (moving home at
# the same time). Self-clearance is reported but NOT required: the left arm's wrist
# unrolls from the flip orientation to the parked one and brushes its own forearm by about
# 10 mm on the way for every end state available - and Gazebo does not simulate self
# contact for this model - so requiring it would only make the search trade clearance
# elsewhere for nothing.
# ---------------------------------------------------------------------------
HOME_MARGIN = 0.030
HOME_SAMPLES = 16


def line_capsules(q0, q1, side, finger_pos=0.0, n=HOME_SAMPLES):
    """Capsules at n+1 evenly spaced points of the straight joint-space line q0 -> q1."""
    return [arm_capsules_fine([a + i / float(n) * (b - a) for a, b in zip(q0, q1)],
                              side, finger_pos) for i in range(n + 1)]


def home_line_clearance(q, q_home, side, finger_pos=0.0, other_line=None,
                        obstacles=(), n=HOME_SAMPLES):
    """Worst floor / base / mast / arm-arm / object clearance along the joint line q -> q_home.

    Used for BOTH joint-space moves the planner does not plan: parked -> the first waypoint,
    and the last waypoint -> parked. ``other_line`` is the other arm's :func:`line_capsules`
    for its own simultaneous move. ``obstacles`` is a list of object capsules, all links.
    """
    worst = 1e9
    for i, caps in enumerate(line_capsules(q, q_home, side, finger_pos, n)):
        terms = [floor_clearance(caps)[0], base_clearance(caps)[0], mast_clearance(caps)[0]]
        if other_line is not None:
            terms.append(arm_arm_clearance(caps, other_line[i])[0])
        for obs in obstacles:
            terms.append(obstacle_clearance(caps, obs))
        worst = min(worst, min(terms))
    return worst


def home_move_report(ql, qr, home_l, home_r, finger_pos=0.0, obstacles=(),
                     n=HOME_SAMPLES):
    """What a joint-space move (to or from parked) will clear, for the log and the tests.

    Returns a dict: 'left' and 'right' hard clearances (as required by the planner) and
    'self_left' / 'self_right', the self-clearance, which is reported only.
    """
    line_l = line_capsules(ql, home_l, +1, finger_pos, n)
    line_r = line_capsules(qr, home_r, -1, finger_pos, n)
    return {
        'left': home_line_clearance(ql, home_l, +1, finger_pos, None, obstacles, n),
        'right': home_line_clearance(qr, home_r, -1, finger_pos, line_l, obstacles, n),
        'self_left': min(self_clearance(c, optimistic=False)[0] for c in line_l),
        'self_right': min(self_clearance(c, optimistic=False)[0] for c in line_r),
    }


# THE PATH SEARCH'S OBJECTIVE: a penalty for every waypoint that is not comfortably clear,
# summed over the whole path - not the single worst clearance.
#
# It used to be the worst clearance (maximin), and that has a blind spot which the planned
# move home exposed: once ONE waypoint is unavoidably tight, every path shares that worst
# value and the search stops caring about anything else. Planning the move home put the left
# wrist's 11 mm unroll past its own forearm into the path - it had always been there, hidden
# in an unplanned joint-space swing - and the settle then drifted from 15 mm off the floor to
# -0.2 mm, because nothing distinguished the two. A summed penalty counts every waypoint:
# under CLEAR_TARGET it costs the square of the shortfall, and anything inside OVERLAP_BUFFER
# of touching costs OVERLAP_WEIGHT times more, so contact is never traded for many shallow
# gains. The weight was set by measurement: at 20 the hand-to-forearm clearance through the
# carry settled at -0.5 mm; at 400 it is +1.9 / +1.8 mm - back to the +2 mm the seven-DOF
# design was built on - with no change to any other clearance, step or timing.
CLEAR_TARGET = 0.020
OVERLAP_WEIGHT = 400.0
OVERLAP_BUFFER = 0.003
# AND A COST FOR EVERY BIG STEP. The controller moves between waypoints in joint space, which
# the search only checks at the waypoints themselves; a 42 deg swing of the elbow between two
# of them - allowed by the 45 deg cap - carried the forearm 4.5 mm past the box it had just
# put down, between two waypoints that each cleared it by 24 mm. Charging the square of the
# largest joint step makes the search reconfigure only where that buys real clearance, and
# spread it out otherwise. Scale: a 40 deg step costs about as much as four waypoints sitting
# at zero clearance. Measured: the tightest point anywhere on the way out went from 4.5 mm to
# 18.6 mm, the largest step from 45 to 41 deg, at no cost in planning time.
STEP_WEIGHT = 3e-3
_INF = float('inf')


def _penalty(g):
    short = max(0.0, CLEAR_TARGET - g)
    into = max(0.0, OVERLAP_BUFFER - g)
    return short * short + OVERLAP_WEIGHT * into * into


def _best_chain(options, seed_q=None):
    """Pick one arm per waypoint: the chain with the smallest summed penalty (see
    CLEAR_TARGET), among chains whose every step is inside JOINT_MAX_STEP.

    WHY THIS IS A PATH PROBLEM AND NOT A PER-WAYPOINT ONE.
    Choosing the roomiest arm at each waypoint in turn is greedy, and greedy gets stuck:
    it climbs to whatever local maximum is nearest, then cannot cross the valley to reach
    the region the flip will need, so it arrives at the coaxial pose in the wrong part of
    the swivel circle and has nowhere to go. That left 5-9 self-collisions no amount of
    tuning the step cap removed, because the fault was the horizon, not the cap.

    So optimise the CHAIN: a shortest path over a small layered graph - one layer per
    waypoint, one node per (psi, wrist branch, shoulder solution), an edge wherever the RAW
    joint travel between two nodes is inside the cap - and the cost carried forward is the
    penalty summed so far. Additive, so the dynamic programme is exact.
    """
    layers = []
    for k, opts in enumerate(options):
        if not opts:
            raise ik.IKError(f'no arm solves at waypoint {k}')
        if k == 0:
            # THE SEED CONSTRAINS CONFIGURATION, NOT POSITION - see CONFIG_JOINTS. The
            # first waypoint of a leg is usually far from the parked pose in j1/j2/j3 and
            # that is fine; what must not change is which wrist branch and roughly where
            # on the swivel circle the arm sits.
            first = [((_penalty(g) if seed_q is None
                       or _config_step(q, seed_q) <= SEED_MAX_STEP else _INF), None, q, g)
                     for q, g in opts]
            if all(c[0] == _INF for c in first):
                raise ik.IKError(
                    'no arm at waypoint 0 keeps the configuration the arms are already '
                    'in within %.0f deg on the swivel and wrist'
                    % math.degrees(SEED_MAX_STEP))
            layers.append(first)
            continue
        prev = layers[-1]
        # predecessors in order of cost: once a predecessor's cost alone is no better than
        # the best total found, no later one can win, because the step cost is never negative
        order = sorted((pi for pi in range(len(prev)) if prev[pi][0] != _INF),
                       key=lambda pi: prev[pi][0])
        cap = JOINT_MAX_STEP
        cur = []
        for q, g in opts:
            best = None
            for pi in order:
                pc = prev[pi][0]
                if best is not None and pc >= best[0]:
                    break
                pq = prev[pi][2]
                step = 0.0
                for a, b in zip(q, pq):
                    d = a - b if a > b else b - a
                    if d > cap:
                        break
                    if d > step:
                        step = d
                else:
                    total = pc + STEP_WEIGHT * step * step
                    if best is None or total < best[0]:
                        best = (total, pi)
            cur.append((best[0] + _penalty(g) if best else _INF,
                        best[1] if best else None, q, g))
        if all(c[0] == _INF for c in cur):
            raise ik.IKError(
                f'the arm cannot get from waypoint {k - 1} to {k} within '
                f'{math.degrees(JOINT_MAX_STEP):.0f} deg on every joint')
        layers.append(cur)
    last = layers[-1]
    end = min(range(len(last)), key=lambda n: last[n][0])
    if last[end][0] == _INF:
        raise ik.IKError('no continuous chain of arms exists')
    # walk back from the chosen end state
    chain = []
    n = end
    for k in range(len(layers) - 1, -1, -1):
        _cost, back, q, _g = layers[k][n]
        chain.append(q)
        n = back
        if n is None and k:
            raise ik.IKError('no continuous chain of arms exists')
    chain.reverse()
    return chain


def solve_path(pairs, finger_pos=0.0, floor_z=0.0, seed_l=None, seed_r=None,
               obstacles=None, pin_start=False, pin_end=None, wide=None):
    """Joint solutions for a whole waypoint list, swivel and wrist branch both resolved.

    The left arm is planned first on its own constraints, then the right against the
    left's ACTUAL capsules, so arm-arm is enforced on the pair that will really be
    commanded rather than on some other pair that might have been chosen.

    ``finger_pos`` is one closure for every waypoint, or a list of one per waypoint - the
    hands are OPEN while travelling and CLOSED while carrying, and a closed hand checked at
    the parked pose reads 1.5 mm into its own forearm when the real, open one clears it.

    ``obstacles`` - one list per waypoint of ``(capsule, links)`` - are the objects in the
    world, enforced on every waypoint with OBJECT_MARGIN (see pose_clearance).

    ``pin_start`` fixes the FIRST waypoint to the seeds themselves and ``pin_end`` - a
    ``(q_left, q_right)`` pair - the LAST one. That is how a cycle is planned parked to
    parked: the arms leave the exact parked joint vector and come back to it, so there is no
    joint-space jump at either end for the planner not to have seen.

    ``wide`` - one bool per waypoint - also searches the second shoulder solution there
    (ik.solve, both_shoulders). The parked pose needs it; it doubles the search, so only the
    legs to and from parked ask for it.
    """
    n = len(pairs)
    obs_at = obstacles if obstacles is not None else [()] * n
    fp_at = list(finger_pos) if isinstance(finger_pos, (list, tuple)) else [finger_pos] * n
    wide_at = list(wide) if wide is not None else [False] * n
    assert len(obs_at) == n and len(fp_at) == n and len(wide_at) == n, (
        'one obstacle list, closure and search width per waypoint')

    def options(i, p, R, side, other, fixed):
        if fixed is not None:
            q = list(fixed)
            return [(q, _pose_clearance_capped(arm_capsules_fine(q, side, fp_at[i]), other,
                                               floor_z, obs_at[i], CLEAR_TARGET))]
        return _arm_options(p, R, side, other, fp_at[i], floor_z, obs_at[i], wide_at[i])

    def pinned(i, seed, end):
        if pin_start and i == 0:
            assert seed is not None, 'pin_start needs a seed'
            return seed
        if pin_end is not None and i == n - 1:
            return end
        return None

    left = _best_chain([options(i, p, R, +1, None, pinned(i, seed_l, pin_end and pin_end[0]))
                        for i, ((p, R), _) in enumerate(pairs)], seed_l)
    caps = [arm_capsules_fine(q, +1, fp_at[i]) for i, q in enumerate(left)]
    right = _best_chain([options(i, p, R, -1, caps[i],
                                 pinned(i, seed_r, pin_end and pin_end[1]))
                         for i, (_, (p, R)) in enumerate(pairs)], seed_r)
    return left, right


def nearest_objects(ql, qr, finger_pos, named_obstacles, waypoints=None):
    """How close the planned arms come to each named object, and where.

    ``named_obstacles``: ``{name: [(capsule, links), ...]}``, applying to the waypoints in
    ``waypoints[name]`` (all of them if absent). Returns ``{name: (clearance, index, side)}``
    WITHOUT the planner's OBJECT_MARGIN, i.e. the real gap.
    """
    fp_at = list(finger_pos) if isinstance(finger_pos, (list, tuple)) else [finger_pos] * len(ql)
    out = {}
    for name, obs in named_obstacles.items():
        idx = range(len(ql)) if not waypoints or name not in waypoints else waypoints[name]
        best = (1e9, -1, '')
        for i in idx:
            for q, side, tag in ((ql[i], +1, 'left'), (qr[i], -1, 'right')):
                caps = arm_capsules_fine(q, side, fp_at[i])
                g = min(obstacle_clearance(caps, c, links) for c, links, _m in _entries(obs))
                if g < best[0]:
                    best = (g, i, tag)
        out[name] = best
    return out


def check_pose(ql, qr, finger_pos=0.0, obj=None, floor_z=0.0):
    """Full clearance report for one instant of the motion.

    Returns a dict of named signed clearances plus ``worst`` and ``worst_name``.
    ``obj``, if given, is a capsule from :func:`object_capsule` and is checked
    against the floor and both arms — a held object dragging on the ground is a
    failure the arms' own clearances cannot reveal.
    """
    # THE FINE MODEL, because every link here is a BOX and a circular capsule enclosing one
    # over-reports by up to 20 mm in the thin axis - the finger plate is 12 mm thick and its
    # capsule is 42 mm across. Checking with the coarse capsules reported +3.7 mm of arm-arm
    # clearance at a pose the box model measures at +22.7 mm, which is the difference between
    # "almost touching" and "comfortable". arm_capsules() is kept for the guards that
    # deliberately compare the two models against each other.
    cl = arm_capsules_fine(ql, +1, finger_pos)
    cr = arm_capsules_fine(qr, -1, finger_pos)

    aa, aa_who = arm_arm_clearance(cl, cr)
    fl, fl_who = floor_clearance(cl + cr, floor_z)
    bs, bs_who = base_clearance(cl + cr)
    ms, ms_who = mast_clearance(cl + cr)

    result = {
        'arm_arm': aa, 'arm_arm_pair': aa_who,
        'floor': fl, 'floor_link': fl_who,
        'base': bs, 'base_link': bs_who,
        'mast': ms, 'mast_link': ms_who,
    }
    if obj is not None:
        _, o0, o1, orad = obj
        result['object_floor'] = min(o0[2], o1[2]) - orad - floor_z
        # the object is HELD, so contact with the holding hands is expected;
        # only the far links matter
        worst = 1e9
        who = None
        for name, a, b, r in cl + cr:
            if name.startswith(('palm', 'finger', 'wrist')):
                continue
            gap = segment_distance(a, b, o0, o1) - r - orad
            if gap < worst:
                worst, who = gap, name
        result['object_arm'] = worst
        result['object_arm_link'] = who

    checks = {k: v for k, v in result.items() if isinstance(v, float)}
    result['worst'] = min(checks.values())
    result['worst_name'] = min(checks, key=checks.get)
    return result


def check_waypoints(pairs, finger_pos=0.0, obj_of=None, floor_z=0.0):
    """Run :func:`check_pose` over a whole waypoint sequence.

    ``obj_of`` is an optional ``index -> capsule`` callable for the held object.
    Returns ``(worst_clearance, detail_dict_at_that_index, index)``.
    """
    # PLAN THE WHOLE PATH, exactly as arm_commander does. Solving each waypoint on its own
    # with the default swivel measures a robot that will never be commanded: it is the
    # six-DOF elbow-up arm, which is precisely the configuration that collides. This test
    # has to see the arms that will actually be sent.
    try:
        QL, QR = solve_path(pairs, finger_pos, floor_z)
    except ik.IKError:
        return 1e9, None, -1              # reachability is a separate test
    worst, detail, at = 1e9, None, -1
    for i, (ql, qr) in enumerate(zip(QL, QR)):
        rep = check_pose(ql, qr, finger_pos,
                         obj=None if obj_of is None else obj_of(i),
                         floor_z=floor_z)
        if rep['worst'] < worst:
            worst, detail, at = rep['worst'], rep, i
    return worst, detail, at
