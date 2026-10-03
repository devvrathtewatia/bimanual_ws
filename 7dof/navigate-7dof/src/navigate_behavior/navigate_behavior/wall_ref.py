"""Absolute heading from the LiDAR — ROS-free so it can be tested offline.

WHY THIS EXISTS
---------------
Section A's base never translates; it only rotates in place to face each object and
each shelf slot. Every target is therefore expressed as a world azimuth, and the robot
turns until its **odometry** heading matches. That works for the first object and
degrades after that: spinning a two-wheel robot with two casters slips, and the error
accumulates over the ~530 degrees of turning one full run needs. The symptoms were
exactly what accumulating heading error predicts — the first grasp clean, later ones a
few millimetres off, and objects placed short of their slot so they overhang the shelf.

The fix is to stop trusting odometry for heading. A flat wall behind the shelf gives an
**absolute** reference: fit a line to the LiDAR returns, and the direction of that
line's normal *is* the robot's heading, with no dependence on how far it has turned or
how much the wheels slipped.

WHY A FLAT WALL AND A LINE FIT
------------------------------
A line fit uses every beam that hits the wall, so it is insensitive to a few dropped
returns or to part of the wall being occluded by the robot's own mast. Picking out the
wall's two ends instead would work in principle but depends on exactly which beams
land on the edges, which is the least reliable part of the measurement.

It also helps that in this world the LiDAR sits at z = 0.46 m and the objects are at
most 0.09 m tall, so the wall is the *only* thing the sensor can see. There is nothing
to disambiguate.

CONVENTIONS
-----------
Beam ``i`` is at angle ``angle_min + i * angle_increment`` in the LiDAR frame, measured
from the robot's +x axis. The LiDAR is mounted without rotation relative to the base,
so bearings in the two frames are identical and its translation along the mast does not
affect any angle — a line's direction is unchanged by a translation. That is why the
mount offset never appears below.
"""
from __future__ import annotations

import math

# The wall's inward normal in WORLD terms. The wall is the plane x = -WALL_X, so the
# normal pointing from it back towards the robot is +x, i.e. world azimuth 0.
WALL_NORMAL_AZIMUTH = 0.0


def scan_points(ranges, angle_min, angle_increment, range_min, range_max,
                band=None):
    """Valid returns as ``[(x, y, bearing, range), ...]`` in the LiDAR frame.

    ``band`` optionally restricts the accepted range to ``(lo, hi)``, which is how the
    wall is separated from anything else that might appear later.
    """
    out = []
    for i, r in enumerate(ranges):
        if r is None or not (range_min < r < range_max):
            continue
        if not math.isfinite(r):
            continue
        if band is not None and not (band[0] <= r <= band[1]):
            continue
        th = angle_min + i * angle_increment
        out.append((r * math.cos(th), r * math.sin(th), th, r))
    return out


def fit_line(points):
    """Total-least-squares line fit. Returns ``(normal_bearing, distance)``.

    The normal is chosen to point from the line back towards the origin, which for a
    wall in front of the sensor is the direction the robot is facing it from.

    Uses the eigenvector of the 2x2 scatter matrix rather than a least-squares fit of
    ``y = mx + c``: the latter blows up for a wall that happens to lie parallel to the
    y axis, which is exactly the orientation this wall has when the robot faces it
    squarely.
    """
    n = len(points)
    if n < 8:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = syy = sxy = 0.0
    for x, y, _, _ in points:
        dx, dy = x - mx, y - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    # smaller eigenvector of [[sxx, sxy], [sxy, syy]] is the normal direction
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = max(0.0, tr * tr / 4.0 - det)
    lam = tr / 2.0 - math.sqrt(disc)
    if abs(sxy) > 1e-12:
        nx, ny = lam - syy, sxy
    elif sxx <= syy:
        nx, ny = 1.0, 0.0
    else:
        nx, ny = 0.0, 1.0
    norm = math.hypot(nx, ny)
    if norm < 1e-12:
        return None
    nx, ny = nx / norm, ny / norm
    d = nx * mx + ny * my            # signed distance from the origin to the line
    if d > 0.0:                      # make the normal point back towards the origin
        nx, ny, d = -nx, -ny, -d
    return math.atan2(ny, nx), abs(d)


def wall_fit_quality(points, normal_bearing, distance):
    """RMS distance of the points from the fitted line, metres.

    A curved or cluttered return set fits a line badly, so this is the number to gate
    on before trusting the result.
    """
    nx, ny = math.cos(normal_bearing), math.sin(normal_bearing)
    if not points:
        return float('inf')
    total = sum((nx * x + ny * y + distance) ** 2 for x, y, _, _ in points)
    return math.sqrt(total / len(points))


def heading_from_scan(ranges, angle_min, angle_increment, range_min, range_max,
                      band=None, max_rms=0.02, min_beams=12,
                      max_spread=math.radians(150)):
    """Absolute robot heading from a scan of the back wall.

    Returns ``(heading, info)``, or ``(None, info)`` with a reason if the wall could
    not be identified. ``heading`` is in the same convention as odometry: the world
    angle the robot's +x axis points along.

    Derivation. The wall's inward normal is world azimuth 0. If the robot's heading is
    ``h`` then that same direction, expressed in the robot frame, is at bearing ``-h``.
    The fit returns exactly that bearing, so ``h = -normal_bearing``.
    """
    pts = scan_points(ranges, angle_min, angle_increment, range_min, range_max, band)
    info = {'beams': len(pts)}
    if len(pts) < min_beams:
        info['reject'] = f'only {len(pts)} beams in the range band'
        return None, info
    bearings = sorted(p[2] for p in pts)
    spread = bearings[-1] - bearings[0]
    # a wrapped set (returns either side of +/-pi) needs the gap-based spread instead
    gaps = [b - a for a, b in zip(bearings, bearings[1:])]
    if gaps and max(gaps) > math.radians(20):
        spread = 2 * math.pi - max(gaps)
    info['spread_deg'] = math.degrees(spread)
    if spread > max_spread:
        info['reject'] = f'returns span {math.degrees(spread):.0f} deg - not one wall'
        return None, info

    fit = fit_line(pts)
    if fit is None:
        info['reject'] = 'line fit failed'
        return None, info
    normal_bearing, distance = fit
    rms = wall_fit_quality(pts, normal_bearing, distance)
    info.update({'rms': rms, 'distance': distance,
                 'normal_bearing': normal_bearing})
    if rms > max_rms:
        info['reject'] = f'line fit rms {rms * 1000:.0f} mm - not a flat wall'
        return None, info
    return -normal_bearing + WALL_NORMAL_AZIMUTH, info


# --------------------------------------------------------------- FULL POSE
# The two reference planes. A is x = -WALL_A_X with inward normal +x (azimuth 0);
# B is y = -WALL_B_Y with inward normal +y (azimuth pi/2). Together they form an
# L-corner, which is what makes a full pose observable.
#
# THESE ARE THE WALL FACES THE BEAMS HIT, NOT THE WALL CENTRES. Both walls - in
# reorient.sdf and in the pick-place shelf model, which are the same two walls - are
# 20 mm boxes centred at -0.46, so the faces the LiDAR sees are at
# -0.45. This used to read 0.46 - the centre - so every pose fix put the base 10 mm too
# far back in x AND in y, and every grasp was aimed at (+10, +10) mm in the world:
# 7.1 mm across the closing axis for the bottle (axis 75 deg), 12-14 mm along the axis
# for both objects. Against a ~+/-3 mm pad window that is the one-finger-in, one-flush
# overlap in the operator's photos. The harness could not see it because its synthetic
# scan used the same constant. test_the_wall_constants_are_the_faces_in_the_world now
# derives both faces from the SDF.
WALL_A_X = 0.45
WALL_B_Y = 0.45


def _fit_rms(points):
    """Fit a line and return ``(bearing, distance, rms)``, or None."""
    f = fit_line(points)
    if f is None:
        return None
    return f[0], f[1], wall_fit_quality(points, f[0], f[1])


def pose_from_scan(ranges, angle_min, angle_increment, range_min, range_max,
                   lidar_offset=(-0.10, 0.0), band=None, max_rms=0.02,
                   min_beams=24, min_each=10, expect_heading=0.0):
    """Full robot pose ``(x, y, heading)`` from two perpendicular reference walls.

    Returns ``(pose, info)``, or ``(None, info)`` with a reason.

    HOW THE TWO WALLS ARE SEPARATED, AND WHY NOT BY SEARCHING FOR THE CORNER.
    The first version looked for the bend: it swept every split of the
    bearing-ordered returns and kept the one minimising the combined residual. That
    works on a clean two-wall scan and failed almost every time in simulation -
    "could not separate two perpendicular walls" on nine attempts out of ten. A real
    scan is not two tidy segments: the robot's own mast occludes part of it, returns
    drop out, and any third surface in the band adds a segment the search tries to
    treat as a wall.

    Using the heading PRIOR instead makes it robust. Odometry is wrong by tens of
    degrees but never by ninety, and that is all the accuracy needed to say which
    returns belong to which wall: wall A's surface lies around bearing
    ``-expect_heading`` and wall B's around ``pi/2 - expect_heading``, so each point
    is assigned to whichever of those two directions it is nearer. Each wall is then
    fitted on its own points, independently, and judged on its own residual.

    A pose is returned ONLY if both walls fit. There is deliberately no partial
    result: an earlier version returned ``(x, None, heading)`` when wall B was
    missing, and the very first caller did arithmetic on that None and crashed the
    whole task before it touched an object. A half-known pose is not worth the hazard
    of a tuple that sometimes contains None - the caller falls back to its nominal aim
    instead, which is well defined and safe.
    """
    pts = scan_points(ranges, angle_min, angle_increment, range_min, range_max,
                      band)
    info = {'beams': len(pts)}
    if len(pts) < min_beams:
        info['reject'] = f'only {len(pts)} beams in the range band'
        return None, info

    # THE DIRECTION A WALL IS SEEN IN IS OPPOSITE ITS INWARD NORMAL. Wall A's inward
    # normal (wall towards robot) is world +x, so the robot LOOKS at wall A along
    # world azimuth pi - i.e. bearing pi - heading in its own frame. Wall B's normal
    # is world +y, so it is looked at along -pi/2 - heading. Centring the sectors on
    # the normals instead put every point in the wrong group and every fit failed with
    # a huge residual.
    want_a = -expect_heading                      # where wall A's NORMAL points
    want_b = math.pi / 2.0 - expect_heading       # where wall B's NORMAL points
    look_a = math.pi - expect_heading             # where wall A IS
    look_b = -math.pi / 2.0 - expect_heading      # where wall B IS

    def angdiff(x, y):
        return abs(math.remainder(x - y, 2 * math.pi))

    group_a, group_b = [], []
    for q in pts:
        (group_a if angdiff(q[2], look_a) <= angdiff(q[2], look_b)
         else group_b).append(q)

    # REFINE. The bearing split puts the boundary half way between the two look
    # directions, which lands a handful of points from one wall in the other group -
    # near the corner the two are seen along almost the same bearing. A few stray
    # points barely move a long wall's fit but badly corrupt a short one, which is why
    # wall B's residual kept failing. Two passes of reassigning each point to whichever
    # fitted line it is actually closer to cleans that up.
    for _ in range(2):
        if len(group_a) < min_each or len(group_b) < min_each:
            break
        la, lb = fit_line(group_a), fit_line(group_b)
        if la is None or lb is None:
            break
        na = (math.cos(la[0]), math.sin(la[0]))
        nb_ = (math.cos(lb[0]), math.sin(lb[0]))
        new_a, new_b = [], []
        for q in pts:
            da_ = abs(na[0] * q[0] + na[1] * q[1] + la[1])
            db_ = abs(nb_[0] * q[0] + nb_[1] * q[1] + lb[1])
            (new_a if da_ <= db_ else new_b).append(q)
        if len(new_a) < min_each or len(new_b) < min_each:
            break
        group_a, group_b = new_a, new_b

    info['beams_a'], info['beams_b'] = len(group_a), len(group_b)
    fa = _fit_rms(group_a) if len(group_a) >= min_each else None
    fb = _fit_rms(group_b) if len(group_b) >= min_each else None
    if fa is None or fa[2] > max_rms:
        info['reject'] = ('wall A not found'
                          if fa is None else
                          f'wall A fit rms {fa[2] * 1000:.0f} mm')
        if fa:
            info['rms'] = fa[2]
        return None, info

    ba, da, rms_a = fa
    if angdiff(ba, want_a) > math.radians(60):
        info['reject'] = (f'wall A normal {math.degrees(ba):.0f} deg is '
                          f'{math.degrees(angdiff(ba, want_a)):.0f} deg from '
                          f'the prior')
        return None, info

    heading = -ba
    c, sn = math.cos(heading), math.sin(heading)
    ox, oy = lidar_offset
    lx = da - WALL_A_X
    info.update({'heading': heading, 'dist_a': da, 'rms': rms_a,
                 'normal_a': ba})

    if fb is None or fb[2] > max_rms or angdiff(fb[0], want_b) > math.radians(60):
        info['reject'] = ('wall B not found (%d beams)' % info['beams_b']
                          if fb is None else
                          f'wall B fit rms {fb[2] * 1000:.0f} mm '
                          f'({info["beams_b"]} beams)')
        return None, info

    bb, db, rms_b = fb
    ly = db - WALL_B_Y
    info.update({'dist_b': db, 'normal_b': bb, 'rms': max(rms_a, rms_b)})
    x = lx - (ox * c - oy * sn)
    y = ly - (ox * sn + oy * c)
    return (x, y, heading), info


def aim_target(pose_fix, azimuth, radius, nominal_reach,
               nominal_axis=math.pi / 2.0):
    """Where to aim, in the base frame, given a pose fix that may be absent.

    Returns ``(x, y, axis, corrected)``. This is the whole decision, kept as a pure
    function so it can be tested without ROS - both runtime failures in this section
    lived in exactly this logic and neither was visible to a geometry-only test:

    * a stale fix was reused because the guard asked "is it None" rather than "is it
      fresh", aiming the arm using a pose measured 60 degrees earlier;
    * a partial fix returned ``(x, None, heading)`` and the caller did arithmetic on
      the None, crashing before the robot touched anything.

    ``pose_fix`` must therefore be either a complete ``(x, y, heading)`` or ``None``.
    Anything else raises rather than silently misbehaving.
    """
    if pose_fix is None:
        return nominal_reach, 0.0, nominal_axis, False
    if len(pose_fix) != 3 or any(v is None for v in pose_fix):
        raise ValueError(f'pose_fix must be a complete (x, y, heading) or None, '
                         f'got {pose_fix!r}')
    world = (radius * math.cos(azimuth), radius * math.sin(azimuth))
    tx, ty = world_to_base(world, pose_fix)
    axis = math.remainder(azimuth + math.pi / 2.0 - pose_fix[2], 2 * math.pi)
    return tx, ty, axis, True


def world_to_base(point, pose):
    """Express a world point in the robot's base frame, given its pose."""
    x, y, h = pose
    dx, dy = point[0] - x, point[1] - y
    c, s = math.cos(h), math.sin(h)
    return (c * dx + s * dy, -s * dx + c * dy)


# ------------------------------------------------------------------ test aid
def synth_scan(true_heading, wall_x, half_width, lidar_xy=(-0.10, 0.0),
               n=360, range_min=0.12, range_max=8.0, noise=0.0, seed=0,
               base_xy=(0.0, 0.0), wall_b_y=None, wall_b_span=None):
    """Synthetic 360-degree scan of one or two flat walls.

    Wall A is the plane ``x = -wall_x`` spanning ``|y| <= half_width``. If ``wall_b_y``
    is given, a second wall on the plane ``y = -wall_b_y`` spanning
    ``wall_b_span = (x_lo, x_hi)`` is added, forming the L-corner that makes a full
    pose observable.

    Beams that miss both walls return ``inf``, as a real scan does. Used by the test
    suite to drive the fit from a known pose and check the pose comes back.
    """
    import random
    rng = random.Random(seed)
    c, s = math.cos(true_heading), math.sin(true_heading)
    lx = base_xy[0] + lidar_xy[0] * c - lidar_xy[1] * s
    ly = base_xy[1] + lidar_xy[0] * s + lidar_xy[1] * c
    out = []
    for i in range(n):
        th = -math.pi + i * (2 * math.pi / n)
        dx = math.cos(true_heading + th)
        dy = math.sin(true_heading + th)
        hits = []
        if abs(dx) > 1e-9:
            t = (-wall_x - lx) / dx
            if range_min < t < range_max and abs(ly + t * dy) <= half_width:
                hits.append(t)
        if wall_b_y is not None and abs(dy) > 1e-9:
            t = (-wall_b_y - ly) / dy
            lo, hi = wall_b_span if wall_b_span else (-1e9, 1e9)
            if range_min < t < range_max and lo <= lx + t * dx <= hi:
                hits.append(t)
        if not hits:
            out.append(float('inf'))
            continue
        r = min(hits)
        out.append(r + (rng.gauss(0.0, noise) if noise else 0.0))
    return out, -math.pi, 2 * math.pi / n
