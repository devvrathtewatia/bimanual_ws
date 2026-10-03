"""Where the robot is in a ROOM, from the LiDAR - ROS-free, so it can be tested offline.

WHY NOT wall_ref
----------------
The fixed-base sections localise off an L of two walls a few decimetres behind the robot
(wall_ref.pose_from_scan). That works because the base never leaves the spot the L was built
round. This section drives across a room several metres wide: from most places in it the L
would be out of range, and nothing guarantees which two walls are nearest. So the reference
is the room itself - a rectangle of walls whose FACES are known - and from anywhere inside it
the LiDAR sees three or four of them.

HOW
---
1. PREDICT. From a prior pose (the last fix carried forward by odometry) every beam is cast
   into the room: which wall should it hit, and at what range?
2. ASSIGN. A return within ``gate`` of its prediction belongs to that wall. Returns near a
   corner - where two walls are seen along almost the same bearing - and returns far from
   any prediction (anything that is not a wall) are dropped.
3. FIT. Each wall's returns get wall_ref.fit_line, the same total-least-squares line as the
   L-corner: its normal gives the heading, its distance one coordinate. The headings of all
   the walls are averaged; the coordinates are solved together, so two non-parallel walls fix
   the pose and more over-determine it.
4. REFINE. Predict again from THAT pose with a tight gate and refit, which removes whatever
   the prior's error let into the wrong wall.

A pose is returned only if at least two non-parallel walls fitted, each on at least
``min_each`` returns with an rms under ``max_rms``, and they agree on the heading. There is
no partial result, for the reason wall_ref gives: a half-known pose is a hazard.

CONVENTIONS as wall_ref: beam ``i`` is at ``angle_min + i * angle_increment`` in the LiDAR
frame, which is the base frame translated by ``lidar_offset``; headings as odometry.
"""
from __future__ import annotations

import math

from . import wall_ref as wr


class Room:
    """A rectangular room: the FACES of its four walls, ``x_min < x < x_max``, ``y_min < y < y_max``.

    Each wall is ``(name, phi, e, a, b)``: its inward normal ``n = (cos phi, sin phi)`` - from
    the wall into the room - the plane ``n . p = e``, and its end points ``a``, ``b``. Inside the
    room ``n . p > e`` for every wall, and ``n . p - e`` is the distance to it.
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        c = ((self.x_min, self.y_min), (self.x_max, self.y_min),
             (self.x_max, self.y_max), (self.x_min, self.y_max))
        self.walls = [
            ('south', math.pi / 2.0, self.y_min, c[0], c[1]),
            ('east', math.pi, -self.x_max, c[1], c[2]),
            ('north', -math.pi / 2.0, -self.y_max, c[2], c[3]),
            ('west', 0.0, self.x_min, c[3], c[0]),
        ]

    def wall_distance(self, p):
        """Distance from a point to the NEAREST wall face (negative outside the room)."""
        return min(math.cos(phi) * p[0] + math.sin(phi) * p[1] - e
                   for _n, phi, e, _a, _b in self.walls)

    def cast(self, origin, direction):
        """The first wall a ray meets: ``(index, range, hit point)`` or None."""
        best = None
        for k, (_n, phi, e, a, b) in enumerate(self.walls):
            nx, ny = math.cos(phi), math.sin(phi)
            nd = nx * direction[0] + ny * direction[1]
            if nd > -1e-12:
                continue                          # moving away from (or along) this wall
            t = (e - (nx * origin[0] + ny * origin[1])) / nd
            if t <= 0.0:
                continue
            p = (origin[0] + t * direction[0], origin[1] + t * direction[1])
            tx, ty = b[0] - a[0], b[1] - a[1]
            s = ((p[0] - a[0]) * tx + (p[1] - a[1]) * ty) / (tx * tx + ty * ty)
            if -1e-9 <= s <= 1.0 + 1e-9 and (best is None or t < best[1]):
                best = (k, t, p, s * math.hypot(tx, ty), math.hypot(tx, ty))
        return best


def _lidar(pose, offset):
    x, y, h = pose
    c, s = math.cos(h), math.sin(h)
    return (x + offset[0] * c - offset[1] * s, y + offset[0] * s + offset[1] * c)


def _assign(pts, room, pose, offset, gate, corner):
    """Returns per wall index the LiDAR-frame points that belong to it, predicted from ``pose``."""
    L = _lidar(pose, offset)
    groups = {}
    for x, y, th, r in pts:
        d = (math.cos(pose[2] + th), math.sin(pose[2] + th))
        hit = room.cast(L, d)
        if hit is None:
            continue
        k, t, _p, along, length = hit
        if abs(r - t) > gate:
            continue                              # not this wall - clutter, or a bad prior
        if along < corner or along > length - corner:
            continue                              # near a corner: two walls on one bearing
        groups.setdefault(k, []).append((x, y, th, r))
    return groups


def _solve(groups, room, offset, max_rms, min_each, heading_ref, max_prior_dev):
    """Fit each wall, agree a heading, solve the position. ``(pose, info)`` or ``(None, info)``."""
    fits = []
    info = {'beams': sum(len(g) for g in groups.values()), 'walls': []}
    for k, g in sorted(groups.items()):
        if len(g) < min_each:
            continue
        f = wr.fit_line(g)
        if f is None:
            continue
        bearing, dist = f
        rms = wr.wall_fit_quality(g, bearing, dist)
        name, phi, e, _a, _b = room.walls[k]
        if rms > max_rms:
            info.setdefault('bad', []).append(f'{name} rms {rms * 1000:.0f} mm')
            continue
        h = math.remainder(phi - bearing, 2 * math.pi)
        if abs(math.remainder(h - heading_ref, 2 * math.pi)) > max_prior_dev:
            info.setdefault('bad', []).append(f'{name} heading off the prior')
            continue
        fits.append((k, h, dist, rms, len(g)))
    # AGREE THE HEADING. A wrongly assigned wall shows up as a heading that disagrees with the
    # rest; drop the worst until the survivors agree to a degree.
    while True:
        if not fits:
            info['reject'] = 'no wall fitted' + (f" ({'; '.join(info['bad'])})"
                                                  if info.get('bad') else '')
            return None, info
        w = sum(f[4] for f in fits)
        ref = fits[0][1]
        h = ref + sum(f[4] * math.remainder(f[1] - ref, 2 * math.pi) for f in fits) / w
        dev = [(abs(math.remainder(f[1] - h, 2 * math.pi)), i) for i, f in enumerate(fits)]
        worst, i = max(dev)
        if worst <= math.radians(1.0) or len(fits) <= 2:
            break
        fits.pop(i)
    if max(d for d, _ in dev) > math.radians(1.0):
        info['reject'] = 'the walls disagree on the heading by %.1f deg' % math.degrees(
            max(d for d, _ in dev))
        return None, info
    # SOLVE THE POSITION: n_k . L = e_k + d_k for every wall, least squares in L.
    a11 = a12 = a22 = b1 = b2 = 0.0
    for k, _h, dist, _rms, n in fits:
        _name, phi, e, _a, _b = room.walls[k]
        nx, ny = math.cos(phi), math.sin(phi)
        rhs = e + dist
        a11 += n * nx * nx
        a12 += n * nx * ny
        a22 += n * ny * ny
        b1 += n * nx * rhs
        b2 += n * ny * rhs
    det = a11 * a22 - a12 * a12
    wsum = a11 + a22
    if det < 0.05 * wsum * wsum:
        info['reject'] = ('only parallel walls fitted (' + ', '.join(
            room.walls[f[0]][0] for f in fits) + ') - one coordinate is unknown')
        return None, info
    lx = (a22 * b1 - a12 * b2) / det
    ly = (a11 * b2 - a12 * b1) / det
    c, s = math.cos(h), math.sin(h)
    x = lx - (offset[0] * c - offset[1] * s)
    y = ly - (offset[0] * s + offset[1] * c)
    info.update({'walls': [room.walls[f[0]][0] for f in fits],
                 'rms': max(f[3] for f in fits),
                 'beams': sum(f[4] for f in fits)})
    return (x, y, math.remainder(h, 2 * math.pi)), info


def pose_in_room(ranges, angle_min, angle_increment, range_min, range_max, room, prior,
                 lidar_offset=(-0.10, 0.0), band=(0.15, 7.5), gate=0.30, fine_gate=0.04,
                 corner=0.12, max_rms=0.02, min_each=10, max_prior_dev=math.radians(40)):
    """Full robot pose ``(x, y, heading)`` in ``room`` from one scan. ``(pose, info)``.

    ``prior`` needs to be good to a few centimetres and about 35 degrees. A square room looks
    the same every 90 degrees, so a prior more than ~45 degrees out cannot be told from a
    neighbouring heading - the search stops at 36 on purpose.

    The heading guesses are tried nearest-first. On total failure the info returned is the
    attempt that got closest, not the last one tried (the last is the worst by construction,
    which is why the old log showed ~155 mm on every wall and hid the real cause).
    """
    pts = wr.scan_points(ranges, angle_min, angle_increment, range_min, range_max, band)
    info = {'beams': len(pts)}
    if len(pts) < 2 * min_each:
        info['reject'] = f'only {len(pts)} beams in the range band'
        return None, info
    best, best_n = info, -1
    for dh in _HEADING_TRIES:
        guess = (prior[0], prior[1], prior[2] + math.radians(dh))
        pose, info = _solve(_assign(pts, room, guess, lidar_offset, gate, corner), room,
                            lidar_offset, max_rms, min_each, guess[2], max_prior_dev)
        if pose is None:
            if info.get('beams', 0) > best_n:
                best, best_n = info, info.get('beams', 0)
            continue
        pose2, info2 = _solve(_assign(pts, room, pose, lidar_offset, fine_gate, corner), room,
                              lidar_offset, max_rms, min_each, pose[2], math.radians(3))
        if pose2 is not None:
            return pose2, info2
        if info2.get('beams', 0) > best_n:
            best, best_n = info2, info2.get('beams', 0)
    best['tried_deg'] = _HEADING_TRIES[-1]
    return None, best


# Prior-heading offsets tried, nearest first. +/-36 covers a 154-164 degree turn on the
# ~0.885 slip ratio even with no slip compensation at all.
_HEADING_TRIES = (0.0, 3.0, -3.0, 7.0, -7.0, 12.0, -12.0, 18.0, -18.0,
                  24.0, -24.0, 30.0, -30.0, 36.0, -36.0)


# ------------------------------------------------------------------ test aid
def synth_scan(room, pose, lidar_offset=(-0.10, 0.0), n=360, range_min=0.12, range_max=8.0,
               noise=0.0, seed=0, clutter=()):
    """A synthetic 360-degree scan of the room from ``pose`` - what the LiDAR would return.

    ``clutter`` is a list of extra line segments ``((x0, y0), (x1, y1))`` in the world - anything
    at LiDAR height that is not a wall. Beams that hit nothing return ``inf``, as a real scan
    does. The suite and the harness drive :func:`pose_in_room` with this from a known pose.
    """
    import random
    rng = random.Random(seed)
    L = _lidar(pose, lidar_offset)
    out = []
    for i in range(n):
        th = -math.pi + i * (2 * math.pi / n)
        d = (math.cos(pose[2] + th), math.sin(pose[2] + th))
        hits = []
        hit = room.cast(L, d)
        if hit is not None:
            hits.append(hit[1])
        for (x0, y0), (x1, y1) in clutter:
            ex, ey = x1 - x0, y1 - y0
            den = d[0] * ey - d[1] * ex
            if abs(den) < 1e-12:
                continue
            t = ((x0 - L[0]) * ey - (y0 - L[1]) * ex) / den
            u = ((x0 - L[0]) * d[1] - (y0 - L[1]) * d[0]) / den
            if t > 0 and 0.0 <= u <= 1.0:
                hits.append(t)
        r = min(hits) if hits else float('inf')
        if not (range_min < r < range_max):
            out.append(float('inf'))
            continue
        out.append(r + (rng.gauss(0.0, noise) if noise else 0.0))
    return out, -math.pi, 2 * math.pi / n