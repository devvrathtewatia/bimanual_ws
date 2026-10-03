"""Camera geometry and object measurement - ROS-free so it can be tested offline.

SHARED FILE, between perceive and navigate (the 'sense' group in SHARED_FILES.sha256). The
same model and fit serve the workspace camera over the reach and the mast camera across the
room: nothing below assumes which camera, how far, or how steeply it looks down.

WHAT THE CAMERA ACTUALLY SEES, AND WHY THE FIRST TWO MEASUREMENTS WERE WRONG
--------------------------------------------------------------------------
The workspace camera looks down at 71 deg, not straight down, so an object's image is the
silhouette of a SOLID seen from an angle - the top AND the near side. The first version
treated the blob as a flat rectangle at the object's mid-height; a second version at least
compared the two resting states, but still against that flat rectangle. For a lying object
that is nearly right (its silhouette really is close to a length x width rectangle). For a
STANDING one it is badly wrong: a 0.24 m bottle seen from 25 deg off vertical is a blob 1.7
times longer than it is wide, pointing at the camera - so it measured as LYING, 59 mm too far
away, with its axis pointing straight at the robot, and was then rejected as ungraspable.

So nothing here assumes a shape for the blob. For each hypothesis - lying at some axis, or
standing - the object's TRUE silhouette is predicted by projecting the solid through the
camera model (the silhouette of a convex body is the convex hull of its projected outline),
its pose is fitted until the predicted silhouette's rectangle lands on the observed one, and
the hypothesis whose silhouette then also matches in SIZE wins. The test suite round-trips
exactly that: render a known pose as a silhouette, measure it, get the pose back.

CONVENTIONS
-----------
Gazebo's camera frame is **+x along the optical axis, +y to the left, +z up**. Image
``u`` therefore increases to the right (along -y) and ``v`` increases downward
(along -z). The camera link is the base frame pitched about +y, so a positive pitch
tips the optical axis down.

A blob is described by the AREA MOMENTS of its convex outline: centroid, principal direction
and the spread along and across it. Not by a minimum-area rectangle: a standing cylinder seen
at an angle is a rounded "stadium", whose minimum rectangle flips between orientations under
a single pixel of quantisation and moves its centre 13 px when it does. Moments are smooth for
any shape, and they are computed the same way here for the observed outline and the predicted
one, so the node never depends on cv2's version-dependent minAreaRect angle convention.
"""
from __future__ import annotations

import math


class CameraModel:
    """A pinhole camera rigidly mounted on the base, pitched down about +y.

    Parameters
    ----------
    position:
        ``(x, y, z)`` of the camera in ``base_link``.
    pitch:
        Downward tilt in radians (positive tips the optical axis towards the floor).
    fx, fy, cu, cv:
        Intrinsics. If only a horizontal field of view is known, use
        :meth:`from_fov`.
    width, height:
        The image size, for deciding whether a projection is inside the frame.
    """

    def __init__(self, position, pitch, fx, fy, cu, cv, width=640, height=480):
        self.position = tuple(float(v) for v in position)
        self.pitch = float(pitch)
        self.fx, self.fy = float(fx), float(fy)
        self.cu, self.cv = float(cu), float(cv)
        self.width, self.height = int(width), int(height)

    @classmethod
    def from_fov(cls, position, pitch, horizontal_fov, width, height):
        """Build from the SDF's ``<horizontal_fov>`` and image size.

        Gazebo derives both focal lengths from the horizontal field of view, so
        ``fy == fx``; using ``height`` to compute ``fy`` separately would give a
        different camera from the one being simulated.
        """
        fx = width / (2.0 * math.tan(horizontal_fov / 2.0))
        return cls(position, pitch, fx, fx, width / 2.0, height / 2.0, width, height)

    # ------------------------------------------------------------ projections
    def ray(self, u, v):
        """Unit-ish direction in BASE coordinates for the ray through pixel (u, v)."""
        d_cam = (1.0, -(u - self.cu) / self.fx, -(v - self.cv) / self.fy)
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        # rotate camera -> base about +y
        return (c * d_cam[0] + s * d_cam[2],
                d_cam[1],
                -s * d_cam[0] + c * d_cam[2])

    def to_plane(self, u, v, plane_z):
        """Intersect the ray through (u, v) with the horizontal plane z = plane_z."""
        d = self.ray(u, v)
        if d[2] > -1e-9:
            return None
        t = (plane_z - self.position[2]) / d[2]
        return (self.position[0] + t * d[0], self.position[1] + t * d[1])

    def to_pixel(self, point):
        """Project a base-frame point to pixels. Inverse of :meth:`to_plane`.

        Returns ``None`` for anything at or behind the image plane.
        """
        px, py, pz = point
        dx = px - self.position[0]
        dy = py - self.position[1]
        dz = pz - self.position[2]
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        # rotate base -> camera about +y (transpose of the above)
        xc = c * dx - s * dz
        yc = dy
        zc = s * dx + c * dz
        if xc <= 1e-9:
            return None
        return (self.cu - self.fx * (yc / xc), self.cv - self.fy * (zc / xc))

    def in_frame(self, px, margin=2.0):
        """Is a pixel inside the image, ``margin`` px clear of its edge?"""
        return (margin <= px[0] <= self.width - 1 - margin
                and margin <= px[1] <= self.height - 1 - margin)

    def metres_per_pixel(self, plane_z):
        """Approximate scale at the plane, for sanity checks and error budgets."""
        a = self.to_plane(self.cu, self.cv, plane_z)
        b = self.to_plane(self.cu + 10.0, self.cv, plane_z)
        if a is None or b is None:
            return float('nan')
        return math.dist(a, b) / 10.0


# --------------------------------------------------------------- 2-D geometry
def convex_hull(points):
    """Convex hull of 2-D points, counter-clockwise, by the monotone chain."""
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def moments(points):
    """Area moments of a polygon (its convex hull): ``(u, v, spread_long, spread_short, deg)``.

    Centroid, then the standard deviations of the area along and across its principal axis,
    and the direction of that axis in image coordinates, folded to [0, 180). For a rectangle
    the spreads are side / sqrt(12); for an ellipse, semi-axis / 2.
    """
    hull = convex_hull(points)
    if len(hull) < 3:
        u = sum(p[0] for p in hull) / max(1, len(hull))
        v = sum(p[1] for p in hull) / max(1, len(hull))
        return u, v, 0.0, 0.0, 0.0
    a = cx = cy = 0.0
    n = len(hull)
    for i in range(n):
        x0, y0 = hull[i]
        x1, y1 = hull[(i + 1) % n]
        c = x0 * y1 - x1 * y0
        a += c
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    a *= 0.5
    cx /= 6.0 * a
    cy /= 6.0 * a
    sxx = syy = sxy = 0.0
    for i in range(n):
        x0, y0 = hull[i][0] - cx, hull[i][1] - cy
        x1, y1 = hull[(i + 1) % n][0] - cx, hull[(i + 1) % n][1] - cy
        c = x0 * y1 - x1 * y0
        sxx += (x0 * x0 + x0 * x1 + x1 * x1) * c
        syy += (y0 * y0 + y0 * y1 + y1 * y1) * c
        sxy += (x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * c
    sxx /= 12.0 * a
    syy /= 12.0 * a
    sxy /= 24.0 * a
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, tr * tr / 4.0 - det))
    l1, l2 = tr / 2.0 + disc, max(0.0, tr / 2.0 - disc)
    th = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return cx, cy, math.sqrt(l1), math.sqrt(l2), math.degrees(th) % 180.0


# --------------------------------------------------------------- the object
def _frame(axis_angle, upright):
    """Orthonormal (along, across, third) axes of an object lying or standing.

    Lying: along its axis in the horizontal plane, a flat face (for a cuboid) down. Standing:
    along +z, ``axis_angle`` then being the yaw of a cuboid's faces (irrelevant for a round
    object).
    """
    c, s = math.cos(axis_angle), math.sin(axis_angle)
    if upright:
        return (0.0, 0.0, 1.0), (c, s, 0.0), (-s, c, 0.0)
    return (c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)


def outline_points(centre, axis_angle, upright, length, half_width, round_, n=32):
    """3-D points whose projections' convex hull IS the object's silhouette.

    A convex solid's silhouette is the convex hull of the projections of its outline: for a
    cylinder, its two end circles; for a cuboid, its eight corners.
    """
    u, v, w = _frame(axis_angle, upright)
    half = length / 2.0
    out = []
    if round_:
        for sgn in (-1.0, 1.0):
            for i in range(n):
                t = 2.0 * math.pi * i / n
                cv_, sv_ = math.cos(t) * half_width, math.sin(t) * half_width
                out.append(tuple(centre[k] + sgn * half * u[k] + cv_ * v[k] + sv_ * w[k]
                                 for k in range(3)))
        return out
    for a in (-1.0, 1.0):
        for b in (-1.0, 1.0):
            for c in (-1.0, 1.0):
                out.append(tuple(centre[k] + a * half * u[k] + b * half_width * v[k]
                                 + c * half_width * w[k] for k in range(3)))
    return out


def silhouette(camera, centre, axis_angle, upright, length, half_width, round_):
    """The object's silhouette in the image: (hull pixels, fully in frame?). None if unseen."""
    px = [camera.to_pixel(p) for p in outline_points(centre, axis_angle, upright, length,
                                                     half_width, round_)]
    if any(p is None for p in px):
        return None
    return convex_hull(px), all(camera.in_frame(p) for p in px)


def centre_height(surface_z, upright, length, half_width):
    """The object's centre height on its surface - from the SURFACE, as everywhere."""
    return surface_z + (length / 2.0 if upright else half_width)


def _solve(J, r):
    """Least-squares step for a small Gauss-Newton: (J^T J) dx = -J^T r."""
    n = len(J[0])
    A = [[sum(J[k][i] * J[k][j] for k in range(len(J))) for j in range(n)] for i in range(n)]
    b = [-sum(J[k][i] * r[k] for k in range(len(J))) for i in range(n)]
    for i in range(n):
        A[i][i] += 1e-9
    # Gaussian elimination, n <= 3
    for i in range(n):
        piv = max(range(i, n), key=lambda k: abs(A[k][i]))
        A[i], A[piv] = A[piv], A[i]
        b[i], b[piv] = b[piv], b[i]
        if abs(A[i][i]) < 1e-15:
            return [0.0] * n
        for k in range(i + 1, n):
            f = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= f * A[i][j]
            b[k] -= f * b[i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x


def _angle_px(a_deg, b_deg, spread):
    """Difference of two line directions (deg, folded to +/-90) as a pixel displacement at
    two spreads out along the axis - so it weighs like a centroid residual."""
    d = (a_deg - b_deg + 90.0) % 180.0 - 90.0
    return math.radians(d) * 2.0 * spread


def fit(outline, length, half_width, round_, camera, surface_z=0.0, iters=10):
    """Turn an observed blob outline into the grasp-target contract.

    ``outline`` is the blob's boundary pixels (any order; only its convex hull matters). For
    each resting state the pose is fitted by Gauss-Newton until the PREDICTED silhouette has
    the observed centroid and (lying) principal direction; the state whose predicted
    silhouette then also has the observed SPREADS wins. Returns a dict: ``found``, ``x``,
    ``y`` (base frame), ``axis`` (the long axis, lying; a standing cuboid's face yaw),
    ``upright``, ``fit_px`` (the winner's mismatch), ``alt_px`` (the other state's best) and
    ``cz`` (the centre height used).
    """
    uc, vc, lo, sh, ang = moments(outline)

    def pose(upright, a0):
        """Fit (x, y) - and the axis, lying - from a starting direction; score the result."""
        cz = centre_height(surface_z, upright, length, half_width)
        g = camera.to_plane(uc, vc, cz)
        if g is None:
            return None
        x, y = g
        a = a0
        free_a = not upright

        def residual(x_, y_, a_):
            s = silhouette(camera, (x_, y_, cz), a_, upright, length, half_width, round_)
            if s is None:
                return None
            pu, pv, plo, psh, pang = moments(s[0])
            r = [pu - uc, pv - vc]
            if free_a:
                r.append(_angle_px(pang, ang, lo))
            return r, (plo, psh)
        for _ in range(iters):
            base = residual(x, y, a)
            if base is None:
                return None
            r0 = base[0]
            cols = []
            for dx, dy, da in ((1e-3, 0, 0), (0, 1e-3, 0), (0, 0, 1e-3))[:3 if free_a else 2]:
                ri = residual(x + dx, y + dy, a + da)
                if ri is None:
                    return None
                h = dx or dy or da
                cols.append([(ri[0][k] - r0[k]) / h for k in range(len(r0))])
            J = [[cols[j][k] for j in range(len(cols))] for k in range(len(r0))]
            step = _solve(J, r0)
            x, y = x + step[0], y + step[1]
            if free_a:
                a += step[2]
            if max(abs(v) for v in step) < 1e-7:
                break
        final = residual(x, y, a)
        if final is None:
            return None
        r, (plo, psh) = final
        # spreads are about a third of the silhouette's sides, so x2 puts the mismatch in
        # units comparable with the blob's size
        size_px = 2.0 * (abs(plo - lo) + abs(psh - sh))
        return (size_px + sum(abs(v) for v in r), upright, x, y, a, cz)

    cands = []
    # LYING: the axis IS the principal direction - start from it, projected onto the plane.
    cz = centre_height(surface_z, False, length, half_width)
    th = math.radians(ang)
    p1 = camera.to_plane(uc + 20.0 * math.cos(th), vc + 20.0 * math.sin(th), cz)
    p2 = camera.to_plane(uc - 20.0 * math.cos(th), vc - 20.0 * math.sin(th), cz)
    if p1 is not None and p2 is not None:
        c = pose(False, math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        if c is not None:
            cands.append(c)
    # STANDING: a cylinder has no direction to find. A cuboid's face yaw cannot be read from
    # the principal direction - standing, every silhouette is elongated radially, towards the
    # camera - so it is searched over a quarter turn (a square section repeats every 90 deg),
    # coarsely and then by golden section round the best, with the position fitted at each.
    if round_:
        c = pose(True, 0.0)
        if c is not None:
            cands.append(c)
    else:
        coarse = [c for c in (pose(True, math.radians(d)) for d in range(0, 90, 15)) if c]
        if coarse:
            best = min(coarse, key=lambda c: c[0])
            lo_a, hi_a = best[4] - math.radians(7.5), best[4] + math.radians(7.5)
            g = (math.sqrt(5.0) - 1.0) / 2.0
            for _ in range(12):
                m1, m2 = hi_a - g * (hi_a - lo_a), lo_a + g * (hi_a - lo_a)
                c1, c2 = pose(True, m1), pose(True, m2)
                if c1 is None or c2 is None:
                    break
                if c1[0] < c2[0]:
                    hi_a = m2
                else:
                    lo_a = m1
            fine = pose(True, 0.5 * (lo_a + hi_a))
            cands.append(min([c for c in (best, fine) if c], key=lambda c: c[0]))
    if not cands:
        return {'found': False}
    cands.sort(key=lambda c: c[0])
    score, upright, x, y, a, cz = cands[0]
    alt = min((c[0] for c in cands if c[1] != upright), default=float('inf'))
    return {
        'found': True,
        'x': round(x, 4),
        'y': round(y, 4),
        'axis': round(math.remainder(a, math.pi), 4),
        'upright': bool(upright),
        'fit_px': round(score, 2),
        'alt_px': round(alt, 2),
        'cz': round(cz, 4),
    }


def observe(camera, centre, axis_angle, upright, length, half_width, round_, quantise=True):
    """A synthetic observation of a known pose: the blob outline a camera would see.

    What the test suite and the harness feed :func:`fit` with. The silhouette is projected
    from the solid, and snapped to whole pixels as a real blob's contour is. Returns
    ``(outline_pixels, fully_in_frame)`` or None if the object is not in front of the camera.
    """
    s = silhouette(camera, centre, axis_angle, upright, length, half_width, round_)
    if s is None:
        return None
    hull, inside = s
    pts = [(round(p[0]), round(p[1])) for p in hull] if quantise else hull
    return pts, inside
