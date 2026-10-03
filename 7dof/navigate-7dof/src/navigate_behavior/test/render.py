"""A small ray-caster: what the workspace camera sees, as an RGB image. Test-only, NumPy.

It exists so the whole perception path - colour threshold, contour, silhouette fit - can be
run on an IMAGE offline, not only on geometry. The objects are the exact solids of the models
(capped cylinders, boxes) in the models' own colours, lit by the world's sun with an ambient
term, over the grey floor; the robot's own arms can be drawn in as occluders.

Pixel (u, v) looks along vision.CameraModel.ray(u, v), so the image and the measurement agree
on the camera by construction; what this checks is everything BETWEEN them.
"""
import math

import numpy as np

FLOOR_RGB = (0.85, 0.85, 0.85)
ARM_RGB = (0.42, 0.45, 0.50)
SUN = np.array([-0.4, 0.2, -0.9]) / np.linalg.norm([-0.4, 0.2, -0.9])
AMBIENT = 0.35


def _rays(camera):
    u, v = np.meshgrid(np.arange(camera.width) + 0.0, np.arange(camera.height) + 0.0)
    d_cam = np.stack([np.ones_like(u), -(u - camera.cu) / camera.fx,
                      -(v - camera.cv) / camera.fy], -1)
    c, s = math.cos(camera.pitch), math.sin(camera.pitch)
    d = np.stack([c * d_cam[..., 0] + s * d_cam[..., 2], d_cam[..., 1],
                  -s * d_cam[..., 0] + c * d_cam[..., 2]], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _frame(axis_angle, upright):
    c, s = math.cos(axis_angle), math.sin(axis_angle)
    if upright:
        return np.array([0, 0, 1.0]), np.array([c, s, 0.0]), np.array([-s, c, 0.0])
    return np.array([c, s, 0.0]), np.array([-s, c, 0.0]), np.array([0, 0, 1.0])


def _hit_box(o, d, centre, frame, half):
    """Slab test in the box's own frame. Returns (t, normal) with t = inf on a miss."""
    R = np.stack(frame)                          # rows: local axes in the world
    ol = (o - centre) @ R.T
    dl = d @ R.T
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (-np.array(half) - ol) / dl
        t2 = (np.array(half) - ol) / dl
    tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
    hit = (tmax >= tmin) & (tmax > 0)
    t = np.where(hit, np.where(tmin > 0, tmin, tmax), np.inf)
    # the face hit: the axis whose slab entry equals tmin
    ent = np.minimum(t1, t2)
    k = np.argmax(ent, axis=-1)
    sgn = -np.sign(np.take_along_axis(dl, k[..., None], -1)[..., 0])
    n_local = np.zeros(d.shape)
    np.put_along_axis(n_local, k[..., None], sgn[..., None], -1)
    return t, n_local @ R


def _hit_cylinder(o, d, centre, frame, half_len, r):
    """Capped cylinder along frame[0]. Returns (t, normal)."""
    u = frame[0]
    oc = o - centre
    du = d @ u
    ou = oc @ u
    dp = d - du[..., None] * u
    op = oc - ou * u
    a = np.sum(dp * dp, -1)
    b = 2 * np.sum(dp * op, -1)
    c = np.sum(op * op) - r * r
    disc = b * b - 4 * a * c
    t_side = np.full(a.shape, np.inf)
    ok = (disc >= 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0))
    t0 = (-b - sq) / np.where(ok, 2 * a, 1)
    z0 = ou + t0 * du
    side = ok & (t0 > 0) & (np.abs(z0) <= half_len)
    t_side = np.where(side, t0, np.inf)
    n_side = (op + t0[..., None] * dp)
    n_side = n_side / np.maximum(np.linalg.norm(n_side, axis=-1, keepdims=True), 1e-12)
    t_cap = np.full(a.shape, np.inf)
    n_cap = np.zeros(d.shape)
    for sg in (-1.0, 1.0):
        with np.errstate(divide='ignore', invalid='ignore'):
            tc = (sg * half_len - ou) / du
        pc = op + tc[..., None] * dp
        inside = (tc > 0) & (np.sum(pc * pc, -1) <= r * r)
        better = inside & (tc < t_cap)
        t_cap = np.where(better, tc, t_cap)
        n_cap = np.where(better[..., None], sg * u, n_cap)
    use_cap = t_cap < t_side
    return np.where(use_cap, t_cap, t_side), np.where(use_cap[..., None], n_cap, n_side)


def render(camera, objects, arms=()):
    """RGB uint8 image. ``objects``: [(centre, axis_angle, upright, length, half_width,
    round_, rgb)]; ``arms``: [(p0, p1, radius)] capsules drawn as grey occluders."""
    d = _rays(camera)
    o = np.array(camera.position)
    with np.errstate(divide='ignore'):
        t_floor = np.where(d[..., 2] < 0, -o[2] / d[..., 2], np.inf)
    best_t = t_floor
    shade = np.ones(best_t.shape) * max(AMBIENT, -SUN[2])
    colour = np.broadcast_to(np.array(FLOOR_RGB), d.shape).copy()
    for centre, a, up, L, hw, rnd, rgb in objects:
        fr = _frame(a, up)
        c = np.array(centre, float)
        if rnd:
            t, n = _hit_cylinder(o, d, c, fr, L / 2.0, hw)
        else:
            t, n = _hit_box(o, d, c, fr, (L / 2.0, hw, hw))
        near = t < best_t
        lam = np.clip(-(n @ SUN), 0, 1)
        best_t = np.where(near, t, best_t)
        shade = np.where(near, AMBIENT + (1 - AMBIENT) * lam, shade)
        colour = np.where(near[..., None], np.array(rgb), colour)
    for p0, p1, r in arms:
        p0, p1 = np.array(p0), np.array(p1)
        ax = p1 - p0
        L = float(np.linalg.norm(ax))
        if L < 1e-9:
            continue
        fr = (ax / L, None, None)
        t, n = _hit_cylinder(o, d, (p0 + p1) / 2.0, fr, L / 2.0, r)
        near = t < best_t
        best_t = np.where(near, t, best_t)
        shade = np.where(near, AMBIENT, shade)
        colour = np.where(near[..., None], np.array(ARM_RGB), colour)
    img = np.clip(colour * shade[..., None], 0, 1)
    return (img * 255).astype(np.uint8)
