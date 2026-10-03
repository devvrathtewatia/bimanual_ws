"""Colour blobs to grasp targets - the image-processing half of perception, ROS-free.

SHARED FILE, between perceive and navigate (the 'sense' group in SHARED_FILES.sha256).

The node (perception_node.py) only moves messages; everything that can be wrong lives here
and in vision.py, where the test suite can drive it with a synthetic image instead of
Gazebo. The first version kept this inside the node, where nothing could test it - and it
read a key it never loaded, so the node died on the first frame that contained an object
and the task reported every object as NOT SEEN.
"""
import cv2
import numpy as np

from . import vision as vz


def object_specs(names, param):
    """Per-object colour window and model, from the node's parameters.

    EVERY KEY THE MEASUREMENT NEEDS IS LOADED HERE, and test_perceive checks that this
    function reads everything detect() uses - the missing ``lying_cz`` / ``upright_cz`` that
    killed the node is exactly the class of fault that check exists for. Heights come from the
    object's SURFACE (``surface_z``), as in every section.
    """
    out = {}
    for n in names:
        lo, hi = param(f'{n}.hsv_lo', None), param(f'{n}.hsv_hi', None)
        if not lo or not hi:
            continue
        out[n] = {
            'hsv_lo': np.array([int(v) for v in lo], np.uint8),
            'hsv_hi': np.array([int(v) for v in hi], np.uint8),
            'length': float(param(f'{n}.length', 0.20)),
            'half_width': float(param(f'{n}.half_width', 0.045)),
            'round': str(param(f'{n}.round', True)).strip().lower() not in ('false', '0', 'no'),
            'surface_z': float(param(f'{n}.surface_z', 0.0)),
        }
    return out


KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# how near another object's blob counts as TOUCHING it: across the one or two antialiased
# pixels at a boundary, which belong to neither colour window
TOUCH = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def detect(img, encoding, specs, camera, min_area=250, edge=3, v_max=None):
    """Every configured object in one image: ``{name: detection}``.

    A detection is vision.fit()'s dict plus ``area_px``, ``touching`` and ``clipped``. CLIPPED
    means the blob touches the image edge, so part of the object is out of view and its
    silhouette - and therefore its fitted pose - is not the whole object's; the task refuses
    those. An object that is simply not in view is ``{'found': False}``.

    ``v_max`` is the lowest image row the camera sees the world in, for a camera whose view
    the robot's own arms cut off from below - the mast camera with the arms in the look pose.
    An arm is not the object's colour, so a blob it cuts does not touch the image edge; it
    just comes out short, and its fitted pose with it. So a blob reaching past ``v_max`` is
    CLIPPED exactly as one touching the edge is.

    A blob that TOUCHES ANOTHER OBJECT'S AND IS BEHIND IT is clipped too. Seen across a room one
    object can stand partly behind another, and the part that shows fits to a pose nowhere near
    the object - a standing box behind a bottle measured 288 mm out. The edge of the hidden part
    is the other object's outline, so the two blobs meet (``touching``). Which is behind: on one
    floor, seen from above it, whatever is nearer reaches LOWER in the image - its foot is lower,
    and anything hidden behind it is hidden above its foot - so of two touching blobs the one
    whose lowest pixel is higher is behind (``behind``). Too close to call, both are refused;
    the one in front is kept, whole.
    """
    code = cv2.COLOR_RGB2HSV if encoding == 'rgb8' else cv2.COLOR_BGR2HSV
    hsv = cv2.cvtColor(img, code)
    h, w = hsv.shape[:2]
    masks = {}
    for name, sp in specs.items():
        mask = cv2.inRange(hsv, sp['hsv_lo'], sp['hsv_hi'])
        # close gaps: shading can split a blob, and the largest fragment would then
        # misreport both the centre and the axis
        masks[name] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    out, foot = {}, {}
    for name, sp in specs.items():
        mask = masks[name]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        det = {'found': False}
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            if area >= min_area:
                pts = [(float(p[0]), float(p[1])) for p in c.reshape(-1, 2)]
                bottom = h - 1 - edge if v_max is None else min(h - 1, v_max) - edge
                clipped = any(p[0] < edge or p[1] < edge or p[0] > w - 1 - edge
                              or p[1] > bottom for p in pts)
                blob = np.zeros_like(mask)
                cv2.drawContours(blob, [c], -1, 255, cv2.FILLED)
                near = cv2.dilate(blob, TOUCH)
                touching = sorted(o for o, m in masks.items()
                                  if o != name and cv2.countNonZero(cv2.bitwise_and(near, m)))
                det = vz.fit(pts, sp['length'], sp['half_width'], sp['round'], camera,
                             sp['surface_z'])
                det['area_px'] = round(area, 1)
                det['touching'] = touching
                det['clipped'] = bool(clipped)
                foot[name] = max(p[1] for p in pts)
        out[name] = det
    for name, det in out.items():
        if not det.get('found'):
            continue
        det['behind'] = sorted(o for o in det['touching']
                               if foot.get(o) is None or foot[name] < foot[o] + 2.0)
        det['clipped'] = bool(det['clipped'] or det['behind'])
    return out
