"""Seeing the objects: the camera half of a task node, SHARED by every section that looks.

SHARED FILE, between perceive_behavior and navigate_behavior (the 'sense' group in
SHARED_FILES.sha256), with vision.py, detect.py, looking.py and perception_node.py. Perceive
looks from a fixed base; navigate looks from wherever it has driven to. What looking IS - arms
out of the view, a median of fresh frames, the pose placed in the world, the world model built
from what was seen, the check against ground truth - is the same, so it is the same code.

A section subclasses SenseTask (a FlipTask, so every object it sees lying it can stand up with
the shared stand-up) and provides ``cycle(name)``.

TWO CAMERAS. ``near`` is the workspace camera, pitched 71 deg down over the reach: precise,
and blind beyond 0.65 m. ``far`` is the forward mast camera: it sees across the room, and
loses an object once it is closer than about 1 m, where the arms in the look pose fill the
bottom of its image. perception_node publishes them on /targets and /targets_far.
"""
import json
import math

from std_msgs.msg import String

from . import arm_commander as ac
from . import collision as col
from . import motion as mo
from .flip_task import FlipTask

# THE LOOK POSE is in looking.py - ROS-free, so the suite can check it clears the camera view.
from .looking import LOOK_LEFT, LOOK_RIGHT, LOOK_SAMPLES  # noqa: E402

CAMERAS = {'near': '/targets', 'far': '/targets_far'}


def _fold(axis):
    """An axis is a LINE: fold its angle to [0, pi)."""
    return axis % math.pi


def _median(got):
    """The median of detections of one object - position, a LINE's angle, and the state vote."""
    xs = sorted(g['x'] for g in got)
    ys = sorted(g['y'] for g in got)
    ups = sum(1 for g in got if g['upright'])
    m = len(got) // 2
    ax0 = got[-1]['axis']
    # median of a LINE's angle: unwrap each about the latest, then take the median
    axs = sorted(ax0 + math.remainder(g['axis'] - ax0, math.pi) for g in got)
    x, y = xs[m], ys[m]
    spread = max(math.hypot(g['x'] - x, g['y'] - y) for g in got)
    return {'x': x, 'y': y, 'axis': axs[m], 'upright': ups * 2 > len(got),
            'frames': len(got), 'spread': spread,
            'fit_px': got[-1].get('fit_px'), 'alt_px': got[-1].get('alt_px')}


class SenseTask(FlipTask):
    """FlipTask plus eyes: the cameras, the look pose, and the world as it was seen."""

    def __init__(self):
        super().__init__()
        p = self.param
        self.sense_frames = int(p('sense_frames', 5))      # fresh frames to take a median of
        self.sense_timeout = float(p('sense_timeout', 8.0))  # simulated s
        # the stand-up's GRASP envelope: +/-20 deg of tangential, measured by sweep
        self.max_axis_dev = math.radians(float(p('max_axis_dev_deg', 20.0)))
        # turn square-on to the object if it is further than this off the centreline, so the
        # stand-up runs at the geometry it was verified at
        self.square_on = math.radians(float(p('square_on_deg', 1.0)))
        self.reach_window = float(p('reach_window', 0.03))
        self.dt_look = float(p('dt_look', 1.0))
        # (stamp, {name: detection}), newest last, per camera
        self.detections = {cam: [] for cam in CAMERAS}
        self.seen = {}             # name -> (world x, world y, world axis, upright)
        for cam, topic in CAMERAS.items():
            self.create_subscription(String, topic,
                                     lambda msg, c=cam: self._on_targets(msg, c), 10)

    # ------------- perception -------------
    def _on_targets(self, msg, camera='near'):
        try:
            d = json.loads(msg.data)
            self.detections[camera].append((float(d['stamp']), d['objects']))
        except (ValueError, KeyError, TypeError):
            return
        del self.detections[camera][:-60]

    def sense(self, name, after, camera='near'):
        """A MEDIAN of fresh detections of one object, or None.

        Only frames stamped at or after ``after`` (simulation time) count - a pose measured
        before the base stopped, or before the arms were out of view, is a pose of a scene that
        no longer exists. A blob touching the image edge is refused: part of the object is out
        of view, so its silhouette is not the whole object's.
        """
        t0 = self._now()
        while self._now() - t0 < self.sense_timeout:
            got = [objs[name] for st, objs in self.detections[camera]
                   if st >= after and name in objs and objs[name].get('found')
                   and not objs[name].get('clipped')]
            if len(got) >= self.sense_frames:
                return _median(got[-self.sense_frames:])
            self._sleep(0.1)
        return None

    def sense_all(self, after, camera='near'):
        """Every object in ONE fresh set of frames: ``{name: median}`` for those seen whole.

        For a survey, where most objects are not in view at all: waiting for each one's frames
        in turn (sense) would sit out the timeout for every object that is not there. Here the
        next ``sense_frames`` frames are taken, and an object counts only if it is found, whole,
        in a majority of them.
        """
        t0 = self._now()
        frames = []
        while self._now() - t0 < self.sense_timeout:
            frames = [objs for st, objs in self.detections[camera] if st >= after]
            if len(frames) >= self.sense_frames:
                break
            self._sleep(0.1)
        frames = frames[:self.sense_frames]
        out = {}
        for name in self.order:
            got = [f[name] for f in frames if name in f and f[name].get('found')
                   and not f[name].get('clipped')]
            if frames and len(got) * 2 > len(frames):
                out[name] = _median(got)
        return out

    def _look_move(self, name, frm, to, label):
        """A joint-space move between the parked and the look pose, planned in full: the
        straight joint line IS the path the controller takes, and it is checked against every
        object the robot knows about before it is sent."""
        (fl, fr), (tl, tr) = frm, to
        ql = [[a + i / LOOK_SAMPLES * (b - a) for a, b in zip(fl, tl)]
              for i in range(1, LOOK_SAMPLES + 1)]
        qr = [[a + i / LOOK_SAMPLES * (b - a) for a, b in zip(fr, tr)]
              for i in range(1, LOOK_SAMPLES + 1)]
        known = [c for n in self.order for c in self._object_capsules(n)]
        if known:
            rep = col.home_move_report(fl, fr, tl, tr, self.open_pos, known)
            gap = min(rep['left'], rep['right'])
            self.log(f'  CLEARANCE: {label} - arms {gap * 1000:+.0f} mm from the nearest known '
                     f'object  -> ' + ('clear' if gap >= 0.010 else
                                       '*** WITHIN 10 mm - the arm may touch it ***'))
        self.last_cmd_l, self.last_cmd_r = list(tl), list(tr)
        ok = self.send_pair(ac.joint_traj(ac.LEFT_JOINTS, ql, self.dt_look, 1.0),
                            ac.joint_traj(ac.RIGHT_JOINTS, qr, self.dt_look, 1.0))
        self.settle_arms(name, label)
        return ok

    def observe(self, name, camera='near'):
        """Arms out of the camera's view, a fresh median measurement, arms back to parked."""
        parked = (list(mo.TRAVEL_LEFT), list(mo.TRAVEL_RIGHT))
        self._look_move(name, parked, (LOOK_LEFT, LOOK_RIGHT), 'the look pose')
        det = self.sense(name, self._now(), camera)
        self._look_move(name, (LOOK_LEFT, LOOK_RIGHT), parked, 'the parked pose')
        return det

    def observe_all(self, label, camera='near'):
        """observe() for every object in view at once: arms out, one set of frames, arms back."""
        parked = (list(mo.TRAVEL_LEFT), list(mo.TRAVEL_RIGHT))
        self._look_move(label, parked, (LOOK_LEFT, LOOK_RIGHT), 'the look pose')
        dets = self.sense_all(self._now(), camera)
        self._look_move(label, (LOOK_LEFT, LOOK_RIGHT), parked, 'the parked pose')
        return dets

    def _world_of(self, det):
        """A base-frame detection as (world x, world y, world axis, upright)."""
        wx, wy = self._to_world(det['x'], det['y'])
        return (wx, wy, _fold(det['axis'] + self.heading_now()), bool(det['upright']))

    def check_perception(self, name, det, tol_mm=10.0, tol_deg=5.0, label='PERCEPTION CHECK'):
        """Is what the camera measured where Gazebo says the object is? CHECKS ONLY.

        Compared in the base frame, with the robot's TRUE pose, so neither the wall fit nor the
        perception is judged in a frame that shares its errors.
        """
        t = self.truth.get(name)
        if not t or self.robot_truth is None:
            self.log(f'  {label} [{name}]: no ground truth')
            return True
        (ox, oy, _oz), q = t
        rx, ry, rh = self.robot_truth
        c, s = math.cos(rh), math.sin(rh)
        bx, by = c * (ox - rx) + s * (oy - ry), -s * (ox - rx) + c * (oy - ry)
        vx, vy, vz = self._axis_vec(q)
        true_up = abs(vz) > math.cos(math.radians(45.0))
        err = math.hypot(det['x'] - bx, det['y'] - by) * 1000.0
        ok = err <= tol_mm and det['upright'] == true_up
        ax_txt = ''
        if not true_up:
            want = math.atan2(vy, vx) - rh
            d_ax = abs(math.degrees(math.remainder(det['axis'] - want, math.pi)))
            ok = ok and d_ax <= tol_deg
            ax_txt = f', axis {d_ax:.1f} deg off'
        self.log(f'  {label} [{name}]: measured {err:.1f} mm from the truth{ax_txt}, '
                 f'{"UPRIGHT" if det["upright"] else "LYING"} (truly '
                 f'{"upright" if true_up else "lying"})  -> '
                 + ('accurate' if ok else '*** PERCEPTION IS WRONG - the grasp is aimed off ***'))
        return ok

    # ------------- the world model -------------
    def _object_capsules(self, name, standing_at=None):
        """One object as obstacle shapes in the BASE frame - from what the camera SAW.

        Standing where this node stood it up (or ``standing_at``), else as it was last seen:
        standing or lying, at the measured position and axis. An object never seen is not in
        the model at all; the survey exists so that, by the time the arms move, every object
        has been seen.
        """
        o = self.obj[name]
        rnd = o.get('round', True)
        if standing_at is None:
            standing_at = self.placed.get(name)
        seen = self.seen.get(name)
        if standing_at is None and seen is not None and seen[3]:
            standing_at = seen[:2]
        if standing_at is not None:
            (bx, by), _h = self._to_base(*standing_at)
            return col.object_capsules((bx, by, o['surface_z'] + o['upright_cz']),
                                       (0.0, 0.0, 1.0), o['length'], o['half_width'], rnd)
        if seen is None:
            return []
        (bx, by), h = self._to_base(seen[0], seen[1])
        a = seen[2] - h
        return col.object_capsules((bx, by, o['surface_z'] + o['cz']),
                                   (math.cos(a), math.sin(a), 0.0),
                                   o['length'], o['half_width'], rnd)
