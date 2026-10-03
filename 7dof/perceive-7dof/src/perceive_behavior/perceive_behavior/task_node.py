"""SECTION C - perceive: find each object with the camera, decide, and stand it up.

Nothing in the config says where any object is or which way it lies. Per object:

    look at its rough bearing -> SENSE its pose from the workspace camera
    -> turn square-on to it and sense again
    -> decide:  already UPRIGHT  -> nothing to do; it is left standing
                LYING            -> stand it up with both hands, where it lies

The stand-up is not this section's own. It is flip_task.stand_up() - SHARED with the reorient
section byte for byte, as is every primitive under it (node_base.py) and the planner, IK and
localisation under that. SEEING is shared too, with the navigate section: sense_task.py holds
the cameras, the look pose and the world model built from what was seen. This file adds only
the fixed-base survey and the decision. That is what lets the sections be combined:
pick-place, reorient, perceive and navigate are one robot running one set of code.

What is NEW here versus reorient:
  * the object's position, axis angle and resting state are MEASURED, not configured;
  * the robot SURVEYS first, so the planner knows every object before the arms move;
  * the arms move out of the camera's view to LOOK - parked, the hands hide one end of any
    object in front, which is how lying objects were once "measured at half their length";
  * an object whose axis is too far from tangential is REPORTED, not attempted: rotating a
    fixed base cannot change the angle between an object's axis and the direction to it.
"""
import math

from .node_base import LATERAL, spin
from .sense_task import SenseTask, _fold


class TaskNode(SenseTask):

    ROBOT_MODEL = 'perceive_bimanual'

    def __init__(self):
        super().__init__()
        self.start()

    def object_params(self, n):
        """The shared keys, plus WHERE TO LOOK - a rough bearing, not a position."""
        o = super().object_params(n)
        o['look_azimuth'] = math.radians(float(self.param(f'{n}.look_azimuth', 0.0)))
        return o

    # ------------- sense, decide, act -------------
    def before_cycles(self):
        """SURVEY: look at every object once before touching any, so the planner knows them."""
        self.phase = 'survey'
        self.log('SURVEY: looking at every object before the arms touch any')
        for name in self.order:
            if not self.face(self.obj[name]['look_azimuth']):
                self.log(f'  SURVEY [{name}]: heading not confirmed - skipped')
                continue
            det = self.observe(name)
            if det is None:
                self.log(f'  SURVEY [{name}]: *** NOT SEEN at its bearing ***')
                continue
            self.seen[name] = self._world_of(det)
            wx, wy, wa, up = self.seen[name]
            self.log(f'  SURVEY [{name}]: {"standing" if up else "lying"} at world '
                     f'({wx:+.3f}, {wy:+.3f})' + ('' if up else
                                                  f', axis {math.degrees(wa):.0f} deg'))

    def cycle(self, name):
        """One object: sense it, decide, and stand it up with both hands if it is lying."""
        o = self.obj[name]
        self.phase = '1/9'
        self.abnormal = []
        self.log(f'[{name}] 1/9 recentring, then looking at it')
        self.travel_pose()
        self.recentre()
        if not self.face(o['look_azimuth']):
            self.log(f'[{name}] ABORT: heading not confirmed - refusing to act on a view '
                     f'that cannot be placed in the world')
            return False
        det = self.observe(name)
        if det is None:
            self.log(f'[{name}] NOT SEEN by the workspace camera at its bearing - skipping')
            return False
        # SQUARE-ON. The stand-up was verified with the object dead ahead; if it is off the
        # centreline, turn to face it and look again rather than grasp at a skew.
        bearing = math.atan2(det['y'], det['x'])
        if abs(bearing) > self.square_on:
            self.log(f'  it is {math.degrees(bearing):+.1f} deg off the centreline - turning '
                     f'square-on and looking again')
            if not self.face(self.heading_now() + bearing):
                self.log(f'[{name}] ABORT: could not turn square-on to it')
                return False
            det = self.observe(name)
            if det is None:
                self.log(f'[{name}] lost it after turning square-on - skipping')
                return False
        self.relocalise(prior=self.heading_now())
        self.check_localisation(name)
        self.check_perception(name, det)
        self.seen[name] = self._world_of(det)
        r = math.hypot(det['x'], det['y'])
        self.log(f'  SENSED [{name}]: {"UPRIGHT" if det["upright"] else "LYING"} at base-frame '
                 f'({det["x"]:+.3f}, {det["y"]:+.3f}), {r:.3f} m out'
                 + ('' if det['upright'] else
                    f', axis {math.degrees(_fold(det["axis"])):.1f} deg')
                 + f' [median of {det["frames"]} frames, spread {det["spread"] * 1000:.1f} mm, '
                   f'fit {det["fit_px"]} px vs {det["alt_px"]} px for the other state]')

        # DECIDE
        if det['upright']:
            self.log(f'[{name}] already UPRIGHT - nothing to do; it is left standing')
            self.placed[name] = self.seen[name][:2]
            return True
        if abs(r - self.reach) > self.reach_window:
            self.log(f'[{name}] it lies {r:.3f} m out, outside {self.reach:.2f} +/- '
                     f'{self.reach_window:.2f} where the stand-up is verified - moving the base '
                     f'to it is section D. Skipping.')
            return False
        psi_pick = _fold(det['axis'])
        dev = psi_pick - LATERAL
        if abs(dev) > self.max_axis_dev:
            self.log(f'[{name}] AXIS {math.degrees(dev):+.0f} deg FROM TANGENTIAL - outside the '
                     f'+/-{math.degrees(self.max_axis_dev):.0f} deg a two-handed grasp reaches '
                     f'from here. Rotating the base cannot change that angle; it needs the base '
                     f'to move sideways (section D). Skipping.')
            return False
        return self.stand_up(
            name, (det['x'], det['y']), psi_pick,
            # where it lies, so the drift check measures "upright WHERE IT STARTED"
            self.truth.get(name, (None, None))[0],
            math.degrees(self.heading_now() + LATERAL),
            f'  grasp frame: hands along {math.degrees(psi_pick):+.1f} deg in the base frame, '
            f'as PERCEIVED')


def main(args=None):
    spin(TaskNode, args)


if __name__ == '__main__':
    main()
