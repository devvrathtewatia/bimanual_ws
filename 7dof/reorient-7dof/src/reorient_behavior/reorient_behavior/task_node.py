"""SECTION B - collaborative in-place reorientation.

Scope, deliberately narrow so a failure has only one possible home:
  * the base NEVER translates on purpose. It rotates in place to face each object (and
    recentres if wheel slip has moved it).
  * NO perception. Geometry comes from config/objects.yaml.
  * NO transfer. Each object is stood upright ON THE SURFACE IT LIES ON - the floor here,
    a shelf or table as soon as one is declared - at the spot where it lay.

Per object, and EVERY step with both hands:
    face it -> hover -> both hands descend onto its two ends -> settle -> close
    -> attach -> lift -> yaw to the flip axis -> ROTATE 90 deg -> lower to the surface
    -> release -> settle it down -> both hands open -> both withdraw -> travel pose

The whole cycle is ONE list of legs, built by choreography.cycle_legs() and used both
here and by the offline suite, so what is tested is what runs. From the grasp to the
moment the object stands on its surface both hands are on it; the suite and the harness
assert the grasp span between them on every one of those waypoints.

EVERYTHING THE CYCLE IS MADE OF - time, grippers, grasp and release, every runtime check,
localisation, planning and sending a leg - is in node_base.py, SHARED with the pick-place and
perceive sections byte for byte. The stand-up itself - the flip, its checks, the plan - is in
flip_task.py, SHARED with the perceive section, which runs the same motion on objects it has
found with the camera. This file holds only what is particular to this section: finding each
object from the config, and the world model of where the config says things are.

Two orderings are load-bearing and must not be swapped:
  1. the fingers close onto the object FIRST and the suction attaches second - the
     attachment freezes whatever relative pose exists when it is made;
  2. the suction releases just ABOVE the surface and the last few millimetres are settled
     with the fingers still shut, so the object is guided down rather than pressed into
     the surface by a position controller.
"""
import math

from . import collision as col
from .flip_task import FlipTask
from .node_base import LATERAL, spin


class TaskNode(FlipTask):

    ROBOT_MODEL = 'reorient_bimanual'

    def __init__(self):
        super().__init__()
        self.start()

    def object_params(self, n):
        """The shared keys, plus where to face it."""
        o = super().object_params(n)
        o['azimuth'] = math.radians(float(self.param(f'{n}.azimuth', 0.0)))
        return o

    def _object_capsules(self, name, standing_at=None):
        """One object as obstacle capsules in the BASE frame.

        Lying where the config puts it, or standing at ``standing_at`` (a world point) /
        where this node stood it up. The config's position is all section B has; the same
        call will take a perceived pose later.
        """
        o = self.obj[name]
        rnd = o.get('round', True)
        if standing_at is None:
            standing_at = self.placed.get(name)
        # HEIGHTS FROM ITS SURFACE: cz and upright_cz are above surface_z, not the floor
        if standing_at is not None:
            (bx, by), _h = self._to_base(*standing_at)
            return col.object_capsules((bx, by, o['surface_z'] + o['upright_cz']),
                                       (0.0, 0.0, 1.0), o['length'], o['half_width'], rnd)
        wx = self.reach * math.cos(o['azimuth'])
        wy = self.reach * math.sin(o['azimuth'])
        (bx, by), h = self._to_base(wx, wy)
        a = o['axis_yaw'] - h
        return col.object_capsules((bx, by, o['surface_z'] + o['cz']),
                                   (math.cos(a), math.sin(a), 0.0),
                                   o['length'], o['half_width'], rnd)

    def cycle(self, name):
        """One object, start to finish, BOTH hands on it from the grasp to its surface."""
        o = self.obj[name]
        self.phase = '1/9'
        # per-object digest: cleared here so each object reports its OWN events
        self.abnormal = []
        self.log(f'[{name}] 1/9 recentring, then facing the object')
        self.travel_pose()
        self.recentre()
        # GATE THE CYCLE ON A CONFIRMED HEADING. The grasp angle is
        # (axis_yaw - heading), and in this section a heading error becomes final TILT
        # because the flip turns the object about the axis perpendicular to its own.
        if not self.face(o['azimuth']):
            self.log(f'[{name}] ABORT: heading not confirmed - refusing to grasp at an '
                     f'angle that cannot be trusted')
            self.release_all(name)
            self.travel_pose()
            return False

        # THE POSITION IS MEASURED, not assumed to be (reach, 0). The base is only ever
        # commanded to rotate, but an in-place turn on a differential drive slips, and slip
        # that rotates also shifts. aim() relocalises off the two walls and converts the
        # object's world position into the base frame.
        xy, aimed = self.aim(o['azimuth'], self.reach)
        if not aimed:
            self.log(f'[{name}] ABORT: no position fix - refusing to grasp at a '
                     f'place that cannot be trusted')
            self.release_all(name)
            self.travel_pose()
            return False
        self.check_localisation(name)

        # ------------------------------------------------------------------
        # THE GRASP FRAME IS RESOLVED HERE, AFTER THE TURN AND THE POSITION FIX.
        #
        # This is the box bug. psi = axis_yaw - heading cancels the heading only if
        # the heading used is the one the robot is actually at WHEN THE HANDS ARE
        # PLACED. It used to be computed before face(), i.e. with the heading from
        # before the turn, so:
        #
        #   bottle  azimuth   0 -> the robot barely turns, pre == post, error 0.0 deg
        #   box     azimuth -55 -> the robot turns 53 deg, so psi was 51.45 instead
        #                          of 104.78. The hand line sat 53.3 deg across the
        #                          object's axis and each hand landed 67 mm from the
        #                          end it was aiming at.
        #
        # It hid because the only object that ever worked is the one whose azimuth is
        # zero, where the bug is arithmetically invisible. It is also taken AFTER aim(),
        # whose relocalisation is the latest heading there is, and it is used for the whole
        # cycle: once held, the object turns with the base, so its base-frame angle is the
        # one it was picked at until the yaw leg changes it.
        psi_pick = self.base_frame_yaw(o['axis_yaw'])
        return self.stand_up(
            name, xy, psi_pick,
            # where it lies, so the drift check measures "upright WHERE IT STARTED"
            self.truth.get(name, (None, None))[0],
            math.degrees(o['azimuth'] + LATERAL),
            f'  grasp frame: hands along {math.degrees(psi_pick):+.1f} deg in the '
            f'base frame = {math.degrees(psi_pick) + math.degrees(self.heading_now()):+.1f} '
            f'deg in the world (object axis is '
            f'{math.degrees(o["axis_yaw"]):+.1f})')


def main(args=None):
    spin(TaskNode, args)


if __name__ == '__main__':
    main()
