"""SECTION A - fixed-base bimanual pick and place.

Scope, deliberately narrow:
  * the base NEVER translates on purpose. It rotates in place to face a target (and
    recentres if wheel slip has moved it).
  * NO perception. Target geometry comes from config/targets.yaml.
  * NO reorientation. Each object is placed in the same orientation it was picked up in -
    a bottle lying on the floor ends up lying on the shelf, yaw-aligned to its slot.

Per object, EVERY step with both hands:
    face the object -> hover -> settle -> descend -> settle -> close -> attach -> lift
    -> face the shelf slot -> carry over it -> yaw-align -> lower -> release above the
    crest -> settle it in -> let go -> rise clear -> planned path home

The cycle is two halves of legs, built by choreography.pick_legs() and place_legs(), used
both here and by the offline suite. The base turns between them, so each half is planned as
one continuous path: the first from the exact parked joint vector, the second back to it.

EVERYTHING THE CYCLE IS MADE OF - time, grippers, grasp and release, every runtime check,
localisation, recentring, planning and sending a leg - is in node_base.py, SHARED with the
reorient section byte for byte. Until 2026-09-28 this file was its own copy of all that, and
it had drifted: waits on the wall clock with a 2 s grace, no settle before the fingers
closed, the flat box's fingertips curled into its face, no ground truth for the robot, a
blind joint-space jump home. The box "closing at its top, grasping nothing" was those.

Two orderings are load-bearing and must not be swapped back:
  1. the fingers close onto the object FIRST and the suction attaches second - the
     attachment freezes whatever relative pose exists when it is made;
  2. the suction releases just ABOVE the barrier crest and the last few millimetres are
     settled with the fingers still shut, so the chamfers can seat the object instead of
     the arm pressing it into the shelf.
"""
import math

from . import arm_commander as ac
from . import choreography as ch
from . import collision as col
from . import motion as mo
from .node_base import TaskBase, spin


class TaskNode(TaskBase):

    ROBOT_MODEL = 'pnp_bimanual'

    def __init__(self):
        super().__init__()
        p = self.param
        self.lift_z = float(p('lift_z', ch.LIFT_Z))
        # TIMING. Waypoints are sampled 25 mm apart, so seconds-per-waypoint sets the hand
        # speed: 1.0 and 0.7 give 25 and 36 mm/s. LOADED moves stay slower than empty ones.
        # SIMULATED seconds, like every wait now.
        self.dt_slow = float(p('dt_slow', 1.0))
        self.dt_fast = float(p('dt_fast', 0.7))
        self.start()

    def object_params(self, n):
        """The shared keys, plus where it is picked and placed and what it is placed on."""
        o = super().object_params(n)
        p = self.param
        o['pick_azimuth'] = math.radians(float(p(f'{n}.pick_azimuth', 0.0)))
        o['place_azimuth'] = math.radians(float(p(f'{n}.place_azimuth', 180.0)))
        # THE SURFACE IT IS PUT ON - the shelf's top. Its resting centre height is this
        # plus cz, exactly as its lying height on the floor is surface_z plus cz.
        o['place_surface_z'] = float(p(f'{n}.place_surface_z', 0.10))
        # Grasp height above the object's centre. Without this the grasp is aimed at the
        # centre and the palm plate, which sits 0.027 m behind the grasp point, is driven
        # 18 mm INTO the top of a 90 mm object - the arm presses a rigid body down and the
        # reaction lifts the base.
        o['grip_dz'] = float(p(f'{n}.grip_dz', ch.GRIP_DZ))
        return o

    # ------------- section checks -------------
    def check_placed(self, name, azimuth, place_z):
        """Where did it actually end up, split the way the shelf cares about.

        Radially the cradle gives about 11 mm before the object rides a chamfer;
        tangentially the pedestal is far more forgiving. One combined number hid an
        object hanging 33 mm off the end of its pedestal.
        """
        if name not in self.truth:
            self.log(f'  PLACE CHECK [{name}]: no ground truth, cannot verify')
            return True
        (x, y, z), _q = self.truth[name]
        ex, ey = self.reach * math.cos(azimuth), self.reach * math.sin(azimuth)
        dx, dy = x - ex, y - ey
        ca, sa = math.cos(azimuth), math.sin(azimuth)
        d_rad = dx * ca + dy * sa
        d_tan = -dx * sa + dy * ca
        dz = z - place_z
        ok = abs(d_rad) <= 0.015 and abs(d_tan) <= 0.040 and abs(dz) <= 0.020
        self.log(f'  PLACE CHECK [{name}]: radial {d_rad * 1000:+.0f} mm '
                 f'(tol 15) | tangential {d_tan * 1000:+.0f} mm (tol 40) | '
                 f'vertical {dz * 1000:+.0f} mm (tol 20)  -> '
                 + ('seated' if ok else '*** OFF ITS SLOT ***'))
        return ok

    # ------------- the world model -------------
    def _object_shapes(self, name, placed=None):
        """One object as obstacle shapes in the BASE frame: lying where the config says,
        or placed on its slot - where this node put it - once it has been handled.

        Heights from the surface: ``surface_z + cz`` lying, ``place_surface_z + cz`` placed.
        The config's position is all section A has; the same call takes a perceived pose
        later.
        """
        o = self.obj[name]
        if placed is None:
            placed = self.placed.get(name)
        if placed is not None:
            wx, wy, axis = placed
            z = o['place_surface_z'] + o['cz']
        else:
            wx = self.reach * math.cos(o['pick_azimuth'])
            wy = self.reach * math.sin(o['pick_azimuth'])
            axis, z = o['axis_yaw'], o['surface_z'] + o['cz']
        (bx, by), h = self._to_base(wx, wy)
        a = axis - h
        return col.object_capsules((bx, by, z), (math.cos(a), math.sin(a), 0.0),
                                   o['length'], o['half_width'], o['round'])

    def _slot_fixture(self, wx, wy):
        """The fixture a world point is placed ON - the pedestal whose footprint holds it."""
        for f, (c, yaw, size) in self.fixtures.items():
            dx, dy = wx - c[0], wy - c[1]
            u = math.cos(yaw) * dx + math.sin(yaw) * dy
            v = -math.sin(yaw) * dx + math.cos(yaw) * dy
            if abs(u) <= size[0] / 2.0 and abs(v) <= size[1] / 2.0 and size[2] < 0.5:
                return f
        return None

    def plan_half(self, name, legs, lying=(), placed=(), slot_at=None, home=False):
        """Plan HALF a cycle as one continuous path, from where the arms are.

        The pick half starts at the exact parked joint vector; the place half ends there
        (``home``), so no joint-space jump happens that the planner did not see. Every
        fixture and every other known object is an obstacle - see choreography.plan_inputs().
        Falling back to per-leg planning is allowed but reported, because a silent fallback
        would hide the discontinuity this exists to prevent.
        """
        o = self.obj[name]
        flat = [w for _, leg in legs for w in leg]
        others = {n: self._object_shapes(n) for n in self.order if n != name}
        slot = self._slot_fixture(*slot_at) if slot_at is not None else None
        fixtures = dict(zip(self.fixtures, self.fixture_boxes()))
        slot_box = [fixtures.pop(slot)] if slot in fixtures else []
        fingers, obs, wide = ch.plan_inputs(
            legs, float(o['close']), self.open_pos, mo.release_closure(o['half_width']),
            lying, placed, [c for caps in others.values() for c in caps],
            list(fixtures.values()), slot_box)
        try:
            ql, qr = ac.solve_pairs(flat, seed_l=self.last_cmd_l, seed_r=self.last_cmd_r,
                                    finger_pos=fingers, obstacles=obs, pin_start=True,
                                    pin_end=((ch.TRAVEL_LEFT, ch.TRAVEL_RIGHT) if home
                                             else None), wide=wide)
        except Exception as exc:                                  # noqa: BLE001
            self.log(f'  PLAN: half-cycle plan failed ({exc}) - each leg will be planned '
                     f'as it is sent, which is less continuous')
            self._plan = {}
            return False
        self._plan = {self._wp_key(w): (ql[i], qr[i]) for i, w in enumerate(flat)}
        self.log(f'  PLAN: {len(flat)} waypoints solved as one continuous path, '
                 + ('to parked' if home else 'from parked'))
        at, k = {}, 0
        for leg_name, seq in legs:
            at[leg_name] = list(range(k, k + len(seq)))
            k += len(seq)
        named = {n: [(c, None) for c in caps] for n, caps in others.items()}
        where = {}
        if placed:
            named[f'{name} as placed'] = [(c, col.ARM_LINKS) for c in placed]
            named[f'{name} on the way home'] = [(c, None) for c in placed]
            where[f'{name} as placed'] = [i for n in ch.LEAVE_LEGS for i in at[n]]
            where[f'{name} on the way home'] = [i for n in ch.HOME_LEGS for i in at[n]]
        # fixtures by their ARMS: the hands work on the shelf by design, like on the floor
        for fx, box in fixtures.items():
            named[f'{fx} (arms)'] = [(box, col.ARM_LINKS)]
        if slot_box:
            named[f'{slot}, the slot (arms)'] = [(slot_box[0], col.ARM_LINKS)]
        self.report_clearance(name, legs, ql, qr, fingers, named, where)
        return True

    # ------------- one pick-and-place cycle -------------
    def cycle(self, name):
        """One object, floor to shelf, BOTH hands on it from the grasp to the shelf."""
        o = self.obj[name]
        # Hands sit end_clear in from each end, so they get a long moment arm on the
        # object and it cannot pivot in the grip. Derived from the length, never tuned.
        s = ch.grip_half_separation(o['length'])
        # GRASP HEIGHTS FROM THE SURFACES: on the floor where it lies, on the shelf where
        # it is put - each surface plus its centre height plus the grasp offset.
        gz = o['surface_z'] + o['cz'] + o['grip_dz']
        place_z = o['place_surface_z'] + o['cz']
        pz = place_z + o['grip_dz']
        self.abnormal = []

        self.phase = '1/8'
        self.log(f'[{name}] 1/8 recentring, then facing the object')
        self.travel_pose()
        self.recentre()
        # GATE THE WHOLE CYCLE ON A CONFIRMED HEADING. The grasp axis is
        # (axis_yaw - heading), so a 23 degree heading error rotates the grasp line by 23
        # degrees and one hand closes on air. Refusing to start is the honest outcome.
        if not self.face(o['pick_azimuth']):
            self.log(f'[{name}] ABORT: heading not confirmed within tolerance - '
                     f'refusing to grasp at an angle that cannot be trusted')
            self.release_all(name)
            self.travel_pose()
            return False
        xy, aimed = self.aim(o['pick_azimuth'], self.reach)
        if not aimed:
            self.log(f'[{name}] ABORT: no position fix - refusing to grasp at a place '
                     f'that cannot be trusted')
            self.release_all(name)
            self.travel_pose()
            return False
        self.check_localisation(name)

        # THE GRASP ANGLE IS MEASURED, NOT ASSUMED, and resolved AFTER the turn and the
        # position fix: psi = axis_yaw - heading cancels the heading only if the heading is
        # the one the robot is at when the hands are placed.
        psi_pick = self.base_frame_yaw(o['axis_yaw'])
        # Pinned ONCE here and reused by every later phase, including the yaw rotation.
        # Re-deciding which arm takes which end flips when the two ends pass through equal
        # y, and the arms trade ends in a single waypoint. See E19.
        sl = ch.left_sign(psi_pick)
        self.plan_finger_pos = float(o['close'])
        legs = ch.pick_legs(xy[0], xy[1], gz, psi_pick, s, sl, self.hover_dz, self.lift_z)
        leg = dict(legs)
        self.plan_half(name, legs, lying=self._object_shapes(name))
        self.log(f'  grasp frame: hands along {math.degrees(psi_pick):+.1f} deg in the base '
                 f'frame (object axis {math.degrees(o["axis_yaw"]):+.1f} in the world)')

        self.phase = '2/8'
        self.log(f'[{name}] 2/8 both hands move over its two ends, settle, then descend')
        self.grippers(self.open_pos, settle=0.8)
        # THE HOVER IS ITS OWN MOVE, and the arms SETTLE there before descending. It used
        # to be one trajectory from parked, timed on the wall clock with a 2 s grace, so
        # on a laptop that simulates below real time the descent was abandoned part-way and
        # the fingers closed above the object - which is what the box showed.
        self.move(leg['unpark'], self.dt_travel)
        self.settle_arms(name, 'the hover')
        if not self.move(leg['approach'], self.dt_fast):
            danger = self.overload()
            if danger is not None:
                self.log(f'[{name}] ABORT: descend is unsafe - {danger}')
                self.release_all(name)
                self.travel_pose()
                return False
            self.log(f'[{name}] descend lagged its command but the base is level and '
                     f'the wrists are unloaded - continuing')
        # AND LET THE ARMS ARRIVE before anything touches the object - see SETTLE_TOL.
        self.settle_arms(name, 'the grasp pose')

        # ORDER MATTERS: close onto the object first, then attach.
        self.phase = '3/8'
        self.log(f'[{name}] 3/8 both hands close on it, then the suction attaches')
        # THE BOX'S FINGERTIPS CURLED INTO ITS FACE. This node never read which objects are
        # flat, so the distal phalanx curled on the box too, drove into its side and held
        # the pads off it.
        self.curl_this_object = bool(o['round'])
        self.log(f"  JAW [{name}]: {'round' if self.curl_this_object else 'flat'} "
                 f"cross-section -> distal curl "
                 f"{'engaged' if self.curl_this_object else 'DISABLED, the pad bears '
                     'over its full height'}")
        self.grippers(o['close'], settle=1.0)
        self.check_grip(name, o['close'])
        self.check_grasp_centring(name, psi_pick)
        self.check_wrist_load(name, 'LOAD CHECK (closed on it)')
        self.check_base_level(name, 'BASE CHECK (at grasp)')
        self.friction_margin(name)
        if not self.engage_grasp(name):
            self.log(f'[{name}] ABORT: grasp not confirmed')
            self.release_all(name)
            self.travel_pose()
            return False

        self.phase = '4/8'
        self.log(f'[{name}] 4/8 both hands lift it to carry height')
        self.move(leg['lift'], self.dt_slow)
        self.check_lifted(name, self.lift_z)

        self.phase = '5/8'
        self.log(f'[{name}] 5/8 recentring, then facing the shelf slot')
        self.recentre()
        if not self.face(o['place_azimuth']):
            self.log(f'[{name}] WARN heading not confirmed for the placement; the slot '
                     f'angle may be off')
        qxy, aimed = self.aim(o['place_azimuth'], self.reach)
        if not aimed:
            # carrying: aborting here would only drop it, so place at the nominal point
            self.log(f'[{name}] placing at the nominal slot position - no position fix')
        # THE OBJECT TURNED WITH THE BASE, so in the base frame it is still at the angle it
        # was grasped at; the target is tangential to the slot, from the measured heading.
        psi_place = self.base_frame_yaw(o['place_azimuth'] + math.pi / 2)
        legs2 = ch.place_legs(xy, qxy, pz, psi_pick, psi_place, s, sl, self.lift_z,
                              self.place_drop, o['half_width'])
        leg2 = dict(legs2)
        wslot = self._to_world(qxy[0], qxy[1])
        placed_axis = psi_place + self.heading_now()
        placed = self._object_shapes(name, placed=(wslot[0], wslot[1], placed_axis))
        self.plan_half(name, legs2, placed=placed, slot_at=wslot, home=True)

        self.phase = '6/8'
        self.log(f'[{name}] 6/8 both hands carry it over the slot and yaw-align it: '
                 f'{math.degrees(psi_pick):+.1f} -> {math.degrees(psi_place):+.1f} deg')
        self.move(leg2['carry'], self.dt_slow)
        self.move(leg2['yaw'], self.dt_slow)
        # tangential to the slot, in the WORLD, so it is compared with ground truth directly
        self.check_yaw(name, math.degrees(o['place_azimuth']) + 90.0)
        self.check_both_hands_holding(name, 'BIMANUAL CHECK (over the slot)')

        # RELEASE ORDER MATTERS. The suction comes off while the object is still 2 mm above
        # the barrier crest, so it is a free body before it can touch a chamfer; the hands
        # then lower the last place_drop with the fingers still closed, so a chamfer can
        # slide it into the slot instead of levering against the arm.
        self.phase = '7/8'
        self.log(f'[{name}] 7/8 both hands lower it, release above the crest, settle it in, '
                 f'let go')
        self.move(leg2['lower'], self.dt_slow)
        self.release_grasp(name)
        self._sleep(0.3)
        self.move(leg2['settle'], self.dt_slow)
        # LET GO BY OPENING JUST CLEAR OF IT, tips straight. Opening fully here swings the
        # fingertips 20 mm out, into the anti-roll barriers 11 mm outside the object; a
        # half-curled tip would hook under a round object's equator and lift it.
        self.grippers(mo.release_closure(o['half_width']), settle=0.8, straight=True)
        self.check_base_level(name, 'BASE CHECK (after release)')
        self.check_wrist_load(name, 'LOAD CHECK (after release)')
        self.check_placed(name, o['place_azimuth'], place_z)
        # it is an obstacle, lying on its slot, for every cycle after this one
        self.placed[name] = (wslot[0], wslot[1], placed_axis)

        self.phase = '8/8'
        self.log(f'[{name}] 8/8 both hands rise straight up clear of it, then a planned '
                 f'path home')
        self.move(leg2['rise'], self.dt_fast)
        self.grippers(self.open_pos, settle=0.5)
        self.move(leg2['park'], self.dt_travel)
        self.travel_pose()           # already there - the plan ends AT the parked pose
        self.report_abnormal_digest(name)
        self.log(f'[{name}] done')
        return True


def main(args=None):
    spin(TaskNode, args)


if __name__ == '__main__':
    main()
