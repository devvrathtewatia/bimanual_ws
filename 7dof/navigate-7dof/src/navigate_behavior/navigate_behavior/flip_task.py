"""The fully bimanual STAND-UP, SHARED by every section that stands objects upright.

SHARED FILE, between reorient_behavior and perceive_behavior (the 'flip' group in
SHARED_FILES.sha256). Reorient finds an object from its config; perceive finds it with the
camera. What happens once it has been found - plan the whole cycle parked to parked, hover,
settle, descend, settle, close, attach, lift, yaw to the flip axis, flip 90 deg with both
hands, lower it to its surface, release, settle, let go together, slide clear and go home by a
planned path - is the same motion, so it is the same code: this file.

A section subclasses FlipTask and provides:
  * ``_object_capsules(name, standing_at=None)`` - its world model: the object as obstacle
    shapes in the base frame, lying where it is known to lie, or standing at a world point;
  * ``cycle(name)`` - how it finds the object, ending in :meth:`stand_up`.
"""
import math

from . import arm_commander as ac
from . import choreography as ch
from . import collision as col
from .node_base import TaskBase, LATERAL


class FlipTask(TaskBase):
    """TaskBase plus the stand-up: the flip's parameters, its checks and the cycle."""

    def __init__(self):
        super().__init__()
        p = self.param
        self.flip_z = float(p('flip_z', ch.FLIP_Z))
        self.dt_grasp = float(p('dt_grasp', 2.0))     # s per waypoint, empty-handed moves
        self.dt_flip = float(p('dt_flip', 2.5))       # s per waypoint, loaded rotation
        self.tilt_tol_deg = float(p('tilt_tol_deg', 8.0))
        self.move_tol = float(p('move_tol', 0.030))
        self.settle_recheck = float(p('settle_recheck_s', 2.0))
        # THERE IS NO HANDOVER ANGLE ANY MORE. It used to be read here as
        # flip_handover_deg and written into ch.FLIP_HANDOVER, while the switch that
        # decided whether the handover RAN was computed from the module's own value at
        # import - so a config of 70 would have flipped to 70 and then skipped the
        # handover, leaving a 20 deg jump. Both hands now always take it to 90 deg, and a
        # stale config entry is reported rather than obeyed.
        if p('flip_handover_deg', None) is not None:
            self.get_logger().info('flip_handover_deg in the config is IGNORED - the flip '
                                   'is always fully bimanual to 90 deg')

    def object_params(self, n):
        """The shared keys, plus how tall it stands."""
        o = super().object_params(n)
        # centre height ABOVE ITS SURFACE once standing - half its length
        o['upright_cz'] = float(self.param(f'{n}.upright_cz', 0.10))
        return o

    def check_flip_progress(self, name, commanded_deg):
        """Is the object ACTUALLY turning, and by how much, part-way through the flip?

        WHY THIS EXISTS. During the rotation the only instrument running was the slip
        monitor, and slip measures the GRIP - how far the object has moved relative to the
        hand. Nothing measured the TASK. So a failed flip reported exactly one thing:

            FLIP CHECK [bottle]: axis 90.0 deg from vertical  -> *** NOT UPRIGHT ***

        which is true and tells you nothing. 90 deg from vertical is the pose it STARTED
        in, so the object either never turned, or turned and came back, or was dropped at
        some unknown point - and the log could not distinguish those.

        Sampling here turns that into a sequence: tracked to 23 deg, diverged at 47 deg,
        lost by 70 deg. The commanded angle is measured from horizontal, so the expected
        tilt from vertical is 90 - commanded.
        """
        if name not in self.truth:
            self.log(f'  FLIP PROGRESS [{name}]: no ground truth')
            return True
        want = 90.0 - commanded_deg
        got = self._tilt_deg(name)
        err = abs(got - want)
        ok = err <= self.tilt_tol_deg
        self.log(f'  FLIP PROGRESS [{name}]: commanded {commanded_deg:.0f} deg of turn, '
                 f'object is {got:.1f} deg from vertical (expected {want:.1f}, '
                 f'off {err:.1f})  -> '
                 + ('tracking' if ok else '*** NOT FOLLOWING - it is slipping in the grip '
                                          'or the arms are not getting there ***'))
        return ok

    def check_upright(self, name, label='FLIP CHECK'):
        """The central claim of this section: is the object's axis VERTICAL?"""
        if name not in self.truth:
            self.log(f'  {label} [{name}]: no ground truth')
            return True
        tilt = self._tilt_deg(name)
        ok = tilt <= self.tilt_tol_deg
        self.log(f'  {label} [{name}]: axis {tilt:.1f} deg from vertical '
                 f'(tol {self.tilt_tol_deg:.0f})  -> '
                 + ('upright' if ok else '*** NOT UPRIGHT ***'))
        return ok

    def check_stands(self, name, start_xy, settle=2.0):
        """Upright AND still standing, checked TWICE.

        An object 0.24 m tall on a 0.09 m base has an aspect ratio of 2.7. It can pass
        an instantaneous tilt check at the moment of release and topple a second later,
        so a single sample would report success for something that is lying down by the
        time anyone looks. Also checks it did not get dragged across the floor.
        """
        first = self.check_upright(name, 'STAND CHECK (at release)')
        self._sleep(settle)
        second = self.check_upright(name, 'STAND CHECK (after settling)')
        drift_ok = True
        if name in self.truth and start_xy is not None:
            now = self.truth[name][0]
            d = math.dist(now[:2], start_xy[:2])
            drift_ok = d <= self.move_tol
            self.log(f'  DRIFT CHECK [{name}]: moved {d * 1000:.0f} mm from where it '
                     f'started (tol {self.move_tol * 1000:.0f})  -> '
                     + ('in place' if drift_ok else '*** DRAGGED ***'))
        return first and second and drift_ok

    def plan_cycle(self, name, legs, xy):
        """Plan the WHOLE cycle, parked to parked, in one call; each leg is sent from it.

        WHY NOT PLAN EACH LEG AS IT IS SENT. Two reasons, both measured.

        Continuity: planning a leg in isolation lets the solver take the other wrist branch
        at the boundary - identical hand pose, (q4+pi, -q5, q6+pi), 180 deg of wrist. Across
        leg boundaries that measured 179.8 deg, and Gazebo showed it as right_j4 off 87.7 deg
        with the right hand spinning about the wrist and sweeping through the bottle it was
        holding, which dragged the object out of the grip and tripped the slip stop.

        Clearance: seeding each leg from the previous one fixes the branch but then the
        planner can no longer reach the configuration the flip needs, because it is tied to
        where the last leg happened to end. That measured -15.7 mm of self-collision against
        +2.0 mm for a single whole-cycle plan. One plan gets both: continuous AND clear.

        THE WORLD IS IN IT. Every fixture (walls, shelves, tables) and every other known
        object is an obstacle for the whole cycle, and the handled one for the parts where the
        hands are not meant to be on it - see choreography.plan_inputs(). Both ends are pinned
        to the exact parked pose, so no joint-space jump happens that the planner did not see.

        Falling back is allowed but reported, because a silent fallback would hide exactly
        the divergence this exists to prevent - see the guard in the test suite.
        """
        o = self.obj[name]
        flat = [w for _, leg in legs for w in leg]
        if not flat:
            return True
        wx, wy = self._to_world(xy[0], xy[1])
        lying = self._object_capsules(name)
        standing = self._object_capsules(name, standing_at=(wx, wy))
        # every OTHER object the section knows about - one it has not seen has no shapes, and
        # is left out rather than handed on empty (the report takes a minimum over each)
        others = {}
        for n in self.order:
            caps = self._object_capsules(n) if n != name else []
            if caps:
                others[n] = caps
        fixtures = dict(zip(self.fixtures, self.fixture_boxes()))
        fingers, obs, wide = ch.plan_inputs(legs, float(o.get('close', 0.0)), self.open_pos,
                                            lying, standing,
                                            [c for caps in others.values() for c in caps],
                                            list(fixtures.values()))
        try:
            ql, qr = ac.solve_pairs(flat, seed_l=self.last_cmd_l, seed_r=self.last_cmd_r,
                                    finger_pos=fingers, obstacles=obs, pin_start=True,
                                    pin_end=(ch.TRAVEL_LEFT, ch.TRAVEL_RIGHT), wide=wide)
        except Exception as exc:                                  # noqa: BLE001
            self.log(f'  PLAN: whole-cycle plan failed ({exc}) - each leg will be '
                     f'planned as it is sent, which is less continuous')
            self._plan = {}
            return False
        self._plan = {self._wp_key(w): (ql[i], qr[i]) for i, w in enumerate(flat)}
        self.log(f'  PLAN: {len(flat)} waypoints solved as one continuous path, parked to '
                 f'parked')
        # HOW CLOSE IT COMES TO EVERYTHING, per object. Printed every cycle so that when an
        # object does get touched the log says which, by how much and in which leg.
        at, k = {}, 0
        for leg_name, seq in legs:
            at[leg_name] = list(range(k, k + len(seq)))
            k += len(seq)
        leave = [i for n in ch.LEAVE_LEGS for i in at[n]]
        named = {n: [(c, None) for c in caps] for n, caps in others.items()}
        named[f'{name} as it stands'] = [(c, col.ARM_LINKS) for c in standing]
        named[f'{name} on the way home'] = [(c, None) for c in standing]
        # fixtures by their ARMS: the hands work on a surface by design, like on the floor
        for fx, box in fixtures.items():
            named[f'{fx} (arms)'] = [(box, col.ARM_LINKS)]
        home = [i for n in ch.HOME_LEGS for i in at[n]]
        self.report_clearance(name, legs, ql, qr, fingers, named,
                              {f'{name} as it stands': leave,
                               f'{name} on the way home': home})
        return True

    def stand_up(self, name, xy, psi_pick, start_xy, yaw_world_deg, note):
        """Stand one object up where it lies, BOTH hands on it from the grasp to its surface.

        ``xy`` is its centre in the base frame and ``psi_pick`` its axis in the base frame,
        however they were found. ``start_xy`` is its ground-truth position before it is
        touched (for the drift check), ``yaw_world_deg`` the world angle its axis should have
        after the yaw alignment (for the yaw check), and ``note`` a line for the log saying
        how the grasp frame was obtained. Returns True if the cycle ran to the end.
        """
        o = self.obj[name]
        s = ch.grip_half_separation(o['length'])
        # Closure the planner should assume while the object is carried.
        self.plan_finger_pos = float(o.get('close', 0.0))
        # THE WHOLE CYCLE, from the ONE definition of it, planned now as one continuous
        # path and then sent leg by leg. Every leg below is looked up in this plan; a leg
        # that is not in it is reported as a PLAN MISS and the harness fails on any.
        # HEIGHTS FROM THE SURFACE it lies on - the floor at 0, or a shelf's top. The base
        # stands on the floor, so a world height is a base-frame height.
        sz = o['surface_z']
        legs = ch.cycle_legs(xy, o['cz'], o['upright_cz'], psi_pick, s, self.flip_z,
                             self.hover_dz, self.place_drop, o['half_width'], sz)
        flip_at = ch.flip_height(sz, o['upright_cz'], o['half_width'], self.flip_z)
        leg = dict(legs)
        self.plan_cycle(name, legs, xy)
        self.log(note)

        self.phase = '2/9'
        self.log(f'[{name}] 2/9 both hands move over its two ends, settle, then descend')
        self.grippers(self.open_pos, settle=0.8)
        # THE HOVER IS ITS OWN MOVE. Parked-to-hover is up to 76 deg of joint travel, and it
        # used to run straight into the descent in one trajectory, so the hands came down
        # while still swinging into line. Now they stop over the object and settle first.
        self.move(leg['unpark'], self.dt_travel)
        self.settle_arms(name, 'the hover')
        if not self.move(leg['approach'], self.dt_grasp):
            # NON-CONVERGENCE ALONE IS NOT A REASON TO ABORT. At gain 1.9 every phase of
            # this robot misses its tolerance for a while; what justified an abort on
            # 2026-09-19 was base tilt 4.02 deg with the fingers pressing into the floor,
            # which is measurable directly. So the gate is danger, not a missed deadline.
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

        # ORDER MATTERS: close onto the object first, then attach. Attaching first
        # freezes whatever gap exists and the grasp only looks like one.
        self.phase = '3/9'
        self.log(f'[{name}] 3/9 both hands close on it, then the suction attaches')
        self.curl_this_object = bool(o.get('round', True))
        self.log(f"  JAW [{name}]: {'round' if self.curl_this_object else 'flat'} "
                 f"cross-section -> distal curl "
                 f"{'engaged' if self.curl_this_object else 'DISABLED, the pad bears '
                     'over its full height'}")
        self.grippers(o['close'], settle=1.0)
        self.check_grip(name, o['close'])
        self.check_grasp_centring(name, psi_pick)
        self.check_wrist_load(name, 'LOAD CHECK (closed on it)')
        self.check_base_level(name, 'BASE CHECK (at grasp)')
        # VERIFY THE GRASP BEFORE TRUSTING IT WITH THE PART, which is what a real
        # cell does. This used to weld and assume.
        self.friction_margin(name)
        if not self.engage_grasp(name):
            self.log(f'[{name}] ABORT: grasp not confirmed')
            self.grippers(self.open_pos, settle=0.5)
            self.travel_pose()
            return False

        self.phase = '4/9'
        self.log(f'[{name}] 4/9 both hands lift it to the flip station')
        self.move(leg['lift'], self.dt_flip)
        self.check_lifted(name, flip_at)

        # YAW-ALIGN FIRST. The flip envelope is one-sided - axis lines above 90 deg are
        # unreachable at the top of the rotation, because the upper wrist sits 95 mm along
        # the object's axis and that pushes it away from the shoulder. The GRASP is
        # symmetric over +/-20 deg, so the variability is absorbed here: both hands orbit
        # the object's centre to the canonical axis, then flip in the frame that is verified.
        self.phase = '5/9'
        self.log(f'[{name}] 5/9 both hands turn it to the flip axis: '
                 f'{math.degrees(psi_pick):+.1f} -> {math.degrees(LATERAL):+.1f} deg')
        self.move(leg['yaw'], self.dt_flip)
        # the canonical axis, expressed in the WORLD so it can be compared with
        # ground truth directly rather than through the robot's own estimate
        self.check_yaw(name, yaw_world_deg)

        # DRIVEN IN SEGMENTS so the object's actual tilt can be sampled part-way. One single
        # move gave no visibility at all: if the object came out of the grip at 30 deg the
        # log looked identical to never having turned.
        self.phase = '6/9'
        total = math.degrees(ch.FLIP_ANGLE)
        self.log(f'[{name}] 6/9 COLLABORATIVE FLIP - both hands stand it upright, '
                 f'{total:.0f} deg together')
        converged = True
        for k in range(1, ch.FLIP_SEGMENTS + 1):
            if not self.move(leg['flip%d' % k], self.dt_flip):
                converged = False
            self.check_flip_progress(name, total * k / ch.FLIP_SEGMENTS)
            self.report_self_collision(name, total * k / ch.FLIP_SEGMENTS)
        if not converged:
            self.log(f'[{name}] WARNING: flip did not fully converge')
        # ASK THE SENSORS while both hands are supposed to be on it.
        self.check_both_hands_holding(name, 'BIMANUAL CHECK (upright)')

        # BOTH HANDS TAKE IT DOWN. This used to be the left alone: the right hand was sent
        # to a parked pose at the first lowering waypoint, fingers still closed round the
        # bottom of the object, and on screen the set-down looked exactly like the old 70
        # deg handover. The lower hand is not trapped - with the object on its surface its
        # grasp point is 35 mm up and its lowest plate 15 mm clear, and the planner keeps
        # it off the floor and every fixture.
        self.phase = '7/9'
        self.log(f'[{name}] 7/9 both hands lower it to just above its surface')
        self.move(leg['lower'], self.dt_flip)
        self.check_both_hands_holding(name, 'BIMANUAL CHECK (lowered)')

        # RELEASE ORDER MATTERS. Lowering an ATTACHED object until its base is on the surface
        # drives a rigid body into a rigid surface and the reaction lifts the base. So
        # stop place_drop short, release the suction there, settle the rest with all four
        # fingers still closed so the object is guided down rather than dropped, and only
        # then open - both hands at once.
        self.phase = '8/9'
        self.log(f'[{name}] 8/9 release just above the surface, both hands set it down '
                 f'and let go together')
        self.release_grasp(name)
        self._sleep(0.3)
        self.move(leg['settle'], self.dt_flip)
        self.check_upright(name)
        self.grippers(self.open_pos, settle=0.8)
        self.check_base_level(name, 'BASE CHECK (after release)')
        self.check_stands(name, start_xy, self.settle_recheck)
        # it is an obstacle, standing here, for every cycle after this one
        self.placed[name] = self._to_world(xy[0], xy[1])

        # BOTH HANDS LEAVE, by a sequence that is safe whatever is around - see the note
        # above choreography.release_waypoints(). Nothing moves sideways near the surface, and
        # the trip to parked is a planned path, not a joint-space jump.
        self.phase = '9/9'
        self.log(f'[{name}] 9/9 both hands leave it: slide up clear, the lower hand backs '
                 f'out, both rise, both withdraw, then a planned path home')
        self.move(leg['release'], self.dt_grasp)
        self.move(leg['clear'], self.dt_grasp)
        self.move(leg['retreat'], self.dt_grasp)
        self.move(leg['withdraw'], self.dt_grasp)
        self.move(leg['park'], self.dt_travel)
        self.travel_pose()           # already there - the plan ends AT the parked pose
        self.report_abnormal_digest(name)
        self.log(f'[{name}] done')
        return True
