"""SECTION D - navigate: scan a room, drive to every lying object, stand it up, rescan.

The robot and its three objects start metres apart in a walled room. Nothing in the config
says where any object is, which way it lies, or which way to look. The run:

    SCAN     at the scan point, turn through 8 headings; at each, arms out of the view, and
             the MAST camera measures every object in sight. A standing object is noted as an
             obstacle and otherwise ignored.
    per LYING object, the nearest by route first:
      ROUTE    a path of straight legs clear of every other object and the walls
      LOOK     1.15 m out, square to its side: the mast camera's last whole view of it, which
               corrects the scan's estimate before the part of the approach that cannot see
      BLIND    on to 0.42 m on the wall fix alone. Between 1.0 and 0.45 m NEITHER camera sees
               the object whole: the mast camera's bottom rows are arm, and the workspace
               camera's view ends. So nothing is measured there, by design.
      STAGE    the WORKSPACE camera measures it precisely
      PARK     the last 0.12 m so it is 0.30 m dead ahead with its axis across the reach,
               measured and corrected until it is inside the stand-up's window
      STAND UP flip_task.stand_up() - the reorient section's own cycle, byte for byte
      BACK UP  0.35 m straight back before turning anywhere: it now stands 0.30 m ahead, and
               the parked hands turning in place would pass it by 4 mm
    RESCAN   back to the scan point and scan again; the job is done when all three are seen
             standing.

WHAT IS SHARED, and so the same robot as the other sections: every primitive (node_base), the
stand-up (flip_task, choreography), the verification, and all of the seeing (sense_task,
vision, detect, looking, perception_node). What this file adds is only what driving adds -
the room localisation (room_ref), the approach geometry and routes (navigation) and the order
of work.
"""
import math
import time

from geometry_msgs.msg import Twist

from . import navigation as nav
from . import room_ref as rr
from ros_gz_interfaces.msg import Contacts
from .node_base import CONTACT_FRESH_S, FINGER_SENSORS, LATERAL, norm_ang, spin
from .sense_task import SenseTask, _fold


class TaskNode(SenseTask):

    ROBOT_MODEL = 'navigate_bimanual'

    def __init__(self):
        super().__init__()
        self._init_tip_contacts()
        p = self.param
        box = [float(v) for v in p('room', [-2.5, 2.5, -2.5, 2.5])]
        self.room = rr.Room(*box)              # the wall FACES: x_min, x_max, y_min, y_max
        self.room_band = tuple(float(v) for v in p('room_band', [0.15, 7.5]))
        self.room_max_rms = float(p('room_max_rms', 0.02))
        self.scan_point = tuple(float(v) for v in p('scan_point', [0.0, 0.0]))
        self.scan_headings = int(p('scan_headings', 8))
        self.backup = float(p('backup', 0.35))
        self.nav_speed = float(p('nav_speed', 0.15))           # m/s, in transit
        self.approach_speed = float(p('approach_speed', 0.05))  # m/s, near an object
        self.nav_accel = float(p('nav_accel', 0.25))           # m/s^2
        self.max_rounds = int(p('max_rounds', 3))
        self.park_tol = float(p('park_tol', 0.010))            # m of reach error accepted
        self.park_axis_tol = math.radians(float(p('park_axis_tol_deg', 5.0)))
        self.park_attempts = int(p('park_attempts', 3))
        # THE LAST FIX CARRIED FORWARD. relocalise() discards pose_fix on a failed fit, which is
        # right for aiming; but the next fit still needs a PRIOR, and between fixes the drive
        # loop needs a pose. So the last good fix is kept with the odometry at that moment.
        self._last_fix = None
        self._fix_odom = None
        # Odometry OVER-reports in-place rotation: measured true/odom = 0.884-0.885 on every
        # turn in the section D log. Scale the carried heading change by it so the prior handed
        # to the wall fit is ~1 deg out after a 150 deg turn instead of ~18.
        self.turn_slip = float(p('turn_slip', 0.885))
        self._scan_fitted = None     # the scan message the drive loop last fitted
        self.all_standing = False
        self.park_report = {}        # name -> (true reach err m, true lateral m, axis dev deg)
        self.start()

    # ------------- localisation: the room, not the L -------------
    def fit_pose(self, sc, prior):
        """Whichever of the room's walls are in view, predicted from the last fix carried
        forward by odometry - see room_ref. ``prior``, a heading, is used only before the first
        fix, when the carried pose is odometry alone."""
        guess = self.pose_estimate()
        if guess is None:
            return None, {'reject': 'no odometry yet', 'beams': 0}
        if self._last_fix is None and prior is not None:
            guess = (guess[0], guess[1], prior)
        pose, info = rr.pose_in_room(sc.ranges, sc.angle_min, sc.angle_increment,
                                     sc.range_min, sc.range_max, self.room, guess,
                                     band=self.room_band, max_rms=self.room_max_rms)
        if pose is not None:
            self._last_fix, self._fix_odom = pose, tuple(self.odom)
        return pose, info

    def pose_estimate(self):
        """Where the base is NOW: the last wall fix moved on by the odometry since it."""
        if self.odom is None:
            return None
        if self._last_fix is None:
            return tuple(self.odom)            # the world frame IS odom's until the first fix
        (fx, fy, fh), (ox, oy, oh) = self._last_fix, self._fix_odom
        r = fh - oh
        dx, dy = self.odom[0] - ox, self.odom[1] - oy
        c, s = math.cos(r), math.sin(r)
        dh = norm_ang(self.odom[2] - oh) * self.turn_slip
        return (fx + c * dx - s * dy, fy + s * dx + c * dy, norm_ang(fh + dh))

    # ------------- contact: the curled tip is part of the hand -------------
    def _init_tip_contacts(self):
        """Listen to the sensors on the distal tips (see arm.xacro, <name>_tip_contact)."""
        self.tip_contact_at = {f: 0.0 for f in FINGER_SENSORS}
        for f in FINGER_SENSORS:
            self.create_subscription(
                Contacts, f'/contact/{f}_tip',
                lambda msg, key=f: self._on_tip_contact(key, msg), 10)

    def _on_tip_contact(self, key, msg):
        if len(msg.contacts) > 0:
            self.tip_contact_at[key] = time.time()

    def touching(self, sensor):
        """Is this finger - pad OR curled tip - touching something right now?

        On a round object the tip bears and the proximal pad is held a few mm off, so the
        pad sensor alone read 0/2 on a hand that was gripping (the finger still stalled).
        """
        if super().touching(sensor):
            return True
        return (time.time() - self.tip_contact_at.get(sensor, 0.0)) < CONTACT_FRESH_S

    # ------------- turning: in legs, with a wall fix between them -------------
    MAX_LEG = math.radians(70)      # longest single turn made between two wall fixes

    def face(self, azimuth, tol=None, max_turns=8):
        """node_base.face, but a turn longer than MAX_LEG is made in legs.

        Odometry over-reports an in-place rotation by ~12 %, so the pose prior after a turn is
        out by that fraction of it: ~5 deg after 45, ~18 after 154. room_ref only recovers from
        a prior about 35 deg out and the old search stopped at 25. Legs keep every prior small.
        """
        if tol is None:
            tol = self.heading_tol
        while self.odom is None:
            time.sleep(0.1)
        for attempt in range(max_turns):
            self.relocalise(prior=norm_ang(self.odom[2] - self.heading_bias))
            target_odom = norm_ang(azimuth + self.heading_bias)
            err0 = norm_ang(target_odom - self.odom[2])
            if abs(err0) < tol and attempt > 0:
                break
            leg_only = abs(err0) > self.MAX_LEG
            if leg_only:                       # part of the way, then re-measure
                target_odom = norm_ang(self.odom[2] + math.copysign(self.MAX_LEG, err0))
            while True:
                err = norm_ang(target_odom - self.odom[2])
                if abs(err) < tol:
                    break
                t = Twist()
                t.angular.z = max(-self.rot_speed, min(self.rot_speed, 1.5 * err))
                floor = 0.10 if abs(err) > 0.05 else 0.035
                if abs(t.angular.z) < floor:
                    t.angular.z = floor if err > 0 else -floor
                self.cmd_pub.publish(t)
                time.sleep(0.02)
            self.cmd_pub.publish(Twist())
            self._sleep(0.4)
            if not self.relocalise(prior=azimuth):
                self.log(f'  could not confirm the heading on pass {attempt + 1}; retrying')
                continue
            true_h = norm_ang(self.odom[2] - self.heading_bias)
            residual = norm_ang(azimuth - true_h)
            self.log(f'  facing {math.degrees(azimuth):+.0f} deg: pass {attempt + 1}, '
                     f'measured {math.degrees(true_h):+.2f}, residual '
                     f'{math.degrees(residual):+.2f} deg' + ('  (leg)' if leg_only else ''))
            if not leg_only and abs(residual) < math.radians(0.6):
                return True
        return False

    def _fix_quick(self):
        """One fit of the newest scan, for the drive loop: no retries, no pause, no log line.

        relocalise() is for a base standing still - it retries with a pause between tries, and
        a pause while driving is the base coasting blind on its last command. On a failed fit
        the pose is discarded as relocalise() would, and the loop steers on the carried fix.
        """
        pose, _info = self.fit_pose(self.scan, None)
        if pose is None:
            self.pose_fix = None
            return False
        self.pose_fix = pose
        self.heading_bias = norm_ang(self.odom[2] - pose[2])
        return True

    def _frame(self):
        return self.pose_fix if self.pose_fix is not None else self.pose_estimate()

    def _to_base(self, wx, wy):
        """As node_base's, but without a fix it falls back to the CARRIED fix, not raw
        odometry - after metres of driving, odometry's frame is not the world's."""
        px, py, h = self._frame()
        dx, dy = wx - px, wy - py
        c, s = math.cos(h), math.sin(h)
        return (c * dx + s * dy, -s * dx + c * dy), h

    def _to_world(self, bx, by):
        px, py, h = self._frame()
        c, s = math.cos(h), math.sin(h)
        return (px + c * bx - s * by, py + s * bx + c * by)

    # ------------- the world, for the base -------------
    def nav_obstacles(self):
        """Every known object's footprint on the floor, in the world: standing where this node
        stood it (or saw it standing), else as last seen."""
        out = {}
        for n in self.order:
            o = self.obj[n]
            if n in self.placed:
                x, y = self.placed[n]
                out[n] = nav.footprint(x, y, 0.0, True, o['length'], o['half_width'])
            elif n in self.seen:
                x, y, a, up = self.seen[n]
                out[n] = nav.footprint(x, y, a, up, o['length'], o['half_width'])
        return out

    # ------------- driving -------------
    def _stop(self):
        self.cmd_pub.publish(Twist())

    def drive_to(self, goal, v_max, reverse=False, guard=(), label=''):
        """Drive to a point, closed on the wall fix, steering onto it; stop when there.

        The pose is refitted on every new scan (quietly - that is many a second) and carried by
        odometry between scans. ``guard`` lists objects the base must not come within 30 mm of
        on the way, measured from that pose; if it would, the base stops and the leg fails.
        Timed in SIMULATED seconds. Two seconds without a single wall fix and the base stops:
        odometry alone is what put the old build a metre from where it believed it was.
        """
        obstacles = self.nav_obstacles()
        start = self.pose_estimate()
        dist = math.dist(start[:2], goal)
        deadline = self._now() + dist / 0.02 + 10.0
        v_now = 0.0
        t_last = t_fix = self._now()
        while self._now() < deadline:
            if self.scan is not None and self.scan is not self._scan_fitted:
                self._scan_fitted = self.scan
                if self._fix_quick():
                    t_fix = self._now()
            if self._now() - t_fix > 2.0:
                self._stop()
                self.log(f'  *** LOST THE WALL FIX while driving {label} - stopping ***')
                return False
            pose = self.pose_estimate()
            v, w, along, lat = nav.drive_command(pose, goal, v_max, reverse=reverse)
            if v == 0.0:
                break
            for n in guard:
                if n in obstacles:
                    a, b, r = obstacles[n]
                    gap = nav.seg_point(pose[:2], a, b) - r - nav.BASE_RADIUS
                    if gap < 0.030:
                        self._stop()
                        self.log(f'  *** DRIVE GUARD: the base is {gap * 1000:.0f} mm from the '
                                 f'{n} - stopping {label} ***')
                        return False
            now = self._now()
            dv = self.nav_accel * max(0.02, now - t_last)
            t_last = now
            v_now = max(v_now - dv, min(v_now + dv, v))
            t = Twist()
            t.linear.x = v_now
            t.angular.z = w
            self.cmd_pub.publish(t)
            self._sleep(0.05)
        self._stop()
        self._sleep(0.4)
        self.relocalise(prior=self.heading_now())
        pose = self.pose_estimate() if self.pose_fix is None else self.pose_fix
        miss = math.dist(pose[:2], goal)
        ok = miss <= 0.02
        self.log(f'  ARRIVED {label}: {miss * 1000:.0f} mm from the goal by the wall fix'
                 + ('' if ok else '  *** DID NOT GET THERE ***'))
        return ok

    def go_to(self, xy, heading, name, label, clear=nav.OBJECT_CLEAR, skip=(), guard=None,
              speed=None):
        """Plan a route to ``xy`` clear of every known object and the walls, drive it leg by leg
        (turn, then drive), and finish facing ``heading`` (None: as it arrives)."""
        self.relocalise(prior=self.heading_now())
        here = self.pose_estimate()
        obstacles = self.nav_obstacles()
        if math.dist(here[:2], xy) > 0.015:
            route = nav.plan_route(here[:2], xy, obstacles, self.room, skip=skip, clear=clear)
            if route is None:
                self.log(f'[{name}] *** NO CLEAR ROUTE to {label} at ({xy[0]:+.2f}, '
                         f'{xy[1]:+.2f}) ***')
                return False
            gap, wall, who = nav.route_clearance(route, obstacles, self.room, skip)
            self.log(f'  ROUTE [{name}] to {label} ({xy[0]:+.3f}, {xy[1]:+.3f}): '
                     f'{len(route) - 1} leg(s), {nav.route_length(route):.2f} m; closest '
                     + (f'{who} {gap * 1000:+.0f} mm (base edge), ' if who else '')
                     + f'walls {wall * 1000:+.0f} mm (arms)')
            if guard is None:
                guard = tuple(n for n in obstacles if n not in skip)
            for a, b in zip(route, route[1:]):
                if math.dist(a, b) < 0.015:
                    continue
                cur = self.pose_estimate()
                self.face(math.atan2(b[1] - cur[1], b[0] - cur[0]))
                if not self.drive_to(b, speed or self.nav_speed, guard=guard,
                                     label=f'at {label}' if b == route[-1] else 'at a waypoint'):
                    return False
        if heading is not None and not self.face(heading):
            self.log(f'[{name}] WARN heading at {label} not confirmed')
            return False
        return True

    def drive_straight(self, dist, name, label):
        """Straight ahead (or back, ``dist < 0``) along the current heading, slowly."""
        self.relocalise(prior=self.heading_now())
        x, y, h = self.pose_estimate()
        goal = (x + dist * math.cos(h), y + dist * math.sin(h))
        gap, wall, who = nav.segment_clearance((x, y), goal, self.nav_obstacles(), self.room,
                                               skip=(name,))
        if gap < 0.03 or wall < 0.03:
            self.log(f'[{name}] *** cannot drive {dist:+.2f} m {label}: '
                     + (f'{who} in the way ***' if gap < 0.03 else 'a wall in the way ***'))
            return False
        self.log(f'  {label}: {abs(dist):.2f} m straight {"back" if dist < 0 else "ahead"}')
        return self.drive_to(goal, self.approach_speed, reverse=dist < 0, label=label)

    # ------------- looking -------------
    def _far_tol(self, r):
        """What the mast camera is good for at range r (m): mm and deg. Measured offline
        (test_the_mast_camera_measures_across_the_room) at under 5 mm and 2.5 deg to 1.6 m,
        11 mm and 5 deg at 2.2 m, 22 mm and 7 deg at 2.4 m. It only has to find an object; the
        look point re-measures it from 1.15 m before anything depends on the axis."""
        return max(10.0, 15.0 * r), max(4.0, 3.5 * r)

    def camera_note(self, name, r):
        """Which camera can see it whole from here, and say so."""
        if nav.FAR_MIN <= r <= nav.FAR_MAX:
            which = 'the MAST camera sees it whole here'
        elif r <= nav.NEAR_MAX:
            which = 'the WORKSPACE camera takes over'
        elif r < nav.FAR_MIN:
            which = 'NEITHER camera sees it whole here - driving on the wall fix alone'
        else:
            which = 'too far for either camera to trust'
        self.log(f'  CAMERA [{name}]: {r:.2f} m away - {which} (mast {nav.FAR_MIN:.2f}-'
                 f'{nav.FAR_MAX:.2f} m, workspace under {nav.NEAR_MAX:.2f} m)')

    def scan_room(self, label):
        """SCAN: every heading, arms out, the mast camera; the most central whole view of each
        object it finds, placed in the world. Returns ``{name: seen tuple}``."""
        self.phase = 'scan'
        here = self.pose_estimate()
        self.log(f'SCAN ({label}): {self.scan_headings} headings from '
                 f'({here[0]:+.2f}, {here[1]:+.2f}), mast camera')
        best = {}
        for k in range(self.scan_headings):
            h = norm_ang(2.0 * math.pi * k / self.scan_headings)
            if not self.face(h):
                self.log(f'  SCAN {math.degrees(h):+.0f} deg: heading not confirmed - skipped')
                continue
            dets = self.observe_all(f'scan {math.degrees(h):+.0f}', 'far')
            for name, d in sorted(dets.items()):
                r = math.hypot(d['x'], d['y'])
                b = abs(math.atan2(d['y'], d['x']))
                if not nav.FAR_MIN <= r <= nav.FAR_MAX:
                    self.log(f'  SCAN [{name}]: seen {r:.2f} m away, outside the '
                             f'{nav.FAR_MIN:.1f}-{nav.FAR_MAX:.1f} m the mast camera is '
                             f'trusted over - ignored')
                    continue
                tol_mm, tol_deg = self._far_tol(r)
                self.check_perception(name, d, tol_mm, tol_deg, 'PERCEPTION CHECK (mast)')
                if name not in best or b < best[name][1]:
                    best[name] = (self._world_of(d), b, r, d)
        for name in self.order:
            if name not in best:
                self.log(f'  SCAN [{name}]: not in view from here')
                continue
            (wx, wy, wa, up), b, r, d = best[name]
            self.seen[name] = (wx, wy, wa, up)
            self.log(f'  SCAN [{name}]: {"STANDING" if up else "LYING"} at world '
                     f'({wx:+.3f}, {wy:+.3f})' + ('' if up else f', axis {math.degrees(wa):.0f} '
                                                  f'deg') + f', {r:.2f} m away '
                     f'[{math.degrees(b):.0f} deg off centre, fit {d["fit_px"]} px vs '
                     f'{d["alt_px"]} px for the other state]'
                     + ('  -> an obstacle, nothing to do' if up else ''))
        return {n: self.seen[n] for n in best}

    # ------------- the order of work -------------
    def cycles(self):
        """Scan, stand up every lying object nearest first, go back and scan again - until a
        scan sees every object standing, or ``max_rounds`` scans have been made."""
        results = {}
        for rnd in range(1, self.max_rounds + 1):
            if not self.go_to(self.scan_point, None, 'scan', 'the scan point'):
                self.log('  WARN could not return to the scan point - scanning from here')
            found = self.scan_room(f'round {rnd}')
            # SOMETHING HIDDEN. An object the scan cannot see is behind another - the walls
            # hide nothing, and the LiDAR cannot see objects at all. So move sideways from each
            # object that could be hiding it, and scan again from there.
            hidden = any(n not in found for n in self.order)
            for k, vp in enumerate(self.vantage_points(found) if hidden else []):
                if all(n in found for n in self.order):
                    break
                missing = [n for n in self.order if n not in found]
                self.log(f'  NOT FOUND: {", ".join(missing)} - something may be hiding it; '
                         f'scanning again from the side, at ({vp[0]:+.2f}, {vp[1]:+.2f})')
                if not self.go_to(vp, None, 'scan', f'vantage point {k + 1}'):
                    continue
                more = self.scan_room(f'round {rnd}, vantage point {k + 1}')
                found.update({n: v for n, v in more.items() if n not in found})
            # what was seen from the scan point stands - nearer, and so better - over what a
            # vantage point saw of the same object from further off
            self.seen.update(found)
            lying = [n for n in self.order if n in found and not found[n][3]]
            missing = [n for n in self.order if n not in found]
            self.log(f'SCAN RESULT (round {rnd}): '
                     + ', '.join(f'{n} {"lying" if n in lying else "standing"}'
                                 for n in self.order if n in found)
                     + (f'; NOT FOUND: {", ".join(missing)}' if missing else ''))
            if not lying and not missing:
                self.all_standing = True
                self.log(f'ALL STANDING: every object seen standing - the job is done '
                         f'({rnd} scan{"s" if rnd > 1 else ""})')
                break
            if missing:
                self.log(f'  *** NOT FOUND from anywhere it looked: {", ".join(missing)} ***')
            if not lying or rnd == self.max_rounds:
                self.log('*** STOPPING WITH WORK LEFT: '
                         + (f'lying {lying}' if lying else '')
                         + (f' not found {missing}' if missing else '') + ' ***')
                break
            while lying:
                here = self.pose_estimate()[:2]
                costs = {}
                for n in lying:
                    x, y, a, _up = self.seen[n]
                    plan, _why = nav.plan_approach(n, (x, y, a), here, self.nav_obstacles(),
                                                   self.room)
                    costs[n] = plan['length'] if plan else float('inf')
                n = min(lying, key=lambda k: costs[k])
                lying.remove(n)
                key = n if rnd == 1 else f'{n} (round {rnd})'
                results[key] = self.run_cycle(n)
        return results

    def vantage_points(self, found, offset=0.8, limit=2):
        """Where to scan from next when something was not seen: beside each object that could be
        hiding it - ``offset`` across the line from here to it, either side - that is clear and
        can be reached. Nearest first, at most ``limit``."""
        here = self.pose_estimate()[:2]
        obstacles = self.nav_obstacles()
        cands = []
        for n, (x, y, _a, _up) in found.items():
            d = math.hypot(x - here[0], y - here[1])
            if d < 1e-6:
                continue
            px, py = -(y - here[1]) / d, (x - here[0]) / d
            for sgn in (1.0, -1.0):
                p = (here[0] + sgn * offset * px, here[1] + sgn * offset * py)
                gap, wall, _who = nav.point_clearance(p, obstacles, self.room)
                if gap < nav.OBJECT_CLEAR + 0.2 or wall < nav.WALL_CLEAR + 0.2:
                    continue
                if nav.plan_route(here, p, obstacles, self.room) is None:
                    continue
                cands.append(p)
        cands.sort(key=lambda p: math.dist(p, here))
        return cands[:limit]

    def run_cycle(self, n):
        """node_base's guarded cycle - and then, however it ended, make sure the base is not
        left where turning would sweep the parked hands past an object. A cycle that succeeds
        has already backed up; one that stopped part way may be parked 0.30 m from it."""
        ok = super().run_cycle(n)
        self.relocalise(prior=self.heading_now())
        here = self.pose_estimate()
        for m, (a, b, r) in self.nav_obstacles().items():
            gap = nav.seg_point(here[:2], a, b) - r - nav.ARM_RADIUS
            if gap < 0.10:
                self.phase = 'backup'
                self.log(f'  [{n}] the parked hands are {gap * 1000:+.0f} mm from the {m} - '
                         f'backing up before anything turns')
                self.drive_straight(-self.backup, m, 'backing up clear of it')
                break
        return ok

    def return_home(self):
        """Back to where the run began - the scan point, facing the way it started."""
        self.log('returning to the starting pose')
        self.travel_pose()
        if not self.go_to(self.scan_point, 0.0, 'home', 'the starting pose'):
            self.log('  WARN could not confirm the starting pose')
            return False
        self.log('  back at the starting pose, arms parked')
        if self.park_report:
            self.log('NAVIGATION SUMMARY (true park, from ground truth): ' + '; '.join(
                f'{n} {e[0] * 1000:+.1f} mm reach, {e[1] * 1000:+.1f} mm off centre, axis '
                f'{e[2]:+.1f} deg' for n, e in self.park_report.items()))
        self.log('ALL THREE STANDING' if self.all_standing else
                 '*** NOT EVERY OBJECT WAS SEEN STANDING AT THE END ***')
        return True

    # ------------- one object -------------
    def _report_park(self, name):
        """Where the object REALLY is relative to the base, from ground truth. CHECKS ONLY."""
        t = self.truth.get(name)
        if not t or self.robot_truth is None:
            return
        (ox, oy, _oz), q = t
        rx, ry, rh = self.robot_truth
        c, s = math.cos(rh), math.sin(rh)
        bx, by = c * (ox - rx) + s * (oy - ry), -s * (ox - rx) + c * (oy - ry)
        vx, vy, _vz = self._axis_vec(q)
        dr, _b, dev = nav.park_error(bx, by, math.atan2(vy, vx) - rh)
        self.park_report[name] = (dr, by, math.degrees(dev))
        ok = abs(dr) <= self.reach_window and abs(by) <= 0.010 and \
            abs(dev) <= self.max_axis_dev
        self.log(f'  PARKED [{name}]: truly {math.hypot(bx, by):.3f} m out ({dr * 1000:+.1f} mm), '
                 f'{by * 1000:+.1f} mm off the centreline, axis {math.degrees(dev):+.1f} deg from '
                 f'across the reach  -> ' + ('inside the stand-up\'s window' if ok else
                                            '*** OUTSIDE THE STAND-UP\'S WINDOW ***'))

    def _look(self, name, camera):
        """Observe with one camera and place the result in the world; None if not seen whole."""
        det = self.observe(name, camera)
        if det is None:
            return None
        r = math.hypot(det['x'], det['y'])
        if camera == 'far':
            tol_mm, tol_deg = self._far_tol(r)
            self.check_perception(name, det, tol_mm, tol_deg, 'PERCEPTION CHECK (mast)')
        else:
            self.check_perception(name, det)
        self.seen[name] = self._world_of(det)
        return det

    def _park_pose(self, name, heading):
        """The park pose on the side the base is already on (nearest ``heading``)."""
        x, y, a, _up = self.seen[name]
        return min(nav.approach_poses((x, y), a, nav.REACH),
                   key=lambda p: abs(norm_ang(p[2] - heading)))

    def _drive_to_park(self, name):
        """From about the staging distance to the park pose, from the LATEST measurement.

        The object itself is left out of the route - the park pose is inside the margin transit
        keeps from it - so this leg is checked against it on its own: the base edge must stay
        50 mm off it all the way, and the drive stops if it comes within 30.
        """
        park = self._park_pose(name, self.heading_now())
        here = self.pose_estimate()
        own = {name: self.nav_obstacles()[name]}
        gap, _w, _who = nav.segment_clearance(here[:2], park[:2], own, self.room)
        if gap < 0.05:
            self.log(f'[{name}] *** the drive to the park pose would pass {gap * 1000:.0f} mm '
                     f'from it ***')
            return False
        return self.go_to(park[:2], park[2], name, 'the park pose', skip=(name,), clear=0.05,
                          guard=(name,), speed=self.approach_speed)

    def cycle(self, name):
        """One lying object: approach it in stages, park, stand it up, back up."""
        o = self.obj[name]
        self.abnormal = []
        self.phase = 'nav'
        if name not in self.seen:
            self.log(f'[{name}] NOT SEEN - nothing to approach')
            return False
        x, y, a, up = self.seen[name]
        if up:
            self.log(f'[{name}] already UPRIGHT - nothing to do; it is left standing')
            self.placed[name] = (x, y)
            return True
        self.travel_pose()
        here = self.pose_estimate()
        plan, why = nav.plan_approach(name, (x, y, a), here[:2], self.nav_obstacles(), self.room)
        if plan is None:
            self.log(f'[{name}] *** NO WAY TO STAND IT UP: {why} ***')
            return False
        self.log(f'[{name}] NAV: lying at ({x:+.3f}, {y:+.3f}), axis {math.degrees(a):.0f} deg - '
                 f'approaching from side {plan["side"]}, facing '
                 f'{math.degrees(plan["park"][2]):+.0f} deg, {plan["length"]:.2f} m to go')

        # LOOK - the mast camera's last whole view, before the blind stretch
        if plan['look'] is not None:
            lx, ly, lh = plan['look']
            if not self.go_to((lx, ly), lh, name, 'the look point'):
                return False
            self.camera_note(name, math.dist((lx, ly), (x, y)))
            det = self._look(name, 'far')
            if det is None:
                self.log(f'  [{name}] not seen whole by the mast camera from the look point - '
                         f'going on the scan\'s estimate')
            elif det['upright']:
                self.log(f'[{name}] seen STANDING from the look point - nothing to do')
                self.placed[name] = self.seen[name][:2]
                return True
        else:
            self.log(f'  [{name}] no clear look point on this side - straight to staging')

        # BLIND, then STAGE - the workspace camera
        x, y, a, _up = self.seen[name]
        stage = min(nav.approach_poses((x, y), a, nav.STAGE_REACH),
                    key=lambda p: abs(norm_ang(p[2] - plan['park'][2])))
        self.log(f'  BLIND LEG [{name}]: {math.dist(self.pose_estimate()[:2], (x, y)):.2f} -> '
                 f'{nav.STAGE_REACH:.2f} m on the wall fix alone - inside {nav.FAR_MIN:.2f} m the '
                 f'arms cut off the mast camera\'s view of it, beyond {nav.NEAR_MAX:.2f} m the '
                 f'workspace camera\'s view ends: NEITHER camera sees it whole')
        if not self.go_to(stage[:2], stage[2], name, 'the staging point', speed=self.nav_speed):
            return False
        self.camera_note(name, nav.STAGE_REACH)
        det = self._look(name, 'near')
        if det is None:
            # a standing object is whole in the workspace view only inside 0.35 m
            self.log(f'  [{name}] not seen whole from the staging point - 60 mm closer, look '
                     f'again')
            if not self.drive_straight(0.06, name, 'closing in'):
                return False
            det = self._look(name, 'near')
        if det is None:
            self.log(f'[{name}] *** LOST - the workspace camera does not see it whole ***')
            return False
        if det['upright']:
            self.log(f'[{name}] seen STANDING by the workspace camera - nothing to do')
            self.placed[name] = self.seen[name][:2]
            return True

        # PARK - drive in, measure, correct, measure
        self.phase = '1/9'
        self.log(f'[{name}] 1/9 parking: the object 0.30 m dead ahead, its axis across the '
                 f'reach')
        if not self._drive_to_park(name):
            return False
        for attempt in range(1, self.park_attempts + 1):
            det = self._look(name, 'near')
            if det is None:
                self.log(f'[{name}] *** LOST at the park pose ***')
                return False
            dr, bearing, dev = nav.park_error(det['x'], det['y'], _fold(det['axis']))
            if abs(bearing) > self.square_on:
                self.log(f'  it is {math.degrees(bearing):+.1f} deg off the centreline - turning '
                         f'square-on and looking again')
                self.face(self.heading_now() + bearing)
                det = self._look(name, 'near')
                if det is None:
                    self.log(f'[{name}] lost it after turning square-on - skipping')
                    return False
                dr, bearing, dev = nav.park_error(det['x'], det['y'], _fold(det['axis']))
            ok = abs(dr) <= self.park_tol and abs(dev) <= self.park_axis_tol
            self.log(f'  PARK CHECK [{name}] attempt {attempt}: {dr * 1000:+.1f} mm from '
                     f'{nav.REACH:.2f} m, bearing {math.degrees(bearing):+.1f} deg, axis '
                     f'{math.degrees(dev):+.1f} deg from across  -> '
                     + ('parked' if ok else 'correcting'))
            if ok or attempt == self.park_attempts:
                break
            # CORRECT BY THE ERROR THERE IS. Dead ahead now, so a reach error is a straight
            # drive. An AXIS error needs the base moved sideways, which a differential base
            # does by backing off and approaching again - never by turning towards a point a
            # few millimetres away, whose bearing is noise.
            if abs(dev) > self.park_axis_tol:
                if not self.drive_straight(-(nav.STAGE_REACH - nav.REACH), name,
                                           'backing off to approach again'):
                    return False
                if not self._drive_to_park(name):
                    return False
            elif not self.drive_straight(dr, name, 'correcting the reach'):
                return False
        self.relocalise(prior=self.heading_now())
        self.check_localisation(name)
        self._report_park(name)
        r = math.hypot(det['x'], det['y'])
        self.log(f'  SENSED [{name}]: LYING at base-frame ({det["x"]:+.3f}, {det["y"]:+.3f}), '
                 f'{r:.3f} m out, axis {math.degrees(_fold(det["axis"])):.1f} deg '
                 f'[median of {det["frames"]} frames, spread {det["spread"] * 1000:.1f} mm, '
                 f'fit {det["fit_px"]} px vs {det["alt_px"]} px for the other state]')
        psi_pick = _fold(det['axis'])
        if abs(r - self.reach) > self.reach_window or \
                abs(psi_pick - LATERAL) > self.max_axis_dev:
            self.log(f'[{name}] *** NOT PARKED INSIDE THE STAND-UP\'S WINDOW after '
                     f'{self.park_attempts} attempts - not attempting it ***')
            return False
        ok = self.stand_up(
            name, (det['x'], det['y']), psi_pick,
            # where it lies, so the drift check measures "upright WHERE IT STARTED"
            self.truth.get(name, (None, None))[0],
            math.degrees(self.heading_now() + LATERAL),
            f'  grasp frame: hands along {math.degrees(psi_pick):+.1f} deg in the base frame, '
            f'as PERCEIVED after driving there')
        # BACK UP before anything turns: it now stands 0.30 m ahead
        self.phase = 'backup'
        self.drive_straight(-self.backup, name, 'backing up clear of it')
        return ok


def main(args=None):
    spin(TaskNode, args)


if __name__ == '__main__':
    main()