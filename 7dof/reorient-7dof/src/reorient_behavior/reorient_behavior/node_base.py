"""The task primitives SHARED by every section - one robot, one set of reflexes.

SHARED FILE. Byte-for-byte the same in every section - reorient, pick-place, perceive and
navigate; the suite in each fails if it is not (SHARED_FILES.sha256 at the section root). They
are one robot, and until 2026-09-28 reorient and pick-place were two copies of it that had
drifted apart: the pick-place copy still waited on the wall clock with a 2 s grace, closed its
fingers without letting the arms settle, curled its fingertips into the flat box, had no ground
truth for its own pose and swung home blind - all of it already fixed in the reorient copy.

So everything a cycle is BUILT FROM lives in this one file: time, the grippers, the grasp and
its release, every runtime check, localisation, facing, recentring, planning and sending a leg,
the protective stop, the slip monitor and the run loop. A section's task_node.py adds only what
makes it that section - its own parameters, its world model and its cycle().

Edit it here, copy it to the other sections, and regenerate the manifest in all of them.
"""
import math
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import Imu
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Empty, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory

from . import arm_commander as ac
from . import collision as col
from . import ik
from . import motion as mo
from . import wall_ref as wr

ALL_JOINTS = ac.LEFT_JOINTS + ac.RIGHT_JOINTS + ac.FINGER_JOINTS
LATERAL = math.pi / 2          # an object's axis in the base frame, once faced: across the reach

FINGER_SENSORS = ['left_finger_l', 'left_finger_r',
                  'right_finger_l', 'right_finger_r']


def norm_ang(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# THE SMALLEST SHARE OF THE LOAD THE UNWELDED HAND CAN TAKE AND STILL COUNT AS
# HOLDING. An evenly shared object gives each wrist about half. 25% is the point
# below which the hand is clearly along for the ride: the 2026-09-20 run measured
# 1.7 N against 6.1 N, a 22% share, while both hands read 2/2 fingers in contact.
MIN_LOAD_SHARE = 0.25

# THE FRICTION BUDGET, from the model rather than from the simulator.
# FINGER_EFFORT_N is the effort limit on each finger joint in arm.xacro - the fingers are
# commanded past the object's surface, so they stall and deliver up to this. FRICTION_MU
# is the conservative mix of the pad's 1.8 and the objects' 1.5. Four stalled fingers
# therefore give 12 N of normal force and 18 N of friction, against 4.41 N for the bottle
# and 4.91 N for the box. If any of those three numbers changes, this changes with it.
FINGER_EFFORT_LEFT_N = 6.0   # the hand that GRIPS      # matches arm.xacro; the GRASP QUALITY line reports it
FINGER_EFFORT_RIGHT_N = 1.5  # the hand that SUPPORTS - weak on purpose
PAD_HALF_LEN = 0.020         # half the pad length along the object axis
# ALL THREE MUST MATCH arm.xacro and robot.urdf.xacro. The torque and GRASP QUALITY
# lines would otherwise report margins the hardware does not have. Guarded by
# test_the_reported_constants_match_the_hardware.
# EFFECTIVE coefficient, which is the MINIMUM of the two surfaces - Gazebo combines
# them that way, so the softer surface governs. The silicone pads are built at mu 2.2 but
# the objects are 1.5, so 1.5 is what the grasp actually gets. Reporting the pad's 2.2
# would overstate every GRASP QUALITY line by 47%.
PAD_MU = 2.2
OBJECT_MU = 1.5
FRICTION_MU = min(PAD_MU, OBJECT_MU)
FRICTION_MARGIN_MIN = 2.0

# HOW RECENTLY A FINGER MUST HAVE REPORTED CONTACT TO COUNT AS TOUCHING. The contact
# sensors update at the simulation rate, so a live contact restamps many times a
# second; 0.4 s is long enough to ride out a dropped message and short enough that a
# hand which let go is reported as having let go within one check.
CONTACT_FRESH_S = 0.4

# LET THE ARMS FINISH ARRIVING BEFORE THE FINGERS CLOSE. send_pair() returns once every
# joint is within settle_tol - 0.08 rad, which at the shoulder's 0.43 m lever is 34 mm of
# hand position - and the fingers used to start closing straight away. At gain 1.9 the arm
# is a 0.53 s first-order lag, so the first finger met the object while the hand was still
# travelling, pushed it (a lying bottle ROLLS), and the suction then froze that offset for
# the whole cycle. 0.004 rad is under 2 mm at the hand; from 0.08 it takes about 1.6 s.
SETTLE_TOL = 0.004
SETTLE_TIMEOUT = 5.0

# SUCTION CARRIES, SO THE FINGERS NEED NOT SQUEEZE. `close` puts each pad 3 mm inside the
# surface on purpose - the fingers stall on it, which is what GRIP CHECK measures and what
# holds the object by friction during the settle. But once the suction cup attaches,
# Gazebo stops computing finger-object contact, so the fingers go all the way to that
# command and sit visibly 3 mm INSIDE the object for the whole carry. So after the grip
# check they back off to just outside the surface (0.5 mm), and squeeze again just before
# the suction releases. Only in suction mode: under friction the squeeze IS the grasp.
CARRY_EASE = 0.0035



def _as_bool(v, default=True):
    """Parse a config boolean without falling for bool('false') being True.

    YAML loaded through a plain text parser hands back the STRING 'false', and
    bool('false') is True - so a flat-sided object declared `round: false` would still
    have had its distal phalanx curl into the face. Caught by the harness reporting
    "round cross-section" for a cuboid.
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    t = str(v).strip().lower()
    if t in ('false', 'no', '0', 'off', ''):
        return False
    if t in ('true', 'yes', '1', 'on'):
        return True
    return default


class TaskBase(Node):
    """What every section's task node is made of.

    A section subclasses this, reads its own parameters in ``__init__`` after calling
    ``super().__init__()``, extends :meth:`object_params` with its own per-object keys,
    implements ``cycle(name)``, and finally calls :meth:`start`.
    """

    # The robot's Gazebo model name - its ground-truth pose topic is /model/<this>/pose.
    ROBOT_MODEL = 'robot'

    def __init__(self):
        super().__init__('task_node',
                         automatically_declare_parameters_from_overrides=True)
        p = self.param

        self.reach = float(p('reach', 0.30))
        # HARD CEILING ON THE AIM RADIUS. A drift correction pushed far enough out makes the
        # target unreachable and the cycle dies on IK, which is worse than placing a few mm
        # short. 1.0 m - i.e. no clamp - unless a section configures one.
        self.reach_max = float(p('reach_max', 1.0))
        self.hover_dz = float(p('hover_dz', mo.HOVER_DZ))
        self.open_pos = float(p('open_pos', 0.065))
        self.rot_speed = float(p('rotate_speed', 0.30))
        self.tol = float(p('settle_tol', 0.08))
        # ---- friction grasp and slip-reactive control ----
        self.grasp_mode = str(p('grasp_mode', 'suction'))
        # _as_bool, not bool(): bool('false') is True, and the harness hands config
        # values over as strings exactly as a plain-text YAML reader would.
        self.weld_rescue = _as_bool(p('weld_rescue', False), default=False)
        self.slip_warn = float(p('slip_warn_mm', 5.0)) / 1000.0
        self.slip_abort = float(p('slip_abort_mm', 15.0)) / 1000.0
        self.regrasp_attempts = int(p('regrasp_attempts', 2))
        self.regrasp_deeper = float(p('regrasp_deeper', 0.004))
        self._slip_warned = False
        self.grasp_ref = {}        # name -> object offset from the hand midpoint
        self.holding = None        # which object the slip monitor should watch
        self.welded = set()        # objects the weld is actually carrying
        # set when the grasp mode pulls the object into the robot tree, which
        # makes Gazebo filter finger-object contacts and silences the sensors
        self.contacts_are_filtered = False
        # set per object from its declared cross-section; see _curl_for()
        self.curl_this_object = True

        # WHERE THE ARMS ALREADY ARE. Handed to the planner as the seed for the next leg so
        # the swivel and the wrist branch stay continuous ACROSS legs, not just within one.
        # They start parked, which is also what the URDF loads them at.
        self.last_cmd_l = list(mo.TRAVEL_LEFT)
        self.last_cmd_r = list(mo.TRAVEL_RIGHT)
        # Finger closure to plan collisions against. Zero (open) until a closure is
        # commanded, otherwise the planner checks clearances for a hand that is not the
        # shape the hand will actually be in while carrying.
        self.plan_finger_pos = 0.0
        # The whole-cycle plan, keyed by waypoint. Empty means "not planned yet", which
        # makes move() fall back to per-leg planning rather than fail.
        self._plan = {}
        self.plan_misses = 0
        # abnormal-event digest, and the phase label it is tagged with
        self.abnormal = []
        self.phase = 'startup'
        # last commanded left-arm joints, for the self-collision report
        self.last_left_q = None
        self.slip_events = []      # (name, phase, mm) for the end-of-run summary
        self.close_now = {}        # current finger command per object, for regrasps
        self.dt_travel = float(p('dt_travel', 1.5))   # s per waypoint, to and from parked
        # WHERE THIS NODE HAS PUT OBJECTS, in the world. Every object the robot knows about
        # is an obstacle to the planner - where the config says until it has been handled,
        # where it was put afterwards. Perception replaces the config later; the planner
        # does not care where the positions come from.
        self.placed = {}
        # WAS MISSING ENTIRELY once - read by the release sequence and never assigned,
        # because the patch that added it anchored on a default that did not match. Same
        # failure as E22/E23 in the catalogue; test_the_node_assigns_every_attribute_it_reads.
        self.place_drop = float(p('place_drop', 0.010))
        self.wall_band = tuple(float(v) for v in p('wall_band', [0.15, 1.60]))
        self.wall_max_rms = float(p('wall_max_rms', 0.03))
        self.heading_tol = float(p('heading_tol', 0.004))
        # RECENTRING. The base is never commanded to translate, but wheel slip that rotates
        # also shifts it; below this the direction of a correction is measurement noise, so
        # it is not attempted at all. See recentre().
        self.pos_deadband = float(p('position_deadband', 0.025))
        self.v_lin = float(p('drive_speed', 0.08))
        # Corrected pose (x, y, heading) from the two walls, or None. Never stale:
        # relocalise() clears it on failure rather than reusing an old measurement.
        self.pose_fix = None
        # Odometry heading minus TRUE heading. Zero until the first fix. The base rotates
        # to face each object and the wheels slip, so without this the assumed grasp angle
        # drifts - one hand then meets the object and the other closes on air.
        self.heading_bias = 0.0
        self.tilt_warn_deg = float(p('tilt_warn_deg', 1.5))
        self.tilt_abort_deg = float(p('tilt_abort_deg', 4.0))
        self.force_warn_n = float(p('force_warn_n', 15.0))
        self.force_abort_n = float(p('force_abort_n', 30.0))
        self.scan = None
        self.order = list(p('order', ['bottle', 'box']))
        self.obj = {n: self.object_params(n) for n in self.order}
        # THE STATIC WORLD - walls, shelves, tables - as boxes, for the planner.
        self.fixtures = self.fixture_params()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.left_pub = self.create_publisher(
            JointTrajectory, '/left_arm_controller/joint_trajectory', 10)
        self.right_pub = self.create_publisher(
            JointTrajectory, '/right_arm_controller/joint_trajectory', 10)
        self.grip_pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10)
        self.verify_pub = self.create_publisher(Empty, '/verify/run', 10)
        self.attach_pub, self.detach_pub = {}, {}
        for n in self.order:
            self.attach_pub[n] = self.create_publisher(
                Empty, f'/grasp/{n}/attach', 10)
            self.detach_pub[n] = self.create_publisher(
                Empty, f'/grasp/{n}/detach', 10)

        self.odom = None
        self.joints = {}
        self.grasp_state = {}
        self.create_subscription(Odometry, '/odom', self._on_odom, 20)
        self.create_subscription(JointState, '/joint_states', self._on_js, 20)
        # FORCE STATE. A position controller has no idea it is pressing on anything -
        # it just integrates error and pushes harder - so these are the only signals
        # that can tell us. base_tilt IS the lift-off fault, measured directly.
        self.base_tilt = 0.0                  # deg from level, from the base IMU
        self.wrist_force = {'left': 0.0, 'right': 0.0}    # N, magnitude at each wrist
        self.create_subscription(Imu, '/imu', self._on_imu, 20)
        for side in ('left', 'right'):
            self.create_subscription(
                Wrench, f'/ft/{side}',
                lambda msg, k=side: self.wrist_force.__setitem__(
                    k, math.sqrt(msg.force.x ** 2 + msg.force.y ** 2
                                 + msg.force.z ** 2)), 20)
        # CONTACT IS A FRESHNESS-GATED SIGNAL, NOT A LATCHED BOOLEAN.
        #
        # The callback sets contact = len(msg.contacts) > 0, which self-clears only if
        # Gazebo publishes a message when contact CEASES. Across every run we have,
        # every single report reads "2/2 touching" - never 0/2, never 1/2, including a
        # run where the right gripper was watched losing the object completely. The
        # flag latches high after the first touch and never comes back down.
        #
        # So time-stamp it instead: record WHEN contact was last reported, and treat
        # anything older than CONTACT_FRESH_S as not touching. That is correct either
        # way round - if Gazebo does publish empties the stamp stops advancing, and if
        # it only publishes on contact then silence IS the absence of contact. This is
        # also how you would read a real sensor that reports events rather than state.
        self.contact_at = {f: 0.0 for f in FINGER_SENSORS}
        for f in FINGER_SENSORS:
            self.create_subscription(
                Contacts, f'/contact/{f}',
                lambda msg, key=f: self._on_contact(key, msg), 10)
        self.create_subscription(LaserScan, '/scan',
                                 lambda m: setattr(self, 'scan', m), 5)
        # GROUND TRUTH for the runtime checks below. The node otherwise works blind:
        # it commands a pose and assumes it happened, which is how a grasp that closed
        # on air, or an object that slipped instead of turning, used to be discovered
        # only by watching the video.
        self.truth = {}
        # AND THE ROBOT'S OWN - (x, y, heading) from Gazebo, for checks only.
        self.robot_truth = None
        self.robot_model = str(p('robot_model', self.ROBOT_MODEL))
        self.create_subscription(PoseStamped, f'/model/{self.robot_model}/pose',
                                 self._on_robot_truth, 20)
        for n in self.order:
            self.create_subscription(
                PoseStamped, f'/model/{n}/pose',
                lambda msg, name=n: self._on_truth(name, msg), 20)
        for n in self.order:
            self.create_subscription(
                String, f'/grasp/{n}/state',
                lambda m, k=n: self.grasp_state.__setitem__(
                    k, m.data.strip().lower()), 10)

    # ------------- configuration -------------
    def param(self, name, default):
        """A parameter's value, or ``default`` if it is unset."""
        try:
            v = self.get_parameter(name).value
            return default if v is None else v
        except Exception:  # noqa: BLE001
            return default

    def object_params(self, n):
        """The per-object keys every section reads. A section extends the dict.

        HEIGHTS ARE FROM THE SURFACE. ``surface_z`` is the height of whatever the object is
        lying on - 0 for the floor, a shelf's top otherwise - and ``cz`` is the object's
        centre height ABOVE it. Nothing in the cycle assumes the floor.
        """
        p = self.param
        return {
            # LENGTH, not a hand-tuned separation: the grip half separation is
            # derived so the hands always sit the same distance in from the ends.
            'length': float(p(f'{n}.length', 0.24)),
            # The object's axis in the WORLD frame. The base-frame grasp angle
            # is this minus the MEASURED heading, so drift cancels instead of
            # rotating the grasp line.
            'axis_yaw': math.radians(float(p(f'{n}.axis_yaw', 90.0))),
            'close': float(p(f'{n}.close', 0.048)),
            # half the width across the closing axis: what the move home must clear, and
            # the surface the fingers ease back to in suction
            'half_width': float(p(f'{n}.half_width', 0.045)),
            # CROSS-SECTION. round -> the surface narrows with depth so the distal
            # phalanx must curl to stay in contact; flat -> the face is vertical, the
            # pad already bears over its full height, and curling only digs in
            # (16.1 mm of penetration on a 90 mm cuboid vs the 2.0 mm intended).
            # Defaults to True so an object that forgets to declare it still wraps.
            'round': _as_bool(p(f'{n}.round', True)),
            'cz': float(p(f'{n}.cz', 0.045)),
            'surface_z': float(p(f'{n}.surface_z', 0.0)),
            'mass': float(p(f'{n}.mass', 0.45)),
        }

    def fixture_params(self):
        """The static world: ``{name: (centre_xyz, yaw_rad, size_xyz)}``, WORLD frame.

        Listed by name under ``fixtures`` and written the way an SDF ``<box>`` is - centre,
        yaw, full size - so the config can be checked against the world file directly.
        """
        names = self.param('fixtures', [])
        if isinstance(names, str):
            names = [names] if names.strip() else []
        out = {}
        for f in names:
            c = [float(v) for v in self.param(f'{f}.centre', [0.0, 0.0, 0.0])]
            size = [float(v) for v in self.param(f'{f}.size', [0.0, 0.0, 0.0])]
            out[f] = (c, math.radians(float(self.param(f'{f}.yaw', 0.0))), size)
        return out

    def fixture_boxes(self, skip=()):
        """Every fixture as a planner obstacle in the BASE frame, from the current pose."""
        fx = {k: v for k, v in self.fixtures.items() if k not in skip}
        return list(mo.fixture_obstacles(fx, lambda x, y: self._to_base(x, y)[0],
                                         self.heading_now()).values())

    def start(self):
        """Launch the task, unless autostart is off. Call last in a section's __init__."""
        if _as_bool(self.param('autostart', True)):
            threading.Thread(target=self._safe_run, daemon=True).start()
        else:
            self.get_logger().info('autostart=false: task idle')

    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        self.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                     math.atan2(2 * (q.w * q.z + q.x * q.y),
                                1 - 2 * (q.y * q.y + q.z * q.z)))

    def _on_js(self, msg):
        for n, v in zip(msg.name, msg.position):
            self.joints[n] = v

    def _now(self):
        """Simulation time in seconds once /clock is running, wall time before that."""
        try:
            t = self.get_clock().now().nanoseconds * 1e-9
        except Exception:                                           # noqa: BLE001
            t = 0.0
        return t if t > 0.0 else time.time()

    def _sleep(self, dt):
        """Wait ``dt`` seconds of SIMULATION time."""
        t0 = self._now()
        while self._now() - t0 < dt:
            time.sleep(min(0.02, max(0.001, dt)))

    def log(self, s):
        self.get_logger().info(s)
        # DIGEST. Every abnormal line is starred, but a run is ~100 lines and the reader
        # has to scan it and then work out the ORDER things went wrong in. Recording them
        # here means the cycle can print a digest: what failed, in sequence, with values.
        if '***' in str(s):
            self.abnormal.append((self.phase, str(s).strip()))

    def _curl_for(self, closure):
        """How far the distal phalanx should be curled for a given closure command.

        Zero when the hand is open, CURL_CLOSED when it is shut on the object, linear in
        between, so the tips sweep inward as the hand closes and finish wrapped UNDER the
        object's equator. That is the whole point of the second phalanx: the rigid plate
        stops 15 mm past the widest point of a 90 mm body and cannot reach below it.
        """
        # SHAPE-AWARE. A flat face does not narrow with depth, so the pad already bears
        # over its whole height and curling the tip inward only drives it into the
        # surface - 16.1 mm of penetration on a 90 mm cuboid against the 2.0 mm intended
        # for a cylinder. That is what put 19 N into the box's wrist and left 0/2 pads
        # touching: the flaps held the pads off the object.
        if not self.curl_this_object:
            return 0.0
        span = self.open_pos - ac.CLOSE_REF
        if span <= 0:
            return 0.0
        f = (self.open_pos - float(closure)) / span
        return max(0.0, min(1.0, f)) * ac.CURL_CLOSED

    def grippers(self, val, settle=1.0, ramp=0.8, steps=16, straight=False):
        """Command both hands' fingers, RAMPED rather than stepped.

        JointGroupPositionController forwards the commanded position straight through,
        so a single 17 mm jump is a commanded discontinuity - an impulse that no effort
        limit can soften, and the cause of the jerk seen at every grasp and release.
        Ramping over ``ramp`` seconds turns it into a motion the controller can track.

        ``straight`` uncurls the distal phalanx whatever the closure. Letting go of a round
        object by opening only part way (motion.release_closure) would otherwise leave the
        tips half curled - under its equator - and the hands would lift it as they rose. The
        uncurl is ramped with the fingers, not stepped: a jump in the tip command is the same
        impulse as a jump in the finger's.
        """
        cur = [self.joints.get(j, self.open_pos) for j in ac.FINGER_JOINTS]
        start = sum(cur) / len(cur) if cur else self.open_pos
        curl0 = self._curl_for(start)
        for i in range(1, steps + 1):
            v = start + (float(val) - start) * i / steps
            self.grip_pub.publish(Float64MultiArray(
                data=[float(v)] * 4
                + [curl0 * (1.0 - i / steps) if straight else self._curl_for(v)] * 4))
            self._sleep(ramp / steps)
        self.grip_pub.publish(Float64MultiArray(
            data=[float(val)] * 4 + [0.0 if straight else self._curl_for(val)] * 4))
        self._sleep(settle)

    def _on_robot_truth(self, msg):
        q, p = msg.pose.orientation, msg.pose.position
        self.robot_truth = (p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y),
                                                 1 - 2 * (q.y * q.y + q.z * q.z)))

    def check_localisation(self, name, tol_mm=3.0, tol_deg=0.3):
        """Is the wall fit where Gazebo says the base is? Every grasp is aimed from it.

        THE CHECK THAT WAS MISSING when the wall constants were the wall CENTRES instead
        of their faces: the fit was then 10 mm out in x and y, every grasp inherited it,
        and no instrument could say so, because everything was measured in that same frame.
        """
        if self.robot_truth is None or self.pose_fix is None:
            self.log(f'  LOCALISATION CHECK [{name}]: '
                     + ('no ground truth for the robot' if self.robot_truth is None
                        else 'no wall fix to check'))
            return True
        tx, ty, th = self.robot_truth
        fx, fy, fh = self.pose_fix
        dx, dy = (fx - tx) * 1000.0, (fy - ty) * 1000.0
        dh = math.degrees(norm_ang(fh - th))
        ok = math.hypot(dx, dy) <= tol_mm and abs(dh) <= tol_deg
        self.log(f'  LOCALISATION CHECK [{name}]: wall fix vs ground truth '
                 f'dx {dx:+.1f} mm, dy {dy:+.1f} mm, heading {dh:+.2f} deg '
                 f'(tol {tol_mm:.0f} mm, {tol_deg:.1f} deg)  -> '
                 + ('accurate' if ok else
                    '*** THE WALL FIX IS OFF - every grasp is aimed off by this much ***'))
        return ok

    def _on_truth(self, name, msg):
        q, p = msg.pose.orientation, msg.pose.position
        self.truth[name] = ((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))

    @staticmethod
    def _axis_vec(q):
        """The object's long axis as a world unit vector (its local +z)."""
        x, y, z, w = q
        return (2.0 * (x * z + w * y), 2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y))

    def _tilt_deg(self, name):
        """Angle between the object's axis and vertical, in degrees."""
        vx, vy, vz = self._axis_vec(self.truth[name][1])
        n = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
        return math.degrees(math.acos(min(1.0, abs(vz) / n)))

    def _on_imu(self, msg):
        """Base roll/pitch from the IMU, as a single angle from level.

        Tipping onto the casters is exactly what "the robot lifted off" means, and it
        is one number. Sign does not matter - either direction is the same fault.
        """
        q = msg.orientation
        roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z),
                          1.0 - 2.0 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
        self.base_tilt = math.degrees(math.hypot(roll, pitch))

    def overload(self):
        """Is the robot pressing on something it should not be? Returns a reason or None.

        THE PROTECTIVE STOP, which is what real cobots do above a force threshold: halt
        rather than push harder. Without it a position controller drives to its commanded
        pose regardless of what is in the way, which is how the palm ended up 18 mm
        inside an object, a finger 25 mm into the floor, and an object pressed against a
        barrier - each time levering the base up off its wheels.

        Thresholds. Holding the heaviest object shares about 2.2 N between the wrists,
        plus roughly 2 N of gripper weight, so normal running is under about 5 N; 30 N is
        six times that and still only half the ~56 N that would tip the chassis
        (an upward reaction at 0.30 m ahead of the axle against a 5 kg robot). Base tilt
        on flat ground is ~0, so 4 deg is unambiguous. Both want confirming against a
        real run - they are derived, not measured.
        """
        if self.base_tilt > self.tilt_abort_deg:
            return (f'base tilted {self.base_tilt:.1f} deg (limit '
                    f'{self.tilt_abort_deg:.1f}) - the arm is levering the robot up')
        for side, f in self.wrist_force.items():
            if f > self.force_abort_n:
                return (f'{side} wrist at {f:.1f} N (limit {self.force_abort_n:.0f}) '
                        f'- pressing on something')
        return None

    def check_base_level(self, name, label='BASE CHECK'):
        """Did the base stay flat? This is the lift-off fault, as a number."""
        ok = self.base_tilt <= self.tilt_warn_deg
        self.log(f'  {label} [{name}]: base {self.base_tilt:.2f} deg from level '
                 f'(warn {self.tilt_warn_deg:.1f})  -> '
                 + ('flat' if ok else '*** THE ROBOT IS TIPPING ***'))
        return ok

    def check_wrist_load(self, name, label='LOAD CHECK'):
        """What are the wrists actually carrying?

        A grasp that holds shows a steady few newtons. A wrist pressing on the floor,
        a barrier or the object's own top shows far more, and that is the signature of
        every lift-off in this project.
        """
        l, r = self.wrist_force['left'], self.wrist_force['right']
        ok = max(l, r) <= self.force_warn_n
        self.log(f'  {label} [{name}]: wrists {l:.1f} / {r:.1f} N '
                 f'(warn {self.force_warn_n:.0f})  -> '
                 + ('normal' if ok else '*** PRESSING ON SOMETHING ***'))
        return ok

    def check_grip(self, name, close):
        """Did the fingers close on SOMETHING, and did BOTH hands?

        Measurable rather than a matter of opinion: the fingers are commanded inward
        past the object's surface, so when the object is there they stall and the joint
        rests ABOVE the command. A hand that reports its commanded position almost
        exactly closed on air. Reported per hand, because the failure that matters is
        one hand on the object and the other beside it - an averaged number hides it.
        """
        self._sleep(0.4)                      # let contact settle
        out, contact = [], {}
        for side, joints, sensors in (('left', ac.FINGER_JOINTS[:2],
                                       FINGER_SENSORS[:2]),
                                      ('right', ac.FINGER_JOINTS[2:],
                                       FINGER_SENSORS[2:])):
            worst = max(self.joints.get(j, close) - close for j in joints)
            touching = [s for s in sensors if self.touching(s)]
            stalled = worst > 0.001
            # THE SENSOR LEADS, the stall cross-checks. Stall is indirect: it cannot
            # tell the object from anything else the finger jams on, and it assumes the
            # commanded over-travel exceeds the detection threshold. Contact is a fact.
            # They are reported separately on purpose - sensor without stall is a touch
            # too light to stop the finger, and stall without sensor means the finger is
            # jammed on something that is not the object. Either disagreement is worth
            # seeing rather than collapsing into one verdict.
            contact[side] = bool(touching)
            out.append(f'{side} {len(touching)}/2 touching, stall '
                       f'{worst * 1000:+.1f} mm'
                       + ('' if bool(touching) == stalled
                          else ' <-- SENSOR/STALL DISAGREE'))
        both = all(contact.values())
        self.log(f'  GRIP CHECK [{name}]: ' + ' | '.join(out)
                 + ('  -> both hands in contact' if both
                    else '  -> *** A HAND IS NOT TOUCHING THE OBJECT ***'))
        return both

    def check_lifted(self, name, want_z, tol=0.05):
        """Did the object come up with the hands, or is it still lying where it was?"""
        if name not in self.truth:
            self.log(f'  LIFT CHECK [{name}]: no ground truth')
            return True
        z = self.truth[name][0][2]
        ok = abs(z - want_z) <= tol
        self.log(f'  LIFT CHECK [{name}]: z={z:.3f}, wanted {want_z:.3f} '
                 f'+/-{tol:.2f}  -> '
                 + ('lifted' if ok else '*** NOT LIFTED - the grasp did not hold ***'))
        return ok

    def check_yaw(self, name, want_world_deg, tol=10.0):
        """Did the yaw alignment actually TURN the object, or did it slip in the grip?

        If it slipped, what follows runs on an axis the object does not have: in reorient an
        axis error becomes final tilt one-for-one, in pick-place the object meets its shelf
        slot crosswise.
        """
        if name not in self.truth:
            self.log(f'  YAW CHECK [{name}]: no ground truth')
            return True
        vx, vy, _ = self._axis_vec(self.truth[name][1])
        got = math.degrees(math.atan2(vy, vx)) % 180.0
        want = want_world_deg % 180.0
        off = abs(((got - want + 90.0) % 180.0) - 90.0)
        ok = off <= tol
        self.log(f'  YAW CHECK [{name}]: axis {got:.1f} deg, wanted {want:.1f}, '
                 f'off {off:.1f} (tol {tol:.0f})  -> '
                 + ('aligned' if ok else '*** NOT ALIGNED - it slipped in the grip ***'))
        return ok

    def _on_contact(self, key, msg):
        """Stamp the moment contact was last reported for one finger."""
        if len(msg.contacts) > 0:
            self.contact_at[key] = time.time()

    def touching(self, sensor):
        """Is this finger touching something RIGHT NOW, rather than ever having?"""
        return (time.time() - self.contact_at.get(sensor, 0.0)) < CONTACT_FRESH_S

    def _obj_in_base(self, name):
        """The object's centre in the BASE frame, or None without ground truth.

        The slip measurement has to be in the robot's frame, because that is the frame
        the hands are in. Converting the object instead of the hands keeps the hand
        positions exact - they come from forward kinematics, not from a fit.
        """
        if name not in self.truth or self.pose_fix is None:
            return None
        wx, wy, wz = self.truth[name][0]
        rx, ry, h = self.pose_fix
        dx, dy = wx - rx, wy - ry
        ch_, sh_ = math.cos(h), math.sin(h)
        return (dx * ch_ + dy * sh_, -dx * sh_ + dy * ch_, wz)

    def _obj_in_left_hand(self, name):
        """The object's centre expressed in the LEFT HAND'S OWN FRAME.

        THIS HAS TO BE IN A ROTATING FRAME, and the first version was not - which made
        the slip monitor fire on every flip, in both grasp modes, and abort the rotation
        at 35 degrees.

        The mistake: it compared the object's centre against the midpoint of the two
        grasp points, in the BASE frame. But _ends places both grasp points grip_dz off
        the object's axis along the object's OWN up, so the midpoint is
        centre + Rt.(grip_dz * up) where Rt is the flip rotation. That offset therefore
        turns with the object, and the apparent movement is

            |offset_now - offset_ref| = 2 * grip_dz * sin(theta/2)

        which is 6.5 mm at 15 degrees and exactly 15.0 mm at 34.9 degrees - matching the
        logged numbers to the millimetre. The weld run proved it beyond argument: an
        object rigidly joined to the palm reported 15.0 mm of "slip".

        Expressing the object in the hand's own frame cancels the rotation, because a
        held object is stationary in that frame by definition. Whatever is left over IS
        slip. This is the same construction the harness uses to carry a held object.
        """
        obj = self._obj_in_base(name)
        if obj is None:
            return None
        try:
            pl, Rl = ik.fk([self.joints[j] for j in ac.LEFT_JOINTS], +1)
        except Exception:                                  # noqa: BLE001
            return None
        d = [obj[k] - pl[k] for k in range(3)]
        # R^T . d, i.e. world/base vector into the hand's frame
        return tuple(sum(Rl[r][k] * d[r] for r in range(3)) for k in range(3))

    def capture_grasp_reference(self, name):
        """Record where the object sits relative to the hands, at the grasp.

        Everything about slip is measured against this. It is taken once, immediately
        after the fingers have closed and settled, so it captures the pose the grip
        actually achieved rather than the pose that was commanded.
        """
        rel = self._obj_in_left_hand(name)
        if rel is None:
            self.grasp_ref.pop(name, None)
            self.log(f'  WARN [{name}] no grasp reference - slip cannot be measured')
            return False
        self.grasp_ref[name] = rel
        return True

    def slip_mm(self, name):
        """How far the object has moved in the grip since the grasp, in metres.

        Returns None when it cannot be measured, which is NOT the same as zero and is
        reported as such - a slip monitor that silently reads zero is worse than none.
        """
        if name not in self.grasp_ref:
            return None
        now = self._obj_in_left_hand(name)
        if now is None:
            return None
        ref = self.grasp_ref[name]
        return math.sqrt(sum((now[k] - ref[k]) ** 2 for k in range(3)))

    def report_self_collision(self, name, deg):
        """Does the arm intersect ITSELF at this point in the flip?

        NOT CHECKED ANYWHERE UNTIL NOW. collision.py compares arm-against-arm, floor, base
        and mast, never an arm against itself, which is why the finger intersecting its own
        forearm went unseen through every run while being plainly visible on screen.

        Bracketed, because a circular capsule around a flat plate over-reports by up to
        20 mm - the finger plate is 12 mm thick, its capsule 42 mm across. Only where the
        OPTIMISTIC bound is negative is a collision certain.
        """
        try:
            q = self.last_left_q
            if q is None:
                names = ac.LEFT_JOINTS
                if not all(n in self.joints for n in names):
                    return True
                q = [self.joints[n] for n in names]
            # ONE implementation, in collision.py. Computing it inline here is how three
            # wrong versions came to exist.
            worst, who = col.self_clearance(col.arm_capsules(q, +1), optimistic=True)
            if who is None:
                return True
            ok = worst > 0.0
            self.log(f'  SELF-COLLISION [{name}] at {deg:.0f} deg: {worst * 1000:+.1f} mm '
                     f'between {who[0]} and {who[1]}  -> '
                     + ('clear' if ok else
                        '*** INTERSECTING - known open fault, see DESIGN.txt ***'))
            return ok
        except Exception as exc:                                    # noqa: BLE001
            self.log(f'  SELF-COLLISION [{name}]: not evaluated ({exc})')
            return True

    def report_abnormal_digest(self, name):
        """Everything that went wrong this cycle, in order, in one place.

        A run is about a hundred lines. Abnormal events are starred, but the reader still
        has to scan for them and then reconstruct the ORDER - and the order is what tells
        you which failure was causal and which were consequences. The box is the example:
        19 N on a wrist, 0/2 pads touching and a slip stop are three lines far apart in the
        log, and they are all one fault.
        """
        if not self.abnormal:
            self.log(f'  DIGEST [{name}]: nothing abnormal reported this cycle')
            return
        self.log(f'  DIGEST [{name}]: {len(self.abnormal)} abnormal event(s), in order:')
        for i, (phase, line) in enumerate(self.abnormal, 1):
            # the stars are STRIPPED here: this is a summary of events already
            # reported, not a new report, and re-emitting the marker made every
            # abnormal event count twice to anything scanning for it.
            self.log(f'    {i}. [step {phase}] ' + line.replace('***', '').strip())

    def check_grasp_centring(self, name, psi, tol=0.004):
        """How far the object sits off the LINE BETWEEN THE PADS, across the closing axis.

        WHY THIS EXISTS. The log kept saying `left 1/2 touching, right 1/2 touching` - one
        pad per hand in contact, on both hands - and in Gazebo the SAME-SIDE finger of each
        hand was seen overlapping the bottle while the other seated cleanly. Both fingers
        are commanded to the same closure, so that pattern can only mean the object is not
        centred between them: the near pad touches and stalls, the far one keeps closing
        into the surface. Both hands show it on the same side because _ends() places both
        from ONE centre, so a single centring error shifts them together.

        Nothing measured this. The axis was checked (0.5 deg out) and the reach was checked,
        but a LATERAL offset perpendicular to the axis moves neither - which is why it
        survived every check while being visible on screen.

        The number that matters: the pad face sits 42 mm from the hand centre against a
        45 mm object radius, so the capture window is about +/-3 mm. Anything past that puts
        one pad in free air and drives the other in.
        """
        t = self.truth.get(name)
        if not t or t[0] is None:
            self.log(f'  CENTRING [{name}]: no ground truth available')
            return True
        try:
            pl, _Rl = ik.fk([self.joints[j] for j in ac.LEFT_JOINTS], +1)
            pr, _Rr = ik.fk([self.joints[j] for j in ac.RIGHT_JOINTS], -1)
        except Exception:                                          # noqa: BLE001
            return True
        mid = [(pl[i] + pr[i]) / 2.0 for i in range(3)]
        # ONE FRAME. ik.fk() is in base_link; self.truth is in the WORLD. Comparing them
        # directly is only right when the heading is zero, which is true of exactly one
        # object here - the bottle - and it reported +2.1 mm while the box, at a heading of
        # -54.8 deg, came out 185 mm off. That is the same base-versus-world confusion that
        # produced the 53 deg grasp error and the phantom 15 mm of slip, so: lift the hand
        # midpoint into the world and compare there.
        # AND WITH THE ROBOT'S TRUE POSE, not raw odometry and not the wall fit. Odometry
        # heading drifts about 29 deg over a run - 5 mm per degree at this reach - and the
        # wall fit is what the grasp was AIMED with, so judging the aim in that frame is
        # blind to any bias in it: that is how a 10 mm error in the wall constants went
        # unseen. Ground truth first, the wall fit only if there is none.
        if self.robot_truth is not None:
            bx, by, bh = self.robot_truth
        elif self.pose_fix is not None:
            bx, by, bh = self.pose_fix
        elif self.odom is not None:
            bx, by, bh = self.odom[0], self.odom[1], self.heading_now()
        else:
            return True
        cb, sb = math.cos(bh), math.sin(bh)
        mid = [bx + mid[0] * cb - mid[1] * sb, by + mid[0] * sb + mid[1] * cb, mid[2]]
        ox, oy = t[0][0], t[0][1]
        # decompose the error into ALONG the object's axis and ACROSS it. Along is harmless
        # (it just shifts which part of the end is gripped); across is what decides whether
        # both pads can reach.
        ca, sa = math.cos(psi + bh), math.sin(psi + bh)
        ex, ey = ox - mid[0], oy - mid[1]
        along = ex * ca + ey * sa
        across = -ex * sa + ey * ca
        ok = abs(across) <= tol
        self.log(f'  CENTRING [{name}]: object is {across * 1000:+.1f} mm ACROSS the '
                 f'closing axis and {along * 1000:+.1f} mm along it '
                 f'(capture window about +/-3 mm)  -> '
                 + ('centred' if ok else
                    '*** OFF CENTRE - the near pad will stall and the far one will be '
                    'driven into the surface; expect 1/2 touching ***'))
        return ok

    def friction_margin(self, name, label='GRASP QUALITY'):
        """Is there enough friction to carry this object, before we lift it?

        A real cell verifies the grasp before trusting it with the part. The numbers are
        derived from the model, not measured in the sim: four fingers stalled against the
        effort limit give the normal force, the pad and object friction give the
        coefficient, and the object's weight is the load. Reported so the margin is on
        the record next to the run that used it.
        """
        o = self.obj[name]
        normal = 2 * FINGER_EFFORT_LEFT_N + 2 * FINGER_EFFORT_RIGHT_N
        avail = FRICTION_MU * normal
        load = o.get('mass', 0.45) * 9.81
        margin = avail / load if load > 0 else 0.0
        ok = margin >= FRICTION_MARGIN_MIN
        self.log(f'  {label} [{name}]: {normal:.0f} N grip x mu {FRICTION_MU:.1f} '
                 f'= {avail:.1f} N of friction against {load:.2f} N of weight '
                 f'-> {margin:.1f}x  '
                 + ('sufficient' if ok else
                    f'*** BELOW THE {FRICTION_MARGIN_MIN:.1f}x MINIMUM - this grasp '
                    f'is not expected to hold ***'))
        return ok

    def check_slip(self, name, label='SLIP CHECK'):
        """Report slip since the grasp, and say whether it is acceptable."""
        d = self.slip_mm(name)
        if d is None:
            self.log(f'  {label} [{name}]: not measurable (no ground truth or no '
                     f'grasp reference)')
            return True
        if d > self.slip_abort:
            verdict = (f'*** SLIPPING OUT OF THE GRIP - past the '
                       f'{self.slip_abort * 1000:.0f} mm limit ***')
        elif d > self.slip_warn:
            verdict = f'moved, past the {self.slip_warn * 1000:.0f} mm warning'
        else:
            verdict = 'held'
        self.log(f'  {label} [{name}]: object has moved {d * 1000:.1f} mm in the grip '
                 f'since the grasp  -> {verdict}')
        return d <= self.slip_abort

    def regrasp(self, name):
        """Open slightly, close deeper, and re-reference. What a gripper does on slip.

        The fingers are position-commanded past the object's surface and stall against
        the effort limit, so closing further raises the normal force up to that limit.
        Opening first lets the object reseat instead of being dragged.
        """
        cur = self.close_now.get(name, self.obj[name]['close'])
        deeper = max(0.010, cur - self.regrasp_deeper)
        self.log(f'  REGRASP [{name}]: easing off then closing to {deeper:.3f} '
                 f'(was {cur:.3f})')
        self.grippers(min(self.open_pos, cur + 0.006), settle=0.4)
        self.grippers(deeper, settle=0.8)
        self.close_now[name] = deeper
        return self.capture_grasp_reference(name)

    def engage_grasp(self, name):
        """Take hold of the object, by friction or by weld depending on the mode."""
        self.holding = name
        self.close_now.setdefault(name, self.obj[name]['close'])
        if self.grasp_mode == 'suction':
            # A suction gripper IS a detachable attachment, so the DetachableJoint models
            # it honestly rather than standing in for a grasp it cannot achieve - which is
            # what the old 'weld' mode was doing.
            area = math.pi * ac.CUP_RADIUS ** 2
            hold = ac.SUCTION_KPA * 1000.0 * area * ac.CUPS_PER_HAND
            load = self.obj[name].get('mass', 0.45) * 9.81
            self.log(f'  SUCTION [{name}]: {ac.CUPS_PER_HAND} cups x '
                     f'{ac.CUP_RADIUS * 2000:.0f} mm at -{ac.SUCTION_KPA:.0f} kPa '
                     f'= {hold:.1f} N against {load:.2f} N -> {hold / load:.1f}x')
            # THE HONEST CAVEAT. Attaching pulls the object into the robot's kinematic
            # tree, so Gazebo filters finger-object collisions and the contact sensors go
            # quiet. GRIP CHECK is therefore NOT meaningful in this mode - it will read
            # 0/2 on hands that are holding perfectly well, which is exactly the false
            # reading that made an earlier BIMANUAL CHECK vacuous. The grasp is verified
            # here by the object MOVING WITH the hand, which the slip monitor measures.
            self.contacts_are_filtered = True
            # BACK THE FINGERS OFF THE SURFACE BEFORE ATTACHING - see CARRY_EASE. They were
            # stalled on it, so easing now moves nothing; after the attach Gazebo would let
            # them sink to the full squeeze instead, 3 mm inside the object on both sides.
            carry = self.carry_closure(name)
            self.close_now[name] = carry
            self.grippers(carry, settle=0.4)
            ok = self.set_grasp(name, True)
            if ok:
                self.welded.add(name)
            self.capture_grasp_reference(name)
            return ok
        if self.grasp_mode == 'weld':
            ok = self.set_grasp(name, True)
            if ok:
                self.welded.add(name)
            self.capture_grasp_reference(name)
            return ok
        self.log(f'  [{name}] holding by FRICTION - no weld. Fingers are stalled on it '
                 f'at the effort limit; slip is monitored throughout.')
        return self.capture_grasp_reference(name)

    def carry_closure(self, name):
        """The finger command that sits just OUTSIDE the object's surface (suction carry)."""
        return min(self.open_pos, self.obj[name]['close'] + CARRY_EASE)

    def release_grasp(self, name):
        """Let go, and undo a rescue weld if one was engaged.

        In suction mode the fingers squeeze again FIRST, while the cup still holds, so the
        moment it releases the object is gripped by all four fingers and the settle can
        carry it the last few millimetres instead of dropping it.
        """
        self.holding = None
        if self.grasp_mode == 'suction' and name in self.welded:
            self.grippers(self.obj[name]['close'], settle=0.5)
            self.close_now[name] = self.obj[name]['close']
        if name in self.welded:
            self.set_grasp(name, False)
            self.welded.discard(name)
        elif self.grasp_mode == 'weld':
            self.set_grasp(name, False)

    def weld_rescue_now(self, name, why):
        """Last resort: engage the weld so the object is not dropped, and say so."""
        if not self.weld_rescue or name in self.welded:
            return False
        self.log(f'  *** WELD RESCUE [{name}]: {why}. Engaging the rigid joint so the '
                 f'object is not dropped. THIS RUN IS NOT A FRICTION-ONLY RESULT. ***')
        if self.set_grasp(name, True):
            self.welded.add(name)
            self.capture_grasp_reference(name)
            return True
        return False

    def hand_contact(self, side):
        """How many of this hand's two fingers are touching anything, right now."""
        sensors = FINGER_SENSORS[:2] if side == 'left' else FINGER_SENSORS[2:]
        return sum(1 for sensor in sensors if self.touching(sensor))

    def check_both_hands_holding(self, name, label='BIMANUAL CHECK'):
        """Are BOTH hands still on the object partway through the carry?

        THIS IS THE CHECK THAT WAS MISSING, and its absence is why the right hand
        could leave the object unnoticed for an entire run. The contact switches and
        wrist force sensors were fitted, bridged and publishing, and then consulted
        only at the grasp and after release - nothing asked them anything while the
        object was being carried, which is the part a bimanual robot exists to show.

        Only the LEFT palm carries the weld, so the object's pose is rigidly slaved to
        the left hand. The right hand's contact is therefore the only evidence that
        the rotation is bimanual at all, and it is the only one that can be lost.
        """
        if getattr(self, "contacts_are_filtered", False):
            # the object is in the robot tree, so Gazebo filters finger-object
            # collisions and the sensors read empty on hands that ARE holding.
            # Saying so beats reporting a failure that is an artefact - a
            # contact count taken here was what made an earlier version of this
            # check vacuous.
            self.log(f'  {"both hands holding".upper()} [{self.holding}]: not measurable in '
                     f'suction mode - contacts are filtered once the object '
                     f'joins the robot tree; slip is the live evidence')
            return True
        left, right = self.hand_contact('left'), self.hand_contact('right')
        lf = self.wrist_force.get('left', 0.0)
        rf = self.wrist_force.get('right', 0.0)

        # CONTACT IS NOT THE SAME AS HOLDING, and counting contacts alone let this
        # check pass on "left 2/2 (6.1 N) | right 2/2 (1.7 N)" - the right hand was
        # in contact while carrying 22% of the object. A finger brushing the surface
        # registers exactly like a finger bearing load. Only the LEFT palm is welded,
        # so the left hand's share is guaranteed and the right hand's is the
        # measurement that means anything: if it is near zero the object is being
        # rotated one-handed with the other hand alongside for appearance.
        total = lf + rf
        share = (rf / total) if total > 0.5 else 0.0
        touching = left > 0 and right > 0
        loaded = share >= MIN_LOAD_SHARE
        ok = touching and loaded
        if not touching:
            verdict = '*** ONE HAND HAS LET GO - the rotation is not bimanual ***'
        elif not loaded:
            verdict = (f'*** THE RIGHT HAND IS ONLY RESTING ON IT - {share * 100:.0f}% '
                       f'of the load against a {MIN_LOAD_SHARE * 100:.0f}% minimum, so '
                       f'it is touching but not carrying ***')
        else:
            verdict = f'both hands carrying it (right takes {share * 100:.0f}%)'
        self.log(f'  {label} [{name}]: left {left}/2 touching ({lf:.1f} N) | '
                 f'right {right}/2 touching ({rf:.1f} N)  -> {verdict}')
        return ok

    def release_all(self, name=None):
        """Drop anything held and open the hands. Safe to call at any time.

        A cycle that dies holding an object used to leave it welded to the palm, so the
        next cycle grasped a second object with the first still attached.
        """
        for n in ([name] if name else list(self.order)):
            try:
                self.set_grasp(n, False, tries=2)
            except Exception:  # noqa: BLE001
                pass
        try:
            self.grippers(self.open_pos, settle=0.2)
        except Exception:  # noqa: BLE001
            pass

    def base_frame_yaw(self, world_yaw):
        """Convert an angle known in WORLD terms into the base frame.

        Uses the MEASURED heading. This is what stops the assumed grasp angle drifting:
        the object's axis is a fact about the world, so subtracting where the robot
        actually points gives the angle to grasp at, however far the base has turned or
        slipped. Falls back to bias-corrected odometry, which is still better than
        assuming the base is where it thinks it is.
        """
        if self.pose_fix is not None:
            heading = self.pose_fix[2]
        else:
            heading = norm_ang(self.odom[2] - self.heading_bias)
        return norm_ang(world_yaw - heading)

    def heading_now(self):
        """Best available heading: the measured one, else bias-corrected odometry."""
        if self.pose_fix is not None:
            return self.pose_fix[2]
        return norm_ang(self.odom[2] - self.heading_bias)

    def set_grasp(self, name, attach, tries=6):
        pub = self.attach_pub[name] if attach else self.detach_pub[name]
        want = 'attached' if attach else 'detached'
        for _ in range(tries):
            pub.publish(Empty())
            t0 = self._now()
            while self._now() - t0 < 0.6:
                if self.grasp_state.get(name) == want:
                    return True
                time.sleep(0.02)
        self.log(f'WARN [{name}] grasp "{want}" not confirmed')
        return False

    def send_pair(self, tl, tr, extra_wait=5.0):
        self._slip_warned = False
        # 5.0, NOT 2.0. I cut this to 2.0 chasing speed, and at gain 1.9 the arm
        # simply needs longer than that to settle: phases that had been quiet for
        # twenty runs started reporting "arms did not converge", and one of them was
        # gated on convergence so it aborted. The grace is not the bottleneck.
        self.left_pub.publish(tl)
        self.right_pub.publish(tr)
        target = {}
        target.update(ac.final_positions(tl))
        target.update(ac.final_positions(tr))
        # SIMULATED seconds - see _now()
        deadline = self._now() + max(ac.total_time(tl), ac.total_time(tr)) + extra_wait
        while self._now() < deadline:
            # PROTECTIVE STOP. Checked every cycle of the wait, not just at the ends,
            # because by the time a motion finishes the damage is done. Aborting leaves
            # the arm wherever it stopped, which is recoverable; pushing on is not.
            reason = self.overload()
            if reason is not None:
                # WHAT TO DO ON A PROTECTIVE STOP DEPENDS ON WHETHER WE ARE HOLDING
                # SOMETHING, and getting this wrong is what turned one bad transient into
                # a wrecked run on 2026-09-20.
                #
                # CARRYING: freeze. The object's pose is slaved to the welded palm, so
                # moving is how it gets crushed or dropped. Holding still is safe and
                # recoverable, and a human can see exactly where it stopped.
                #
                # EMPTY-HANDED: retreat to the parked pose. Freezing mid-transit strands
                # the arm in a pose that was never a valid place to be - on 2026-09-20 it
                # froze both arms fully extended at r=0.52 m, and the base then rotated
                # with them stuck out like that, sweeping the objects across the room and
                # shoving itself off its spot. There is nothing to protect when the hands
                # are empty, so the safe state is the parked one.
                carrying = any(v == 'attached' for v in self.grasp_state.values())
                self.left_pub.publish(ac.hold(ac.LEFT_JOINTS, self.joints))
                self.right_pub.publish(ac.hold(ac.RIGHT_JOINTS, self.joints))
                if carrying:
                    self.log(f'  *** PROTECTIVE STOP: {reason} - holding position '
                             f'(carrying) ***')
                else:
                    self.log(f'  *** PROTECTIVE STOP: {reason} - empty-handed, '
                             f'retreating to the parked pose ***')
                    self.left_pub.publish(
                        ac.joint_traj(ac.LEFT_JOINTS, [mo.TRAVEL_LEFT], 2.5, t0=2.5))
                    self.right_pub.publish(
                        ac.joint_traj(ac.RIGHT_JOINTS, [mo.TRAVEL_RIGHT], 2.5, t0=2.5))
                    self._sleep(3.0)
                return False
            # SLIP-REACTIVE CONTROL. Checked inside the wait, because a grasp fails
            # DURING a motion - by a phase boundary the object is already out. Past the
            # warning the pace eases so the grip can recover; past the abort the motion
            # stops and the caller can regrasp.
            if self.holding is not None:
                d = self.slip_mm(self.holding)
                if d is not None and d > self.slip_abort:
                    self.slip_events.append((self.holding, 'motion', d * 1000.0))
                    self.log(f'  *** SLIP STOP [{self.holding}]: object has moved '
                             f'{d * 1000:.1f} mm in the grip, past the '
                             f'{self.slip_abort * 1000:.0f} mm limit - stopping ***')
                    self.left_pub.publish(ac.hold(ac.LEFT_JOINTS, self.joints))
                    self.right_pub.publish(ac.hold(ac.RIGHT_JOINTS, self.joints))
                    # RECOVER, DO NOT JUST STOP. regrasp() and weld_rescue_now() were
                    # written and then never called, so the first friction run reported
                    # "no weld rescue needed" when no rescue could possibly have fired,
                    # and every slip ended as a drop.
                    held = self.holding
                    for attempt in range(self.regrasp_attempts):
                        self.log(f'  recovery attempt {attempt + 1} of '
                                 f'{self.regrasp_attempts}')
                        if self.regrasp(held):
                            d2 = self.slip_mm(held)
                            if d2 is None or d2 <= self.slip_abort:
                                self.log(f'  REGRASP HELD [{held}] - resuming')
                                return False       # caller re-issues the motion
                    self.weld_rescue_now(held, 'the friction grasp could not be '
                                               'recovered by regrasping')
                    return False
                if d is not None and d > self.slip_warn and not self._slip_warned:
                    self._slip_warned = True
                    self.slip_events.append((self.holding, 'warn', d * 1000.0))
                    self.log(f'  SLIP WARNING [{self.holding}]: {d * 1000:.1f} mm - '
                             f'easing off the pace')
                    self._sleep(0.5)
            if all(abs(self.joints.get(j, 99.0) - v) < self.tol
                   for j, v in target.items()):
                return True
            time.sleep(0.05)
        errs = sorted(((abs(self.joints.get(j, 99.0) - v), j)
                       for j, v in target.items()), reverse=True)
        self.log('WARN arms did not converge - '
                 + ', '.join(f'{j} off {e:.3f}' for e, j in errs[:3]))
        return False

    def settle_arms(self, name, where='the grasp pose', tol=SETTLE_TOL,
                    timeout=SETTLE_TIMEOUT):
        """Wait until every arm joint is within ``tol`` of where it was last sent.

        Called at the hover, before descending, and at the grasp, before the fingers close -
        see SETTLE_TOL for why the loose convergence gate of send_pair() is not enough
        there. Timed in SIMULATED seconds.
        """
        target = dict(zip(ac.LEFT_JOINTS, self.last_cmd_l))
        target.update(zip(ac.RIGHT_JOINTS, self.last_cmd_r))
        t0 = self._now()
        err = 9.9
        while True:
            err = max(abs(self.joints.get(j, 99.0) - v) for j, v in target.items())
            if err <= tol or self._now() - t0 >= timeout:
                break
            if self.overload() is not None:
                break
            time.sleep(0.02)
        ok = err <= tol
        self.log(f'  ARMS SETTLED [{name}]: worst joint {err:.4f} rad from {where} '
                 f'after {self._now() - t0:.1f} s (want {tol:.3f})  -> '
                 + ('settled' if ok else
                    '*** STILL MOVING - the hands are not where they were sent ***'))
        return ok

    @staticmethod
    def _wp_key(pair):
        """A stable key for a waypoint pair, so a pre-planned solution can be found again.

        Waypoints are produced by pure functions of the same arguments, so rebuilding a leg
        gives bit-identical numbers; rounding is belt and braces.
        """
        (pl, Rl), (pr, Rr) = pair
        return (tuple(round(v, 9) for v in pl) + tuple(round(v, 9) for v in pr)
                + tuple(round(Rl[i][0], 9) for i in range(3))
                + tuple(round(Rr[i][0], 9) for i in range(3)))

    def move(self, pairs, dt, t0=1.5):
        """Plan and send one leg, SEEDED from where the arms already are.

        The seed is the whole reason this is not a one-liner. Planning each leg from
        scratch let the solver take the other wrist branch at a phase boundary - identical
        hand pose, 180 deg of wrist - and Gazebo showed it as right_j4 off 87.7 deg with
        the right hand spinning through the bottle it was holding.
        """
        keys = [self._wp_key(w) for w in pairs]
        if self._plan and all(k in self._plan for k in keys):
            ql = [self._plan[k][0] for k in keys]
            qr = [self._plan[k][1] for k in keys]
            tl = ac.joint_traj(ac.LEFT_JOINTS, ql, dt, t0)
            tr = ac.joint_traj(ac.RIGHT_JOINTS, qr, dt, t0)
        else:
            if self._plan:
                self.plan_misses += len(pairs)
                self.log(f'  * PLAN MISS: a leg of {len(pairs)} waypoints starting at '
                         f'left {[round(v, 3) for v in pairs[0][0][0]]} was not in the '
                         f'whole-cycle plan, so it is being planned on its own - the '
                         f'joint path across this boundary is not guaranteed continuous *')
            # BOTH SHOULDER SOLUTIONS, or a leg that starts or ends parked cannot be solved
            # at all on its own - the parked pose needs the second one (see ik.solve).
            tl, tr = ac.pair_trajs(pairs, dt, t0=t0,
                                   seed_l=self.last_cmd_l, seed_r=self.last_cmd_r,
                                   finger_pos=self.plan_finger_pos,
                                   wide=[True] * len(pairs))
        self.last_cmd_l = [float(v) for v in tl.points[-1].positions]
        self.last_cmd_r = [float(v) for v in tr.points[-1].positions]
        return self.send_pair(tl, tr)

    def travel_pose(self):
        # This commands the parked pose DIRECTLY, bypassing the planner, so the recorded
        # seed has to be reset with it or the next planned leg would be seeded from
        # wherever the arms were before parking.
        self.last_cmd_l = list(mo.TRAVEL_LEFT)
        self.last_cmd_r = list(mo.TRAVEL_RIGHT)
        self.plan_finger_pos = 0.0
        return self.send_pair(
            ac.joint_traj(ac.LEFT_JOINTS, [mo.TRAVEL_LEFT], 2.0, t0=2.0),
            ac.joint_traj(ac.RIGHT_JOINTS, [mo.TRAVEL_RIGHT], 2.0, t0=2.0))

    def face(self, azimuth, tol=None, max_turns=4):
        """Rotate until the MEASURED heading matches ``azimuth``, iterating.

        WHY THIS HAS TO BE A LOOP. The wall fit can only be trusted near the heading it
        is measuring, because it needs a prior good to about 50 degrees to tell the two
        walls apart. A single measure-then-turn cannot work: before the turn the prior is
        wrong by the whole turn angle, that fix is rejected, the bias stays stale, and
        the turn servos to a stale target and stops short. Pick-place landed 14 to 20
        degrees off exactly that way.

        So: turn on the best bias available, MEASURE, and turn again by the residual.
        Each turn is smaller than the last, so each prior is better than the last.
        """
        if tol is None:
            tol = self.heading_tol
        while self.odom is None:
            time.sleep(0.1)
        for attempt in range(max_turns):
            self.relocalise(prior=norm_ang(self.odom[2] - self.heading_bias))
            target_odom = norm_ang(azimuth + self.heading_bias)
            if abs(norm_ang(target_odom - self.odom[2])) < tol and attempt > 0:
                break
            while True:
                err = norm_ang(target_odom - self.odom[2])
                if abs(err) < tol:
                    break
                t = Twist()
                t.angular.z = max(-self.rot_speed,
                                  min(self.rot_speed, 1.5 * err))
                # Graded floor: a flat 0.10 rad/s cannot settle inside the tolerance,
                # it hunts. Creep for the last fraction of a degree.
                floor = 0.10 if abs(err) > 0.05 else 0.035
                if abs(t.angular.z) < floor:
                    t.angular.z = floor if err > 0 else -floor
                self.cmd_pub.publish(t)
                time.sleep(0.02)
            self.cmd_pub.publish(Twist())
            self._sleep(0.4)
            if not self.relocalise(prior=azimuth):
                # DO NOT GIVE UP. This used to `break`, abandoning the loop after one
                # open-loop turn made with a stale bias; pick-place landed 23 degrees off
                # that way. The thing that broke the fit was transient.
                self.log(f'  could not confirm the heading on pass {attempt + 1}; '
                         f'retrying')
                continue
            true_h = norm_ang(self.odom[2] - self.heading_bias)
            residual = norm_ang(azimuth - true_h)
            self.log(f'  facing {math.degrees(azimuth):+.0f} deg: pass '
                     f'{attempt + 1}, measured {math.degrees(true_h):+.2f}, '
                     f'residual {math.degrees(residual):+.2f} deg')
            if abs(residual) < math.radians(0.6):
                return True
        return False

    def _to_base(self, wx, wy):
        """A world point in the base frame, from the measured pose (the wall fix)."""
        px, py, h = self.pose_fix if self.pose_fix is not None else (
            self.odom[0], self.odom[1], self.heading_now())
        dx, dy = wx - px, wy - py
        c, s = math.cos(h), math.sin(h)
        return (c * dx + s * dy, -s * dx + c * dy), h

    def _to_world(self, bx, by):
        px, py, h = self.pose_fix if self.pose_fix is not None else (
            self.odom[0], self.odom[1], self.heading_now())
        c, s = math.cos(h), math.sin(h)
        return (px + c * bx - s * by, py + s * bx + c * by)

    def fit_pose(self, sc, prior):
        """One scan's pose fit: ``(pose, info)``, ``pose`` None with ``info['reject']``.

        The fixed-base sections' world is the two-wall L of wall_ref. A section in a different
        world overrides this and nothing else - every fix goes through relocalise(), so facing,
        aiming, recentring and every check use whatever this measures.
        """
        return wr.pose_from_scan(
            sc.ranges, sc.angle_min, sc.angle_increment,
            sc.range_min, sc.range_max, band=self.wall_band,
            max_rms=self.wall_max_rms,
            expect_heading=(prior if prior is not None
                            else norm_ang(self.odom[2] - self.heading_bias)))

    def relocalise(self, tries=8, prior=None):
        """Measure the TRUE pose off the two walls and update the odometry bias.

        The base rotates to face each object and the wheels slip, so the odometry heading
        drifts - the first grasp was clean and later ones were millimetres out. The walls
        give an absolute reference that cannot drift, however far the robot has turned.

        ``prior`` is a rough heading used ONLY to decide which returns belong to which
        wall. Pass the heading just commanded when there is one: the default of
        ``odom - heading_bias`` is stale, because the bias was last measured before the
        turn and is therefore wrong by the whole turn angle. Measured once, that produced a
        61 deg prior error, the fit locked onto the wrong wall and reported a heading 19 deg
        out.

        On failure the POSE is discarded. Keeping it is worse than having none - a pose
        measured at one heading and reused after a 60 degree turn aims the arms at a
        completely wrong point (the can was once aimed behind the robot that way).
        """
        reason = 'no scan yet'
        for _ in range(tries):
            sc = self.scan
            if sc is not None:
                pose, info = self.fit_pose(sc, prior)
                if pose is not None:
                    self.pose_fix = pose
                    self.heading_bias = norm_ang(self.odom[2] - pose[2])
                    drift = math.hypot(pose[0] - self.odom[0], pose[1] - self.odom[1])
                    self.log(f'  wall fix: true pose ({pose[0]:+.3f}, {pose[1]:+.3f}, '
                             f'{math.degrees(pose[2]):+.2f} deg); odometry says '
                             f'({self.odom[0]:+.3f}, {self.odom[1]:+.3f}, '
                             f'{math.degrees(self.odom[2]):+.2f}) - position off by '
                             f'{drift * 1000:.0f} mm, heading by '
                             f'{math.degrees(abs(self.heading_bias)):.2f} deg '
                             f'[{info["beams"]} beams, rms {info["rms"] * 1000:.0f} mm]')
                    return True
                reason = info.get('reject', 'no fix')
            time.sleep(0.25)
        self.pose_fix = None
        self.log(f'  WARN no wall fix ({reason}); pose discarded')
        return False

    def aim(self, azimuth, radius):
        """Base-frame position of a target known in WORLD terms, and whether it was measured.

        The base is only ever commanded to rotate, but slip that rotates also shifts, so
        it does not stay on the origin. This converts the target's world position into the
        frame the robot is ACTUALLY in, so a few centimetres of drift no longer moves the
        aim point. Falls back to the nominal ``(reach, 0)`` when there is no fix.
        """
        self.relocalise(prior=azimuth)
        tx, ty, _axis, corrected = wr.aim_target(
            self.pose_fix, azimuth, radius, self.reach, LATERAL)
        if not corrected:
            self.log(f'  WARN no pose fix - cannot aim; nominal would be '
                     f'({self.reach:.3f}, 0.000)')
        else:
            self.log(f'  aim: target at base-frame ({tx:+.3f}, {ty:+.3f})')
        # CLAMP THE RADIUS. Pushed far enough out, the correction makes the target
        # unreachable and the whole cycle dies on IK - strictly worse than placing a few mm
        # short. Only binds where a section configures reach_max.
        r = math.hypot(tx, ty)
        if r > self.reach_max:
            k = self.reach_max / r
            self.log(f'  aim radius {r:.3f} exceeds the {self.reach_max:.3f} ceiling; '
                     f'clamping to keep the lift reachable')
            tx, ty = tx * k, ty * k
        # THE FLAG IS RETURNED, not swallowed. Falling back to the nominal point
        # silently is how a measured-position design degrades into the open-loop one
        # it replaced, with nothing in the log to say so.
        return (tx, ty), corrected

    def stop_base(self):
        self.cmd_pub.publish(Twist())

    def drive(self, dist):
        """Drive straight by ``dist`` metres, closed on the wall fix rather than odometry."""
        start = self.pose_fix
        if start is None:
            return False
        t0 = self._now()
        while self._now() - t0 < 12.0:
            self.relocalise(tries=2)
            if self.pose_fix is None:
                break
            moved = math.hypot(self.pose_fix[0] - start[0], self.pose_fix[1] - start[1])
            if moved >= abs(dist) - 0.004:
                break
            t = Twist()
            t.linear.x = (1.0 if dist > 0 else -1.0) * self.v_lin
            self.cmd_pub.publish(t)
            time.sleep(0.05)
        self.stop_base()
        self._sleep(0.3)
        return True

    def averaged_fix(self, n=5, need=3):
        """Mean (x, y) over several wall fixes, or None if too few succeed.

        One fix is good to a few millimetres - fine for deciding WHERE the robot is, not for
        deciding which way to drive a correction of similar size: that direction is atan2 of
        two noisy small numbers. Refusing to act on fewer than ``need`` fixes stops a lucky
        outlier from sending the base off on its own.
        """
        xs, ys = [], []
        for _ in range(n):
            if self.relocalise(tries=2):
                xs.append(self.pose_fix[0])
                ys.append(self.pose_fix[1])
        if len(xs) < need:
            return None
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def recentre(self):
        """Drive the base back to the world origin if it has drifted off it.

        WHY A BASE THAT "NEVER TRANSLATES" MOVES. It is never commanded to, but it moves
        anyway: the wheels slip badly enough to accumulate 29 deg of heading error in one
        run, and slip that rotates also shifts. The arm's aim correction absorbs only about
        20 mm before a target passes the reach limit, so beyond that the drift is removed
        rather than reached around. Rotate-drive-rotate, because the base is differential;
        every leg is closed on the wall fix, not on the odometry it is correcting. Inside
        ``position_deadband`` nothing is done - the direction would be noise.
        """
        for _ in range(3):
            fix = self.averaged_fix()
            if fix is None:
                self.log('  WARN cannot recentre without a wall fix')
                return False
            x, y = fix
            err = math.hypot(x, y)
            if err <= self.pos_deadband:
                self.log(f'  {err * 1000:.0f} mm from the origin - inside the '
                         f'{self.pos_deadband * 1000:.0f} mm deadband, not correcting')
                return True
            self.log(f'  recentring: {err * 1000:.0f} mm off ({x:+.3f}, {y:+.3f})')
            self.face(math.atan2(-y, -x))      # point at the origin
            self.drive(err)
        self.relocalise()
        if self.pose_fix:
            self.log(f'  recentre residual '
                     f'{math.hypot(self.pose_fix[0], self.pose_fix[1]) * 1000:.0f} mm')
        return True

    def report_clearance(self, name, legs, ql, qr, fingers, named, where):
        """Log how close a plan comes to each named obstacle, which leg and which arm.

        Printed every cycle, so that when something does get touched the log says what, by
        how much and where. ``named`` and ``where`` are as col.nearest_objects() takes them.
        """
        at, k = {}, 0
        for leg_name, seq in legs:
            at[leg_name] = range(k, k + len(seq))
            k += len(seq)

        def leg_of(i):
            return next((n for n, r in at.items() if i in r), '?')
        for obj_name, (gap, i, side) in col.nearest_objects(ql, qr, fingers, named,
                                                            where).items():
            self.log(f'  CLEARANCE [{name}]: {obj_name} {gap * 1000:+.0f} mm at the '
                     f'closest ({side} arm, {leg_of(i)} leg)  -> '
                     + ('clear' if gap >= 0.010 else
                        '*** WITHIN 10 mm - the arm may touch it ***'))

    # ------------- top level -------------
    def _safe_run(self):
        try:
            self.run()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'task aborted: {exc!r}')
            self.stop_base()

    def return_home(self):
        """Put the robot back where it started: parked arms, original heading.

        The base is only ever commanded to rotate, so "home" is the heading it began
        at - azimuth zero, which is the direction it faced when the world loaded. After
        the last object the robot was left pointing at wherever that object happened to
        be (-55 deg for the box), which makes the end of a run look unfinished and makes
        two consecutive runs start from different places.

        Arms first, then the turn. Turning with the arms still out at the placement pose
        is how the 2026-09-20 run swept its objects across the room, so the order here is
        deliberate rather than incidental.
        """
        self.log('returning to the starting pose')
        self.travel_pose()
        if not self.face(0.0):
            self.log('  WARN could not confirm the starting heading; '
                     'leaving the base where it is')
            return False
        self.log('  back at the starting heading, arms parked')
        return True

    def before_cycles(self):
        """Anything a section does once before its first object - perceive surveys the room.
        Nothing by default."""

    def run_cycle(self, n):
        """One object's cycle, guarded. Returns what cycle() returned, False if it raised."""
        ok = False
        try:
            ok = self.cycle(n)
        except ik.IKError as exc:
            self.log(f'[{n}] IK failure: {exc}')
            ok = False
        except Exception as exc:  # noqa: BLE001
            self.log(f'[{n}] FAILED: {exc!r}')
            ok = False
        finally:
            self.stop_base()
            # A FAILED CYCLE MUST NOT CARRY STATE INTO THE NEXT. One that died holding
            # its object used to leave it attached, and the next cycle then picked up a
            # second object with the first still hanging from the palm.
            if ok is not True:
                self.release_all(n)
        return ok

    def cycles(self):
        """Every object, in the configured order: ``{name: result}``. A section that decides
        for itself which object comes next - navigate, from what its scan found - overrides
        this and runs each one through run_cycle()."""
        return {n: self.run_cycle(n) for n in self.order}

    def run(self):
        self.log('waiting for simulation interfaces...')
        while self.odom is None:
            time.sleep(0.1)
        while not all(j in self.joints for j in ALL_JOINTS):
            time.sleep(0.2)
        self.log(f'all {len(ALL_JOINTS)} joints reporting; controllers ready')
        time.sleep(2.0)

        # arm.xacro sets <suppress_initial_attach>, so these load detached. Releasing
        # anyway makes the intended starting state explicit.
        for n in self.order:
            self.set_grasp(n, False)
            self.welded.discard(n)
        self.grippers(self.open_pos, settle=0.5)
        self.travel_pose()
        self.before_cycles()
        results = self.cycles()

        # EVERYTHING LET GO, THEN BACK TO THE START. Releasing first matters: turning
        # the base while still holding something would carry it round with the robot.
        for n in self.order:
            self.release_grasp(n)
        self.grippers(self.open_pos, settle=0.5)
        self.return_home()

        self.log('TASK COMPLETE: ' + ', '.join(
            f'{k}={"ok" if v else "failed"}' for k, v in results.items()))
        if self.slip_events:
            warned = [e for e in self.slip_events if e[1] == 'warn']
            stopped = [e for e in self.slip_events if e[1] != 'warn']
            worst = max(e[2] for e in self.slip_events)
            self.log(f'SLIP SUMMARY: {len(warned)} warnings, {len(stopped)} stops, '
                     f'worst {worst:.1f} mm'
                     + (f', weld rescued {sorted(self.welded)}' if self.welded
                        else ', no weld rescue needed'))
        else:
            self.log('SLIP SUMMARY: no slip detected at any point')
        time.sleep(2.0)
        self.verify_pub.publish(Empty())


def spin(node_cls, args=None):
    """The entry point every section's task node uses."""
    rclpy.init(args=args)
    node = node_cls()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # Ctrl-C under ros2 launch shuts the context down while a message is being taken,
        # and rclpy raises "Unable to convert call argument" from take_message(). Harmless at
        # shutdown - re-raised if it happens while ROS is still running.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
