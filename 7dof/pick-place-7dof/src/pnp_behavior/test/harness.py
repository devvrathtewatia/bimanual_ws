"""Headless harness (section A): runs the REAL task_node cycle against a fake robot.

WHY THIS EXISTS
---------------
Four bugs have reached the user's laptop that could only ever surface by RUNNING the
node: a label string passed where a float was expected, a missing ``sign_left``, three
attributes read but never assigned, and a yaw alignment that commanded a 148 degree
rotation. The offline suite passed through every one of them, because it tested
geometry against its own fixtures and never imported task_node.

So this stubs out rclpy and the message packages, instantiates the real TaskNode, and
drives its real ``cycle()`` against a fake robot that:

  * integrates /cmd_vel into a true heading, so face() actually has to converge
  * synthesises a LiDAR scan from that heading, so relocalise() actually has to fit
  * echoes commanded joint positions back as joint states, so send_pair() completes
  * reports object poses, so the runtime checks have something to read

It is not a physics simulation and does not pretend to be. It answers exactly one
question: does the code do what it thinks it does? Anything about contact dynamics
still needs Gazebo.

Run: python3 test/harness.py
"""
import math
import math as _m          # used by the fake robot below
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)


# --------------------------------------------------------------------- stubs
def _ns(**fields):
    """A message-like object: attributes, default-constructed."""
    class M:
        def __init__(self, **kw):
            for k, v in fields.items():
                setattr(self, k, v() if callable(v) else v)
            for k, v in kw.items():
                setattr(self, k, v)
    return M


class _Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Quat:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x, self.y, self.z, self.w = x, y, z, w


class _Pose:
    def __init__(self):
        self.position = _Vec()
        self.orientation = _Quat()


class _Twist:
    def __init__(self):
        self.linear = _Vec()
        self.angular = _Vec()


class _Logger:
    def __init__(self):
        self.lines = []

    def _emit(self, kind, msg):
        self.lines.append(f'[{kind}] {msg}')

    def info(self, m):
        self._emit('info', m)

    def warn(self, m):
        self._emit('warn', m)

    def error(self, m):
        self._emit('ERROR', m)


class _Param:
    def __init__(self, v):
        self.value = v


class _Pub:
    def __init__(self, sink):
        self._sink = sink

    def publish(self, msg):
        if self._sink:
            self._sink(msg)


class FakeNode:
    """Just enough rclpy.node.Node for the task node to construct and run."""

    _params = {}
    _sinks = {}

    def __init__(self, name, **kw):
        self._name = name
        self._logger = _Logger()
        self.subs = []
        self.timers = []

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return types.SimpleNamespace(now=lambda: types.SimpleNamespace(
            nanoseconds=int(SIM_T[0] * 1e9)))

    def get_parameter(self, name):
        if name not in FakeNode._params:
            raise KeyError(name)
        return _Param(FakeNode._params[name])

    def create_publisher(self, _t, topic, _q=10):
        return _Pub(lambda msg, tp=topic: FakeNode.SINK(tp, msg))

    SINK = staticmethod(lambda topic, msg: None)

    def create_subscription(self, _t, topic, cb, _q=10):
        self.subs.append((topic, cb))
        return object()

    def create_timer(self, _period, cb):
        t = types.SimpleNamespace(cancel=lambda: None, cb=cb)
        self.timers.append(t)
        return t


def install_stubs():
    """Put fake ROS modules into sys.modules BEFORE task_node is imported."""
    rclpy = types.ModuleType('rclpy')
    rclpy.init = lambda **kw: None
    rclpy.spin = lambda n: None
    rclpy.try_shutdown = lambda: None
    node_mod = types.ModuleType('rclpy.node')
    node_mod.Node = FakeNode
    rclpy.node = node_mod
    sys.modules['rclpy'] = rclpy
    sys.modules['rclpy.node'] = node_mod
    rclpy.ok = lambda: True
    ex_mod = types.ModuleType('rclpy.executors')
    ex_mod.ExternalShutdownException = type('ExternalShutdownException', (Exception,), {})
    rclpy.executors = ex_mod
    sys.modules['rclpy.executors'] = ex_mod

    def mod(name, **types_):
        m = types.ModuleType(name)
        for k, v in types_.items():
            setattr(m, k, v)
        sys.modules[name] = m
        parent = name.rsplit('.', 1)[0]
        if parent != name and parent in sys.modules:
            setattr(sys.modules[parent], name.rsplit('.', 1)[1], m)
        return m

    for pkg in ('geometry_msgs', 'nav_msgs', 'sensor_msgs', 'std_msgs',
                'trajectory_msgs', 'ros_gz_interfaces', 'tf2_msgs',
                'builtin_interfaces'):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))

    mod('geometry_msgs.msg', Twist=_Twist, Wrench=_ns(force=_Vec, torque=_Vec),
        PoseStamped=_ns(pose=_Pose, header=lambda: types.SimpleNamespace()))
    mod('nav_msgs.msg', Odometry=_ns(pose=lambda: types.SimpleNamespace(
        pose=_Pose()), twist=lambda: types.SimpleNamespace(twist=_Twist())))
    mod('sensor_msgs.msg',
        JointState=_ns(name=list, position=list, velocity=list, effort=list),
        LaserScan=_ns(ranges=list, angle_min=0.0, angle_increment=0.0,
                      range_min=0.12, range_max=8.0),
        Imu=_ns(orientation=_Quat, angular_velocity=_Vec,
                linear_acceleration=_Vec))
    mod('std_msgs.msg', Empty=_ns(), String=_ns(data=''),
        Float64MultiArray=_ns(data=list))
    mod('trajectory_msgs.msg',
        JointTrajectory=_ns(joint_names=list, points=list),
        JointTrajectoryPoint=_ns(positions=list, velocities=list,
                                 time_from_start=lambda: types.SimpleNamespace(
                                     sec=0, nanosec=0)))
    mod('ros_gz_interfaces.msg', Contacts=_ns(contacts=list))
    mod('tf2_msgs.msg', TFMessage=_ns(transforms=list))
    mod('builtin_interfaces.msg', Duration=_ns(sec=0, nanosec=0))


install_stubs()

import time as _time                                          # noqa: E402
# A SIMULATED CLOCK. The node now waits on simulation time (its get_clock()), so sleeping
# advances this clock instead of the wall - the run stays instant and every wait still ends.
SIM_T = [1.0]


def _fake_sleep(s=0.0, *_a, **_k):
    SIM_T[0] += max(float(s), 1e-4)


_time.sleep = _fake_sleep

from pnp_behavior import arm_commander as ac                   # noqa: E402
from pnp_behavior import choreography as ch                    # noqa: E402
from pnp_behavior import ik                                    # noqa: E402
from pnp_behavior import wall_ref as wr                        # noqa: E402


# --------------------------------------------------------------------- config
def load_params(path):
    """Flatten the task_node block of the YAML the way ROS would present it."""
    out, stack = {}, []
    with open(path) as fh:
        lines = fh.readlines()
    in_task = False
    for raw in lines:
        if raw.startswith('task_node:'):
            in_task = True
            continue
        if raw and not raw[0].isspace() and not raw.startswith('task_node'):
            if in_task and raw.strip():
                in_task = False
        if not in_task:
            continue
        line = raw.split('#')[0].rstrip()
        if not line.strip() or line.strip() == 'ros__parameters:':
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(':')
        val = val.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        prefix = '.'.join(k for _, k in stack)
        full = f'{prefix}.{key}' if prefix else key
        if val == '':
            stack.append((indent, key))
            continue
        if val.startswith('['):
            items = [v.strip().strip("'\"") for v in val[1:-1].split(',') if v.strip()]
            try:
                out[full] = [float(v) for v in items]
            except ValueError:
                out[full] = items
        else:
            try:
                out[full] = float(val)
            except ValueError:
                out[full] = val.strip("'\"")
    return out


# ------------------------------------------------------------------ fake robot
def world_wall_faces():
    """The wall FACES the LiDAR actually sees, read from the world file.

    NOT from wall_ref. This fake used to scan walls placed at wall_ref's own constants, so
    the node's localisation was checked against itself and could not be wrong - which is
    how constants holding the wall CENTRES (0.46) instead of the faces (0.45) survived
    every run of this harness while biasing every grasp in Gazebo by 10 mm.
    Returns (wall_a_x, wall_a_half_width, wall_b_y, wall_b_span).
    """
    import re as _re
    # the walls are part of the shelf model in this section
    sdf = open(os.path.join(os.path.dirname(PKG), 'pnp_gazebo', 'models', 'shelf',
                            'model.sdf')).read()

    def box(name):
        m = _re.search(r'<collision name="%s"><pose>([^<]+)</pose>\s*<geometry><box><size>'
                       r'([^<]+)</size>' % name, sdf)
        assert m, f'wall {name} not found in the shelf model'
        return [float(v) for v in m.group(1).split()], [float(v) for v in m.group(2).split()]

    (ax, ay, _az, *_), (sx, sy, _sz) = box('backwall')
    (bx, by, _bz, *_), (tx, ty, _tz) = box('sidewall')
    # the face towards the robot, which sits at the origin
    return (-(ax + sx / 2.0), sy / 2.0, -(by + ty / 2.0), (bx - tx / 2.0, bx + tx / 2.0))


WALL_A, WALL_HALF, WALL_B, WALL_B_SPAN = world_wall_faces()


ROBOT_MODEL = 'pnp_bimanual'
# WHERE A FINGER STALLS ON AN OBJECT: its pad face meets the 45 mm surface when the finger is
# at half_width + PAD_HALF. Letting go opens to motion.release_closure(), 4 mm past that, so
# a fake that stalled any wider would think the hands were still holding and lift the object
# off the shelf with them on the way up.
BIND = 0.045 + 0.006


class FakeRobot:
    """A robot that is wrong in the ways the real one is, and right about nothing else.

    It integrates commanded velocity, synthesises a scan from where it actually is, and
    echoes joint commands. It deliberately models ONE physical effect: a finger stalls
    when it meets an object, because the grasp check depends on that and nothing else
    here can tell us whether that logic is right.
    """

    def __init__(self, objects, fixtures=None):
        self.fixtures = fixtures or {}  # name -> (centre, yaw rad, size), WORLD frame
        self.heading = 0.0
        self.xy = [-0.004, -0.007]        # a few mm off, as the real one always is
        self.joints = {}
        self.objects = objects            # name -> dict(xy, z, axis_deg, held)
        self.welded = {n: False for n in objects}
        # A FRICTION GRASP IS ALSO A GRASP. This fake used to carry an object only when
        # the weld's /attach topic fired, so with grasp_mode: friction nothing was ever
        # picked up - and the slip monitor correctly reported the object still on the
        # floor, 195 mm from the hands. That was the monitor working, not a bug.
        # Modelled here as a PERFECT friction grasp: closing the fingers past the
        # object's surface holds it. Real slip physics belongs in Gazebo; the harness's
        # job is to prove the choreography and the monitoring plumbing are sound.
        self.gripped = {n: False for n in objects}
        self.hold = {}
        self.grasp_axis_err = {}        # object pose in the LEFT hand's frame, on attach
        # THE BIMANUAL RECORD. At the grasp: where the RIGHT hand sits in the LEFT hand's
        # frame. Two hands holding one rigid object keep that constant, so any change
        # while the object is held means one hand has left it.
        self.span_ref = {}
        self.span_dev = []              # (phase, name, deviation m) after every move
        self.let_go = {}                # name -> (z, tilt deg) when the fingers opened
        # WHAT EVERY COMMANDED TRAJECTORY ACTUALLY SWEEPS, sampled in joint space between
        # its points - the path the controller really takes, not just where it ends.
        self.current = None             # the object the node is working on
        self.slot = None                # the pedestal it is being put on
        self.phase_of = lambda: ''      # reads the node's step label
        self.max_raw = (0.0, '')        # largest joint jump between consecutive points
        self.near = {}                  # (object, what) -> (clearance m, phase)
        self.finger_cmd = 0.065
        self.node = None
        self.published = []
        self.ik_failures = []
        self.yaw_rotations = []

    # ---- outputs the node reads ----
    def odom_msg(self):
        from nav_msgs.msg import Odometry
        m = Odometry()
        m.pose.pose.position.x, m.pose.pose.position.y = self.xy
        h = self.heading
        m.pose.pose.orientation.z = math.sin(h / 2.0)
        m.pose.pose.orientation.w = math.cos(h / 2.0)
        return m

    def scan_msg(self):
        from sensor_msgs.msg import LaserScan
        ranges, amin, ainc = wr.synth_scan(
            self.heading, WALL_A, WALL_HALF, base_xy=tuple(self.xy),
            wall_b_y=WALL_B, wall_b_span=WALL_B_SPAN)
        m = LaserScan()
        m.ranges, m.angle_min, m.angle_increment = ranges, amin, ainc
        m.range_min, m.range_max = 0.12, 8.0
        return m

    def robot_pose_msg(self):
        """Where the base REALLY is - what Gazebo's PosePublisher reports."""
        from geometry_msgs.msg import PoseStamped
        m = PoseStamped()
        m.pose.position.x, m.pose.position.y = self.xy
        m.pose.orientation.z = math.sin(self.heading / 2.0)
        m.pose.orientation.w = math.cos(self.heading / 2.0)
        return m

    def joint_msg(self):
        from sensor_msgs.msg import JointState
        m = JointState()
        m.name = list(self.joints)
        m.position = [self.joints[k] for k in m.name]
        return m

    def pose_msg(self, name):
        from geometry_msgs.msg import PoseStamped
        o = self.objects[name]
        m = PoseStamped()
        m.pose.position.x, m.pose.position.y = o['xy']
        m.pose.position.z = o['z']
        if 'axis3' in o:
            # build a quaternion whose local +z is the object's actual axis, so a
            # flipped object really does read as vertical
            ax, ay, az = o['axis3']
            vx, vy, vz = -ay, ax, 0.0           # (0,0,1) x axis
            sn = math.sqrt(vx * vx + vy * vy + vz * vz)
            q = m.pose.orientation
            if sn < 1e-9:
                q.x = q.y = q.z = 0.0
                q.w = 1.0 if az > 0 else 0.0
                if az <= 0:
                    q.x = 1.0
            else:
                ang = math.acos(max(-1.0, min(1.0, az)))
                sh = math.sin(ang / 2.0)
                q.x, q.y, q.z = vx / sn * sh, vy / sn * sh, vz / sn * sh
                q.w = math.cos(ang / 2.0)
            return m
        # lying body not yet grasped: axis in the horizontal plane, as the SDF encodes it
        a = math.radians(o['axis_deg'] + 90.0)
        r, p, y = math.pi / 2, 0.0, a
        cr, sr = math.cos(r / 2), math.sin(r / 2)
        cp, sp = math.cos(p / 2), math.sin(p / 2)
        cy, sy = math.cos(y / 2), math.sin(y / 2)
        q = m.pose.orientation
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        q.w = cr * cp * cy + sr * sp * sy
        return m

    def contacts_msg(self, stalled):
        from ros_gz_interfaces.msg import Contacts
        m = Contacts()
        m.contacts = [object()] if stalled else []
        return m

    # ---- feed the node ----
    def feed(self):
        for topic, cb in self.node.subs:
            if topic == '/odom':
                cb(self.odom_msg())
            elif topic == '/scan':
                cb(self.scan_msg())
            elif topic == '/joint_states':
                cb(self.joint_msg())
            elif topic.startswith('/model/') and topic.endswith('/pose'):
                name = topic.split('/')[2]
                if name in self.objects:
                    cb(self.pose_msg(name))
                elif name == ROBOT_MODEL:
                    cb(self.robot_pose_msg())
            elif topic.startswith('/contact/'):
                bind = BIND
                cb(self.contacts_msg(self.finger_cmd < bind - 0.0005
                                     and any(self.welded.values()) is not None))
            elif topic == '/imu':
                from sensor_msgs.msg import Imu
                cb(Imu())                       # perfectly level
            elif topic.startswith('/ft/'):
                from geometry_msgs.msg import Wrench
                w = Wrench()
                w.force.z = -2.2                # a held object, shared
                cb(w)


    def _grasp_midpoint(self):
        """Midpoint of the two grasp points, for deciding what is between the hands."""
        try:
            pl, _ = ik.fk([self.joints[j] for j in ac.LEFT_JOINTS], +1)
            pr, _ = ik.fk([self.joints[j] for j in ac.RIGHT_JOINTS], -1)
        except Exception:                                  # noqa: BLE001
            return None
        base = [(pl[k] + pr[k]) / 2.0 for k in range(3)]
        ch_, sh_ = _m.cos(self.heading), _m.sin(self.heading)
        return (self.xy[0] + base[0] * ch_ - base[1] * sh_,
                self.xy[1] + base[0] * sh_ + base[1] * ch_,
                base[2])

    def _capture(self, name):
        """Freeze the object's CURRENT pose in the left hand's frame.

        That is exactly what a DetachableJoint does, and what a friction grip does if it
        holds. THE FIRST VERSION DID NOT DO THIS: it captured on the first arm trajectory
        AFTER the grasp, taking the object's centre as the hands' midpoint - but each arm's
        trajectory is published separately, so at that instant the left arm was already at
        the END of the lift while the right was still at the grasp. The captured offset was
        wrong by half a lift, which is why the harness reported the object 97 mm out of the
        grip, 49 deg off vertical after a correct flip, and 197 mm dragged - and why every
        one of those checks had to be excluded. They are live again.
        """
        import math as _m
        try:
            pl, Rl = ik.fk([self.joints[j] for j in ac.LEFT_JOINTS], +1)
            pr, _Rr = ik.fk([self.joints[j] for j in ac.RIGHT_JOINTS], -1)
        except Exception:                                  # noqa: BLE001
            return
        o = self.objects[name]
        ch_, sh_ = _m.cos(self.heading), _m.sin(self.heading)
        dx, dy = o['xy'][0] - self.xy[0], o['xy'][1] - self.xy[1]
        c_base = [dx * ch_ + dy * sh_, -dx * sh_ + dy * ch_, o['z']]
        if 'axis3' in o:
            wx, wy, wz = o['axis3']
        else:
            a = _m.radians(o['axis_deg'])
            wx, wy, wz = _m.cos(a), _m.sin(a), 0.0
        a_base = [wx * ch_ + wy * sh_, -wx * sh_ + wy * ch_, wz]
        rel_c = [sum(Rl[r][k] * (c_base[r] - pl[r]) for r in range(3)) for k in range(3)]
        rel_a = [sum(Rl[r][k] * a_base[r] for r in range(3)) for k in range(3)]
        self.hold[name] = (rel_c, rel_a)
        self.span_ref[name] = [sum(Rl[r][k] * (pr[r] - pl[r]) for r in range(3))
                               for k in range(3)]
        # THE CHECK THAT WOULD HAVE CAUGHT THE BOX BUG. At the instant of the grasp the
        # line between the two hands must lie along the object's TRUE axis in the WORLD.
        # Recorded once, while the object still lies down: at the coaxial upright pose the
        # azimuth of that line is numerical noise.
        axis = [pl[k] - pr[k] for k in range(3)]
        if name not in self.grasp_axis_err and _m.hypot(axis[0], axis[1]) > 0.030:
            _wx = axis[0] * ch_ - axis[1] * sh_
            _wy = axis[0] * sh_ + axis[1] * ch_
            _got = _m.degrees(_m.atan2(_wy, _wx)) % 180.0
            # NO SILENT FALLBACK - comparing against a missing key with a default of _got
            # would compare the measurement with itself and pass unconditionally.
            assert 'axis_deg' in o, f'no configured axis for {name}'
            _want = o['axis_deg'] % 180.0
            self.grasp_axis_err[name] = (_got, _want,
                                         abs((_got - _want + 90.0) % 180.0 - 90.0))

    def span_deviation(self, name):
        """How far the right hand has moved from where it gripped, in the LEFT hand's frame."""
        if name not in self.span_ref:
            return None
        try:
            pl, Rl = ik.fk([self.joints[j] for j in ac.LEFT_JOINTS], +1)
            pr, _ = ik.fk([self.joints[j] for j in ac.RIGHT_JOINTS], -1)
        except Exception:                                  # noqa: BLE001
            return None
        now = [sum(Rl[r][k] * (pr[r] - pl[r]) for r in range(3)) for k in range(3)]
        return math.dist(now, self.span_ref[name])

    def _carry(self):
        """Move any held object rigidly with the LEFT hand, from the pose captured at the grasp.

        A DetachableJoint is a RIGID attachment to ONE link - the left palm - so the honest
        model is to derive the object's world pose from that hand alone. Averaging the two
        hands instead goes wrong the instant they stop agreeing, which is exactly the fault
        this harness now has to be able to see.
        """
        import math as _m
        for name in list(self.objects):
            held = self.welded.get(name) or self.gripped.get(name)
            if not held:
                continue
            if name not in self.hold:
                self._capture(name)
            if name not in self.hold:
                continue
            try:
                pl, Rl = ik.fk([self.joints[j] for j in ac.LEFT_JOINTS], +1)
            except Exception:                                  # noqa: BLE001
                continue
            o = self.objects[name]
            rel_c, rel_a = self.hold[name]
            bc = [pl[k] + sum(Rl[k][r] * rel_c[r] for r in range(3)) for k in range(3)]
            ba = [sum(Rl[k][r] * rel_a[r] for r in range(3)) for k in range(3)]
            ch_, sh_ = _m.cos(self.heading), _m.sin(self.heading)
            o['xy'] = [self.xy[0] + bc[0] * ch_ - bc[1] * sh_,
                       self.xy[1] + bc[0] * sh_ + bc[1] * ch_]
            o['z'] = bc[2]
            wx = ba[0] * ch_ - ba[1] * sh_
            wy = ba[0] * sh_ + ba[1] * ch_
            o['axis3'] = (wx, wy, ba[2])
            o['axis_deg'] = _m.degrees(_m.atan2(wy, wx)) % 180.0

    def _object_caps_base(self, name):
        """An object's capsules in the base frame, from where the fake says it really is."""
        from pnp_behavior import collision as col
        o = self.objects[name]
        ch_, sh_ = math.cos(self.heading), math.sin(self.heading)
        dx, dy = o['xy'][0] - self.xy[0], o['xy'][1] - self.xy[1]
        c = (dx * ch_ + dy * sh_, -dx * sh_ + dy * ch_, o['z'])
        if 'axis3' in o:
            wx, wy, wz = o['axis3']
        else:
            a = math.radians(o['axis_deg'])
            wx, wy, wz = math.cos(a), math.sin(a), 0.0
        ax = (wx * ch_ + wy * sh_, -wx * sh_ + wy * ch_, wz)
        return col.object_capsules(c, ax, o['L'], o['hw'], o['round'])

    def _fixture_boxes_base(self):
        """Every fixture in the base frame, from where the fake robot really is."""
        from pnp_behavior import collision as col
        ch_, sh_ = math.cos(self.heading), math.sin(self.heading)
        out = {}
        for n, (c, yaw, size) in self.fixtures.items():
            dx, dy = c[0] - self.xy[0], c[1] - self.xy[1]
            out[n] = col.box_obstacle((dx * ch_ + dy * sh_, -dx * sh_ + dy * ch_, c[2]),
                                      yaw - self.heading, size)
        return out

    def _sweep(self, msg):
        """Check the joint-space path of one arm trajectory against the objects and fixtures."""
        from pnp_behavior import collision as col
        names = list(msg.joint_names)
        side = +1 if names[0].startswith('left') else -1
        start = [self.joints.get(j, v) for j, v in zip(
            names, ch.TRAVEL_LEFT if side > 0 else ch.TRAVEL_RIGHT)]
        pts = [start] + [list(pt.positions) for pt in msg.points]
        phase = self.phase_of()
        for a, b in zip(pts, pts[1:]):
            step = max(abs(x - y) for x, y in zip(a, b))
            if step > self.max_raw[0]:
                self.max_raw = (step, phase)
        park = all(abs(x - y) < 1e-6 for x, y in zip(
            pts[-1], ch.TRAVEL_LEFT if side > 0 else ch.TRAVEL_RIGHT))
        fp = self.finger_cmd
        boxes = self._fixture_boxes_base()
        for a, b in zip(pts, pts[1:]):
            for k in range(1, 7):
                q = [x + k / 6.0 * (y - x) for x, y in zip(a, b)]
                caps = col.arm_capsules_fine(q, side, fp)
                # FIXTURES - walls, pedestals - with the whole arm, except the pedestal the
                # object is being put on while the hands work into it (its ARMS then)
                for fx, box in boxes.items():
                    into = fx == self.slot and (phase == '7/8' or (phase == '8/8' and not park))
                    g = col.obstacle_clearance(caps, box, col.ARM_LINKS if into else None)
                    what = 'slot, arm' if into else 'fixture'
                    if g < self.near.get((fx, what), (9.9, ''))[0]:
                        self.near[(fx, what)] = (g, phase)
                for n in self.objects:
                    held = self.welded.get(n) or self.gripped.get(n)
                    if n == self.current:
                        if held or phase != '8/8':
                            continue          # the hands are on it, or going onto it
                        what, links = ('placed, arm', col.ARM_LINKS)
                        if park:
                            what, links = ('placed, whole arm', None)
                    else:
                        what, links = ('other', None)
                    g = min(col.obstacle_clearance(caps, c, links)
                            for c in self._object_caps_base(n))
                    key = (n, what)
                    if g < self.near.get(key, (9.9, ''))[0]:
                        self.near[key] = (g, phase)

    # ---- inputs the node writes ----
    def on_publish(self, topic, msg):
        self.published.append(topic)
        if topic == '/cmd_vel':
            self.heading = math.remainder(
                self.heading + msg.angular.z * 0.02, 2 * math.pi)
            if abs(msg.linear.x) > 1e-9:
                self.xy[0] += msg.linear.x * 0.02 * math.cos(self.heading)
                self.xy[1] += msg.linear.x * 0.02 * math.sin(self.heading)
            # THE BASE TURNS WITH THE OBJECT HELD in this section - to face the shelf - so
            # a held object has to go round with it, not wait for the next arm move
            self._carry()
            self.feed()
        elif topic.endswith('_arm_controller/joint_trajectory'):
            self._sweep(msg)
            self.joints.update(ac.final_positions(msg))
            self._carry()
            self.feed()
        elif topic.endswith('gripper_controller/commands'):
            v = float(msg.data[0])
            bind = BIND
            self.finger_cmd = v
            # THE ONE PHYSICAL EFFECT MODELLED: a finger cannot close past the object.
            held = v if v >= bind else bind
            for j in ac.FINGER_JOINTS:
                self.joints[j] = held
            # CLOSING ON IT IS TAKING HOLD OF IT; OPENING IS LETTING GO. The threshold
            # is the bind width, i.e. the command has been driven past the surface, which
            # is exactly the condition under which the real fingers stall on the object.
            closing = v < bind
            for n in self.objects:
                if closing and not self.gripped[n]:
                    mid = self._grasp_midpoint()
                    o = self.objects[n]
                    if mid is not None and _m.dist(
                            (o['xy'][0], o['xy'][1], o['z']), mid) < 0.06:
                        self.gripped[n] = True
                        if n not in self.hold:
                            self._capture(n)
                elif not closing and self.gripped[n]:
                    self.gripped[n] = False
                    if not self.welded.get(n):
                        # LET GO: record where and how it was left, then it stays there
                        o = self.objects[n]
                        az = o.get('axis3', (1.0, 0.0, 0.0))[2]
                        self.let_go[n] = (o['z'], _m.degrees(_m.acos(min(1.0, abs(az)))),
                                          tuple(o['xy']))
                        self.hold.pop(n, None)
                        self.span_ref.pop(n, None)
            self.feed()
        elif '/attach' in topic:
            n = topic.split('/')[2]
            self.welded[n] = True
            if n not in self.hold:
                self._capture(n)
        elif '/detach' in topic:
            n = topic.split('/')[2]
            self.welded[n] = False
            # the fingers still hold it by friction, so the relative pose is unchanged
            if not self.gripped.get(n):
                self.hold.pop(n, None)
                self.span_ref.pop(n, None)
        elif topic.endswith('/state'):
            pass

    def grasp_state_msg(self, name):
        from std_msgs.msg import String
        m = String()
        # THE REAL PLUGIN PUBLISHES THE STATE OF THE JOINT, not whether pads are
        # touching. Reporting a friction pinch as 'attached' made release_grasp()
        # wait forever for 'detached' as soon as BOTH hands kept hold to the end -
        # which is what a fully bimanual flip does. The weld is what attaches.
        m.data = 'attached' if self.welded[name] else 'detached'
        return m


# ------------------------------------------------------------------ the driver
def run():
    cfg = os.path.join(PKG, 'config', 'targets.yaml')
    params = load_params(cfg)
    params['autostart'] = False          # do NOT start the run thread
    FakeNode._params = params
    fixtures = {f: ([float(v) for v in params[f'{f}.centre']],
                    math.radians(float(params.get(f'{f}.yaw', 0.0))),
                    [float(v) for v in params[f'{f}.size']])
                for f in params.get('fixtures', [])}

    objects = {}
    for n in params.get('order', ['bottle', 'box']):
        objects[n] = {
            'xy': [0.30 * math.cos(math.radians(params[f'{n}.pick_azimuth'])),
                   0.30 * math.sin(math.radians(params[f'{n}.pick_azimuth']))],
            'z': float(params.get(f'{n}.surface_z', 0.0)) + params[f'{n}.cz'],
            'axis_deg': params[f'{n}.axis_yaw'],
            'L': params[f'{n}.length'],
            'hw': params.get(f'{n}.half_width', 0.045),
            'round': str(params.get(f'{n}.round', 'true')).lower() == 'true',
        }
    robot = FakeRobot(objects, fixtures)
    FakeNode.SINK = robot.on_publish
    # the fake starts parked, which is where the URDF loads the real arms
    for j, v in zip(ac.LEFT_JOINTS + ac.RIGHT_JOINTS, ch.TRAVEL_LEFT + ch.TRAVEL_RIGHT):
        robot.joints[j] = v

    from pnp_behavior import task_node as tn                   # noqa: E402
    node = tn.TaskNode()
    robot.node = node
    robot.phase_of = lambda: node.phase

    # the grasp plugin's confirmation comes back on a topic the node subscribes to
    for topic, cb in node.subs:
        if topic.startswith('/grasp/') and topic.endswith('/state'):
            name = topic.split('/')[2]
            node.grasp_state[name] = 'detached'
    robot.feed()

    # record what the arms are actually asked to do
    # sections name this differently: move_arms in pick-place, move in reorient
    move_name = 'move_arms' if hasattr(node, 'move_arms') else 'move'
    real_move = getattr(node, move_name)
    real_yaw = ch.yaw_waypoints

    def reachable7(p, R, side):
        # SEVEN-DOF reachability: some swivel, either shoulder solution. Asking only psi = 0
        # is asking whether the six-DOF elbow-up arm can reach it, which is not the robot.
        return any(ik.solve_branches(p, R, side, math.radians(d), True)
                   for d in range(0, 360, 5))

    def spy_move(pairs, dt):
        for (pl, Rl), (pr, Rr) in pairs:
            for p, R, sd in ((pl, Rl, +1), (pr, Rr, -1)):
                if not reachable7(p, R, sd):
                    robot.ik_failures.append(f'unreachable at {[round(v, 3) for v in p]} '
                                             f'side {sd}')
        out = real_move(pairs, dt)
        for n in list(robot.span_ref):
            d = robot.span_deviation(n)
            if d is not None:
                robot.span_dev.append((node.phase, n, d))
        return out

    def spy_yaw(*args, **kw):
        # the two sections differ: pick-place passes (cx, cy, psi_from, psi_to, ...)
        # and reorient passes (centre_xy, psi_from, psi_to, ...). Read the angles by
        # position relative to the tuple, not by a fixed index - guessing wrong made
        # the harness report an 85 degree rotation that was never commanded.
        if isinstance(args[0], (tuple, list)):
            a, b = args[1], args[2]
        else:
            a, b = args[2], args[3]
        robot.yaw_rotations.append(
            math.degrees(abs(math.remainder(b - a, 2 * math.pi))))
        return real_yaw(*args, **kw)

    setattr(node, move_name, spy_move)
    ch.yaw_waypoints = spy_yaw

    # drive the grasp confirmation: whenever attach/detach is published, echo the state
    base_on_publish = robot.on_publish

    def on_pub(topic, msg):
        base_on_publish(topic, msg)
        if '/attach' in topic or '/detach' in topic:
            name = topic.split('/')[2]
            for t, cb in node.subs:
                if t == f'/grasp/{name}/state':
                    cb(robot.grasp_state_msg(name))
    FakeNode.SINK = on_pub

    results = {}
    def slot_of(name):
        az = math.radians(float(params[f'{name}.place_azimuth']))
        wx, wy = 0.30 * math.cos(az), 0.30 * math.sin(az)
        for f, (c, yaw, size) in fixtures.items():
            dx, dy = wx - c[0], wy - c[1]
            if (abs(math.cos(yaw) * dx + math.sin(yaw) * dy) <= size[0] / 2
                    and abs(-math.sin(yaw) * dx + math.cos(yaw) * dy) <= size[1] / 2
                    and size[2] < 0.5):
                return f
        return None

    try:
        for name in params.get('order', ['bottle', 'box']):
            robot.current = name
            robot.slot = slot_of(name)
            try:
                results[name] = node.cycle(name)
            except Exception as exc:                               # noqa: BLE001
                results[name] = f'{type(exc).__name__}: {exc}'
        # AND THE RETURN HOME, which run() does after the last object. Driving only cycle()
        # left return_home() as code no offline check ever executed - the same gap that let
        # four bugs reach the laptop before this harness existed.
        robot.current = None
        try:
            results['_return_home'] = node.return_home()
        except Exception as exc:                               # noqa: BLE001
            results['_return_home'] = f'{type(exc).__name__}: {exc}'
    finally:
        ch.yaw_waypoints = real_yaw          # a second run must not spy on the spy
    return node, robot, results


# ------------------------------------------------------------------ assertions
def main():
    fails = []
    evaluate(*run(), label='floor to a 0.10 m shelf', fails=fails)
    print()
    print(f'RESULT: {len(fails)} failure(s)')
    return 1 if fails else 0


def evaluate(node, robot, results, label, fails, show_log=True):
    def check(ok, msg):
        print(f'  [{"ok " if ok else "FAIL"}] {msg}')
        if not ok:
            fails.append(f'{label}: {msg}')

    print('=' * 78)
    print(f'HEADLESS HARNESS (section A) - the real task_node cycle, fake robot, {label}')
    print('=' * 78)
    for name, r in results.items():
        if name == '_return_home':
            check(r is True, f'return_home() returned {r!r} - the robot must end the '
                             f'run parked and back at its starting heading')
            continue
        check(r is True, f'{name}: cycle returned {r!r}')
    check(not robot.ik_failures,
          f'no IK failures in any commanded waypoint '
          f'({len(robot.ik_failures)} found: {robot.ik_failures[:2]})')
    check(all(not v for v in robot.welded.values()),
          f'no object left welded at the end ({robot.welded})')
    worst = max(robot.yaw_rotations) if robot.yaw_rotations else 0.0
    check(worst <= 95.0,
          f'largest commanded yaw rotation {worst:.1f} deg '
          f'(all: {[round(v) for v in robot.yaw_rotations]})')
    # THE GRASP FRAME. Asserted in the WORLD, against the object's true axis, because
    # that is the frame the error appears in - every base-frame number looked fine.
    for nm, (got, want, err) in sorted(robot.grasp_axis_err.items()):
        check(err <= 6.0,
              f'{nm}: hands gripped along {got:.1f} deg in the world, object axis is '
              f'{want:.1f} deg, off by {err:.1f} (tol 6)')
    # THE WHOLE CYCLE MUST BE ONE PLAN. A miss means some leg was solved on its own, and a
    # leg solved on its own is free to take the other wrist branch at its first waypoint -
    # identical hand pose, 180 deg of wrist. That is what Gazebo showed as right_j4 off
    # 87.7 deg with the right hand spinning through the bottle it was holding.
    misses = len([ln for ln in node.get_logger().lines if 'PLAN MISS' in ln])
    check(misses == 0,
          'every leg came from the whole-cycle plan (%d legs re-planned on their own)'
          % misses)

    # FULLY BIMANUAL, MEASURED ON THE REAL CYCLE. From the grasp until the fingers open,
    # the right hand must stay exactly where it gripped relative to the left - the two
    # hold one rigid object. Sampled after EVERY move of the real task node.
    for nm in [k for k in results if not k.startswith('_')]:
        devs = [(ph, d) for ph, n, d in robot.span_dev if n == nm]
        phases = sorted({ph for ph, _ in devs})
        worst_dev = max((d for _, d in devs), default=None)
        check(worst_dev is not None and worst_dev < 0.002,
              f'{nm}: both hands stay on it from grasp to release - right hand moved '
              f'{(worst_dev or 0) * 1000:.1f} mm relative to the left over {len(devs)} moves')
        want = {'4/8', '6/8', '7/8'}
        check(want <= set(phases),
              f'{nm}: held two-handed through lift, carry, yaw, lower and settle '
              f'(phases seen: {phases})')
        lg = robot.let_go.get(nm)
        P = FakeNode._params
        rest = float(P.get(f'{nm}.place_surface_z', 0.10)) + float(P.get(f'{nm}.cz', 0.045))
        az = math.radians(float(P[f'{nm}.place_azimuth']))
        slot = (0.30 * math.cos(az), 0.30 * math.sin(az))
        off = math.dist(lg[2], slot) if lg else float('nan')
        check(lg is not None and abs(lg[0] - rest) < 0.005 and lg[1] > 85.0 and off < 0.010,
              f'{nm}: the hands let go only once it lay ON the shelf, in its slot '
              f'(z {lg[0] if lg else float("nan"):.3f} vs {rest:.3f}, tilt '
              f'{lg[1] if lg else float("nan"):.1f} deg from vertical, {off * 1000:.1f} mm '
              f'from the slot)')

    # NO JOINT JUMPS THE LONG WAY ROUND - the 321 deg spin of right_j2b on 2026-09-27.
    step, where = robot.max_raw
    check(step <= math.radians(60.0),
          f'no commanded joint moves more than 60 deg between trajectory points '
          f'(largest {math.degrees(step):.0f} deg, in step {where})')
    # AND NOTHING THE ARMS SWEEP TOUCHES AN OBJECT THEY ARE NOT HANDLING - the box struck on
    # the way out, and the standing bottle knocked over on the way home.
    for (nm, what), (g, ph) in sorted(robot.near.items()):
        check(g >= 0.005, f'{nm} ({what}): the arms stay {g * 1000:+.0f} mm clear at the '
                          f'closest, in step {ph} (want at least 5 mm)')
    check(all(any(n == f and w in ('fixture', 'slot, arm') for n, w in robot.near)
              for f in robot.fixtures),
          f'every fixture was checked along the swept path ({sorted(robot.fixtures)})')
    check(any(w == 'placed, whole arm' for _, w in robot.near),
          'the way home was checked against the object just stood up')
    objs = [k for k in results if not k.startswith('_')]
    check(len(robot.grasp_axis_err) == len(objs),
          f'a grasp frame was recorded for every object '
          f'({sorted(robot.grasp_axis_err)} vs {sorted(objs)})')

    errs = [ln for ln in node.get_logger().lines if ln.startswith('[ERROR')]
    check(not errs, f'no ERROR log lines ({errs[:2]})')

    # EVERY RUNTIME CHECK MUST PASS. The node marks its own failures with ***, so this
    # holds the harness to the same standard the node holds itself to - otherwise a
    # cycle could "succeed" while reporting that the grasp closed on air.
    # NOTHING IS EXCLUDED ANY MORE. FLIP, STAND, DRIFT, YAW, LIFT and SLIP used to be,
    # because the fake captured the held object's pose at the wrong instant (see
    # FakeRobot._capture); with that fixed they are measured like everything else.
    UNMODELLED = ()
    starred = [ln.split(']', 1)[-1].strip()
               for ln in node.get_logger().lines
               if '***' in ln and not any(u in ln for u in UNMODELLED)]
    check(not starred, f'no runtime check reported a failure ({len(starred)}): '
                       f'{starred[:3]}')

    # and the warnings the node emits for degraded operation
    # 'flip did not fully converge' is excluded ONLY because it is a direct cascade from
    # the SLIP STOP above, which is itself excluded because this fake's carry frame has a
    # constant offset the slip monitor correctly detects. The slip monitor returns False
    # from the motion wait, move() propagates that, and the flip reports non-convergence.
    # Nothing else is excluded, and if the fake's carry transform is ever fixed both of
    # these should be removed together.
    warns = [ln for ln in node.get_logger().lines
             if 'WARN' in ln and 'deadband' not in ln]
    check(not warns, f'no WARN lines ({len(warns)}): {[w[-70:] for w in warns[:2]]}')

    if show_log:
        print()
        print('  --- what the node logged ---')
        for ln in node.get_logger().lines:
            print('   ', ln[7:] if ln.startswith('[info]') else ln)
    print()


if __name__ == '__main__':
    sys.exit(main())
