"""Headless harness (section D): the REAL task_node, a fake robot that DRIVES, and two fake cameras.

The section's point is getting there, so the fake has what that needs:

  * a BASE that integrates the commanded velocity over simulated time - and ODOMETRY that is
    wrong about it the way wheels are: 3 % on turns, 2 % on distance, and a sideways creep
    while turning in place. Anything the node steers by odometry alone lands off target;
  * a LiDAR SCAN of the room's four walls from where the base REALLY is, read from the world
    file, so the room localisation has to find the pose from the walls;
  * the MAST camera and the WORKSPACE camera, each projecting every object's true silhouette
    through the node's own camera model, hiding every outline point whose sight line the
    robot's own arms (or another object) block, refusing blobs cut off by the frame or - on the
    mast camera - by the arm band at its bottom, and fitting the rest with vision.fit();
  * joint states that echo commands, a finger that stalls on an object, a grasp that carries it.

and it records what matters about driving: how close the base ever came to every object and
wall, whether it ever moved with the arms out, and where it truly parked.

It is not a physics simulation and does not pretend to be. It answers one question: does the
code do what it thinks it does, the whole run from the first scan to the final one? Anything
about contact dynamics or wheel slip in earnest still needs Gazebo.

Run: python3 test/harness.py
"""
import json
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


ROBOT = [None]          # the fake robot, moved on by every simulated sleep


def _fake_sleep(s=0.0, *_a, **_k):
    dt = max(float(s), 1e-4)
    SIM_T[0] += dt
    if ROBOT[0] is not None:
        ROBOT[0].advance(dt)


_time.sleep = _fake_sleep

from navigate_behavior import arm_commander as ac                   # noqa: E402
from navigate_behavior import choreography as ch                    # noqa: E402
from navigate_behavior import ik                                    # noqa: E402
from navigate_behavior import navigation as nav                     # noqa: E402
from navigate_behavior import room_ref as rr                        # noqa: E402


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
WORLD_SDF = os.path.join(os.path.dirname(PKG), 'navigate_gazebo', 'worlds', 'navigate.sdf')


def room_from_world():
    """The room's wall FACES, read from the WORLD FILE - not from the config the node reads, so
    the node's localisation is checked against the world, not against itself (the lesson of
    the wall constants that held the wall centres instead of their faces)."""
    import re as _re
    sdf = open(WORLD_SDF).read()
    faces = {}
    for wall in ('wall_n', 'wall_s', 'wall_e', 'wall_w'):
        m = _re.search(r'<collision name="%s"><pose>([^<]+)</pose>\s*<geometry><box><size>'
                       r'([^<]+)</size>' % wall, sdf)
        assert m, f'{wall} not found in navigate.sdf'
        (cx, cy, *_), (sx, sy, _sz) = ([float(v) for v in m.group(1).split()],
                                       [float(v) for v in m.group(2).split()])
        faces[wall] = (cx, cy, sx, sy)
    return rr.Room(faces['wall_w'][0] + faces['wall_w'][2] / 2.0,
                   faces['wall_e'][0] - faces['wall_e'][2] / 2.0,
                   faces['wall_s'][1] + faces['wall_s'][3] / 2.0,
                   faces['wall_n'][1] - faces['wall_n'][3] / 2.0)


ROOM = room_from_world()
CAMERAS = {}           # 'near' / 'far' -> vision.CameraModel, from the perception config
FAR_V_MAX = [479]
FAR_MIN_AREA = [120]
NEAR_MIN_AREA = [250]

# HOW THE FAKE'S ODOMETRY IS WRONG. Turning on two wheels and two casters slips; the real run
# measured 29 deg of heading error over a run of turning alone. These are gentler than that and
# still far past what the park can absorb, so a node that trusted odometry would miss.
ODOM_TURN_ERR = 0.03       # odometry reports 3 % more turn than happened
ODOM_DIST_ERR = -0.02      # and 2 % less distance
TURN_CREEP = 0.004         # m of sideways creep per radian turned in place


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
        # THE BASE: the last commanded velocity, held until the next (as DiffDrive does), and
        # what odometry believes - integrated with the errors above
        self.v = self.w = 0.0
        self.odom_xy = [0.0, 0.0]
        self.odom_h = 0.0
        self.path = []                  # (x, y) every step the base moved
        self.base_near = {}             # object -> (base edge gap m, phase), the closest
        self.wall_near = (9.9, '')      # parked arms to the nearest wall, the closest
        self.moved_unparked = []        # phases in which the base moved with the arms out
        self._scan_cache = None
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
        self.phase_of = lambda: ''      # reads the node's step label
        self.max_raw = (0.0, '')        # largest joint jump between consecutive points
        self.near = {}                  # (object, what) -> (clearance m, phase)
        self.finger_cmd = 0.065
        self.node = None
        self.published = []
        self.ik_failures = []
        self.yaw_rotations = []
        self.occluded = 0               # outline points hidden in a detection the node got
        self.sensing = None             # the object the node is measuring right now

    # ---- outputs the node reads ----
    def odom_msg(self):
        """What the WHEELS say - not where the base is."""
        from nav_msgs.msg import Odometry
        m = Odometry()
        m.pose.pose.position.x, m.pose.pose.position.y = self.odom_xy
        h = self.odom_h
        m.pose.pose.orientation.z = math.sin(h / 2.0)
        m.pose.pose.orientation.w = math.cos(h / 2.0)
        return m

    def scan_msg(self):
        """The room's walls from where the base REALLY is. A new message only when it moved."""
        from sensor_msgs.msg import LaserScan
        key = (round(self.xy[0], 7), round(self.xy[1], 7), round(self.heading, 8))
        if self._scan_cache is None or self._scan_cache[0] != key:
            ranges, amin, ainc = rr.synth_scan(ROOM, (self.xy[0], self.xy[1], self.heading))
            m = LaserScan()
            m.ranges, m.angle_min, m.angle_increment = ranges, amin, ainc
            m.range_min, m.range_max = 0.12, 8.0
            self._scan_cache = (key, m)
        return self._scan_cache[1]

    def arms_parked(self):
        return all(abs(self.joints.get(j, v) - v) < 1e-6 for j, v in zip(
            ac.LEFT_JOINTS + ac.RIGHT_JOINTS, ch.TRAVEL_LEFT + ch.TRAVEL_RIGHT))

    def advance(self, dt):
        """Move the base on by ``dt`` simulated seconds at the held velocity."""
        if abs(self.v) < 1e-12 and abs(self.w) < 1e-12:
            return
        if not self.arms_parked():
            self.moved_unparked.append(self.phase_of())
        steps = max(1, int(dt / 0.01 + 0.5))
        h_dt = dt / steps
        for _ in range(steps):
            dth = self.w * h_dt
            ds = self.v * h_dt
            mid = self.heading + dth / 2.0
            self.xy[0] += ds * math.cos(mid)
            self.xy[1] += ds * math.sin(mid)
            if abs(ds) < 1e-9:           # turning in place creeps sideways
                self.xy[0] += TURN_CREEP * abs(dth) * -math.sin(self.heading)
                self.xy[1] += TURN_CREEP * abs(dth) * math.cos(self.heading)
            self.heading = math.remainder(self.heading + dth, 2 * math.pi)
            odth = dth * (1.0 + ODOM_TURN_ERR)
            ods = ds * (1.0 + ODOM_DIST_ERR)
            omid = self.odom_h + odth / 2.0
            self.odom_xy[0] += ods * math.cos(omid)
            self.odom_xy[1] += ods * math.sin(omid)
            self.odom_h = math.remainder(self.odom_h + odth, 2 * math.pi)
        self.path.append(tuple(self.xy))
        self._track_clearance()
        self.feed()

    def footprints(self):
        """Every object's footprint on the floor, from where the fake says it truly is."""
        out = {}
        for n, o in self.objects.items():
            if self.welded.get(n) or self.gripped.get(n):
                continue
            if 'axis3' in o and abs(o['axis3'][2]) > 0.7:
                out[n] = nav.footprint(o['xy'][0], o['xy'][1], 0.0, True, o['L'], o['hw'])
            else:
                out[n] = nav.footprint(o['xy'][0], o['xy'][1], math.radians(o['axis_deg']),
                                       False, o['L'], o['hw'])
        return out

    def _track_clearance(self):
        ph = self.phase_of()
        for n, (a, b, r) in self.footprints().items():
            g = nav.seg_point(self.xy, a, b) - r - nav.BASE_RADIUS
            if g < self.base_near.get(n, (9.9, ''))[0]:
                self.base_near[n] = (g, ph)
        g = ROOM.wall_distance(self.xy) - nav.ARM_RADIUS
        if g < self.wall_near[0]:
            self.wall_near = (g, ph)

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

    def targets_msg(self, camera='near'):
        """One frame of one camera: every object's silhouette, arms and objects as occluders.

        As the real pipeline sees it: a blob touching the frame edge - or, on the mast camera,
        reaching into the arm band below FAR_V_MAX - is CLIPPED; one smaller than the camera's
        minimum area is not found. A point hidden behind an arm or another object is simply
        missing from the blob, as it would be in the image; if such a blob is still handed to
        the node as whole, that is counted in ``occluded`` - it is a measurement of the wrong
        silhouette, and the harness fails on it.
        """
        from std_msgs.msg import String
        from navigate_behavior import collision as col
        from navigate_behavior import vision as vz
        cam = CAMERAS[camera]
        v_max = FAR_V_MAX[0] if camera == 'far' else cam.height - 1
        min_area = FAR_MIN_AREA[0] if camera == 'far' else NEAR_MIN_AREA[0]
        caps = []
        for joints, sd in ((ac.LEFT_JOINTS, +1), (ac.RIGHT_JOINTS, -1)):
            if all(j in self.joints for j in joints):
                caps += col.arm_capsules_fine([self.joints[j] for j in joints], sd,
                                              self.finger_cmd)
        others = {n: self._object_caps_base(n) for n in self.objects}
        ch_, sh_ = math.cos(self.heading), math.sin(self.heading)
        out = {}
        for n, o in self.objects.items():
            dx, dy = o['xy'][0] - self.xy[0], o['xy'][1] - self.xy[1]
            c = (dx * ch_ + dy * sh_, -dx * sh_ + dy * ch_, o['z'])
            if 'axis3' in o:
                wx, wy, wz = o['axis3']
            else:
                a = math.radians(o['axis_deg'])
                wx, wy, wz = math.cos(a), math.sin(a), 0.0
            bx, by = wx * ch_ + wy * sh_, -wx * sh_ + wy * ch_
            upright = abs(wz) > 0.7
            ang = 0.0 if upright else math.atan2(by, bx)
            pts = vz.outline_points(c, ang, upright, o['L'], o['hw'], o['round'])
            px, clipped, hidden = [], False, 0
            for p in pts:
                q = cam.to_pixel(p)
                if q is None or not cam.in_frame(q, 0.0):
                    clipped = True
                    continue
                if not cam.in_frame(q, 3.0) or q[1] > v_max - 3:
                    clipped = True
                # BEHIND ANOTHER OBJECT: the edge of what is hidden is that object's outline, so
                # the two blobs meet in the image - which detect() refuses as TOUCHING
                if any(col.segment_solid_distance(cam.position, p, cc) < 0.0
                       for m_, cs in others.items() if m_ != n for cc in cs):
                    clipped = True
                    continue
                # BEHIND AN ARM: an arm is no object's colour, so nothing in the image says so
                if caps and min(col.segment_distance(cam.position, p, a0, a1) - r
                                for _n, a0, a1, r in caps) < 0.0:
                    hidden += 1
                    continue
                px.append((round(q[0]), round(q[1])))
            if len(px) < 6:
                out[n] = {'found': False}
                continue
            hull = vz.convex_hull(px)
            area = 0.5 * abs(sum(hull[i][0] * hull[(i + 1) % len(hull)][1]
                                 - hull[(i + 1) % len(hull)][0] * hull[i][1]
                                 for i in range(len(hull))))
            if area < min_area:
                out[n] = {'found': False}
                continue
            det = vz.fit(px, o['L'], o['hw'], o['round'], cam, 0.0)
            det['clipped'] = clipped
            if hidden and not clipped:
                self.occluded += hidden
            out[n] = det
        m = String()
        m.data = json.dumps({'stamp': SIM_T[0], 'objects': out})
        return m

    def publish_targets(self, camera='near'):
        msg = self.targets_msg(camera)
        topic = '/targets' if camera == 'near' else '/targets_far'
        for t, cb in self.node.subs:
            if t == topic:
                cb(msg)

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
                elif name == 'navigate_bimanual':
                    cb(self.robot_pose_msg())
            elif topic.startswith('/contact/'):
                bind = 0.045 + 0.012
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
        from navigate_behavior import collision as col
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
        from navigate_behavior import collision as col
        ch_, sh_ = math.cos(self.heading), math.sin(self.heading)
        out = {}
        for n, (c, yaw, size) in self.fixtures.items():
            dx, dy = c[0] - self.xy[0], c[1] - self.xy[1]
            out[n] = col.box_obstacle((dx * ch_ + dy * sh_, -dx * sh_ + dy * ch_, c[2]),
                                      yaw - self.heading, size)
        return out

    def _sweep(self, msg):
        """Check the joint-space path of one arm trajectory against the objects and fixtures."""
        from navigate_behavior import collision as col
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
                # FIXTURES - walls, tables, shelves - with the whole arm, always
                for fx, box in boxes.items():
                    g = col.obstacle_clearance(caps, box)
                    if g < self.near.get((fx, 'fixture'), (9.9, ''))[0]:
                        self.near[(fx, 'fixture')] = (g, phase)
                for n in self.objects:
                    held = self.welded.get(n) or self.gripped.get(n)
                    if n == self.current and phase in ('2/9', '3/9', '4/9', '5/9', '6/9',
                                                       '7/9', '8/9'):
                        continue              # the hands are on it, or going onto it
                    if held:
                        continue
                    if n == self.current and phase == '9/9':
                        what, links = ('placed, arm', col.ARM_LINKS)
                        if park:
                            what, links = ('placed, whole arm', None)
                    elif n == self.current:
                        # LOOKING AT IT - the arms to and from the look pose, at the staging
                        # and park poses: the whole arm must stay off it
                        what, links = ('looked at', None)
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
            # HELD until the next command, as DiffDrive holds it; the clock moves the base
            self.v, self.w = float(msg.linear.x), float(msg.angular.z)
            self.feed()
        elif topic.endswith('_arm_controller/joint_trajectory'):
            self._sweep(msg)
            self.joints.update(ac.final_positions(msg))
            self._carry()
            self.feed()
        elif topic.endswith('gripper_controller/commands'):
            v = float(msg.data[0])
            bind = 0.045 + 0.012
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
                        self.let_go[n] = (o['z'], _m.degrees(_m.acos(min(1.0, abs(az)))))
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
def world_objects():
    """The objects as the WORLD places them: name -> [x, y, z, roll, pitch, yaw]."""
    import re as _re
    sdf = open(WORLD_SDF).read()
    out = {}
    for m in _re.finditer(r'<uri>model://(\w+)</uri>\s*<name>(\w+)</name>\s*<pose>([^<]+)</pose>',
                          sdf):
        out[m.group(2)] = [float(v) for v in m.group(3).split()]
    return out


def load_block(path, block):
    """load_params() for another node's block of the same file."""
    import tempfile
    txt = open(path).read()
    txt = txt[txt.index(block + ':'):].replace(block + ':', 'task_node:', 1)
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as fh:
        fh.write(txt)
    return load_params(fh.name)


def run(layout=None):
    """One whole run, first scan to last. ``layout`` overrides the world file's object poses:
    ``{name: (x, y, z, roll, pitch, yaw)}`` in the SDF's own convention."""
    cfg = os.path.join(PKG, 'config', 'task.yaml')
    params = load_params(cfg)
    params['autostart'] = False          # do NOT start the run thread
    FakeNode._params = params
    from navigate_behavior import vision as vz
    pc = load_block(cfg, 'perception_node')
    CAMERAS['near'] = vz.CameraModel.from_fov(
        (pc['cam_x'], pc['cam_y'], pc['cam_z']), pc['cam_pitch'], pc['horizontal_fov'],
        int(pc['image_width']), int(pc['image_height']))
    CAMERAS['far'] = vz.CameraModel.from_fov(
        (pc['far_cam_x'], pc['far_cam_y'], pc['far_cam_z']), pc['far_cam_pitch'],
        pc['far_horizontal_fov'], int(pc['far_image_width']), int(pc['far_image_height']))
    FAR_V_MAX[0] = int(pc['far_v_max'])
    FAR_MIN_AREA[0] = int(pc['far_min_area'])
    NEAR_MIN_AREA[0] = int(pc['min_area'])
    fixtures = {f: ([float(v) for v in params[f'{f}.centre']],
                    math.radians(float(params.get(f'{f}.yaw', 0.0))),
                    [float(v) for v in params[f'{f}.size']])
                for f in params.get('fixtures', [])}

    objects = {}
    placed_in_world = layout or world_objects()
    for n in params.get('order', ['bottle', 'box']):
        x, y, z, roll, _p, yaw = placed_in_world[n]
        standing = abs(roll) < 0.1
        objects[n] = {
            'xy': [x, y],
            'z': z,
            # a lying body's axis is (spawn yaw - 90 deg); a standing one's is vertical
            'axis_deg': math.degrees(yaw) - 90.0,
            'L': params[f'{n}.length'],
            'hw': params.get(f'{n}.half_width', 0.045),
            'round': str(params.get(f'{n}.round', 'true')).lower() == 'true',
        }
        if standing:
            objects[n]['axis3'] = (0.0, 0.0, 1.0)
    start = {n: (list(o['xy']), o['z'], o.get('axis3')) for n, o in objects.items()}
    robot = FakeRobot(objects, fixtures)
    robot.start = start
    FakeNode.SINK = robot.on_publish
    # the fake starts parked, which is where the URDF loads the real arms
    for j, v in zip(ac.LEFT_JOINTS + ac.RIGHT_JOINTS, ch.TRAVEL_LEFT + ch.TRAVEL_RIGHT):
        robot.joints[j] = v

    from navigate_behavior import task_node as tn                   # noqa: E402
    node = tn.TaskNode()
    robot.node = node
    robot.phase_of = lambda: node.phase

    # the grasp plugin's confirmation comes back on a topic the node subscribes to
    for topic, cb in node.subs:
        if topic.startswith('/grasp/') and topic.endswith('/state'):
            name = topic.split('/')[2]
            node.grasp_state[name] = 'detached'
    robot.feed()
    ROBOT[0] = robot

    real_move = node.move
    real_yaw = ch.yaw_waypoints

    def reachable7(p, R, side):
        # SEVEN-DOF reachability: some swivel, either shoulder solution.
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
        a, b = args[1], args[2]
        robot.yaw_rotations.append(
            math.degrees(abs(math.remainder(b - a, 2 * math.pi))))
        return real_yaw(*args, **kw)

    node.move = spy_move
    ch.yaw_waypoints = spy_yaw

    # THE CAMERAS RUN WHILE THE NODE WAITS: frames arrive as it senses, at 15 Hz simulated,
    # taken with the arms wherever the node has put them and the base wherever it is
    real_sense, real_sense_all = node.sense, node.sense_all
    robot.frames = {'near': 0, 'far': 0}

    def frames(camera):
        for _ in range(node.sense_frames + 1):
            _fake_sleep(1.0 / 15.0)
            robot.publish_targets(camera)
            robot.frames[camera] += 1

    def spy_sense(name, after, camera='near'):
        robot.sensing = name
        frames(camera)
        robot.sensing = None
        return real_sense(name, after, camera)

    def spy_sense_all(after, camera='near'):
        frames(camera)
        return real_sense_all(after, camera)
    node.sense, node.sense_all = spy_sense, spy_sense_all

    real_run_cycle = node.run_cycle

    def spy_run_cycle(n):
        robot.current = n
        try:
            return real_run_cycle(n)
        finally:
            robot.current = None
    node.run_cycle = spy_run_cycle

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
    try:
        robot.current = None
        node.travel_pose()
        node.before_cycles()
        results.update(node.cycles())          # what run() does, scan to scan
        try:
            results['_return_home'] = node.return_home()
        except Exception as exc:                               # noqa: BLE001
            results['_return_home'] = f'{type(exc).__name__}: {exc}'
    finally:
        ch.yaw_waypoints = real_yaw          # a second run must not spy on the spy
        ROBOT[0] = None
    return node, robot, results


# ------------------------------------------------------------------ assertions
# THE WORLD FILE, and two more rooms with the objects elsewhere - the node has no positions in
# its config, so it must do the same job wherever they are. The second puts the can close to
# a wall, so the side nearer the robot is too near it and the approach must go round; the third
# puts the box behind the standing bottle as seen from the scan point, so the route must pass
# it, with both lying axes along the room's diagonals.
LAYOUTS = [
    ('the world file: the bottle standing, the can and the box lying', None, ()),
    ('the can near the north wall, the box on the west side', {
        'bottle': (-1.30, -1.20, 0.120, 0.0, 0.0, 0.0),
        'can': (0.60, 1.40, 0.045, 1.5708, 0.0, math.radians(170.0 + 90.0)),
        'box': (-1.50, 0.40, 0.045, 1.5708, 0.0, math.radians(75.0 + 90.0))}, ()),
    ('the box behind the bottle from the scan point', {
        'bottle': (0.00, 1.20, 0.120, 0.0, 0.0, 0.0),
        'can': (1.35, -0.95, 0.045, 1.5708, 0.0, math.radians(45.0 + 90.0)),
        'box': (0.10, 1.95, 0.045, 1.5708, 0.0, math.radians(135.0 + 90.0))},
     # hidden from the scan point, so it must be found by scanning again from the side
     ('scanning again from the side',)),
]


def main():
    fails = []
    which = [a for a in sys.argv[1:] if a.isdigit()]
    for k, (label, layout, expect) in enumerate(LAYOUTS):
        if which and str(k + 1) not in which:
            continue
        evaluate(*run(layout), label=label, fails=fails, show_log=(k == 0 or '-v' in sys.argv),
                 expect=expect)
    print()
    print(f'RESULT: {len(fails)} failure(s)')
    for f_ in fails:
        print('  -', f_)
    return 1 if fails else 0


def evaluate(node, robot, results, label, fails, show_log=True, expect=()):
    def check(ok, msg):
        print(f'  [{"ok " if ok else "FAIL"}] {msg}')
        if not ok:
            fails.append(f'{label}: {msg}')

    lines = node.get_logger().lines
    print('=' * 78)
    print(f'HEADLESS HARNESS (section D) - the real task_node, a driving base, two cameras:')
    print(f'  {label}')
    print('=' * 78)
    standing0 = sorted(n for n, (_xy, _z, ax) in robot.start.items() if ax is not None)
    lying0 = sorted(n for n in robot.start if n not in standing0)

    # ---- THE JOB ----
    for e in expect:
        check(any(e in ln for ln in lines), f'this room makes it: "{e}"')
    first = next((ln for ln in lines if 'SCAN RESULT (round 1)' in ln), '')
    check(all(f'{n} standing' in first for n in standing0)
          and all(f'{n} lying' in first for n in lying0) and 'NOT FOUND' not in first,
          f'the first scan found every object, in the right state ({first.split(": ", 1)[-1]})')
    check(node.all_standing and any('ALL STANDING: every object seen standing' in ln
                                    for ln in lines),
          'a later scan saw all three standing, and the node called the job done')
    worked = sorted(k for k in results if not k.startswith('_'))
    check(worked == lying0 and all(results[k] is True for k in worked),
          f'it stood up exactly the lying objects, each once ({results})')
    for n in standing0:
        (x0, y0), z0, _ax = robot.start[n]
        o = robot.objects[n]
        check(math.dist(o['xy'], (x0, y0)) < 1e-9 and abs(o['z'] - z0) < 1e-9,
              f'{n}: seen STANDING and left alone - never grasped, never moved')
    for n in lying0:
        o = robot.objects[n]
        (x0, y0), _z0, _ax = robot.start[n]
        up = (float(FakeNode._params.get(f'{n}.surface_z', 0.0))
              + float(FakeNode._params.get(f'{n}.upright_cz', 0.0)))
        tilt = math.degrees(math.acos(min(1.0, abs(o.get('axis3', (1, 0, 0))[2]))))
        moved = math.dist(o['xy'], (x0, y0))
        check(tilt < 3.0 and abs(o['z'] - up) < 0.005 and moved <= 0.030,
              f'{n}: standing where it lay - tilt {tilt:.1f} deg, z {o["z"]:.3f} vs {up:.3f}, '
              f'{moved * 1000:.0f} mm from where it lay')
    check(results.get('_return_home') is True
          and math.dist(robot.xy, node.scan_point) <= 0.02
          and abs(math.degrees(robot.heading)) <= 1.0,
          f'it ended back at the start: {math.dist(robot.xy, node.scan_point) * 1000:.0f} mm and '
          f'{math.degrees(robot.heading):+.1f} deg from it')

    # ---- DRIVING ----
    for n in sorted(robot.objects):
        g, ph = robot.base_near.get(n, (9.9, ''))
        check(g >= 0.030, f'{n}: the base never came within 30 mm of it - closest '
                          f'{g * 1000:+.0f} mm (base edge to surface), in step {ph or "-"}')
    g, ph = robot.wall_near
    check(g >= 0.05, f'the parked arms never came within 50 mm of a wall - closest '
                     f'{g * 1000:+.0f} mm, in step {ph or "-"}')
    check(not robot.moved_unparked,
          f'the base never moved with the arms out of the parked pose '
          f'({sorted(set(robot.moved_unparked))})')
    for n in lying0:
        e = node.park_report.get(n)
        check(e is not None and abs(e[0]) <= 0.010 and abs(e[1]) <= 0.010 and abs(e[2]) <= 5.0,
              f'{n}: TRULY parked inside the stand-up\'s geometry - '
              + (f'{e[0] * 1000:+.1f} mm reach, {e[1] * 1000:+.1f} mm off centre, axis '
                 f'{e[2]:+.1f} deg' if e else 'no park report'))
    odo = math.dist(robot.odom_xy, robot.xy)
    check(odo > 0.02, f'odometry was wrong by {odo * 1000:.0f} mm by the end, so every arrival '
                      f'above came from the wall fix, not from the wheels')
    travelled = sum(math.dist(a, b) for a, b in zip(robot.path, robot.path[1:]))
    check(travelled > 3.0, f'it actually drove: {travelled:.1f} m in all')

    # ---- SEEING ----
    check(robot.occluded == 0,
          f'no detection handed to the node had any of its outline hidden behind an arm or '
          f'another object ({robot.occluded} points)')
    check(robot.frames['far'] > 0 and robot.frames['near'] > 0,
          f'both cameras were used (mast {robot.frames["far"]} frames, workspace '
          f'{robot.frames["near"]})')
    cams = [ln for ln in lines if 'CAMERA [' in ln or 'BLIND LEG [' in ln]
    for n in lying0:
        mine = [ln for ln in cams if f'[{n}]' in ln]
        check(any('MAST camera sees it whole' in ln for ln in mine)
              and any('NEITHER camera' in ln for ln in mine)
              and any('WORKSPACE camera takes over' in ln for ln in mine),
              f'{n}: the approach went mast camera -> blind leg -> workspace camera')
    pc = [ln for ln in lines if 'PERCEPTION CHECK' in ln]
    check(len(pc) >= 6 and all('accurate' in ln for ln in pc),
          f'every perception check was accurate ({len(pc)} checks, '
          f'{sum("(mast)" in ln for ln in pc)} of them the mast camera)')
    lc = [ln for ln in lines if 'LOCALISATION CHECK' in ln]
    check(lc and all('accurate' in ln for ln in lc),
          f'every localisation check was accurate ({len(lc)})')

    # ---- THE STAND-UP, as the other sections hold it ----
    check(not robot.ik_failures,
          f'no IK failures in any commanded waypoint '
          f'({len(robot.ik_failures)} found: {robot.ik_failures[:2]})')
    check(all(not v for v in robot.welded.values()),
          f'no object left welded at the end ({robot.welded})')
    worst = max(robot.yaw_rotations) if robot.yaw_rotations else 0.0
    check(worst <= 95.0,
          f'largest commanded yaw rotation {worst:.1f} deg '
          f'(all: {[round(v) for v in robot.yaw_rotations]})')
    for nm, (got, want, err) in sorted(robot.grasp_axis_err.items()):
        check(err <= 6.0,
              f'{nm}: hands gripped along {got:.1f} deg in the world, object axis is '
              f'{want:.1f} deg, off by {err:.1f} (tol 6)')
    check(sorted(robot.grasp_axis_err) == lying0,
          f'a grasp frame was recorded for every LYING object and none other '
          f'({sorted(robot.grasp_axis_err)})')
    misses = len([ln for ln in lines if 'PLAN MISS' in ln])
    check(misses == 0,
          'every leg came from the whole-cycle plan (%d legs re-planned on their own)'
          % misses)
    bad = [ln for ln in lines
           if 'lower hand opens and withdraws' in ln
           or 'one hand alone completes' in ln
           or 'ONE-HAND TORQUE' in ln or 'upper hand lowers it' in ln]
    check(not bad, 'no single-handed phase is ever logged (%s)' % (bad[:1] or 'none'))
    for nm in lying0:
        devs = [(ph, d) for ph, n, d in robot.span_dev if n == nm]
        phases = sorted({ph for ph, _ in devs})
        worst_dev = max((d for _, d in devs), default=None)
        check(worst_dev is not None and worst_dev < 0.002,
              f'{nm}: both hands stay on it from grasp to release - right hand moved '
              f'{(worst_dev or 0) * 1000:.1f} mm relative to the left over {len(devs)} moves')
        check({'4/9', '5/9', '6/9', '7/9', '8/9'} <= set(phases),
              f'{nm}: held two-handed through lift, yaw, flip, lower and settle')
        lg = robot.let_go.get(nm)
        up = (float(FakeNode._params.get(f'{nm}.surface_z', 0.0))
              + float(FakeNode._params.get(f'{nm}.upright_cz', 0.0)))
        check(lg is not None and abs(lg[0] - up) < 0.005 and lg[1] < 3.0,
              f'{nm}: the hands let go only once it stood upright on its surface')
    step, where = robot.max_raw
    check(step <= math.radians(60.0),
          f'no commanded joint moves more than 60 deg between trajectory points '
          f'(largest {math.degrees(step):.0f} deg, in step {where})')
    for (nm, what), (g, ph) in sorted(robot.near.items()):
        check(g >= 0.005, f'{nm} ({what}): the arms stay {g * 1000:+.0f} mm clear at the '
                          f'closest, in step {ph} (want at least 5 mm)')
    check(all(any(n == f and w == 'fixture' for n, w in robot.near)
              for f in robot.fixtures),
          f'every fixture was checked along the swept path ({sorted(robot.fixtures)})')
    check(any(w == 'placed, whole arm' for _, w in robot.near),
          'the way home was checked against the object just stood up')

    errs = [ln for ln in lines if ln.startswith('[ERROR')]
    check(not errs, f'no ERROR log lines ({errs[:2]})')
    starred = [ln.split(']', 1)[-1].strip() for ln in lines if '***' in ln]
    check(not starred, f'no runtime check reported a failure ({len(starred)}): '
                       f'{starred[:3]}')
    warns = [ln for ln in lines if 'WARN' in ln and 'deadband' not in ln]
    check(not warns, f'no WARN lines ({len(warns)}): {[w[-70:] for w in warns[:2]]}')

    if show_log:
        print()
        print('  --- what the node logged ---')
        for ln in lines:
            print('   ', ln[7:] if ln.startswith('[info]') else ln)
    print()


if __name__ == '__main__':
    sys.exit(main())
