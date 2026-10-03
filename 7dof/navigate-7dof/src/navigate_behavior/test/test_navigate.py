"""Offline validation for section D, navigate. No ROS, no Gazebo.

    python3 test/test_navigate.py

WHAT HAS TO BE TRUE FOR THE RUN TO WORK, AND IS CHECKED HERE
------------------------------------------------------------
* The room localisation finds the pose from the walls anywhere in the room, from a prior
  that is centimetres and degrees out, and refuses a scan it cannot place.
* Each camera sees what the approach assumes it sees, and nothing else: the mast camera an
  object whole only beyond 1.0 m with the arms in the look pose (and the arms nowhere above
  its cut-off row), the workspace camera a lying object only inside 0.45 m. The approach's
  staging, look and blind distances are derived from that and checked against it.
* The base's footprint is what the route planner thinks it is, the park pose puts an object
  exactly where the shared stand-up was verified, routes stay clear, and the drive law
  arrives without overshooting.
* The world is one the run can finish: the objects metres apart, each reachable, each with
  room to be stood up; the stand-up plans at every park pose with this world round it.
* The perception is perceive's, checked on what the camera REALLY sees - solid silhouettes,
  and where OpenCV is available a rendered image - and the robot is the other sections' robot:
  the shared files are byte for byte theirs.
And the headless harness runs the real node, first scan to last, in three different rooms.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from navigate_behavior import choreography as ch
from navigate_behavior import collision as col
from navigate_behavior import ik
from navigate_behavior import motion as mo
from navigate_behavior import navigation as nav
from navigate_behavior import room_ref as rr
from navigate_behavior import vision as vz

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SRC = os.path.dirname(PKG)
ROOT = os.path.dirname(SRC)
CFG = os.path.join(PKG, 'config', 'task.yaml')
BEH = os.path.join(PKG, 'navigate_behavior')
WORLD = os.path.join(SRC, 'navigate_gazebo', 'worlds', 'navigate.sdf')
URDF = os.path.join(SRC, 'navigate_description', 'urdf')
LAUNCH = os.path.join(SRC, 'navigate_gazebo', 'launch', 'navigate.launch.py')

REACH = 0.30
OPEN_POS = 0.065
MIN_CLEARANCE = 0.004
OBJECTS = {
    'bottle': dict(length=0.240, half_width=0.045, round=True, cz=0.045, upright_cz=0.120,
                   close=0.048, mass=0.45, rgb=(0.90, 0.35, 0.05)),
    'can': dict(length=0.200, half_width=0.045, round=True, cz=0.045, upright_cz=0.100,
                close=0.048, mass=0.35, rgb=(0.10, 0.35, 0.85)),
    'box': dict(length=0.220, half_width=0.045, round=False, cz=0.045, upright_cz=0.110,
                close=0.048, mass=0.50, rgb=(0.20, 0.70, 0.25)),
}
CAMERA = vz.CameraModel.from_fov((0.05, 0.0, 0.62), 1.244, 1.204, 640, 480)
MAST = vz.CameraModel.from_fov((-0.075, 0.0, 0.42), 0.2618, 1.204, 640, 480)


def _src(*names):
    return '\n'.join(open(os.path.join(BEH, n)).read() for n in names)


def _node_src():
    """What the running task node is made of: its own file, the shared seeing, the shared
    stand-up, the base."""
    return _src('task_node.py', 'sense_task.py', 'flip_task.py', 'node_base.py')


def _cfg_block(node):
    txt = open(CFG).read()
    i = txt.index(node + ':')
    j = min([k for k in (txt.find('\n' + n + ':', i + 1) for n in
                         ('perception_node', 'task_node', 'verify_node')) if k > 0] + [len(txt)])
    return txt[i:j]


def _world_objects():
    sdf = open(WORLD).read()
    return {m.group(2): [float(v) for v in m.group(3).split()] for m in re.finditer(
        r'<uri>model://(\w+)</uri>\s*<name>(\w+)</name>\s*<pose>([^<]+)</pose>', sdf)}


# ============================================================== PERCEPTION
def test_camera_projection_inverts():
    """``to_pixel`` and ``to_plane`` must be inverses, or every measurement is wrong."""
    for x in (0.20, 0.28, 0.36):
        for y in (-0.15, 0.0, 0.15):
            for z in (0.045, 0.120):
                px = CAMERA.to_pixel((x, y, z))
                assert px is not None, f'({x},{y},{z}) not visible'
                back = CAMERA.to_plane(px[0], px[1], z)
                assert back is not None and math.dist(back, (x, y)) < 1e-9


def test_perception_recovers_every_pose_from_its_true_silhouette():
    """THE ROUND TRIP, on what the camera actually sees.

    A known pose is turned into the silhouette of the SOLID - its outline projected through
    the camera model, snapped to whole pixels as a real blob's contour is - and fed back
    through vision.fit(). Across the reach annulus, both resting states, every object:
    the state must always come back right, with a clear margin over the wrong one, and the
    position inside the +/-3 mm the pads can capture.
    """
    import random
    rng = random.Random(7)
    for name, o in OBJECTS.items():
        for upright in (False, True):
            for _ in range(10):
                r, b = rng.uniform(0.27, 0.33), math.radians(rng.uniform(-8, 8))
                x, y = r * math.cos(b), r * math.sin(b)
                a = (math.radians(90 + rng.uniform(-25, 25)) if not upright
                     else rng.uniform(-3, 3))
                cz = vz.centre_height(0.0, upright, o['length'], o['half_width'])
                obs, inside = vz.observe(CAMERA, (x, y, cz), a, upright, o['length'],
                                         o['half_width'], o['round'])
                assert inside, f'{name} at ({x:.3f},{y:.3f}) is not wholly in view'
                d = vz.fit(obs, o['length'], o['half_width'], o['round'], CAMERA)
                assert d['upright'] == upright, (
                    f'{name} {"standing" if upright else "lying"} read as the other state '
                    f'(fit {d["fit_px"]} px vs {d["alt_px"]})')
                assert d['alt_px'] - d['fit_px'] > 10.0, (
                    f'{name}: the resting state is decided by only '
                    f'{d["alt_px"] - d["fit_px"]:.1f} px')
                err = math.dist((d['x'], d['y']), (x, y))
                lim = 0.008 if (upright and not o['round']) else 0.003
                assert err <= lim, (f'{name} {"standing" if upright else "lying"}: '
                                    f'{err * 1000:.1f} mm off')
                if not upright:
                    da = abs(math.degrees(math.remainder(d['axis'] - a, math.pi)))
                    assert da <= 1.5, f'{name}: axis {da:.1f} deg off'


def test_a_standing_bottle_is_not_read_as_lying():
    """THE FAULT BEHIND THE CRASH, pinned. A flat rectangle at mid-height - what the old
    measurement assumed - read the standing bottle as lying, 59 mm away, axis radial."""
    o = OBJECTS['bottle']
    obs, _ = vz.observe(CAMERA, (0.30, 0.0, 0.12), 0.0, True, o['length'], o['half_width'],
                        True)
    d = vz.fit(obs, o['length'], o['half_width'], True, CAMERA)
    assert d['upright'] and math.dist((d['x'], d['y']), (0.30, 0.0)) < 0.002, d
    # and the silhouette really is elongated - the fact the old model ignored
    _u, _v, lo, sh, _a = vz.moments(obs)
    assert lo / sh > 1.4, 'a standing bottle seen at 71 deg is not the round blob assumed'


def test_the_image_pipeline_finds_every_object():
    """THE WHOLE PATH ON AN IMAGE: colour threshold -> contour -> silhouette fit.

    A ray-cast of the scene (render.py): the exact solids in their models' colours, lit by
    the world's sun and ambient, over the grey floor, with the arms drawn in. Run through
    detect.detect() - the node's own code - it must find each object, in the right state,
    within 3 mm. Needs NumPy and OpenCV (both there under ROS); skipped where they are not.
    """
    try:
        import numpy  # noqa: F401
        import cv2  # noqa: F401
    except ImportError:
        print('    (skipped: no numpy/cv2 here - it runs where ROS is installed)')
        return
    import render
    from navigate_behavior import detect as dt
    from navigate_behavior.looking import LOOK_LEFT, LOOK_RIGHT
    specs = dt.object_specs(list(OBJECTS), _param_reader('perception_node'))
    arms = [(a, b, r) for q, sd in ((LOOK_LEFT, 1), (LOOK_RIGHT, -1))
            for _n, a, b, r in col.arm_capsules_fine(q, sd, OPEN_POS)]
    cases = [('bottle', (0.30, 0.0), 0.0, True), ('can', (0.30, 0.01), 75.0, False),
             ('box', (0.30, -0.01), 105.0, False), ('box', (0.29, 0.02), 30.0, True),
             ('bottle', (0.31, -0.02), 80.0, False)]
    for name, xy, a_deg, up in cases:
        o = OBJECTS[name]
        cz = vz.centre_height(0.0, up, o['length'], o['half_width'])
        img = render.render(CAMERA, [((xy[0], xy[1], cz), math.radians(a_deg), up,
                                      o['length'], o['half_width'], o['round'], o['rgb'])],
                            arms)
        det = dt.detect(img, 'rgb8', specs, CAMERA)
        d = det[name]
        assert d['found'] and not d['clipped'], f'{name}: not found in the rendered image'
        assert d['upright'] == up, f'{name}: read in the wrong resting state'
        err = math.dist((d['x'], d['y']), xy)
        assert err < 0.003, f'{name}: {err * 1000:.1f} mm off in the rendered image'
        assert all(not v['found'] for k, v in det.items() if k != name), (
            f'something other than the {name} was detected in an image of the {name}')


def _param_reader(node):
    """A get-parameter function over one node's block of the config (inline maps too)."""
    blk = _cfg_block(node)
    vals = {}
    for m in re.finditer(r'^    (\w+):\s*\{([^}]*)\}', blk, re.M):
        for kv in re.finditer(r'(\w+):\s*(\[[^\]]*\]|[^,]+)', m.group(2)):
            k, v = kv.group(1), kv.group(2).strip()
            vals[f'{m.group(1)}.{k}'] = ([float(x) for x in v[1:-1].split(',')]
                                         if v.startswith('[') else
                                         (v == 'true') if v in ('true', 'false') else float(v))
    for m in re.finditer(r'^    (\w+):\s*([-0-9.]+)\s*(?:#.*)?$', blk, re.M):
        vals[m.group(1)] = float(m.group(2))
    return lambda k, default: vals.get(k, default)


def _hsv(rgb):
    """OpenCV HSV (H 0-180, S and V 0-255) of an RGB colour in 0-1."""
    import colorsys
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return h * 180.0, s * 255.0, v * 255.0


def test_the_colour_windows_contain_the_models_and_exclude_the_robot():
    """Each object's HSV window must hold its colour lit AND in shade, and no window may
    hold any of the robot's own colours - or an arm in view would be detected as an object."""
    P = _param_reader('perception_node')
    robot = [tuple(float(v) for v in m.group(1).split()[:3]) for m in re.finditer(
        r'<color rgba="([^"]+)"', open(os.path.join(URDF, 'robot.urdf.xacro')).read())]
    assert robot, 'no robot colours found'
    for name, o in OBJECTS.items():
        lo, hi = P(f'{name}.hsv_lo', None), P(f'{name}.hsv_hi', None)
        assert lo and hi, f'{name}: no colour window'
        for shade in (0.35, 0.5, 0.75, 1.0):         # the ambient floor to full sun
            h, s, v = _hsv(tuple(c * shade for c in o['rgb']))
            assert lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2], (
                f'{name} at {shade:.0%} light is HSV ({h:.0f},{s:.0f},{v:.0f}), outside '
                f'{lo}-{hi}')
        for rgb in robot:
            for shade in (0.35, 1.0):
                h, s, v = _hsv(tuple(c * shade for c in rgb))
                assert not (lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1]
                            and lo[2] <= v <= hi[2]), (
                    f'the robot colour {rgb} falls inside the {name} window')


def _xacro_props():
    txt = open(os.path.join(URDF, 'robot.urdf.xacro')).read()
    return {m.group(1): m.group(2) for m in
            re.finditer(r'<xacro:property name="(\w+)"\s+value="([^"]+)"', txt)}


def test_the_cameras_are_the_urdf():
    """Perception's two cameras must BE the URDF's: pose, pitch, field of view, image size -
    and the harness and the tests use the same numbers."""
    xac = open(os.path.join(URDF, 'mast_sensors.xacro')).read()
    P = _param_reader('perception_node')
    props = _xacro_props()

    def val(v):
        v = v.strip()
        return float(props[v[2:-1]]) if v.startswith('${') else float(v)
    for joint, sensor, pre, topic in (('workspace_cam_joint', 'workspace_cam', '', 'workspace_cam'),
                                      ('camera_joint', 'camera', 'far_', 'camera')):
        m = re.search(r'<joint name="%s".*?<origin xyz="([^"]+)" rpy="([^"]+)"' % joint, xac, re.S)
        assert m, f'{joint} not found'
        xyz = [val(v) for v in re.findall(r'\$\{[^}]+\}|[-0-9.]+', m.group(1))]
        rpy = [val(v) for v in re.findall(r'\$\{[^}]+\}|[-0-9.]+', m.group(2))]
        sen = xac[xac.index(f'<sensor name="{sensor}"'):]
        fov = float(re.search(r'<horizontal_fov>([0-9.]+)<', sen).group(1))
        w = int(re.search(r'<width>(\d+)<', sen).group(1))
        h = int(re.search(r'<height>(\d+)<', sen).group(1))
        for key, want in (('cam_x', xyz[0]), ('cam_y', xyz[1]), ('cam_z', xyz[2]),
                          ('cam_pitch', rpy[1]), ('horizontal_fov', fov), ('image_width', w),
                          ('image_height', h)):
            k = pre + key
            assert abs(P(k, float('nan')) - want) < 1e-9, f'{k}: config vs URDF {want}'
        assert re.search(rf'<topic>{topic}</topic>', sen), f'the {sensor} topic changed'
        assert f"'/{topic}@sensor_msgs/msg/Image" in open(LAUNCH).read(), f'/{topic} not bridged'
    for cam, pre in ((CAMERA, ''), (MAST, 'far_')):
        assert math.dist(cam.position, (P(pre + 'cam_x', 0), P(pre + 'cam_y', 0),
                                        P(pre + 'cam_z', 0))) < 1e-12
        assert abs(cam.pitch - P(pre + 'cam_pitch', 0)) < 1e-12
    assert re.search(r'^\s*far_camera:\s*true\b', _cfg_block('perception_node'), re.M), (
        'the mast camera is not switched on')


def test_the_perception_node_loads_every_key_it_uses():
    """THE CRASH, as a test: detect() reads per-object keys that object_specs() must load.

    The node used to read ``lying_cz`` and ``upright_cz`` it never loaded - a KeyError on the
    first frame with an object in it, which killed the node. A missing key is a runtime error,
    so compiling passes; this reads the source.
    """
    src = _src('detect.py')
    spec = src[src.index('def object_specs('):src.index('KERNEL =')]
    loaded = set(re.findall(r"^\s*'(\w+)':", spec, re.M))
    used = set(re.findall(r"sp\['(\w+)'\]", src))
    assert used and used <= loaded, f'read but never loaded: {sorted(used - loaded)}'
    node = _src('perception_node.py')
    assert "d['lying_cz']" not in node and 'self.dims' not in node


def test_one_bad_frame_does_not_kill_perception():
    """An exception in the image callback must be logged and skipped, not leave spin()."""
    node = _src('perception_node.py')
    body = node[node.index('    def on_image(self, msg'):node.index('\ndef main(')]
    assert re.search(r'try:\s+img = .*?dt\.detect\(.*?except Exception', body, re.S), (
        'detect() is not guarded - one bad frame kills the node, as before')


def test_the_parked_arms_hide_the_view_and_the_look_pose_does_not():
    """WHY THE ARMS MOVE TO LOOK. Parked, the hands block sight lines to objects in front;
    in the look pose every sight line to every test pose clears both arms by 20 mm."""
    from navigate_behavior.looking import LOOK_LEFT, LOOK_RIGHT

    def worst(ql, qr):
        caps = col.arm_capsules_fine(ql, +1, OPEN_POS) + col.arm_capsules_fine(qr, -1, OPEN_POS)
        g = 9.0
        for r in (0.27, 0.30, 0.33):
            for b in (-8, 0, 8):
                x, y = r * math.cos(math.radians(b)), r * math.sin(math.radians(b))
                for o in OBJECTS.values():
                    poses = [((x, y, o['upright_cz']), 0.0, True)] + [
                        ((x, y, o['cz']), math.radians(a), False) for a in (60, 90, 120)]
                    for c, a, up in poses:
                        for p in vz.outline_points(c, a, up, o['length'], o['half_width'],
                                                   o['round'], n=12):
                            g = min(g, min(col.segment_distance(CAMERA.position, p, a0, a1) - rr
                                           for _n, a0, a1, rr in caps))
        return g
    assert worst(ch.TRAVEL_LEFT, ch.TRAVEL_RIGHT) < 0.0, (
        'the parked arms no longer block the view - then the look pose is unnecessary')
    assert worst(LOOK_LEFT, LOOK_RIGHT) > 0.020, 'the look pose blocks the camera'


def test_the_look_move_is_clear():
    """Parked -> look is a short joint-space line; checked in full, because it IS the path."""
    from navigate_behavior.looking import LOOK_LEFT, LOOK_RIGHT
    change = max(max(abs(a - b) for a, b in zip(ch.TRAVEL_LEFT, LOOK_LEFT)),
                 max(abs(a - b) for a, b in zip(ch.TRAVEL_RIGHT, LOOK_RIGHT)))
    assert change <= math.radians(50), f'the look pose is {math.degrees(change):.0f} deg away'
    lims = (ik.YAW_LIM, ik.PITCH2_LIM, ik.ROLL2_LIM, ik.PITCH3_LIM, ik.W_ROLL_LIM,
            ik.W_PITCH_LIM, ik.W_ROLL_LIM)
    for q in (LOOK_LEFT, LOOK_RIGHT):
        assert all(abs(v) <= lim for v, lim in zip(q, lims)), 'look pose outside joint limits'
    for i in range(41):
        f = i / 40.0
        ql = [a + f * (b - a) for a, b in zip(ch.TRAVEL_LEFT, LOOK_LEFT)]
        qr = [a + f * (b - a) for a, b in zip(ch.TRAVEL_RIGHT, LOOK_RIGHT)]
        rep = col.check_pose(ql, qr, OPEN_POS)
        assert rep['arm_arm'] > 0.040 and rep['base'] > 0.020 and rep['mast'] > 0.020 and \
            rep['floor'] > 0.050, f'the look move meets {rep["worst_name"]} at {f:.0%}'
        for q, sd in ((ql, +1), (qr, -1)):
            g = col.self_clearance(col.arm_capsules_fine(q, sd, OPEN_POS), False)[0]
            # the fingertip passes its own forearm at +0.3 mm near parked, whose own margin is
            # 3 mm; the model has no self-collision in Gazebo, so bounded, not zero
            assert g > -0.001, f'the look move passes {-g * 1000:.1f} mm into the arm at {f:.0%}'


def test_stale_or_clipped_detections_are_refused():
    """Only frames taken after the arms moved and the base stopped, of the whole object."""
    src = _src('sense_task.py')
    body = src[src.index('    def sense(self, name, after'):src.index('    def _look_move(')]
    assert 'st >= after' in body and "not objs[name].get('clipped')" in body
    assert "not f[name].get('clipped')" in body, 'sense_all() takes clipped blobs'
    obs = src[src.index('    def observe(self, name'):src.index('    def _world_of(')]
    assert obs.index("'the look pose'") < obs.index('self.sense(name, self._now()') < \
        obs.index("'the parked pose'"), 'sensing must happen with the arms in the look pose'
    assert obs.index("'the look pose'", obs.index('def observe_all')) < \
        obs.index('self.sense_all(self._now()') < \
        obs.index("'the parked pose'", obs.index('def observe_all')), (
            'observe_all() must sense with the arms in the look pose')


def test_the_axis_gate_is_the_stand_up_envelope():
    """The node refuses what the shared stand-up cannot grasp: more than 20 deg off
    tangential. Rotating the base cannot fix it - the angle between an object's axis and the
    direction to it is invariant under rotation about the base - so it is reported, not tried.
    """
    P = _param_reader('task_node')
    assert P('max_axis_dev_deg', None) == 20.0
    for az in (0.0, 0.7, -1.9):
        c, s = math.cos(az), math.sin(az)
        p, axis = (0.30, 0.05), 0.4
        rp = (c * p[0] - s * p[1], s * p[0] + c * p[1])
        dev0 = math.remainder(axis - math.atan2(p[1], p[0]), math.pi)
        dev1 = math.remainder(axis + az - math.atan2(rp[1], rp[0]), math.pi)
        assert abs(dev0 - dev1) < 1e-12


# ================================================================ ONE ROBOT
def test_shared_files_match_the_manifest():
    """The IK, planner, primitives AND the stand-up are reorient's, byte for byte; the seeing
    (vision, detect, looking, perception_node, sense_task) is the navigate section's too."""
    import hashlib
    man = os.path.join(ROOT, 'SHARED_FILES.sha256')
    assert os.path.exists(man), 'SHARED_FILES.sha256 is missing from the section root'
    mine = ('all', 'flip', 'sense')
    listed = []
    for line in open(man):
        if not line.strip() or line.startswith('#'):
            continue
        digest, fname, group = line.split()
        if group not in mine:
            continue
        listed.append(fname)
        got = hashlib.sha256(open(os.path.join(BEH, fname), 'rb').read()).hexdigest()
        assert got == digest, f'{fname} differs from the shared copy'
    for f in ('ik.py', 'collision.py', 'arm_commander.py', 'wall_ref.py', 'motion.py',
              'node_base.py', 'choreography.py', 'flip_task.py', 'verify_node.py',
              'vision.py', 'detect.py', 'looking.py', 'perception_node.py', 'sense_task.py'):
        assert f in listed, f'{f} is not in the shared manifest'


def test_the_stand_up_is_the_shared_one():
    """This section decides; the reorient section's code acts. No sequence of its own."""
    tn = _src('task_node.py')
    assert 'class TaskNode(SenseTask)' in tn and 'ok = self.stand_up(' in tn
    assert 'class SenseTask(FlipTask)' in _src('sense_task.py')
    assert not re.search(r'\bch\.', tn), 'the task node builds motion itself'


def test_the_node_assigns_every_attribute_it_reads():
    src = _node_src()
    api = {'create_publisher', 'create_subscription', 'create_timer', 'get_logger',
           'get_parameter', 'declare_parameter', 'destroy_node', 'get_clock'}
    assigned = set(re.findall(r'self\.(\w+)\s*(?:=|\[)', src))
    assigned |= set(re.findall(r'^    ([A-Z_]+)\s*=', src, re.M))
    methods = set(re.findall(r'def (\w+)\(', src))
    missing = sorted(set(re.findall(r'self\.(\w+)', src)) - assigned - methods - api)
    assert not missing, f'read but never assigned: {missing}'


def test_every_wait_is_on_simulation_time():
    src = _node_src()
    for fn in ('sense', 'sense_all', 'send_pair', 'settle_arms', 'grippers', 'set_grasp',
               'check_grip', 'drive_to'):
        body = re.search(r'    def %s\(.*?(?=\n    def |\n    @|\Z)' % fn, src, re.S).group(0)
        assert 'time.time()' not in body, f'{fn}() times itself on the wall clock'
        assert 'self._now()' in body or 'self._sleep(' in body, f'{fn}() is not on sim time'


def test_the_config_holds_no_object_positions():
    """Where an object is, how it lies, even which way to look for it: the scan says."""
    blk = _cfg_block('task_node')
    allowed = {'length', 'half_width', 'round', 'mass', 'surface_z', 'cz', 'upright_cz',
               'close'}
    for name in OBJECTS:
        m = re.search(rf'^    {name}:\n((?:      .+\n)+)', blk, re.M)
        assert m, f'{name} missing from task_node'
        keys = set(re.findall(r'^      (\w+):', m.group(1), re.M))
        assert keys <= allowed, f'{name} is told {sorted(keys - allowed)}'
        for k, want in OBJECTS[name].items():
            if k == 'rgb':
                continue
            v = re.search(rf'^      {k}:\s*(\S+)', m.group(1), re.M).group(1)
            got = (v == 'true') if v in ('true', 'false') else float(v)
            assert got == want, f'{name}.{k}: config {got} vs this file {want}'


def test_the_world_spreads_the_objects_out():
    """The job as asked: the robot and its three objects far apart. From the scan point every
    object is inside the range the mast camera is trusted over; they are metres from each other;
    each is on its surface, at least 1 m from every wall; and there is lying work to do."""
    wo = _world_objects()
    assert set(wo) == set(OBJECTS)
    P = _param_reader('task_node')
    room = _room()
    sx, sy = 0.0, 0.0
    lying = 0
    for name, (x, y, z, roll, _p, _yaw) in wo.items():
        o = OBJECTS[name]
        standing = abs(roll) < 0.1
        lying += not standing
        assert abs(z - (o['upright_cz'] if standing else o['cz'])) < 1e-6, f'{name}: height'
        r = math.hypot(x - sx, y - sy)
        assert nav.FAR_MIN + 0.3 <= r <= nav.FAR_MAX - 0.3, f'{name} is {r:.2f} m from the scan point'
        assert room.wall_distance((x, y)) >= 1.0, f'{name} is too near a wall'
    names = sorted(wo)
    gaps = [math.dist(wo[a][:2], wo[b][:2]) for i, a in enumerate(names) for b in names[i + 1:]]
    assert min(gaps) >= 2.0, f'objects only {min(gaps):.2f} m apart'
    assert lying == 2, 'the world should have two lying objects and one standing'


def test_the_fixtures_are_the_world():
    sdf = open(WORLD).read()
    boxes = {m.group(1): ([float(v) for v in m.group(2).split()],
                          [float(v) for v in m.group(3).split()])
             for m in re.finditer(r'<collision name="(\w+)"><pose>([^<]+)</pose>\s*<geometry>'
                                  r'<box><size>([^<]+)</size>', sdf)}
    blk = _cfg_block('task_node')
    names = re.search(r'^\s*fixtures:\s*\[([^\]]*)\]', blk, re.M).group(1)
    for f in [v.strip() for v in names.split(',') if v.strip()]:
        body = re.search(rf'^    {f}:\n((?:      .+\n)+)', blk, re.M).group(1)
        centre = [float(v) for v in re.search(r'centre:\s*\[([^\]]+)\]', body).group(1).split(',')]
        size = [float(v) for v in re.search(r'size:\s*\[([^\]]+)\]', body).group(1).split(',')]
        pose, sz = boxes[f]
        assert max(abs(a - b) for a, b in zip(centre, pose[:3])) < 1e-9, f
        assert max(abs(a - b) for a, b in zip(size, sz)) < 1e-9, f


def _room():
    blk = _cfg_block('task_node')
    box = [float(v) for v in re.search(r'^\s*room:\s*\[([^\]]+)\]', blk, re.M).group(1).split(',')]
    return rr.Room(*box)


def test_the_room_is_the_world():
    """The config's room is the FACES of the world's walls - what the LiDAR hits - not their
    centres: the ten-millimetre mistake the fixed-base sections once made with their L."""
    sdf = open(WORLD).read()

    def box(name):
        m = re.search(r'<collision name="%s"><pose>([^<]+)</pose>\s*<geometry><box><size>'
                      r'([^<]+)</size>' % name, sdf)
        return [float(v) for v in m.group(1).split()], [float(v) for v in m.group(2).split()]
    room = _room()
    (ex, *_), (esx, *_) = box('wall_e')
    (wx, *_), (wsx, *_) = box('wall_w')
    (_nx, ny, *_), (_nsx, nsy, *_) = box('wall_n')
    (_sx, sy, *_), (_ssx, ssy, *_) = box('wall_s')
    assert abs(room.x_max - (ex - esx / 2)) < 1e-9 and abs(room.x_min - (wx + wsx / 2)) < 1e-9
    assert abs(room.y_max - (ny - nsy / 2)) < 1e-9 and abs(room.y_min - (sy + ssy / 2)) < 1e-9
    # tall enough for the LiDAR to see, which everything else in the room is not
    lidar_z = float(_xacro_props()['lidar_z'])
    for w in ('wall_n', 'wall_s', 'wall_e', 'wall_w'):
        (_x, _y, cz, *_), (_a, _b, h) = box(w)
        assert cz + h / 2 > lidar_z + 0.1, f'{w} is below the LiDAR'
    assert all(o['upright_cz'] * 2 < lidar_z for o in OBJECTS.values())


def test_the_object_list_agrees_everywhere():
    """World, task, perception, verification, launch bridges and suction joints: one set."""
    want = sorted(OBJECTS)
    txt = open(CFG).read()
    lists = [sorted(v.strip() for v in m.group(1).split(','))
             for m in re.finditer(r'^\s*(?:order|objects):\s*\[([^\]]+)\]', txt, re.M)]
    assert len(lists) == 3 and all(lst == want for lst in lists), lists
    assert sorted(_world_objects()) == want
    launch = open(LAUNCH).read()
    assert sorted(re.findall(r"'(\w+)'", re.search(r'for obj in \(([^)]*)\)', launch).group(1))) == want
    plug = open(os.path.join(URDF, 'gazebo_plugins.xacro')).read()
    assert sorted(re.findall(r'grasp_joint obj="(\w+)"', plug)) == want
    for m in OBJECTS:
        sdf = open(os.path.join(SRC, 'navigate_gazebo', 'models', m, 'model.sdf')).read()
        assert 'PosePublisher' in sdf, f'{m} publishes no ground truth'


def test_every_sensor_the_nodes_subscribe_to_is_bridged():
    launch = open(LAUNCH).read()
    for topic in ('/workspace_cam@', '/workspace_cam_info@', '/camera@', '/camera_info@',
                  '/imu@', '/ft/left@', '/ft/right@', '/scan@', '/odom@', '/clock@',
                  '/cmd_vel@', '/model/navigate_bimanual/pose@'):
        assert topic in launch, f'{topic.rstrip("@")} is not bridged'
    assert "executable='perception_node'" in launch


def _world_obstacles():
    """The world's objects as the route planner sees them: footprints, lying or standing."""
    out = {}
    for n, (x, y, _z, roll, _p, yaw) in _world_objects().items():
        o = OBJECTS[n]
        out[n] = nav.footprint(x, y, yaw - math.pi / 2, abs(roll) < 0.1, o['length'],
                               o['half_width'])
    return out


def _park(name):
    """Where the node parks for ``name``: the approach it would choose from the scan point."""
    x, y, _z, _r, _p, yaw = _world_objects()[name]
    plan, why = nav.plan_approach(name, (x, y, (yaw - math.pi / 2) % math.pi), (0.0, 0.0),
                                  _world_obstacles(), _room())
    assert plan, f'{name}: no approach - {why}'
    return plan


def _plan(name):
    """The stand-up of a perceived object, planned as the node plans it, AT THE PARK POSE the
    node drives to, with the other objects and the room's walls in the world."""
    wo = _world_objects()
    px, py, h = _park(name)['park']

    def base(wx, wy):
        dx, dy = wx - px, wy - py
        return (math.cos(h) * dx + math.sin(h) * dy, -math.sin(h) * dx + math.cos(h) * dy)
    o = OBJECTS[name]
    x, y, _z, _r, _p, yaw = wo[name]
    bx, by = base(x, y)
    assert math.hypot(bx - REACH, by) < 1e-9, 'the park pose does not put it at the reach'
    psi = (yaw - math.pi / 2 - h) % math.pi
    assert abs(psi - math.pi / 2) < 1e-9, 'the park pose does not put its axis across'
    others = []
    for n, (ox, oy, oz, roll, _pp, oyaw) in wo.items():
        if n == name:
            continue
        on = OBJECTS[n]
        obx, oby = base(ox, oy)
        if abs(roll) < 0.1:
            others += col.object_capsules((obx, oby, oz), (0, 0, 1), on['length'],
                                          on['half_width'], on['round'])
        else:
            a = oyaw - math.pi / 2 - h
            others += col.object_capsules((obx, oby, oz), (math.cos(a), math.sin(a), 0),
                                          on['length'], on['half_width'], on['round'])
    walls = []
    for cx, cy, sx, sy in ((0.0, 2.55, 5.2, 0.1), (0.0, -2.55, 5.2, 0.1),
                           (2.55, 0.0, 0.1, 5.2), (-2.55, 0.0, 0.1, 5.2)):
        walls.append(col.box_obstacle((*base(cx, cy), 0.45), -h, (sx, sy, 0.9)))
    legs = ch.cycle_legs((REACH, 0.0), o['cz'], o['upright_cz'], psi,
                         ch.grip_half_separation(o['length']), ch.FLIP_Z, ch.HOVER_DZ,
                         ch.PLACE_DROP, o['half_width'], 0.0)
    lying = col.object_capsules((REACH, 0, o['cz']), (math.cos(psi), math.sin(psi), 0),
                                o['length'], o['half_width'], o['round'])
    standing = col.object_capsules((REACH, 0, o['upright_cz']), (0, 0, 1), o['length'],
                                   o['half_width'], o['round'])
    fp, obs, wide = ch.plan_inputs(legs, o['close'], OPEN_POS, lying, standing, others, walls)
    flat = [w for _, leg in legs for w in leg]
    QL, QR = col.solve_path(flat, finger_pos=fp, seed_l=ch.TRAVEL_LEFT, seed_r=ch.TRAVEL_RIGHT,
                            obstacles=obs, pin_start=True,
                            pin_end=(ch.TRAVEL_LEFT, ch.TRAVEL_RIGHT), wide=wide)
    return QL, QR, fp, others, walls


def test_the_stand_up_plans_for_every_lying_object_in_this_world():
    """The shared stand-up at every park pose the node drives to, with THIS room round it:
    bodies clear, and along the real path every other object 20 mm and every wall 10 mm off."""
    for name in ('can', 'box'):
        QL, QR, fp, others, walls = _plan(name)
        for i, (ql, qr) in enumerate(zip(QL, QR)):
            rep = col.check_pose(ql, qr, fp[i])
            assert rep['worst'] > MIN_CLEARANCE, f'{name}: {rep["worst_name"]} at {i}'
        g_obj, g_wall = 9.0, 9.0
        for i in range(len(QL) - 1):
            for Q, sd in ((QL, +1), (QR, -1)):
                for f in (0.0, 0.25, 0.5, 0.75):
                    q = [a + f * (b - a) for a, b in zip(Q[i], Q[i + 1])]
                    caps = col.arm_capsules_fine(q, sd, fp[i])
                    g_obj = min(g_obj, min(col.obstacle_clearance(caps, c) for c in others))
                    g_wall = min(g_wall, min(col.obstacle_clearance(caps, w) for w in walls))
        assert g_obj > 0.020, f'{name}: an arm passes {g_obj * 1000:+.1f} mm from another object'
        assert g_wall > 0.010, f'{name}: an arm passes {g_wall * 1000:+.1f} mm from a wall'


def test_every_gripper_command_is_as_wide_as_the_controller():
    ctl = open(os.path.join(SRC, 'navigate_description', 'config', 'controllers.yaml')).read()
    lines = ctl.splitlines(True)
    at = next(i for i, ln in enumerate(lines) if ln.startswith('gripper_controller:'))
    blk = ''.join(lines[at:])
    blk = blk[blk.index('joints:'):]
    n_ctrl = len([x for x in blk[blk.index('[') + 1:blk.index(']')].split(',') if x.strip()])
    src = _node_src()
    chunks = src.split('grip_pub.publish(')[1:]
    assert chunks
    for c in chunks:
        widths = [int(p.strip()[0]) for p in c[:c.index(')))') if ')))' in c else 200].split('*')[1:]
                  if p.strip()[:1].isdigit()]
        assert sum(widths) == n_ctrl, f'a gripper publish is {sum(widths)} wide, not {n_ctrl}'


def test_the_arms_load_already_parked():
    src = open(os.path.join(URDF, 'ros2_control.xacro')).read()
    want = dict(zip([f'left_{j}' for j in ik.ARM_JOINT_SUFFIX]
                    + [f'right_{j}' for j in ik.ARM_JOINT_SUFFIX],
                    list(ch.TRAVEL_LEFT) + list(ch.TRAVEL_RIGHT)))
    found = dict(re.findall(r'<xacro:ctrl_joint name="(\w+)"\s+init="([-0-9.]+)"', src))
    for j, v in want.items():
        assert abs(float(found[j]) - v) < 1e-3, f'{j} loads away from parked'


# ============================================================== NAVIGATION
def test_the_room_fix_finds_the_pose_anywhere_in_the_room():
    """From anywhere in the room, at any heading, from a prior 8 cm and 8 deg out, with 3 mm of
    range noise: the pose to a millimetre and a few hundredths of a degree."""
    import random
    rng = random.Random(3)
    room = _room()
    for i in range(120):
        x, y = rng.uniform(-2.1, 2.1), rng.uniform(-2.1, 2.1)
        h = rng.uniform(-math.pi, math.pi)
        sc, amin, ainc = rr.synth_scan(room, (x, y, h), noise=0.003, seed=i)
        prior = (x + rng.uniform(-0.08, 0.08), y + rng.uniform(-0.08, 0.08),
                 h + math.radians(rng.uniform(-8, 8)))
        pose, info = rr.pose_in_room(sc, amin, ainc, 0.12, 8.0, room, prior)
        assert pose is not None, f'no fix at ({x:.2f}, {y:.2f}, {math.degrees(h):.0f}): {info}'
        assert math.hypot(pose[0] - x, pose[1] - y) < 0.002, f'{info}'
        assert abs(math.remainder(pose[2] - h, 2 * math.pi)) < math.radians(0.1)


def test_the_room_fix_ignores_clutter_and_refuses_what_it_cannot_place():
    """Something at LiDAR height that is not a wall is dropped, not fitted; a scan with only
    parallel walls in it - one coordinate unknown - gives NO pose rather than half of one."""
    room = _room()
    pose = (0.4, -0.3, 0.7)
    clutter = [((1.0, -1.2), (1.6, -1.4)), ((-1.8, 0.5), (-1.3, 1.1))]
    sc, amin, ainc = rr.synth_scan(room, pose, clutter=clutter)
    got, info = rr.pose_in_room(sc, amin, ainc, 0.12, 8.0, room, (0.45, -0.28, 0.72))
    assert got is not None and math.hypot(got[0] - 0.4, got[1] + 0.3) < 0.002, info
    # only the east and west walls in view
    sc, amin, ainc = rr.synth_scan(room, (0.0, 0.0, 0.0))
    only = [r if (abs(math.cos(amin + i * ainc)) > 0.9) else float('inf')
            for i, r in enumerate(sc)]
    got, info = rr.pose_in_room(only, amin, ainc, 0.12, 8.0, room, (0.0, 0.0, 0.0))
    assert got is None and 'parallel' in info['reject'], info


def test_the_mast_camera_measures_across_the_room():
    """The shared silhouette fit, through the MAST camera, over the range it is trusted on:
    the resting state always right, the position and axis inside what the node's mast
    perception check allows (15 mm and 3.5 deg per metre), and good to 5 mm out to 1.6 m."""
    import itertools
    worst = {}
    for (n, o), r, bdeg, up, adeg in itertools.product(
            OBJECTS.items(), (1.0, 1.3, 1.6, 2.0, 2.4), (-20, 0, 20), (False, True),
            (0, 45, 90, 135)):
        if up and adeg:
            continue
        b = math.radians(bdeg)
        x, y = r * math.cos(b), r * math.sin(b)
        cz = vz.centre_height(0.0, up, o['length'], o['half_width'])
        obs = vz.observe(MAST, (x, y, cz), math.radians(adeg), up, o['length'],
                         o['half_width'], o['round'])
        assert obs is not None
        pts, inside = obs
        assert inside and max(p[1] for p in pts) < 330 - 3, f'{n} at {r} m, {bdeg} deg: not whole'
        d = vz.fit(pts, o['length'], o['half_width'], o['round'], MAST, 0.0)
        assert d['upright'] == up, f'{n} at {r} m read in the wrong state'
        e = math.dist((d['x'], d['y']), (x, y))
        assert e * 1000 <= 15.0 * r, f'{n} at {r} m: {e * 1000:.1f} mm'
        if r <= 1.6:
            assert e <= 0.005, f'{n} at {r} m: {e * 1000:.1f} mm'
        if not up:
            da = abs(math.degrees(math.remainder(d['axis'] - math.radians(adeg), math.pi)))
            assert da <= max(4.0, 3.5 * r), f'{n} at {r} m: axis {da:.1f} deg'


def _whole(cam, name, r, bdeg, adeg, up, v_max=None, arms=None):
    o = OBJECTS[name]
    b = math.radians(bdeg)
    c = (r * math.cos(b), r * math.sin(b), vz.centre_height(0, up, o['length'], o['half_width']))
    for p in vz.outline_points(c, math.radians(adeg), up, o['length'], o['half_width'],
                               o['round'], n=16):
        q = cam.to_pixel(p)
        if q is None or not cam.in_frame(q, 3.0) or (v_max is not None and q[1] > v_max - 3):
            return False
        if arms and min(col.segment_distance(cam.position, p, a0, a1) - rr_
                        for _n, a0, a1, rr_ in arms) < 0.0:
            return False
    return True


def test_the_camera_ranges_are_what_the_approach_uses():
    """WHY THE APPROACH HAS A BLIND LEG, measured. With the arms in the look pose:
    the mast camera sees any object whole, at any bearing it is used at, from FAR_MIN out - and
    NOT from much closer; the workspace camera sees a lying object whole to NEAR_MAX and a
    standing one to NEAR_STANDING_MAX, and not beyond. The staging and look distances sit inside
    those with margin, and the blind stretch between them is real."""
    from navigate_behavior.looking import LOOK_LEFT, LOOK_RIGHT
    arms = [c for q, sd in ((LOOK_LEFT, 1), (LOOK_RIGHT, -1))
            for c in col.arm_capsules_fine(q, sd, OPEN_POS)]
    P = _param_reader('perception_node')
    vmax = P('far_v_max', None)
    for n in OBJECTS:
        for b in (-20, 0, 20):
            assert _whole(MAST, n, nav.FAR_MIN, b, 0, True, vmax, arms), n
            for a in (0, 45, 90, 135):
                assert _whole(MAST, n, nav.FAR_MIN, b, a, False, vmax, arms), (n, b, a)
        assert not all(_whole(MAST, n, nav.FAR_MIN - 0.2, b, a, False, vmax, arms)
                       for b in (-20, 0, 20) for a in (0, 45, 90, 135)), (
            f'the mast camera sees {n} whole well inside FAR_MIN - then FAR_MIN is too cautious')
        for b in (-5, 0, 5):
            for a in (0, 45, 90, 135):
                assert _whole(CAMERA, n, nav.NEAR_MAX, b, a, False, None, arms), (n, b, a)
            assert _whole(CAMERA, n, nav.NEAR_STANDING_MAX, b, 0, True, None, arms), n
        assert not all(_whole(CAMERA, n, nav.NEAR_MAX + 0.1, b, a, False, None, arms)
                       for b in (-5, 0, 5) for a in (0, 45, 90, 135))
    # the tallest standing object is the one that leaves the top of the view first
    assert not all(_whole(CAMERA, n, nav.NEAR_STANDING_MAX + 0.07, 0, 0, True, None, arms)
                   for n in OBJECTS)
    assert nav.REACH < nav.STAGE_REACH <= nav.NEAR_MAX - 0.02, 'staging is not in the near view'
    assert nav.FAR_MIN + 0.1 <= nav.LOOK_REACH <= nav.FAR_MAX, 'the look point is not in the far view'
    assert nav.NEAR_MAX < nav.FAR_MIN, 'there is no blind stretch - then the approach can see all the way'


def test_the_look_pose_clears_the_mast_camera_above_far_v_max():
    """The mast camera's cut-off row IS where the look-pose arms begin: no ray above it meets an
    arm (so nothing whole above it is ever cut), and rays just below it do (so it is not set
    needlessly high). And the parked arms fill the middle of its view - why the scan looks with
    the arms out."""
    from navigate_behavior.looking import LOOK_LEFT, LOOK_RIGHT
    vmax = int(_param_reader('perception_node')('far_v_max', 0))

    def blocked(ql, qr, v, step=4):
        caps = col.arm_capsules_fine(ql, 1, OPEN_POS) + col.arm_capsules_fine(qr, -1, OPEN_POS)
        for u in range(0, MAST.width, step):
            d = MAST.ray(u, v)
            k = 3.0 / math.sqrt(sum(c * c for c in d))
            p1 = [MAST.position[i] + k * d[i] for i in range(3)]
            if min(col.segment_distance(MAST.position, p1, a, b) - r for _n, a, b, r in caps) < 0:
                return True
        return False
    assert not any(blocked(LOOK_LEFT, LOOK_RIGHT, v) for v in range(0, vmax + 1, 2))
    assert blocked(LOOK_LEFT, LOOK_RIGHT, vmax + 10), 'far_v_max is set higher than it needs be'
    assert blocked(ch.TRAVEL_LEFT, ch.TRAVEL_RIGHT, 150), 'the parked arms do not block the mast view'


def test_the_image_pipeline_finds_objects_across_the_room():
    """THE WHOLE PATH ON A MAST-CAMERA IMAGE: a ray-cast of objects metres away, the look-pose
    arms drawn in, through detect.detect() with the arm band - each found whole, in the right
    state, within 10 mm. Needs NumPy and OpenCV; skipped where they are not."""
    try:
        import numpy  # noqa: F401
        import cv2  # noqa: F401
    except ImportError:
        print('    (skipped: no numpy/cv2 here - it runs where ROS is installed)')
        return
    import render
    from navigate_behavior import detect as dt
    from navigate_behavior.looking import LOOK_LEFT, LOOK_RIGHT
    P = _param_reader('perception_node')
    specs = dt.object_specs(list(OBJECTS), P)
    arms = [(a, b, r) for q, sd in ((LOOK_LEFT, 1), (LOOK_RIGHT, -1))
            for _n, a, b, r in col.arm_capsules_fine(q, sd, OPEN_POS)]
    cases = [('bottle', 1.6, 12.0, 0.0, True), ('can', 1.2, -8.0, 30.0, False),
             ('box', 2.0, 5.0, 100.0, False), ('box', 1.4, -15.0, 0.0, True),
             ('can', 1.05, 0.0, 90.0, False)]
    for name, r, bdeg, a_deg, up in cases:
        o = OBJECTS[name]
        b = math.radians(bdeg)
        xy = (r * math.cos(b), r * math.sin(b))
        cz = vz.centre_height(0.0, up, o['length'], o['half_width'])
        img = render.render(MAST, [((xy[0], xy[1], cz), math.radians(a_deg), up,
                                    o['length'], o['half_width'], o['round'], o['rgb'])], arms)
        det = dt.detect(img, 'rgb8', specs, MAST, int(P('far_min_area', 0)),
                        v_max=int(P('far_v_max', 0)))
        d = det[name]
        assert d['found'] and not d['clipped'], f'{name} at {r} m: not found whole ({d})'
        assert d['upright'] == up, f'{name} at {r} m: read in the wrong resting state'
        err = math.dist((d['x'], d['y']), xy)
        assert err < 0.010, f'{name} at {r} m: {err * 1000:.1f} mm off in the rendered image'


def test_an_object_partly_behind_another_is_refused_and_the_front_one_kept():
    """A standing box half behind the bottle, seen from across the room: the box's blob is cut
    by the bottle's outline, fits to a pose far from the truth, and must be refused - while the
    bottle in front, whole, is kept. Needs NumPy and OpenCV; skipped where they are not."""
    try:
        import numpy  # noqa: F401
        import cv2  # noqa: F401
    except ImportError:
        print('    (skipped: no numpy/cv2 here - it runs where ROS is installed)')
        return
    import render
    from navigate_behavior import detect as dt
    P = _param_reader('perception_node')
    specs = dt.object_specs(list(OBJECTS), P)
    ob, ox = OBJECTS['bottle'], OBJECTS['box']
    scene = [((1.20, 0.00, ob['upright_cz']), 0.0, True, ob['length'], ob['half_width'], True,
              ob['rgb']),
             ((1.90, 0.06, ox['upright_cz']), 0.3, True, ox['length'], ox['half_width'], False,
              ox['rgb'])]
    img = render.render(MAST, scene)
    det = dt.detect(img, 'rgb8', specs, MAST, int(P('far_min_area', 0)),
                    v_max=int(P('far_v_max', 0)))
    assert det['box']['found'] and det['box']['clipped'] and det['box']['behind'] == ['bottle'], (
        det['box'])
    assert det['bottle']['found'] and not det['bottle']['clipped'], det['bottle']
    assert math.dist((det['bottle']['x'], det['bottle']['y']), (1.20, 0.0)) < 0.010


def test_a_blob_into_the_arm_band_is_clipped():
    """detect() refuses a blob that reaches below far_v_max, as it refuses one at the edge - an
    arm cuts it short without touching the edge. Needs OpenCV; skipped where it is not."""
    try:
        import numpy as np
        import cv2  # noqa: F401
    except ImportError:
        print('    (skipped: no numpy/cv2 here - it runs where ROS is installed)')
        return
    from navigate_behavior import detect as dt
    specs = dt.object_specs(['can'], _param_reader('perception_node'))
    img = np.full((480, 640, 3), 200, np.uint8)
    img[300:340, 280:360] = (26, 89, 217)                   # the can's blue, as RGB
    out = dt.detect(img, 'rgb8', specs, MAST, 120, v_max=330)
    assert out['can']['found'] and out['can']['clipped'], out['can']
    img = np.full((480, 640, 3), 200, np.uint8)
    img[250:290, 280:360] = (26, 89, 217)
    out = dt.detect(img, 'rgb8', specs, MAST, 120, v_max=330)
    assert out['can']['found'] and not out['can']['clipped'], out['can']


def test_the_parked_arms_are_inside_the_base_below_object_height():
    """THE FOOTPRINT THE ROUTES USE. Against objects the robot is its base: every part of the
    parked arms lower than 0.30 m - above the tallest object - is inside the base radius. Against
    walls, taller than the hands, it is the parked arms' whole reach, ARM_RADIUS."""
    caps = (col.arm_capsules_fine(ch.TRAVEL_LEFT, 1, OPEN_POS)
            + col.arm_capsules_fine(ch.TRAVEL_RIGHT, -1, OPEN_POS))
    low = reach = 0.0
    for _n, a, b, r in caps:
        for k in range(41):
            p = [a[i] + k / 40 * (b[i] - a[i]) for i in range(3)]
            e = math.hypot(p[0], p[1]) + r
            reach = max(reach, e)
            if p[2] - r < 0.30:
                low = max(low, e)
    assert max(2 * o['upright_cz'] for o in OBJECTS.values()) < 0.30
    assert low < nav.BASE_RADIUS, f'a parked arm reaches {low:.3f} m below 0.30 m'
    assert reach <= nav.ARM_RADIUS, f'the parked arms reach {reach:.3f} m, past ARM_RADIUS'


def test_a_park_pose_puts_the_object_where_the_stand_up_was_verified():
    """From either side, for any object anywhere at any axis: the object REACH dead ahead with
    its axis exactly across the reach - the geometry the shared stand-up is verified at."""
    import random
    rng = random.Random(5)
    for _ in range(200):
        ox, oy, a = rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0, math.pi)
        for px, py, h in nav.approach_poses((ox, oy), a, nav.REACH):
            dx, dy = ox - px, oy - py
            bx, by = math.cos(h) * dx + math.sin(h) * dy, -math.sin(h) * dx + math.cos(h) * dy
            assert abs(bx - nav.REACH) < 1e-9 and abs(by) < 1e-9
            dr, bearing, dev = nav.park_error(bx, by, a - h)
            assert abs(dr) < 1e-9 and abs(bearing) < 1e-9 and abs(dev) < 1e-9


def test_routes_keep_clear_and_go_round():
    """A route never brings the base edge within OBJECT_CLEAR of an object, nor the parked arms
    within WALL_CLEAR of a wall; it goes round what is in the way; and a goal that is not clear
    has no route at all."""
    room = _room()
    obs = _world_obstacles()
    import random
    rng = random.Random(11)
    n_round = 0
    for _ in range(60):
        a = (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0))
        b = (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0))
        if min(nav.point_clearance(p, obs, room)[0] for p in (a, b)) < nav.OBJECT_CLEAR or \
                min(nav.point_clearance(p, obs, room)[1] for p in (a, b)) < nav.WALL_CLEAR:
            continue
        route = nav.plan_route(a, b, obs, room)
        assert route is not None, f'no route {a} -> {b}'
        g, w, who = nav.route_clearance(route, obs, room)
        assert g >= nav.OBJECT_CLEAR - 1e-9 and w >= nav.WALL_CLEAR - 1e-9, (g, w, who)
        n_round += len(route) > 2
    assert n_round >= 3, 'no route ever had to go round anything - the test proves nothing'
    bx, by = obs['bottle'][0]
    assert nav.plan_route((0.0, 0.0), (bx + 0.1, by), obs, room) is None


def test_every_lying_object_in_this_world_can_be_approached():
    """From the scan point, each lying object has a side to be stood up from, a clear look
    point on it, and a route there - and the park and staging poses are clear of the rest."""
    room = _room()
    obs = _world_obstacles()
    for name, (x, y, _z, roll, _p, yaw) in _world_objects().items():
        if abs(roll) < 0.1:
            continue
        plan = _park(name)
        assert plan['look'] is not None, f'{name}: no look point'
        for key in ('park', 'stage', 'look'):
            gap, wall, who = nav.point_clearance(plan[key][:2], obs, room,
                                                 skip=(name,) if key == 'park' else ())
            assert gap >= nav.OBJECT_CLEAR and wall >= nav.WALL_CLEAR, (name, key, who)
        assert abs(math.dist(plan['park'][:2], (x, y)) - nav.REACH) < 1e-9


def test_the_drive_law_arrives_without_overshooting():
    """Driven by drive_command() on a differential base at 20 Hz, from offsets and headings a
    turn leaves it with, forwards and in reverse: it arrives within 2 mm, never ends up past
    the goal, and never exceeds its speed limit."""
    for gx, gy, h0, rev in ((0.6037, 0.00, 0.0, False), (0.5113, 0.04, 0.05, False),
                            (1.2291, -0.10, -0.08, False), (-0.3519, 0.0, 0.0, True),
                            (0.1207, 0.003, 0.0, False), (0.0413, 0.0, 0.0, False)):
        x, y, h = 0.0, 0.0, h0 + (math.atan2(gy, gx) if not rev else 0.0)
        for _ in range(4000):
            v, w, _along, _lat = nav.drive_command((x, y, h), (gx, gy), 0.15, reverse=rev)
            assert abs(v) <= 0.15 + 1e-12
            if v == 0.0:
                break
            x += v * 0.05 * math.cos(h)
            y += v * 0.05 * math.sin(h)
            h += w * 0.05
        hd = h + (math.pi if rev else 0.0)
        left = (gx - x) * math.cos(hd) + (gy - y) * math.sin(hd)
        assert left > -0.002, f'it overshot the goal ({gx}, {gy}) by {-left * 1000:.1f} mm'
        assert math.hypot(x - gx, y - gy) < 0.003 + abs(gy) * 0.05, (gx, gy, x, y)


def test_the_scan_sees_the_whole_room():
    """The scan headings and the mast camera's field of view leave no bearing unseen: every
    object is well inside the frame from at least one heading, with room for its own width."""
    P = _param_reader('task_node')
    n = int(P('scan_headings', 0))
    half_fov = _param_reader('perception_node')('far_horizontal_fov', 0) / 2
    widest = math.atan2(0.5 * max(o['length'] for o in OBJECTS.values()), nav.FAR_MIN)
    assert math.pi / n + widest < half_fov - math.radians(3), 'the scan leaves gaps'
    for name, (x, y, *_r) in _world_objects().items():
        b = math.atan2(y, x)
        off = min(abs(math.remainder(b - 2 * math.pi * k / n, 2 * math.pi)) for k in range(n))
        assert off <= math.pi / n + 1e-9, name


def test_the_base_only_drives_with_the_arms_parked():
    """Every move of the base starts from the parked pose: the approach parks the arms before
    driving, every look returns them, and the stand-up ends parked. Read from the source: a
    drive is only ever reached through go_to or drive_straight, from cycle, cycles and
    return_home, each of which parks first or follows a routine that ends parked."""
    tn = _src('task_node.py')
    cyc = tn[tn.index('    def cycle(self, name):'):tn.index('\ndef main(')]
    assert cyc.index('self.travel_pose()') < cyc.index('self.go_to('), 'cycle drives before parking'
    home = tn[tn.index('    def return_home(self):'):tn.index('    # ------------- one object')]
    assert home.index('self.travel_pose()') < home.index('self.go_to(')
    st = _src('sense_task.py')
    for fn in ('observe', 'observe_all'):
        body = st[st.index(f'    def {fn}('):]
        body = body[:body.index('\n    def ', 10)]
        assert body.rstrip().splitlines()[-2].strip().startswith('self._look_move(') and \
            "'the parked pose'" in body.rstrip().splitlines()[-2], f'{fn} does not end parked'


def test_the_backup_clears_the_object_before_any_turn():
    """After a stand-up the object stands REACH ahead; turning there would sweep the parked
    hands past it. The backup must leave it clear of the whole parked reach, with margin."""
    P = _param_reader('task_node')
    for o in OBJECTS.values():
        gap = nav.REACH + P('backup', 0) - o['half_width'] - nav.ARM_RADIUS
        assert gap >= 0.10, f'after backing up the parked hands pass it by {gap * 1000:.0f} mm'


def test_the_headless_harness_passes():
    """The REAL task node - scan, drive, look, park, stand up, rescan - against a fake robot
    that drives on wrong odometry and two fake cameras, in three different rooms."""
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HERE, 'harness.py')],
                       capture_output=True, text=True)
    bad = [ln.strip() for ln in r.stdout.splitlines() if '[FAIL]' in ln]
    assert r.returncode == 0 and not bad, ('the headless harness failed:\n  '
                                           + '\n  '.join(bad[:6]) + r.stderr[-300:])


if __name__ == '__main__':
    fails = 0
    for nm, fn in sorted(globals().items()):
        if nm.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS', nm)
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print('FAIL', nm + ':', exc)
    print()
    print('%d checks, %d failed' % (len([n for n in globals() if n.startswith('test_')]), fails))
    sys.exit(1 if fails else 0)
