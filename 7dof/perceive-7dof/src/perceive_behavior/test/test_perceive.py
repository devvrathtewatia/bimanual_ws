"""Offline validation for section C. No ROS, no Gazebo.

    python3 test/test_perceive.py

WHY "IT COULD NOT DETECT ANY OBJECTS"
-------------------------------------
Three faults, one behind the other, none of which the previous suite could see:

1. **The perception node died on the first object it saw.** It read ``lying_cz`` and
   ``upright_cz`` and never loaded them - a KeyError inside the image callback, which left
   rclpy.spin() and killed the node. /targets was never published again and the task logged
   every object NOT SEEN. The camera was fine.
2. **The measurement was of a shape the camera never sees.** It treated each blob as a flat
   rectangle at the object's mid-height. From a camera 71 deg down, a standing bottle is a
   blob 1.7x longer than wide, pointing at the camera - measured as LYING, 59 mm off, with
   its axis pointing at the robot, and then rejected as ungraspable. The old suite tested
   the measurement against the same flat rectangle, so it tested itself.
3. **The robot's own parked hands hide the workspace.** They sit between the camera and
   the near half of anything in front, so a lying object loses an end behind a hand.

Every check below that is about perception drives it with what the camera REALLY sees:
the solid's silhouette, and - where OpenCV is available - a rendered image.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from perceive_behavior import choreography as ch
from perceive_behavior import collision as col
from perceive_behavior import ik
from perceive_behavior import motion as mo
from perceive_behavior import vision as vz

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SRC = os.path.dirname(PKG)
ROOT = os.path.dirname(SRC)
CFG = os.path.join(PKG, 'config', 'task.yaml')
BEH = os.path.join(PKG, 'perceive_behavior')
WORLD = os.path.join(SRC, 'perceive_gazebo', 'worlds', 'perceive.sdf')
URDF = os.path.join(SRC, 'perceive_description', 'urdf')
LAUNCH = os.path.join(SRC, 'perceive_gazebo', 'launch', 'perceive.launch.py')

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
    from perceive_behavior import detect as dt
    from perceive_behavior.looking import LOOK_LEFT, LOOK_RIGHT
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


def test_the_camera_config_is_the_urdf():
    """Perception's camera must BE the camera in the URDF: pose, pitch, field of view, size."""
    xac = open(os.path.join(URDF, 'mast_sensors.xacro')).read()
    m = re.search(r'<joint name="workspace_cam_joint".*?<origin xyz="([^"]+)" rpy="([^"]+)"',
                  xac, re.S)
    assert m, 'workspace_cam_joint not found'
    xyz = [float(v) for v in m.group(1).split()]
    rpy = [float(v) for v in m.group(2).split()]
    sen = xac[xac.index('<sensor name="workspace_cam"'):]
    fov = float(re.search(r'<horizontal_fov>([0-9.]+)<', sen).group(1))
    w = int(re.search(r'<width>(\d+)<', sen).group(1))
    h = int(re.search(r'<height>(\d+)<', sen).group(1))
    P = _param_reader('perception_node')
    for key, want in (('cam_x', xyz[0]), ('cam_y', xyz[1]), ('cam_z', xyz[2]),
                      ('cam_pitch', rpy[1]), ('horizontal_fov', fov), ('image_width', w),
                      ('image_height', h)):
        assert abs(P(key, float('nan')) - want) < 1e-9, f'{key}: config vs URDF {want}'
    assert re.search(r'<topic>workspace_cam</topic>', sen), 'the sensor topic changed'
    assert "'/workspace_cam@sensor_msgs/msg/Image" in open(LAUNCH).read(), 'not bridged'


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
    from perceive_behavior.looking import LOOK_LEFT, LOOK_RIGHT

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
    from perceive_behavior.looking import LOOK_LEFT, LOOK_RIGHT
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
    assert 'class TaskNode(SenseTask)' in tn and 'return self.stand_up(' in tn
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
    for fn in ('sense', 'send_pair', 'settle_arms', 'grippers', 'set_grasp', 'check_grip'):
        body = re.search(r'    def %s\(.*?(?=\n    def |\n    @|\Z)' % fn, src, re.S).group(0)
        assert 'time.time()' not in body, f'{fn}() times itself on the wall clock'
        assert 'self._now()' in body or 'self._sleep(' in body, f'{fn}() is not on sim time'


def test_the_config_holds_no_object_positions():
    """Where an object is, and how it lies, is for the camera to say."""
    blk = _cfg_block('task_node')
    allowed = {'look_azimuth', 'length', 'half_width', 'round', 'mass', 'surface_z', 'cz',
               'upright_cz', 'close'}
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


def test_the_world_is_what_the_stand_up_was_verified_for():
    """The world puts the objects where the robot can look for them and stand them up:
    on their surface, at the verified reach, near their look bearing, lying inside the grasp
    envelope - and 65 deg apart, so a standing neighbour is clear of the lower hand."""
    P = _param_reader('task_node')
    wo = _world_objects()
    assert set(wo) == set(OBJECTS)
    for name, (x, y, z, roll, _p, yaw) in wo.items():
        o = OBJECTS[name]
        blk = re.search(rf'^    {name}:\n((?:      .+\n)+)', _cfg_block('task_node'), re.M)
        look = math.radians(float(re.search(r'look_azimuth:\s*([-0-9.]+)', blk.group(1)).group(1)))
        assert abs(math.hypot(x, y) - REACH) < 0.005, f'{name} is not at the verified reach'
        assert abs(math.remainder(math.atan2(y, x) - look, 2 * math.pi)) < math.radians(3)
        standing = abs(roll) < 0.1
        assert abs(z - (o['upright_cz'] if standing else o['cz'])) < 1e-6, f'{name}: height'
        if not standing:
            psi = math.remainder(yaw - math.pi / 2 - math.atan2(y, x), math.pi) % math.pi
            assert abs(psi - math.pi / 2) <= math.radians(P('max_axis_dev_deg', 20)), name
    az = sorted(math.degrees(math.atan2(v[1], v[0])) for v in wo.values())
    assert min(b - a for a, b in zip(az, az[1:])) >= 60.0, f'objects only {az} apart'


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
        pose, sz = boxes[{'back_wall': 'backwall', 'side_wall': 'sidewall'}[f]]
        assert max(abs(a - b) for a, b in zip(centre, pose[:3])) < 1e-9, f
        assert max(abs(a - b) for a, b in zip(size, sz)) < 1e-9, f


def test_the_wall_constants_are_the_faces_in_the_world():
    from perceive_behavior import wall_ref as wr
    sdf = open(WORLD).read()

    def box(name):
        m = re.search(r'<collision name="%s"><pose>([^<]+)</pose>\s*<geometry><box><size>'
                      r'([^<]+)</size>' % name, sdf)
        return [float(v) for v in m.group(1).split()], [float(v) for v in m.group(2).split()]
    (ax, *_), (sx, *_) = box('backwall')
    (_bx, by, *_), (_tx, ty, *_) = box('sidewall')
    assert abs(wr.WALL_A_X + (ax + sx / 2.0)) < 1e-9 and abs(wr.WALL_B_Y + (by + ty / 2.0)) < 1e-9


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
        sdf = open(os.path.join(SRC, 'perceive_gazebo', 'models', m, 'model.sdf')).read()
        assert 'PosePublisher' in sdf, f'{m} publishes no ground truth'


def test_every_sensor_the_nodes_subscribe_to_is_bridged():
    launch = open(LAUNCH).read()
    for topic in ('/workspace_cam@', '/workspace_cam_info@', '/imu@', '/ft/left@', '/ft/right@',
                  '/scan@', '/odom@', '/clock@', '/model/perceive_bimanual/pose@'):
        assert topic in launch, f'{topic.rstrip("@")} is not bridged'
    assert "executable='perception_node'" in launch


def _plan(name):
    """The stand-up of a perceived object, planned as the node plans it, with the standing
    bottle and the other lying object and the walls in the world."""
    wo = _world_objects()
    x, y, _z, _r, _p, yaw = wo[name]
    h = math.atan2(y, x)

    def base(wx, wy):
        return (math.cos(h) * wx + math.sin(h) * wy, -math.sin(h) * wx + math.cos(h) * wy)
    o = OBJECTS[name]
    psi = (yaw - math.pi / 2 - h) % math.pi
    others = []
    for n, (ox, oy, oz, roll, _pp, oyaw) in wo.items():
        if n == name:
            continue
        on = OBJECTS[n]
        bx, by = base(ox, oy)
        if abs(roll) < 0.1:
            others += col.object_capsules((bx, by, oz), (0, 0, 1), on['length'],
                                          on['half_width'], on['round'])
        else:
            a = oyaw - math.pi / 2 - h
            others += col.object_capsules((bx, by, oz), (math.cos(a), math.sin(a), 0),
                                          on['length'], on['half_width'], on['round'])
    walls = [col.box_obstacle((*base(-0.46, 0.0), 0.45), -h, (0.02, 0.9, 0.9)),
             col.box_obstacle((*base(-0.18, -0.46), 0.45), -h, (0.56, 0.02, 0.9))]
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
    """The shared stand-up, at the perceived geometry, with THIS world round it: bodies clear,
    and along the real path the standing bottle and the other objects at least 20 mm clear."""
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
    ctl = open(os.path.join(SRC, 'perceive_description', 'config', 'controllers.yaml')).read()
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


def test_the_headless_harness_passes():
    """The REAL task node - survey, sense, decide, stand up - against a fake robot and camera."""
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
