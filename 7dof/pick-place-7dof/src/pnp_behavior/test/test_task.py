"""Offline validation for section A. No ROS, no Gazebo.

    python3 test/test_task.py

WHAT CHANGED, AND WHY IT MATTERS
--------------------------------
The first version of this suite passed every check and the section still failed in
simulation, because it modelled each gripper as a POINT. It verified that the two
grasp points were reachable and more than 50 mm apart, while the real hand is a body
140 mm across with fingers reaching 49 mm past the grasp point. It could not see
that those fingers were being driven 25–35 mm into the floor on every grasp.

Every check below that says "clearance" therefore runs against ``collision.py``,
which models each link as a capsule and tests it against the other arm, the floor,
the base and the sensor mast. Reachability alone is no longer accepted as proof.

The values here mirror ``config/targets.yaml``; ``test_config_matches_this_file``
fails if the two drift apart.
"""
import math
import ast
import os
import xml.etree.ElementTree as ET
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pnp_behavior import choreography as ch
from pnp_behavior import collision as col
from pnp_behavior import ik
from pnp_behavior import motion as mo
from pnp_behavior import wall_ref as wr

CFG = os.path.join(os.path.dirname(__file__), '..', 'config', 'targets.yaml')

REACH = 0.30
LATERAL = math.pi / 2
OPEN_POS = 0.065
MIN_CLEARANCE = 0.004        # 4 mm on a conservative capsule model
HALF_WIDTH = 0.045      # every object is 90 mm across
PICK_PSI = {'bottle': 72.0, 'can': 96.0, 'box': 108.0}     # object axis in the BASE frame, deg
PLACE_PSI = 90.0                             # tangential to the slot, always
# HEIGHTS FROM SURFACES: each object lies on surface_z (the floor) and is put on
# place_surface_z (the shelf top, 0.100 since 2026-09-28); cz is its centre height above
# either. place_z - the resting centre height on the shelf - is DERIVED, never configured.
SHELF_TOP = 0.100
OBJECTS = {
    'bottle': dict(cz=0.045, surface_z=0.0, place_surface_z=SHELF_TOP, length=0.240,
                   close=0.047, grip_dz=0.025, half_width=0.045, place_azimuth=130.0,
                   pick_azimuth=0.0, axis_yaw=72.0, round=True, mass=0.45),
    'can': dict(cz=0.045, surface_z=0.0, place_surface_z=SHELF_TOP, length=0.200,
                close=0.048, grip_dz=0.025, half_width=0.045, place_azimuth=180.0,
                pick_azimuth=55.0, axis_yaw=151.0, round=True, mass=0.35),
    'box': dict(cz=0.045, surface_z=0.0, place_surface_z=SHELF_TOP, length=0.220,
                close=0.048, grip_dz=0.025, half_width=0.045, place_azimuth=230.0,
                pick_azimuth=-55.0, axis_yaw=53.0, round=False, mass=0.50),
}
for _o in OBJECTS.values():
    _o['place_z'] = _o['place_surface_z'] + _o['cz']
BASE_RADIUS = 0.1775


def _binding_half_width(o):
    """The widest part of the object that the FINGER PLATE can reach.

    Not the width at the grasp height. The finger is a plate spanning roughly 10 to
    45 mm above the object's axis, so on a cylinder it straddles the shoulder and the
    binding constraint is the widest point inside that span - which is at the BOTTOM of
    it. Sizing the close position from the width at the grasp height instead gave
    close = 0.040 and drove the finger's lower edge 9.9 mm into the object.

    This is the same mistake as the original point model of the gripper, in a smaller
    place: a body with extent was treated as a single coordinate.
    """
    if not o['round']:
        return o['half_width']              # flat sides: same width at every height
    lo = o['grip_dz'] - (col.FINGERTIP_X - col.LG)
    lo = max(0.0, min(lo, o['half_width']))
    return math.sqrt(max(0.0, o['half_width'] ** 2 - lo ** 2))


def test_fingers_do_not_gouge_into_the_object():
    """The finger's lower edge must not be driven into a curved shoulder."""
    for name, o in OBJECTS.items():
        face = o['close'] - 0.006
        bind = _binding_half_width(o)
        pen = bind - face
        assert pen <= 0.004, (
            f'{name}: finger face at {face:.4f} against a binding half-width of '
            f'{bind:.4f} - {pen * 1000:.1f} mm of penetration')
        assert pen >= 0.001, (
            f'{name}: finger face at {face:.4f} never reaches {bind:.4f}, no grip')


def test_grip_holds_the_object_once_the_weld_is_released():
    """During the settle the weld is off, so friction alone carries the object.

    Gripping above a cylinder's centre means the finger meets a sloped surface, so the
    squeeze pushes DOWN as well as in. That wedging force adds to the object's weight,
    and with the wrong close position it exceeded what the fingers could hold.
    """
    n_faces, force, mu = 4, 3.0, 1.0        # two fingers per hand, two hands
    hold = mu * force * n_faces
    for name, o in OBJECTS.items():
        face = o['close'] - 0.006
        if o['round']:
            h = math.sqrt(max(0.0, o['half_width'] ** 2 - face ** 2))
            phi = math.asin(min(1.0, h / o['half_width']))
        else:
            phi = 0.0                        # flat sides wedge not at all
        wedge = force * n_faces * math.sin(phi)
        need = o['mass'] * 9.81 + wedge
        assert hold > need * 1.15, (
            f'{name}: needs {need:.2f} N (weight {o["mass"] * 9.81:.2f} + wedge '
            f'{wedge:.2f}) but the fingers hold only {hold:.2f} N')


def test_palm_clears_the_top_of_the_object():
    """The pickup lift: the palm was being driven into the object.

    The palm plate sits ``LG - 0.032 - 0.011 = 0.027`` m above the grasp point. Aiming
    at a 0.045 m object centre put its underside at 0.072 while the object's top was at
    0.090 - 18 mm of interpenetration on every single grasp. The arm pushed a rigid body
    down against the floor and the reaction tipped the base onto its casters.

    Note there is NO object height that fixes this while grasping at the centre: the
    fingertips need cz > 0.031 to clear the floor and the palm needs cz < 0.022 to clear
    the top. The grasp point itself had to move up.
    """
    palm_above_grasp = col.LG - 0.032 - 0.011
    for name, o in OBJECTS.items():
        top = o['cz'] + o['half_width']
        grasp = o['cz'] + o['grip_dz']
        assert grasp + palm_above_grasp > top + 0.004, (
            f'{name}: palm underside {grasp + palm_above_grasp:.3f} m against an '
            f'object top of {top:.3f} m')
        # and the fingers must still reach far enough down to actually hold it
        fingertip = grasp - (col.FINGERTIP_X - col.LG)
        assert fingertip < top - 0.015, (
            f'{name}: fingers only reach {fingertip:.3f} m, object top {top:.3f}')
        assert fingertip > 0.010, f'{name}: fingertip {fingertip:.3f} m near the floor'


SHELF_SDF = os.path.join(os.path.dirname(__file__), '..', '..', 'pnp_gazebo',
                         'models', 'shelf', 'model.sdf')
PED_TOP = SHELF_TOP
PED_HALF_RADIAL = 0.090   # pedestal is 0.18 radial; see the shelf SDF


def _yaml_float(key):
    """Read a scalar out of the real config, so tests cannot drift from the run."""
    cfg = os.path.join(os.path.dirname(__file__), '..', 'config', 'targets.yaml')
    with open(cfg) as fh:
        for line in fh:
            m = re.match(rf'\s*{re.escape(key)}:\s*([-0-9.]+)', line)
            if m:
                return float(m.group(1))
    raise AssertionError(f'{key} not found in targets.yaml')


def _shelf_ramps():
    """Reconstruct every barrier's top surface from the SDF, in (radial, z).

    Reads the real file rather than repeating the design constants, so if the SDF and
    the intent ever diverge the test fails instead of agreeing with itself.

    A pose of ``0 pitch yaw`` gives R = Rz(yaw)*Ry(pitch), and the pedestals are laid
    out with yaw = slot azimuth, so the plate's local +x maps to the outward radial
    direction and its local +z to the surface normal. A local point (a, 0, c) therefore
    sits at radial ``a*cos(p) + c*sin(p)`` and height ``-a*sin(p) + c*cos(p)``
    relative to the plate centre.
    """
    tree = ET.parse(SHELF_SDF)
    link = tree.getroot().find('model').find('link')
    out = []
    for col in link.findall('collision'):
        name = col.get('name')
        if not name.startswith('ramp'):
            continue
        px, py, pz, roll, pitch, yaw = [float(v) for v in
                                        col.find('pose').text.split()]
        ls, _ly, t = [float(v) for v in
                      col.find('geometry').find('box').find('size').text.split()]
        assert abs(roll) < 1e-9, f'{name}: unexpected roll'
        slot_r = 0.30
        cxs, cys = slot_r * math.cos(yaw), slot_r * math.sin(yaw)
        # radial offset of the plate centre from its slot centre
        c_rad = ((px - cxs) * math.cos(yaw) + (py - cys) * math.sin(yaw))
        corners = []
        for a in (-ls / 2, ls / 2):                    # top face only: c = +t/2
            c = t / 2
            corners.append((c_rad + a * math.cos(pitch) + c * math.sin(pitch),
                            pz - a * math.sin(pitch) + c * math.cos(pitch)))
        mu = float(col.find('surface').find('friction').find('ode').find('mu').text)
        # foot = the end nearer the slot centre in |radial|; crest = the other
        corners.sort(key=lambda rc: abs(rc[0]))
        out.append({'name': name, 'foot': corners[0], 'crest': corners[1],
                    'pitch': pitch, 'mu': mu, 'thick': t})
    return out


def test_shelf_barriers_exist_on_every_slot():
    """Cylinders roll. Removing the barriers is not an acceptable fix."""
    ramps = _shelf_ramps()
    assert len(ramps) == 6, f'expected 2 barriers per slot, found {len(ramps)}'
    for i in (1, 2, 3):
        pair = [r for r in ramps if r['name'].startswith(f'ramp{i}')]
        assert len(pair) == 2, f'slot {i} has {len(pair)} barriers, needs 2'


def test_shelf_barriers_are_properly_chamfered():
    """No barrier may present a horizontal face or a square edge to the object.

    This is the regression test for the worst self-inflicted failure in the project.
    The first barriers were plain rectangular rails: they put a flat top face with a
    square edge inside the region the object is lowered into, the object landed on that
    face instead of the pedestal, and the arm went on pressing a rigid body against a
    rigid obstruction until the robot tipped onto its casters. The barrier is now a
    ramp, so the only surface reachable from above is inclined and converts the arm's
    downward motion into a sideways push that seats the object.
    """
    for r in _shelf_ramps():
        n = r['name']
        foot_r, foot_z = r['foot']
        crest_r, crest_z = r['crest']
        # 1. It must actually be inclined - a level plate is the old failure.
        assert abs(r['pitch']) > math.radians(20), (
            f'{n}: pitch {math.degrees(r["pitch"]):.1f} deg is nearly flat')
        # 2. It must rise going AWAY from the slot centre, or it funnels outward.
        assert abs(crest_r) > abs(foot_r) and crest_z > foot_z, (
            f'{n}: surface does not rise outward '
            f'(foot {foot_r:+.4f}/{foot_z:.4f}, crest {crest_r:+.4f}/{crest_z:.4f})')
        # 3. The foot must meet the pedestal surface with no step to catch on.
        assert foot_z <= PED_TOP + 0.001, (
            f'{n}: foot sits {(foot_z - PED_TOP) * 1000:.1f} mm above the surface, '
            f'which is a new square edge')
        # 4. A correctly placed object must touch nothing.
        assert abs(foot_r) - HALF_WIDTH >= 0.008, (
            f'{n}: only {(abs(foot_r) - HALF_WIDTH) * 1000:.1f} mm clearance at the foot')
        # 5. The mouth must absorb far more error than we actually have (2.7 mm).
        assert abs(crest_r) - HALF_WIDTH >= 0.025, (
            f'{n}: captures only {(abs(crest_r) - HALF_WIDTH) * 1000:.1f} mm of error')
        # 6. It must stay on the pedestal.
        assert abs(crest_r) < PED_HALF_RADIAL - 0.002, (
            f'{n}: crest at {abs(crest_r):.4f} overhangs the pedestal edge')
        # 7. The object must slide back DOWN the chamfer, not stick on it.
        assert r['mu'] < math.tan(abs(r['pitch'])) - 0.15, (
            f'{n}: mu {r["mu"]} too high to slide on a '
            f'{math.degrees(abs(r["pitch"])):.0f} deg slope')


TASK_NODE = os.path.join(os.path.dirname(__file__), '..', 'pnp_behavior',
                         'task_node.py')
NODE_BASE = os.path.join(os.path.dirname(__file__), '..', 'pnp_behavior',
                         'node_base.py')


def _node_src():
    """What the running node is made of: this section's task_node.py AND the shared
    node_base.py it inherits every primitive from."""
    return open(TASK_NODE).read() + '\n' + open(NODE_BASE).read()


def _object_dict_keys():
    """The keys the node puts into each object's dict: the shared object_params() in
    node_base.py plus this section's additions in task_node.py, read from source."""
    base = open(NODE_BASE).read()
    seg = re.search(r'    def object_params\(self, n\):.*?\n        }\n', base, re.S)
    assert seg, 'could not find object_params() in node_base.py'
    keys = set(re.findall(r"^\s*'(\w+)':\s*", seg.group(0), re.M))
    tn = open(TASK_NODE).read()
    seg2 = re.search(r'    def object_params\(self, n\):.*?return o\n', tn, re.S)
    assert seg2, 'could not find object_params() in task_node.py'
    keys |= set(re.findall(r"o\['(\w+)'\]\s*=", seg2.group(0)))
    return keys, _node_src()


LAUNCH = os.path.join(os.path.dirname(__file__), '..', '..', 'pnp_gazebo',
                      'launch', 'pnp.launch.py')


def test_contact_topics_use_the_name_gazebo_actually_publishes():
    """Gazebo IGNORES <topic> for contact sensors and names them itself.

    The first attempt bridged /contact/<finger>, which nothing ever published, so every
    finger reported "0/2 touching" while the joint stall said +4.3 mm. The sensor was
    wrong and the older, indirect check was right. Gazebo prints the real name at
    startup:

        world/<world>/model/<model>/link/<link>/sensor/<sensor>/contact

    This asserts the bridge uses that shape, and that it is remapped to the short name
    the task node subscribes to. A harness cannot catch this - Gazebo is not running -
    so it is checked structurally instead.
    """
    launch = open(LAUNCH).read()
    world, model = 'pick_place', 'pnp_bimanual'
    fingers = ['left_finger_l', 'left_finger_r', 'right_finger_l', 'right_finger_r']
    for f in fingers:
        gz = (f'world/{world}/model/{model}/link/{f}_link'
              f'/sensor/{f}_contact/contact')
        assert gz in launch, (
            f'{f}: the bridge does not use the name Gazebo generates.\n'
            f'  expected to find: {gz}')
        assert f"'/contact/{f}'" in launch, (
            f'{f}: bridged but never remapped to /contact/{f}, which is what the task '
            f'node subscribes to')
    # and the task node must subscribe to the short names
    src = _node_src()
    assert "f'/contact/{f}'" in src or '/contact/' in src, (
        'the task node does not subscribe to any contact topic')


def test_sensor_topics_and_subscriptions_agree():
    """Every topic the node subscribes to must be bridged, and vice versa.

    A subscription to a topic nothing publishes is silent: the value simply stays at
    its default, which is how "0/2 touching" looked like a real measurement.
    """
    launch = open(LAUNCH).read()
    src = _node_src()
    # topics are sometimes built with an f-string (f'/ft/{side}'), so match the stem
    for topic, stem in (('/imu', '/imu'), ('/ft/left', '/ft/'),
                        ('/ft/right', '/ft/'), ('/odom', '/odom'),
                        ('/scan', '/scan')):
        assert stem in src, f'the node does not subscribe to anything matching {stem}'
        assert topic.lstrip('/') in launch, (
            f'{topic} is not bridged in the launch file - a subscription to an '
            f'unbridged topic is silent, the value just stays at its default')


def test_the_headless_harness_passes():
    """Runs the REAL task_node cycle against a fake robot, in this process.

    This is the check that was missing for four days. Every geometry test in this file
    works on its own fixtures and never imports task_node, so four bugs reached the
    user's laptop that could only surface by running the node: a label string passed
    where a float belonged, a missing sign_left, three attributes read but never
    assigned, and a yaw alignment that commanded 148 degrees. All four are caught by
    this in seconds.

    It is not a physics test. It answers one question - does the code do what it thinks
    it does - and says nothing about contact dynamics.
    """
    import subprocess
    import sys
    r = subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(__file__), 'harness.py')],
                       capture_output=True, text=True)
    bad = [ln.strip() for ln in r.stdout.splitlines() if '[FAIL]' in ln]
    assert r.returncode == 0 and not bad, (
        'the headless harness failed:\n  ' + '\n  '.join(bad[:6])
        + (('\n  stderr: ' + r.stderr.strip()[-300:]) if r.stderr.strip() else ''))


def test_the_robot_cannot_see_its_own_arms():
    """The fault that made "rotate to an angle" fail for four days.

    The LiDAR sat at z=0.460 and the robot's own wrists sit at z=0.440, 0.26 m away -
    inside the 0.15..1.60 m band the wall fit accepts. So the scan contained the hands as
    if they were wall, the line fit reported rms 105 mm and was rejected, face() gave up
    after one open-loop turn, and the robot landed 23 degrees off. Because the arms are
    fixed in the BASE frame, which wall's sector they polluted changed with heading -
    which is why wall A failed at some headings and wall B at others, and why it looked
    random.

    The sensor is now above everything the arms can reach. This asserts that, for the
    travel pose AND for the worst case of both arms fully extended upward.
    """
    import re as _re
    urdf = open(os.path.join(os.path.dirname(__file__), '..', '..',
                             'pnp_description', 'urdf', 'robot.urdf.xacro')).read()
    m = _re.search(r'name="lidar_z"\s+value="([0-9.]+)"', urdf)
    assert m, 'lidar_z not found'
    lidar_z = float(m.group(1))

    # highest point either arm reaches at the travel pose
    worst = 0.0
    for q, side in ((ch.TRAVEL_LEFT, +1), (ch.TRAVEL_RIGHT, -1)):
        for _n, a, b, r in col.arm_capsules(q, side):
            worst = max(worst, a[2] + r, b[2] + r)
    assert lidar_z > worst + 0.05, (
        f'the LiDAR at {lidar_z:.3f} is only {(lidar_z - worst) * 1000:.0f} mm above the '
        f'highest arm point at the travel pose ({worst:.3f}) - it will scan its own hands')

    # and the absolute worst case: shoulder height plus full extension plus the hand
    reach_up = 0.240 + ik.L1 + ik.L2 + 0.05
    assert lidar_z > reach_up, (
        f'the LiDAR at {lidar_z:.3f} is below the {reach_up:.3f} the arms can reach, so '
        f'some arm pose will put a link in the scan plane')

    # the walls must be TALLER than the sensor or it sees nothing at all
    shelf = open(SHELF_SDF).read()
    for m2 in _re.finditer(r'name="(?:backwall|sidewall)"><pose>[-0-9. ]+</pose>\s*'
                           r'<geometry><box><size>([0-9. ]+)</size>', shelf):
        h = float(m2.group(1).split()[2])
        assert h > lidar_z + 0.05, (
            f'a wall is only {h:.2f} tall against a LiDAR at {lidar_z:.2f} - the beam '
            f'passes over it and there is nothing to fit')


def test_the_lift_has_margin_for_the_aim_correction():
    """The regression that cost a run: the lift had FOUR mm of margin.

    The lift is the phase that binds reach. At a nominal 0.300 the bottle's wrist hit
    exactly 0.320 against a 0.320 limit - r=0.302 worked and r=0.305 failed. But the aim
    correction deliberately moves the target by up to +/-10 mm to absorb base drift, so a
    4 mm margin against a 10 mm correction is a coin flip: one run the bottle lost it and
    the box won, which looked like a rotation fault and was not.

    This requires the lift to survive the whole correction range, so the nominal reach can
    never again be set where a legitimate correction breaks it.
    """
    # With L=0.180 the reach limit is 0.360, so the lift has real headroom now.
    # Verified against 15 mm, which is more than the aim correction can apply.
    drift = 0.015
    for name, o in OBJECTS.items():
        s_ = _s(o)
        for r in (REACH - drift, REACH, REACH + drift):
            for psi in (_psi(name), math.radians(PLACE_PSI)):
                seq = ch.lift_waypoints(r, 0.0, o['cz'] + o['grip_dz'], psi, s_,
                                        ch.left_sign(psi))
                for (pl, rl), (pr, rr) in seq:
                    try:
                        ik.solve(pl, rl, +1)
                        ik.solve(pr, rr, -1)
                    except ik.IKError as exc:
                        raise AssertionError(
                            f'{name}: the lift fails at aim radius {r:.3f} '
                            f'(nominal {REACH:.3f} +/- {drift * 1000:.0f} mm), '
                            f'psi {math.degrees(psi):.0f} deg: {exc}')
    # and the clamp must sit inside the range that actually works
    m = re.search(r'^\s*reach_max:\s*([0-9.]+)', open(
        os.path.join(os.path.dirname(__file__), '..', 'config',
                     'targets.yaml')).read(), re.M)
    assert m, 'reach_max missing from the config'
    assert float(m.group(1)) <= REACH + drift, (
        f'reach_max {m.group(1)} is outside the verified range')


def test_the_object_list_agrees_everywhere():
    """Four files name the object set, and a mismatch is silent and expensive.

    The can was removed from the world and the config but the robot's
    gazebo_plugins.xacro still declared a DetachableJoint for it. The plugin then logged
    "child model could not be found" on every update - hundreds of lines - and a weld
    system that is erroring is not one the real grasps should depend on. Nothing caught
    it because each file was individually valid.
    """
    here = os.path.dirname(__file__)
    cfg = open(os.path.join(here, '..', 'config', 'targets.yaml')).read()
    world = open(os.path.join(here, '..', '..', 'pnp_gazebo', 'worlds',
                              'pick_place.sdf')).read()
    launch = open(os.path.join(here, '..', '..', 'pnp_gazebo', 'launch',
                               'pnp.launch.py')).read()
    plug = open(os.path.join(here, '..', '..', 'pnp_description', 'urdf',
                             'gazebo_plugins.xacro')).read()

    m = re.search(r'^    order:\s*\[([^\]]+)\]', cfg, re.M)
    assert m, 'order not found in targets.yaml'
    want = sorted(x.strip() for x in m.group(1).split(','))

    spawned = sorted(set(re.findall(r'<uri>model://(\w+)</uri>', world)) - {'shelf'})
    bridged = sorted(re.findall(r"'(\w+)'",
                                re.search(r'for obj in \(([^)]*)\)', launch).group(1)))
    welded = sorted(re.findall(r'grasp_joint obj="(\w+)"', plug))
    verify = sorted(x.strip() for x in
                    re.search(r'^    objects:\s*\[([^\]]+)\]', cfg, re.M)
                    .group(1).split(','))

    assert spawned == want, f'world spawns {spawned}, config order is {want}'
    assert bridged == want, f'launch bridges {bridged}, config order is {want}'
    assert welded == want, (
        f'gazebo_plugins declares welds for {welded} but the object set is {want} - '
        f'a weld for a model that does not exist logs "child model could not be found" '
        f'on every update')
    assert verify == want, f'verify_node checks {verify}, config order is {want}'


def test_task_node_sets_every_object_key_it_reads():
    """Guards the failure that killed section A outright.

    A patch added the line that READS ``o['grip_dz']`` but the companion line that
    PUTS grip_dz into the dict silently failed to apply - it searched for ``{n}`` where
    the loop variable is ``name``, and str.replace reports no error when it matches
    nothing. Result: KeyError on the first cycle, before any motion was commanded, and
    all three objects failed instantly.

    Nothing else caught it. A missing dict key is a runtime error, so py_compile and
    ast.parse pass. And this suite defines its own OBJECTS dict, so every geometry test
    kept passing against test data while the node itself could not start a cycle - the
    tests were testing the tests. This check reads task_node.py's real source instead.
    """
    set_keys, src = _object_dict_keys()
    read_keys = set(re.findall(r"\bo\['(\w+)'\]", src))
    missing = read_keys - set_keys
    assert not missing, (
        f'task_node reads {sorted(missing)} off each object but never sets them; '
        f'it sets {sorted(set_keys)}')


def test_every_config_key_is_loaded_and_every_loaded_key_is_configured():
    """The config and the node must agree in BOTH directions.

    A key in the YAML that the node never loads is dead config that looks live - you
    change it, nothing happens, and you spend a run wondering why. A key the node loads
    that the YAML lacks is only as safe as its hardcoded default, which then silently
    disagrees with everything documented here.
    """
    set_keys, src = _object_dict_keys()
    cfg = open(os.path.join(os.path.dirname(__file__), '..', 'config',
                            'targets.yaml')).read()
    # per-object keys are indented six spaces under each object name
    # per-object keys only - the fixture blocks are indented the same way
    order = re.search(r'^    order:\s*\[([^\]]+)\]', cfg, re.M).group(1)
    cfg_keys = set()
    for name in (v.strip() for v in order.split(',')):
        blk = re.search(rf'^    {name}:\n((?:      .+\n|\s*\n)+)', cfg, re.M)
        cfg_keys |= set(re.findall(r'^      (\w+):', blk.group(1), re.M))
    assert not (cfg_keys - set_keys), (
        f'in targets.yaml but never loaded by task_node: '
        f'{sorted(cfg_keys - set_keys)}')
    assert not (set_keys - cfg_keys), (
        f'loaded by task_node but absent from targets.yaml (running on defaults): '
        f'{sorted(set_keys - cfg_keys)}')


def test_the_test_objects_match_the_shipped_config():
    """This file's OBJECTS dict must mirror targets.yaml.

    Every geometry test in here runs against OBJECTS, not against the config the robot
    actually loads. If the two drift, the suite certifies a machine that does not exist.
    """
    cfg = open(os.path.join(os.path.dirname(__file__), '..', 'config',
                            'targets.yaml')).read()
    for name, o in OBJECTS.items():
        block = re.search(rf'^    {name}:\n((?:      .+\n)+)', cfg, re.M)
        assert block, f'{name} missing from targets.yaml'
        body = block.group(1)
        for key in ('cz', 'surface_z', 'place_surface_z', 'length', 'close', 'grip_dz',
                    'axis_yaw', 'pick_azimuth', 'place_azimuth', 'half_width', 'mass'):
            m = re.search(rf'{key}:\s*([-0-9.]+)', body)
            assert m, f'{name}.{key} missing from targets.yaml'
            assert abs(float(m.group(1)) - o[key]) < 1e-9, (
                f'{name}.{key}: config {m.group(1)} vs this file {o[key]}')


def _world_corners(pose, size):
    """All eight world corners of an SDF box whose pose is ``0 pitch yaw``."""
    px, py, pz, roll, pitch, yaw = pose
    sx, sy, sz = size
    assert abs(roll) < 1e-9, 'roll is assumed zero in this shelf'
    cy, sy_, cp, sp = (math.cos(yaw), math.sin(yaw),
                       math.cos(pitch), math.sin(pitch))
    m = [[cy * cp, -sy_, cy * sp],
         [sy_ * cp, cy, sy_ * sp],
         [-sp, 0.0, cp]]
    out = []
    for a in (-sx / 2, sx / 2):
        for b in (-sy / 2, sy / 2):
            for c in (-sz / 2, sz / 2):
                out.append(tuple(
                    (px, py, pz)[k] + m[k][0] * a + m[k][1] * b + m[k][2] * c
                    for k in range(3)))
    return out


def _shelf_boxes():
    tree = ET.parse(SHELF_SDF)
    link = tree.getroot().find('model').find('link')
    out = {}
    for col in link.findall('collision'):
        pose = [float(v) for v in col.find('pose').text.split()]
        size = [float(v) for v in
                col.find('geometry').find('box').find('size').text.split()]
        out[col.get('name')] = (pose, size)
    return out


def test_barriers_do_not_reach_into_another_slot():
    """A barrier must not stand above the surface inside a NEIGHBOUR's landing zone.

    This is the fault the per-slot chamfer test is structurally unable to see. Every
    quantity that test examines is expressed relative to the barrier's own slot, using
    that barrier's own yaw, so it would agree with itself even if the barrier were
    pointing the wrong way entirely. The fault lives BETWEEN slots and only shows up in
    world coordinates.

    What was wrong: adjacent slot centres are 0.2536 m apart (50 deg at a 0.30 m
    radius) while the barriers were 0.300 m long, so each one reached 3.6 mm into the
    next slot's footprint, standing 9.4 mm proud of the surface. An object lowered into
    slot 2 would land on slot 1's barrier - and that is worse than landing on its own,
    because slot 1's chamfer is oriented for slot 1 and pushes the object AWAY from the
    slot it is entering. The arm then presses a rigid object against a rigid
    obstruction, which is the lift-off during placement.

    Margin at the chosen 0.260 length is +11.7 mm; intrusion starts at 0.290.
    """
    boxes = _shelf_boxes()
    slots = {i: boxes[f'ped{i}'][0] for i in (1, 2, 3)}
    obj_r, obj_t = HALF_WIDTH, 0.120        # object half-footprint on the shelf
    worst = 9.9
    for i in (1, 2, 3):
        for side in ('o', 'i'):
            pose, size = boxes[f'ramp{i}{side}']
            # the barrier must share its slot's yaw, or it is not on its slot at all
            assert abs(pose[5] - slots[i][5]) < 1e-6, (
                f'ramp{i}{side}: yaw {math.degrees(pose[5]):.1f} deg does not match '
                f'pedestal {i} at {math.degrees(slots[i][5]):.1f} deg')
            for j in (1, 2, 3):
                if j == i:
                    continue
                pcx, pcy, _, _, _, yaw = slots[j]
                for (x, y, z) in _world_corners(pose, size):
                    if z <= PED_TOP + 1e-9:
                        continue            # buried in shelf material, harmless
                    dx, dy = x - pcx, y - pcy
                    u = dx * math.cos(yaw) + dy * math.sin(yaw)
                    v = -dx * math.sin(yaw) + dy * math.cos(yaw)
                    outside = max(abs(u) - obj_r, abs(v) - obj_t)
                    worst = min(worst, outside)
                    assert outside > 0.0, (
                        f'ramp{i}{side} stands {(z - PED_TOP) * 1000:.1f} mm above the '
                        f'surface inside slot {j}: radial {u:+.4f}, tangential {v:+.4f}')
    assert worst > 0.005, (
        f'barriers clear neighbouring slots by only {worst * 1000:.1f} mm')


def test_barriers_are_long_enough_to_cradle_their_object():
    """Shortening them to clear the neighbour must not leave them shorter than the
    object they exist to hold."""
    boxes = _shelf_boxes()
    longest = max(o['length'] for o in OBJECTS.values())
    for i in (1, 2, 3):
        for side in ('o', 'i'):
            length = boxes[f'ramp{i}{side}'][1][1]
            assert length >= longest, (
                f'ramp{i}{side} is {length:.3f} long but the longest object is '
                f'{longest:.3f}')


def test_barrier_is_tall_enough_to_stop_a_roll():
    """Escaping the slot means lifting the object's centre of mass by the crest height."""
    ramps = _shelf_ramps()
    crest_h = min(r['crest'][1] for r in ramps) - PED_TOP
    for name, mass in (('bottle', 0.45), ('can', 0.38)):
        escape = mass * 9.81 * crest_h
        roll = 0.75 * mass * 0.15 ** 2      # a solid cylinder rolling at 0.15 m/s
        assert escape > 4 * roll, (
            f'{name}: crest {crest_h * 1000:.1f} mm needs {escape * 1000:.1f} mJ to '
            f'clear but a 0.15 m/s roll carries {roll * 1000:.1f} mJ')


def test_hands_never_reach_barrier_height():
    """The fingers must not be able to touch a barrier - in EITHER axis.

    The previous form demanded 20 mm of vertical clearance, which was an arbitrary
    number that happened to hold when the fingers were short. Now that they reach
    further down, the honest question is geometric: does the volume the finger occupies
    at the lowest point of a placement overlap the volume a barrier occupies? Two boxes
    fail to intersect if they are separated in ANY axis, so clearance in one is enough -
    and reporting both makes it obvious which one is carrying the guarantee.
    """
    # WITH THE DISTAL PHALANX. The old form measured the plate alone (FINGERTIP_X - LG) and
    # reported the tips 16 mm above the crest while the real, longer finger reached 6 mm
    # BELOW it. So: tips as they really are - curled on a round object, straight on the flat
    # box - and the separation that actually holds is RADIAL.
    ramps = _shelf_ramps()
    crest_z = max(r['crest'][1] for r in ramps)
    foot_r = min(abs(r['foot'][0]) for r in ramps)
    slope = math.tan(max(abs(r['pitch']) for r in ramps))
    curl_closed = _commanded_curl()
    for name, o in OBJECTS.items():
        lowest_grasp = o['place_z'] + o['grip_dz']
        curl = curl_closed if o['round'] else 0.0
        tip_z = lowest_grasp - (col.finger_reach(curl) - col.LG)
        finger_out = o['close'] + mo.PAD_HALF    # outer face of the finger, radial
        r_gap = foot_r - finger_out              # positive: finger is inboard of the foot
        assert tip_z > PED_TOP + 0.005, (
            f'{name}: closed, the fingertips reach z={tip_z:.4f}, into the shelf top')
        assert r_gap > 0.001 or tip_z > crest_z + 0.005, (
            f'{name}: closed, the finger reaches z={tip_z:.4f} (crest {crest_z:.4f}) with '
            f'its outer face at {finger_out:.4f} against the barrier foot at {foot_r:.4f} - '
            f'it can touch the barrier')
        # LETTING GO: open only to release_closure, tips STRAIGHT, and the tips must stay
        # above the chamfer where the finger's outer face then is
        out = mo.release_closure(o['half_width']) + mo.PAD_HALF
        tip_open = lowest_grasp - (col.finger_reach(0.0) - col.LG)
        chamfer = PED_TOP + max(0.0, out - foot_r) * slope
        assert tip_open > chamfer + 0.003, (
            f'{name}: letting go, the straight fingertips at z={tip_open:.4f} meet the '
            f'chamfer at {chamfer:.4f} (outer face {out:.4f})')
        # and the node must actually let go that way - part-open, tips straight - and only
        # open fully once the hands have risen clear
        body = open(TASK_NODE).read()
        body = body[body.index('    def cycle(self, name):'):]
        i_rel = body.index("self.grippers(mo.release_closure(o['half_width']), settle=0.8, "
                           "straight=True)")
        i_rise = body.index("self.move(leg2['rise']")
        i_open = body.index('self.grippers(self.open_pos', i_rise)
        assert i_rel < i_rise < i_open, 'the hands must let go part-open, rise, THEN open'
        # and it must buy real margin over opening fully, which leaves almost none
        full = OPEN_POS + mo.PAD_HALF
        spare_full = tip_open - (PED_TOP + max(0.0, full - foot_r) * slope)
        assert (tip_open - chamfer) - spare_full > 0.003, (
            f'{name}: the partial release clears the chamfer by '
            f'{(tip_open - chamfer) * 1000:.1f} mm against {spare_full * 1000:.1f} mm fully '
            f'open - it is not buying anything')


def test_weld_is_released_before_the_object_can_touch_a_barrier():
    """A chamfer can only move an object that is free to move.

    While the DetachableJoint weld is active the object is rigidly part of the arm, so
    a sideways force from the chamfer fights the position controller instead of seating
    the object - which is the tipping failure all over again. So the welded descent has
    to stop above the crest, and the last few millimetres happen after release.
    """
    crest_z = max(r['crest'][1] for r in _shelf_ramps())
    drop = _yaml_float('place_drop')
    for name, o in OBJECTS.items():
        release_underside = o['place_z'] + drop - o['half_width']
        assert release_underside > crest_z, (
            f'{name}: weld released with the underside at {release_underside:.4f}, '
            f'already {(crest_z - release_underside) * 1000:.1f} mm below the '
            f'{crest_z:.4f} crest')
        assert drop <= 0.030, f'place_drop {drop} is too far to settle gently'
    # and the settle leg must actually close that gap, continuously
    for name, o in OBJECTS.items():
        pz = o['place_z'] + o['grip_dz']
        seq = (ch.place_waypoints(REACH, 0.0, pz + drop, LATERAL, _s(o), ch.left_sign(LATERAL))
               + ch.settle_waypoints(REACH, 0.0, pz, LATERAL, _s(o), ch.left_sign(LATERAL), drop)
               + ch.retreat_waypoints(REACH, 0.0, pz, LATERAL, _s(o), ch.left_sign(LATERAL)))
        assert ch.is_continuous(seq), (
            f'{name}: place -> settle -> retreat is not continuous')


def _placed(name):
    """Centre and the two end points of an object once it is on the shelf."""
    o = OBJECTS[name]
    az = math.radians(o['place_azimuth'])
    c = (REACH * math.cos(az), REACH * math.sin(az))
    d = (math.cos(az + math.pi / 2), math.sin(az + math.pi / 2))
    half = o['length'] / 2.0
    return c, [(c[0] + sgn * half * d[0], c[1] + sgn * half * d[1])
               for sgn in (+1, -1)]


def _s(o):
    return ch.grip_half_separation(o['length'])


def _psi(name):
    return math.radians(PICK_PSI[name])


def _cycle(o, name=None):
    """Full cycle at this object's real pick yaw, placed tangential to its slot."""
    if name is None:
        name = 'bottle' if o['length'] == 0.240 else 'box'
    return ch.full_cycle(REACH, o['cz'], o['place_z'], _psi(name),
                         math.radians(PLACE_PSI), o['length'])




# --------------------------------------------------------------- reachability
def _reachable7(p, R, side):
    """SEVEN-DOF reachability: SOME swivel and either shoulder solution solves it."""
    return any(ik.solve_branches(p, R, side, math.radians(d), True)
               for d in range(0, 360, 5))


def test_every_waypoint_is_reachable():
    for name, o in OBJECTS.items():
        for i, ((pl, Rl), (pr, Rr)) in enumerate(_cycle(o)):
            assert _reachable7(pl, Rl, +1), f'{name}: left unreachable at waypoint {i}'
            assert _reachable7(pr, Rr, -1), f'{name}: right unreachable at waypoint {i}'


def test_travel_pose_is_collision_free():
    """The check whose absence let the arms overlap from the moment the model loaded.

    The old travel-pose test verified joint limits, base radius, height and which side
    of the centreline each hand was on - all of which the broken pose satisfied while
    the two palm plates sat 45.7 mm inside each other. Clearance is a body property; it
    needs the body model.
    """
    rep = col.check_pose(ch.TRAVEL_LEFT, ch.TRAVEL_RIGHT, finger_pos=OPEN_POS)
    assert rep['worst'] > 0.010, (
        f'travel pose: {rep["worst_name"]} clearance {rep["worst"]:+.4f} m '
        f'({rep["arm_arm_pair"]})')


def test_the_trips_to_and_from_parked_are_planned_and_clear():
    """The controller interpolates in JOINT space between poses.

    Both endpoints can be clear while the straight line between them is not. The move from
    parked to the first grasp and back used to be ONE blind joint-space jump that no check
    looked at; now both are planned legs that start and end on the exact parked joint
    vector, and this samples the path the controller takes between their waypoints.
    """
    for name in OBJECTS:
        for half in _planned(name):
            QL, QR, fp, at = half['QL'], half['QR'], half['fp'], half['at']
            idx = [i for n in ('unpark', 'park') if n in at for i in at[n]]
            for i in idx:
                j = i - 1 if i > 0 else i
                for f in (0.0, 0.25, 0.5, 0.75):
                    # DERIVED, never 6 or 7 written out: the seventh axis arrived by
                    # inserting a name into ik.ARM_JOINT_SUFFIX
                    il = [a + f * (b - a) for a, b in zip(QL[j], QL[i])]
                    ir = [a + f * (b - a) for a, b in zip(QR[j], QR[i])]
                    rep = col.check_pose(il, ir, finger_pos=fp[i])
                    assert rep['worst'] > MIN_CLEARANCE, (
                        f'{name}: the trip {"from" if half["label"] == "pick" else "to"} '
                        f'parked passes through a collision, {rep["worst"]:+.4f} m '
                        f'({rep["worst_name"]}) near waypoint {i}')


def test_grasp_tolerates_a_few_mm_of_pose_error():
    """The grip must survive the base settling a fraction of a degree off.

    A 0.29 deg heading error is 1.5 mm of lateral offset at a 0.30 m reach. The fingers
    are commanded 6 mm past contact against an 8 N effort limit, so both make contact
    and the object is self-centred rather than pinched on one side and gapped on the
    other.
    """
    for name, o in OBJECTS.items():
        overtravel = o['half_width'] - (o['close'] - 0.006)
        assert overtravel >= 0.002, (
            f'{name}: only {overtravel * 1000:.1f} mm of over-travel - not enough to '
            f'absorb a few mm of pose error')
        # and it must not be so much that the fingers would cross the object's centre
        assert o['close'] - 0.006 > 0.0, f'{name}: fingers would cross the centreline'


def test_travel_pose_is_compact_and_on_side():
    lims = (ik.YAW_LIM, ik.PITCH2_LIM, ik.PITCH3_LIM, ik.W_ROLL_LIM,
            ik.W_PITCH_LIM, ik.W_ROLL_LIM)
    for q, side, ysign in ((ch.TRAVEL_LEFT, +1, +1), (ch.TRAVEL_RIGHT, -1, -1)):
        for v, lim in zip(q, lims):
            assert abs(v) <= lim, f'travel pose violates a joint limit: {v}'
        p, _ = ik.fk(q, side)
        assert math.hypot(p[0], p[1]) < col.BASE_RADIUS, 'travel pose sticks out'
        assert p[2] > 0.20, 'travel pose too low'
        assert p[1] * ysign > 0, 'travel pose crosses the centreline'


# ------------------------------------------------------------------- geometry
def test_collision_model_agrees_with_the_ik():
    """The capsule model's kinematics must match ``ik.fk`` exactly.

    If it drifts, every clearance number below is measuring the wrong robot.
    """
    import random
    rng = random.Random(4242)
    for _ in range(200):
        q = [rng.uniform(-1.2, 1.2) for _ in range(ik.N_ARM_JOINTS)]
        for side in (+1, -1):
            p_ik, R_ik = ik.fk(q, side)
            R, t = col.link_frames(q, side)['palm']
            appr = [R[0][0], R[1][0], R[2][0]]
            p_col = [t[i] + col.LG * appr[i] for i in range(3)]
            assert max(abs(p_ik[i] - p_col[i]) for i in range(3)) < 1e-9


def test_no_body_collides_anywhere_in_the_cycle():
    """Arm-arm, floor, base and mast at every waypoint of the plan the node makes."""
    for name in OBJECTS:
        for half in _planned(name):
            QL, QR, fp = half['QL'], half['QR'], half['fp']
            for i, (ql, qr) in enumerate(zip(QL, QR)):
                rep = col.check_pose(ql, qr, fp[i])
                assert rep['worst'] > MIN_CLEARANCE, (
                    f'{name} ({half["label"]}): {rep["worst_name"]} clearance '
                    f'{rep["worst"]:+.4f} m at waypoint {i} (arm-arm {rep["arm_arm"]:+.4f}, '
                    f'floor {rep["floor"]:+.4f}, base {rep["base"]:+.4f})')


def test_fingertips_stay_above_the_floor():
    """The specific failure that broke the first run, as its own check.

    The fingers reach ``FINGERTIP_X - LG`` past the grasp point. With the grasp
    point at a floor-lying object's centre, anything shallower than that plus the
    finger's own half-thickness puts the tips underground.
    """
    # Measured from the GRASP HEIGHT, which is grip_dz ABOVE the object's axis - not
    # from the axis itself. The fingers deliberately reach well past the grasp point
    # now, so that they straddle the widest part of the body instead of pinching its
    # shoulder; what must never happen is the TIPS going underground.
    overhang = col.FINGERTIP_X - col.LG
    for name, o in OBJECTS.items():
        tip_z = o['cz'] + o.get('grip_dz', 0.0) - overhang
        assert tip_z > 0.015, (
            f'{name}: lying centre {o["cz"]:.3f} m puts the fingertips at '
            f'{tip_z:+.3f} m')


def test_arms_start_on_their_own_sides():
    for name, o in OBJECTS.items():
        (pl, _), (pr, _) = _cycle(o)[0]
        assert pl[1] > 0 > pr[1], f'{name}: hands do not straddle the centreline'


def test_path_is_continuous():
    """No waypoint-to-waypoint jump. A jump is a velocity step, i.e. a jerk."""
    for name, o in OBJECTS.items():
        seq = _cycle(o)
        assert ch.is_continuous(seq), f'{name}: path has a discontinuity'
        step = max(max(math.dist(seq[i][h][0], seq[i + 1][h][0])
                       for h in (0, 1)) for i in range(len(seq) - 1))
        assert step < 0.08, f'{name}: biggest step {step:.3f} m is too coarse'


# --------------------------------------------------------------------- grasp
def test_fingers_actually_reach_the_object():
    """``close`` must put the finger face INSIDE the surface, not short of it.

    The first version closed to ``half_width + 0.008``, an 8 mm air gap per side, so
    the gripper never touched the object and the weld did all the work.
    """
    for name, o in OBJECTS.items():
        face = o['close'] - 0.006           # finger half-thickness
        hw = _binding_half_width(o)
        assert face < hw, (
            f'{name}: finger face at {face:.3f} m never reaches the '
            f'{hw:.3f} m binding surface')
        squeeze = hw - face
        assert 0.002 <= squeeze <= 0.010, (
            f'{name}: commanded over-travel {squeeze * 1000:.1f} mm - too little to '
            f'self-centre, or so much the fingers would pass the object centre')


def test_fingers_clear_the_object_while_descending():
    for name, o in OBJECTS.items():
        assert OPEN_POS - 0.006 > o['half_width'] + 0.005, (
            f'{name}: open fingers do not clear it on the way down')


def test_grip_points_lie_within_the_object():
    for name, o in OBJECTS.items():
        assert _s(o) < o['length'] / 2.0, f'{name}: grip is past the end'
        assert _s(o) > o['length'] * 0.30, f'{name}: grip too close to the middle'


# ------------------------------------------------------- LIDAR HEADING FIX
WALL_X, WALL_HALF = 0.46, 0.45


def test_wall_gives_the_true_heading_from_any_orientation():
    """The measurement that replaces drifting odometry.

    Section A's base turns roughly 530 deg over a run, and wheel slip makes the
    odometry heading drift over that - which is what left later grasps millimetres off
    and objects overhanging their slot. This must recover the heading from ANY
    orientation, since the LiDAR is a full 360 deg scan and the wall is always in view.
    """
    worst = 0.0
    for deg in range(-180, 180, 11):
        h = math.radians(deg)
        ranges, amin, ainc = wr.synth_scan(h, WALL_X, WALL_HALF)
        got, info = wr.heading_from_scan(ranges, amin, ainc, 0.12, 8.0,
                                        band=(0.20, 1.20))
        assert got is not None, f'no fix at {deg} deg: {info.get("reject")}'
        err = abs(math.remainder(got - h, 2 * math.pi))
        worst = max(worst, err)
    assert worst < math.radians(0.01), (
        f'worst heading error {math.degrees(worst):.4f} deg on a clean scan')


def test_wall_fix_survives_range_noise():
    """It has to beat the tolerance it is servoing to, or it buys nothing."""
    for sigma, limit_deg in ((0.002, 0.15), (0.005, 0.30)):
        worst = 0.0
        for deg in range(-180, 180, 23):
            h = math.radians(deg)
            ranges, amin, ainc = wr.synth_scan(h, WALL_X, WALL_HALF,
                                              noise=sigma, seed=deg)
            got, _ = wr.heading_from_scan(ranges, amin, ainc, 0.12, 8.0,
                                         band=(0.20, 1.20))
            assert got is not None, f'no fix at {deg} deg with {sigma} m noise'
            worst = max(worst, abs(math.remainder(got - h, 2 * math.pi)))
        assert math.degrees(worst) < limit_deg, (
            f'{sigma * 1000:.0f} mm noise -> {math.degrees(worst):.2f} deg, '
            f'worse than the {limit_deg} deg budget')


def test_wall_fix_rejects_rubbish_rather_than_guessing():
    """A bad fix must fail loudly. Silently returning a wrong heading is worse than
    keeping the previous one."""
    got, info = wr.heading_from_scan([float('inf')] * 360, -math.pi,
                                    2 * math.pi / 360, 0.12, 8.0)
    assert got is None and 'reject' in info, 'an empty scan should be rejected'
    # a full circular wall is not a flat wall and must not fit a line
    got, info = wr.heading_from_scan([0.5] * 360, -math.pi, 2 * math.pi / 360,
                                    0.12, 8.0, band=(0.20, 1.20))
    assert got is None and 'reject' in info, (
        f'a circular scan fitted a line anyway: {info}')


def test_corner_gives_the_full_pose_not_just_heading():
    """Heading alone was not enough.

    A single flat wall leaves the coordinate PARALLEL to it unobservable - sliding
    along a wall changes nothing about how it looks. The run that motivated this
    corrected heading perfectly (every turn landed within 0.2 deg) and still misplaced
    every object, because the base had also translated: 29 deg of accumulated odometry
    heading error means badly slipping wheels, and wheels that slip shift the base.
    Two perpendicular walls make x, y and heading all observable.
    """
    worst_p = worst_h = 0.0
    for bx, by, deg in ((0, 0, 0), (0.03, -0.04, 30), (-0.05, 0.06, 130),
                        (0.08, 0.05, -100), (-0.02, -0.07, 180),
                        (0.06, -0.03, -55), (0.04, 0.09, -130)):
        h = math.radians(deg)
        ranges, amin, ainc = wr.synth_scan(h, wr.WALL_A_X, 0.45, base_xy=(bx, by),
                                          wall_b_y=wr.WALL_B_Y,
                                          wall_b_span=(-0.46, 0.10))
        pose, info = wr.pose_from_scan(ranges, amin, ainc, 0.12, 8.0,
                                      band=(0.15, 1.60), expect_heading=h)
        assert pose is not None, f'no fix at ({bx},{by},{deg}): {info.get("reject")}'
        worst_p = max(worst_p, math.hypot(pose[0] - bx, pose[1] - by))
        worst_h = max(worst_h, abs(math.remainder(pose[2] - h, 2 * math.pi)))
    assert worst_p < 0.001, f'position error {worst_p * 1000:.1f} mm'
    assert worst_h < math.radians(0.01), f'heading error {math.degrees(worst_h):.3f} deg'


def test_corner_fix_survives_noise():
    worst_p = worst_h = 0.0
    for deg in range(-180, 180, 29):
        h = math.radians(deg)
        ranges, amin, ainc = wr.synth_scan(h, wr.WALL_A_X, 0.45,
                                          base_xy=(0.04, -0.05),
                                          wall_b_y=wr.WALL_B_Y,
                                          wall_b_span=(-0.46, 0.10),
                                          noise=0.005, seed=deg)
        pose, _ = wr.pose_from_scan(ranges, amin, ainc, 0.12, 8.0,
                                   band=(0.15, 1.60), max_rms=0.03,
                                   expect_heading=h)
        assert pose is not None, f'no fix at {deg} deg with 5 mm noise'
        worst_p = max(worst_p, math.hypot(pose[0] - 0.04, pose[1] + 0.05))
        worst_h = max(worst_h, abs(math.remainder(pose[2] - h, 2 * math.pi)))
    assert worst_p < 0.005, f'position error {worst_p * 1000:.1f} mm with noise'
    assert worst_h < math.radians(0.5), f'heading {math.degrees(worst_h):.2f} deg'


def test_aim_correction_absorbs_the_residual_drift():
    """After recentring, the arm must cope with whatever drift is left.

    The base is recentred to within ``position_tol`` (10 mm), so the aim correction
    only has to absorb that much. It cannot absorb much more: at a 0.30 m reach the
    wrist centre is already near its 0.32 m limit, so a target pushed 40 mm further out
    is unreachable. That is exactly why the base is driven back rather than the arm
    stretched.
    """
    for drift in (0.005, 0.010, 0.015, 0.020):
        for dx, dy in ((drift, 0), (-drift, 0), (0, drift), (0, -drift)):
            for az_deg in (0.0, 130.0, 230.0):
                az = math.radians(az_deg)
                pose = (dx, dy, az)
                world = (REACH * math.cos(az), REACH * math.sin(az))
                tx, ty = wr.world_to_base(world, pose)
                axis = math.remainder(az + math.pi / 2 - pose[2], 2 * math.pi)
                for name, o in OBJECTS.items():
                    seq = (ch.approach_waypoints(tx, ty, o['cz'] + o['grip_dz'],
                                                axis, _s(o), ch.left_sign(axis))
                           + ch.place_waypoints(tx, ty, o['place_z'] + 0.010, axis, _s(o), ch.left_sign(axis)))
                    for (pl, Rl), (pr, Rr) in seq:
                        ik.solve(pl, Rl, +1)
                        ik.solve(pr, Rr, -1)
                    w, d, _ = col.check_waypoints(seq, finger_pos=o['close'])
                    assert w > MIN_CLEARANCE, (
                        f'{name} at {drift * 1000:.0f} mm drift, azimuth '
                        f'{az_deg}: {d["worst_name"]} {w:+.4f} m')


def test_release_happens_above_the_resting_height():
    """The arm must not press the object into the shelf.

    The weld freezes whatever vertical offset the grasp happened to have, so lowering
    to the exact resting height often means pressing - and the reaction lifted the
    whole robot off its wheels. Releasing a few mm high lets it drop instead.
    """
    import re as _re
    cfg = open(CFG).read()
    m = _re.search(r'^\s*place_drop:\s*([-0-9.]+)', cfg, _re.M)
    assert m, 'place_drop missing from the config'
    drop = float(m.group(1))
    assert 0.005 <= drop <= 0.020, f'place_drop {drop} m is outside a sane range'


def test_pose_fix_is_never_partially_valid():
    """A pose fix must be a COMPLETE pose or absent - never a tuple holding None.

    This is the bug that crashed the whole task before it touched an object: an earlier
    version returned ``(x, None, heading)`` when only one wall was found, and the first
    caller did arithmetic on the None. Two runtime failures in this section have now
    come from the pose-handling logic rather than from geometry, so it is checked
    directly.
    """
    cases = []
    # good scans, bad scans, and degenerate ones
    for hd in (0, 45, 130, -100, 180):
        h = math.radians(hd)
        cases.append(wr.synth_scan(h, WALL_X, WALL_HALF, base_xy=(0.03, -0.02),
                                  wall_b_y=wr.WALL_B_Y,
                                  wall_b_span=(-0.46, 0.10)))
        cases.append(wr.synth_scan(h, WALL_X, WALL_HALF))        # wall A only
    cases.append(([float('inf')] * 360, -math.pi, 2 * math.pi / 360))
    cases.append(([0.5] * 360, -math.pi, 2 * math.pi / 360))     # circular
    for ranges, amin, ainc in cases:
        pose, info = wr.pose_from_scan(ranges, amin, ainc, 0.12, 8.0,
                                      band=(0.15, 1.60))
        assert pose is None or (len(pose) == 3
                               and all(v is not None for v in pose)), (
            f'pose_from_scan returned a partially valid pose: {pose}')
        if pose is None:
            assert 'reject' in info, 'a rejection must say why'


def test_aim_target_handles_a_missing_fix_without_crashing():
    """The decision that both runtime bugs lived in, tested directly."""
    # no fix at all -> nominal aim, flagged as uncorrected
    tx, ty, axis, corrected = wr.aim_target(None, math.radians(130.0), REACH,
                                           REACH, LATERAL)
    assert (tx, ty) == (REACH, 0.0) and not corrected
    assert abs(axis - LATERAL) < 1e-12

    # a good fix -> corrected aim close to nominal when drift is small
    pose = (0.005, -0.004, math.radians(130.0))
    tx, ty, axis, corrected = wr.aim_target(pose, math.radians(130.0), REACH,
                                           REACH, LATERAL)
    assert corrected
    assert math.hypot(tx - REACH, ty) < 0.02, (tx, ty)

    # a malformed fix must RAISE, not silently misbehave
    for bad in ((0.0, None, 0.0), (0.0, 0.0), (None, None, None)):
        try:
            wr.aim_target(bad, 0.0, REACH, REACH, LATERAL)
        except ValueError:
            pass
        else:
            raise AssertionError(f'aim_target accepted a malformed fix: {bad}')


def test_a_stale_pose_fix_is_worse_than_none():
    """Regression: a pose measured at one heading must never be reused after turning.

    The guard used to be "if pose_fix is None", which is never true again once a single
    fix has succeeded. So when a later fix failed, the arm was aimed using a pose
    measured 60 degrees earlier: the can was aimed at (+0.152, -0.271) and the box
    behind the robot, and both failed IK. This checks the magnitude of that error, so
    the reasoning is recorded even though the guard itself lives in the ROS node.
    """
    stale = (0.0, 0.0, math.radians(115.29))     # measured while facing the shelf
    actual_azimuth = math.radians(55.0)          # but now facing the next object
    world = (REACH * math.cos(actual_azimuth), REACH * math.sin(actual_azimuth))
    tx, ty = wr.world_to_base(world, stale)
    nominal = (REACH, 0.0)
    err = math.hypot(tx - nominal[0], ty - nominal[1])
    assert err > 0.15, (
        'the stale-pose error should be large enough to be obviously fatal, '
        f'got {err * 1000:.0f} mm')
    # and it must be unreachable, i.e. it really does break rather than degrade
    unreachable = False
    try:
        psi = math.remainder(actual_azimuth + math.pi / 2 - stale[2], 2 * math.pi)
        for (pl, Rl), (pr, Rr) in ch.approach_waypoints(
                tx, ty, 0.045, psi, 0.086, ch.left_sign(psi)):
            ik.solve(pl, Rl, +1)
            ik.solve(pr, Rr, -1)
    except ik.IKError:
        unreachable = True
    assert unreachable, 'expected the stale aim to be unreachable'


# ================================================================= REDESIGN GUARDS
def test_the_arms_never_swap_ends():
    """E19, tested as a MECHANISM rather than at the shipped angles.

    The old ``_stations`` ended with
        return (a, b) if a[0][1] >= b[0][1] else (b, a)
    re-deciding which arm takes which end by comparing y at every call. Section B
    suffered this as a 170 mm instantaneous jump when the two ends passed through equal
    y at the top of its flip.

    At the yaws this section actually ships - 72 and 108 degrees sweeping to 90 - sin(psi)
    never changes sign, so a y-sort would coincidentally agree and the fault stays
    dormant. That is luck, not design: widening the envelope, or picking an object whose
    axis is near radial, brings it straight back. So this drives the rotation THROUGH the
    crossing on purpose and requires the hand assignment to survive it.
    """
    s_ = 0.085
    # sweep across 0 deg, where a y-comparison flips sign
    psi0, psi1 = math.radians(-40.0), math.radians(40.0)
    sl = ch.left_sign(psi0)
    seq = ch.yaw_waypoints(REACH, 0.0, psi0, psi1, s_, sl)
    # 1. the left hand's path must be continuous. A swap teleports it by 2*s.
    step = ch.max_step(seq)
    assert step < 2 * s_ * 0.5, (
        f'a hand jumped {step * 1000:.1f} mm through the crossing - that is a swap, '
        f'not a motion (2s would be {2 * s_ * 1000:.0f} mm)')
    # 2. the left hand must stay on the SAME end throughout. Identify the end by the
    #    sign of the projection onto the axis the caller asked for.
    for i, ((pl, _), (pr, _)) in enumerate(seq):
        psi = psi0 + (psi1 - psi0) * (i + 1) / len(seq)
        ax = (math.cos(psi), math.sin(psi))
        mid = [(pl[k] + pr[k]) / 2 for k in range(3)]
        proj = (pl[0] - mid[0]) * ax[0] + (pl[1] - mid[1]) * ax[1]
        assert proj * sl > 0, (
            f'waypoint {i}: the left hand is on the {"negative" if proj > 0 else "positive"}'
            f' end but was pinned to the other one')
    # 3. and the assignment must be a pure function of the PICK yaw, not of anything
    #    that changes during the motion
    assert ch.left_sign(psi0) == sl
    for extra in (psi1, math.radians(0.0), math.radians(89.0)):
        pair = ch._stations(REACH, 0.0, 0.34, extra, s_, sl)
        mid = [(pair[0][0][k] + pair[1][0][k]) / 2 for k in range(3)]
        ax = (math.cos(extra), math.sin(extra))
        proj = ((pair[0][0][0] - mid[0]) * ax[0] + (pair[0][0][1] - mid[1]) * ax[1])
        assert proj * sl > 0, (
            f'_stations re-decided the hand assignment at psi={math.degrees(extra):.0f}')


GRIP_CONTACT_THRESHOLD = 0.001      # must match check_grip() in task_node.py


def test_grasp_detection_has_margin():
    """The runtime grip check infers contact from how far short the finger stalls.

    A finger is commanded inward past the object's surface, so when the object is
    there it stalls and the joint rests ABOVE the command; on air it reaches the
    command exactly. The gap between those two outcomes is the over-travel, and the
    detection threshold has to sit well inside it - otherwise a real grasp reads as
    air, which is worse than no check at all because it would abort a good cycle.
    """
    for name, o in OBJECTS.items():
        bind = _binding_half_width(o)
        stall = bind + 0.006                      # finger half-thickness
        over_travel = stall - o['close']
        assert over_travel > 0, (
            f'{name}: close {o["close"]} is not past the surface at {bind:.4f}, so '
            f'the finger never stalls and contact can never be detected')
        assert over_travel > 2.0 * GRIP_CONTACT_THRESHOLD, (
            f'{name}: over-travel {over_travel * 1000:.1f} mm is less than twice the '
            f'{GRIP_CONTACT_THRESHOLD * 1000:.1f} mm detection threshold - a real '
            f'grasp could be reported as closing on air')


def test_the_runtime_checks_are_actually_wired_in():
    """A check that is never called is worse than none - it reads as reassurance.

    Verified against the real source, the same way the object-dict wiring is, because
    nothing else would notice if a call were dropped by an edit.
    """
    src = _node_src()
    for fn in ('check_grip', 'check_lifted', 'check_yaw', 'check_placed', 'check_grasp_centring',
               'check_localisation', 'settle_arms', 'check_both_hands_holding'):
        assert f'def {fn}(' in src, f'{fn} is not defined'
        calls = src.count(f'self.{fn}(')
        assert calls >= 1, f'{fn} is defined but never called'
    # and the threshold used at runtime must be the one this file reasons about
    m = re.search(r'worst > ([0-9.]+)\s*#', src) or re.search(r'worst > ([0-9.]+)', src)
    assert m, 'could not find the contact threshold in check_grip'
    assert abs(float(m.group(1)) - GRIP_CONTACT_THRESHOLD) < 1e-9, (
        f'task_node uses {m.group(1)} but this file assumes '
        f'{GRIP_CONTACT_THRESHOLD}')


def test_yaw_alignment_actually_reaches_the_slot_angle():
    """The object must finish tangential, or the cradle chamfers cannot hold it."""
    for name, o in OBJECTS.items():
        s_ = _s(o)
        psi0, psi1 = _psi(name), math.radians(PLACE_PSI)
        sl = ch.left_sign(psi0)
        seq = ch.yaw_waypoints(REACH, 0.0, psi0, psi1, s_, sl)
        assert seq, f'{name}: yaw alignment produced no waypoints'
        (pl, _), (pr, _) = seq[-1]
        got = math.atan2(pl[1] - pr[1], pl[0] - pr[0]) % math.pi
        assert abs(((got - psi1 + math.pi / 2) % math.pi) - math.pi / 2) < 1e-6, (
            f'{name}: yaw alignment ended at {math.degrees(got):.2f} deg, '
            f'wanted {PLACE_PSI:.1f}')
        # and it must get there in steps the controller can follow
        assert ch.max_step(seq) <= 1.5 * ch.MAX_STEP, (
            f'{name}: yaw alignment steps {ch.max_step(seq) * 1000:.1f} mm')


def test_pick_yaws_are_inside_the_measured_envelope():
    """Rotating the axis away from tangential swings one end towards the robot.

    Past about 22 degrees a hand strikes the base cylinder. This is measured, not
    assumed: the sweep that produced it is in the session notes, and the check below
    re-derives the clearance rather than trusting the number.
    """
    for name, o in OBJECTS.items():
        off = abs(PICK_PSI[name] - 90.0)
        assert off >= 5.0, (
            f'{name} is only {off:.1f} deg off tangential - too close to the old '
            f'single-orientation case to prove anything')
        assert off <= 22.0, (
            f'{name} is {off:.1f} deg off tangential, outside the measured envelope')
        # re-derive: the grasp itself must clear every body
        s_, psi = _s(o), _psi(name)
        pair = ch._stations(REACH, 0.0, o['cz'] + o['grip_dz'], psi, s_,
                            ch.left_sign(psi))
        w, d, _ = col.check_waypoints([pair], finger_pos=o['close'])
        assert w > 0.008, (
            f'{name} at {PICK_PSI[name]:.0f} deg: clearance {w * 1000:+.1f} mm '
            f'({d["worst_name"]})')


def test_the_two_objects_bracket_tangential():
    """Both objects off the SAME side would only half-test the general path."""
    offs = [PICK_PSI[n] - 90.0 for n in OBJECTS]
    assert min(offs) < 0 < max(offs), (
        f'pick yaws {offs} are all on one side of tangential')


def test_config_axis_yaw_agrees_with_the_world_spawn_pose():
    """N5: if the config and the SDF disagree, every grasp is offset by the difference.

    A body is laid down by a 90 degree roll, which puts its long axis along -y, so the
    axis angle in the world is (spawn yaw - 90) folded into [0, 180). Nothing else in
    the system can detect a mismatch here - the robot would simply grasp at the wrong
    angle and the run would look like a localisation fault.
    """
    world = open(os.path.join(os.path.dirname(__file__), '..', '..', 'pnp_gazebo',
                              'worlds', 'pick_place.sdf')).read()
    for name, o in OBJECTS.items():
        m = re.search(rf'<name>{name}</name>\s*<pose>([-0-9.eE ]+)</pose>', world)
        if m is None:
            m = re.search(rf'<uri>model://{name}</uri>\s*<name>{name}</name>\s*'
                          rf'<pose>([-0-9.eE ]+)</pose>', world)
        assert m, f'{name}: no spawn pose found in the world file'
        vals = [float(v) for v in m.group(1).split()]
        x, y, z, roll, _pitch, spawn_yaw = vals
        assert abs(roll - math.pi / 2) < 1e-3, (
            f'{name}: roll {roll:.4f} is not the 90 deg that lays the body down')
        axis_world = math.degrees(spawn_yaw) - 90.0
        want = o['axis_yaw']
        assert abs(((axis_world - want + 90.0) % 180.0) - 90.0) < 0.5, (
            f'{name}: world spawn yaw {math.degrees(spawn_yaw):.1f} deg implies an '
            f'axis of {axis_world % 180.0:.1f}, but the config says {want:.1f}')
        # and it must spawn where the robot is told to look for it
        az = math.radians(o['pick_azimuth'])
        assert math.dist((x, y), (REACH * math.cos(az), REACH * math.sin(az))) < 0.01, (
            f'{name}: spawned at ({x:+.3f},{y:+.3f}) but the robot aims at azimuth '
            f'{o["pick_azimuth"]:.0f} and radius {REACH}')
        assert abs(z - o['cz']) < 1e-6, f'{name}: spawn z {z} vs cz {o["cz"]}'


def test_config_grip_dz_matches_the_choreography_constant():
    """The choreography converts carry height to grasp height with its OWN constant.

    If the config disagrees, the lift and the place would use different offsets and the
    object would be released at the wrong height.
    """
    for name, o in OBJECTS.items():
        assert abs(o['grip_dz'] - ch.GRIP_DZ) < 1e-9, (
            f'{name}.grip_dz {o["grip_dz"]} vs choreography.GRIP_DZ {ch.GRIP_DZ}')


def test_every_choreography_call_in_the_node_has_the_right_arity():
    """E23's class: a wrong argument list that only fails at runtime.

    The settle call shipped with the label string where dt belonged, so every placement
    died AFTER the weld was released - the object was dropped rather than set down. A
    later edit then omitted sign_left, which would silently have put a float into the
    hand-assignment. Neither is a syntax error and neither is reachable from a geometry
    test, so it is checked here against the real signatures.
    """
    import inspect
    src = open(TASK_NODE).read()
    tree = ast.parse(src)
    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == 'ch'):
            continue
        fn = getattr(ch, node.func.attr, None)
        if not callable(fn):
            continue
        params = list(inspect.signature(fn).parameters.values())
        required = sum(1 for q in params if q.default is inspect.Parameter.empty)
        given = len(node.args) + len(node.keywords)
        checked += 1
        assert required <= given <= len(params), (
            f'line {node.lineno}: ch.{node.func.attr} called with {given} args, '
            f'signature takes {required}..{len(params)} {tuple(p.name for p in params)}')
    names = {n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'ch'}
    assert {'pick_legs', 'place_legs', 'plan_inputs', 'grip_half_separation',
            'left_sign'} <= names, (
        f'only {checked} choreography calls found: {sorted(names)}')


def test_waypoint_steps_are_bounded():
    """E7: a fixed sample count put 67.5 mm between waypoints on the 0.27 m lift.

    The controller renders a large step as a lurch. Sampling is now by distance.
    """
    for name, o in OBJECTS.items():
        seq = _cycle(o, name)
        step = ch.max_step(seq)
        assert step <= 1.5 * ch.MAX_STEP, (
            f'{name}: largest step {step * 1000:.1f} mm, limit '
            f'{1.5 * ch.MAX_STEP * 1000:.1f}')
        assert ch.is_continuous(seq), f'{name}: cycle is not continuous'


def test_hands_stay_on_the_object_along_its_axis():
    """The palm is 0.040 long along the object's axis, so it must not hang off the end."""
    for name, o in OBJECTS.items():
        inboard = o['length'] / 2 - _s(o)
        assert inboard >= 0.020 + 0.005, (
            f'{name}: hand centre is only {inboard * 1000:.1f} mm from the end, '
            f'and the palm needs 20 mm plus margin')
        frac = inboard / o['length']
        assert 0.08 <= frac <= 0.25, (
            f'{name}: gripping {frac * 100:.0f}% in from each end - too close to the '
            f'end to be safe, or too close to the middle to control rotation')


def test_the_rotating_object_clears_the_robot_and_placed_objects():
    """N2: yaw-rotating a held object sweeps a disc, not a point.

    collision.py models the arms, not the payload, so the swept object has to be
    checked separately - against the mast it turns beside, and against anything already
    standing on the shelf.
    """
    mast_xy, mast_r = (-0.100, 0.0), 0.030
    for name, o in OBJECTS.items():
        radius = _s(o) + o['half_width']        # object's own swept radius
        centre = (REACH, 0.0)                   # it rotates over the slot
        gap = math.dist(centre, mast_xy) - radius - mast_r
        assert gap > 0.02, (
            f'{name}: swept object passes within {gap * 1000:.1f} mm of the mast')
        # the carried object's underside must clear a placed object's top
        under = ch.LIFT_Z - o['half_width']
        placed_top = o['place_z'] + o['half_width']
        assert under > placed_top, (
            f'{name}: carried underside {under:.3f} vs placed top {placed_top:.3f}')


# ----------------------------------------------------------- CARRY CLEARANCE
def test_carried_object_clears_an_already_placed_one():
    """The robot rotates with the object at the SAME radius the slots are on.

    So a carried object sweeps directly over whatever is already on the shelf. At the
    old 0.30 m carry height the underside sat at 0.255 against a placed object's top of
    0.260 - it struck the bottle it had just put down.
    """
    placed_top = max(o['place_z'] + o['half_width'] for o in OBJECTS.values())
    for name, o in OBJECTS.items():
        underside = ch.LIFT_Z - o['half_width']
        assert underside > placed_top + 0.020, (
            f'{name}: carried underside {underside:.3f} m against a placed top of '
            f'{placed_top:.3f} m - only {(underside - placed_top) * 1000:.0f} mm')


def test_placed_object_has_real_margin_on_its_pedestal():
    """A pedestal barely wider than the object dangles it off the edge.

    The first version gave the 0.240 m bottle a 0.26 m pedestal - 10 mm a side, so two
    degrees of heading error left it overhanging.
    """
    import re as _re
    sdf = open(os.path.join(os.path.dirname(__file__), '..', '..', 'pnp_gazebo',
                            'models', 'shelf', 'model.sdf')).read()
    sizes = _re.findall(r'name="ped\d"[^>]*>.*?<size>([0-9.]+) ([0-9.]+)',
                        sdf, _re.S)
    assert sizes, 'no pedestals found'
    tangential = float(sizes[0][1])
    longest = max(o['length'] for o in OBJECTS.values())
    margin = (tangential - longest) / 2.0
    assert margin > 0.030, (
        f'only {margin * 1000:.0f} mm of margin either side of the longest object')


# ----------------------------------------------------------------- placement
def test_placed_objects_do_not_overlap():
    """Three objects on an arc need real separation.

    They are placed LYING with the long axis tangential, so the gap between two slot
    centres must exceed the mean of their lengths. The first version used slots 20 deg
    apart, giving 0.104 m where 0.220 m was needed - the objects were being stacked on
    top of one another, and each placement knocked over the previous one.
    """
    names = list(OBJECTS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca, _ = _placed(a)
            cb, _ = _placed(b)
            gap = math.dist(ca, cb)
            need = (OBJECTS[a]['length'] + OBJECTS[b]['length']) / 2.0
            assert gap > need, (f'{a} and {b} only {gap:.3f} m apart, '
                               f'need {need:.3f} m')


def test_placed_objects_clear_the_robot_base():
    """A long object on an arc swings its ends inward; they must miss the base."""
    for name in OBJECTS:
        _, ends = _placed(name)
        for p in ends:
            r = math.hypot(*p)
            assert r > BASE_RADIUS + 0.015, (
                f'{name}: an end lands {r:.3f} m from the base axis')


def test_placed_objects_sit_on_a_pedestal():
    """Every slot must actually have shelf under it."""
    import re as _re
    sdf = open(os.path.join(os.path.dirname(__file__), '..', '..',
                            'pnp_gazebo', 'models', 'shelf',
                            'model.sdf')).read()
    peds = [tuple(float(v) for v in m.split()[:2])
            for m in _re.findall(r'<collision name="ped\d+"><pose>([^<]+)</pose>',
                                 sdf)]
    # At LEAST one pedestal per object. The shelf deliberately keeps three slots while
    # only two are used, so the can can be added back once two objects work end to end.
    assert len(peds) >= len(OBJECTS), (
        f'{len(peds)} pedestals for {len(OBJECTS)} objects - need one each')
    for name in OBJECTS:
        c, _ = _placed(name)
        assert min(math.dist(c, p) for p in peds) < 0.02, (
            f'{name} is not placed over any pedestal')


# -------------------------------------------------------------------- config
def test_config_matches_this_file():
    """Guard against the config and the test drifting apart."""
    txt = open(CFG).read()
    assert re.search(r'^\s*reach:\s*%.2f' % REACH, txt, re.M), 'reach differs'
    assert re.search(r'^\s*open_pos:\s*0\.065', txt, re.M), 'open_pos differs'
    for name, o in OBJECTS.items():
        block = re.search(rf'^    {name}:\n((?:      .+\n)+)', txt, re.M)
        assert block, f'{name} missing from the config'
        body = block.group(1)
        for key in ('cz', 'place_surface_z', 'length', 'close'):
            m = re.search(rf'{key}:\s*([-0-9.]+)', body)
            assert m, f'{name}.{key} missing'
            assert abs(float(m.group(1)) - o[key]) < 1e-9, (
                f'{name}.{key}: config {m.group(1)} vs test {o[key]}')


def test_the_arms_load_already_parked():
    """Every arm joint's initial_value must equal its travel-pose angle, and the gain
    must stay at 1.9.

    This variant shipped with gain 6.0 while all twelve arm joints still loaded at 0.0 -
    both arms straight out, each hand 0.524 m from base centre, 126 deg from parked. In
    section B that exact combination whipped the arms hard enough to trip the protective
    stop before the first cycle printed; the arms froze extended and the rotating base
    swept the objects across the room. The faster timings are fine, the unparked start
    was not.
    """
    import pathlib
    xacro = (pathlib.Path(__file__).resolve().parents[2]
             / 'pnp_description' / 'urdf' / 'ros2_control.xacro')
    src = xacro.read_text()
    want = dict(zip(
        # j2b sits between j2 and j3, so a positional j1..j7 list would pair left_j3's
        # init with the ROLL's travel angle and report a phantom 94 deg error.
        [f'left_{j}' for j in ik.ARM_JOINT_SUFFIX]
        + [f'right_{j}' for j in ik.ARM_JOINT_SUFFIX],
        list(ch.TRAVEL_LEFT) + list(ch.TRAVEL_RIGHT)))
    found = dict(re.findall(r'<xacro:ctrl_joint name="(\w+)"\s+init="([-0-9.]+)"', src))
    missing = sorted(set(want) - set(found))
    assert not missing, (f'no init= on {missing}; those joints load at 0.0, straight out')
    for j, target in want.items():
        got = float(found[j])
        assert abs(got - target) < 1e-3, (
            f'{j} loads at {got:+.4f} but parked is {target:+.4f} - '
            f'{abs(got - target) * 57.3:.0f} deg of error applied as one step')
    gain = re.search(r'position_proportional_gain">([0-9.]+)<', src)
    assert gain and abs(float(gain.group(1)) - 1.9) < 1e-6, (
        f'gain is {gain.group(1) if gain else "missing"}, expected 1.9')


def test_every_gripper_command_is_as_wide_as_the_controller():
    """The gripper array must carry one value per joint the controller declares.

    THIS IS THE FAULT THE 7-DOF PORT INTRODUCED AND OFFLINE TESTS MISSED. Bringing over the
    shared arm added a distal phalanx per finger, so gripper_controller went from four
    joints to eight - but task_node still published four values. A ros2_control controller
    REJECTS a command of the wrong width outright, so the fingers would simply never have
    moved, and every offline check here would still have passed, because none of them looks
    at the width of what is published.

    Counted from the YAML that configures the controller and the source that publishes to
    it, so neither can be changed on its own.
    """
    import pathlib as _pl
    root = _pl.Path(__file__).resolve().parents[2]
    yaml = (root / 'pnp_description' / 'config' / 'controllers.yaml').read_text()
    # Anchor on the DEFINITION, a line with no leading space - 'gripper_controller' also
    # appears indented under controller_manager, and starting there picked up the ARM
    # controller's joint list instead and counted 7.
    lines = yaml.splitlines(True)
    at = next(i for i, l in enumerate(lines)
              if l.startswith('gripper_controller:'))
    blk = ''.join(lines[at:])
    blk = blk[blk.index('joints:'):]
    blk = blk[blk.index('[') + 1:blk.index(']')]
    n_ctrl = len([x for x in blk.replace('\n', ' ').split(',') if x.strip()])
    assert n_ctrl >= 4, 'could not read the gripper controller joint list'

    src = _node_src()
    n_pub = 0
    for chunk in src.split('grip_pub.publish(')[1:]:
        # MATCH PARENTHESES, do not look for '))'. A nested call like
        # _curl_for(float(val)) ends in '))' itself, so a naive scan truncated the
        # expression and this guard reported its own parse as a 4-wide command.
        depth, end = 1, len(chunk)
        for idx, chp in enumerate(chunk):
            if chp == '(':
                depth += 1
            elif chp == ')':
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        expr = chunk[:end]
        widths = []
        for part in expr.split('*')[1:]:
            digits = ''
            for ch_ in part.strip():
                if ch_.isdigit():
                    digits += ch_
                else:
                    break
            if digits:
                widths.append(int(digits))
        assert widths, 'a gripper publish has no explicit width: %s' % expr[:70]
        assert sum(widths) == n_ctrl, (
            'a gripper publish sends %d values but the controller declares %d joints. '
            'A width mismatch is rejected silently and the fingers never move: %s'
            % (sum(widths), n_ctrl, expr.replace('\n', ' ')[:70]))
        n_pub += 1
    assert n_pub, 'no gripper publish found - this test cannot detect anything'



# ================================================================= 2026-09-28
# THE SECTION REBUILT ON THE SHARED CODE. Everything below guards what was ported from the
# reorient section, or what the Gazebo run showed: the box's fingers closing at its top.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))                 # the section root


def _commanded_curl():
    t = open(os.path.join(os.path.dirname(__file__), '..', 'pnp_behavior',
                          'arm_commander.py')).read()
    return float(re.search(r'^CURL_CLOSED\s*=\s*([0-9.]+)', t, re.M).group(1))


def _fixtures():
    """The fixtures the config gives the planner: name -> (centre, yaw rad, size), WORLD."""
    txt = open(CFG).read()
    names = re.search(r'^\s*fixtures:\s*\[([^\]]*)\]', txt, re.M)
    assert names, 'no fixtures declared in the config'
    out = {}
    for f in [v.strip() for v in names.group(1).split(',') if v.strip()]:
        blk = re.search(rf'^    {f}:\n((?:      .+\n)+)', txt, re.M)
        assert blk, f'fixture {f} is listed but not defined'
        body = blk.group(1)
        centre = [float(v) for v in re.search(r'centre:\s*\[([^\]]+)\]', body).group(1).split(',')]
        size = [float(v) for v in re.search(r'size:\s*\[([^\]]+)\]', body).group(1).split(',')]
        yaw = math.radians(float(re.search(r'yaw:\s*([-0-9.]+)', body).group(1)))
        out[f] = (centre, yaw, size)
    return out


def _to_base(h):
    return lambda x, y: (math.cos(h) * x + math.sin(h) * y, -math.sin(h) * x + math.cos(h) * y)


ORDER = ('bottle', 'can', 'box')


def _shapes(n, h, placed):
    """One object as the node models it, from a base facing heading ``h``."""
    o = OBJECTS[n]
    if placed:
        a = math.radians(o['place_azimuth'])
        axis, z = a + math.pi / 2, o['place_surface_z'] + o['cz']
    else:
        a = math.radians(o['pick_azimuth'])
        axis, z = math.radians(o['axis_yaw']), o['surface_z'] + o['cz']
    bx, by = _to_base(h)(REACH * math.cos(a), REACH * math.sin(a))
    return col.object_capsules((bx, by, z), (math.cos(axis - h), math.sin(axis - h), 0.0),
                               o['length'], o['half_width'], o['round'])


def _slot(n):
    a = math.radians(OBJECTS[n]['place_azimuth'])
    wx, wy = REACH * math.cos(a), REACH * math.sin(a)
    for f, (c, yaw, size) in _fixtures().items():
        dx, dy = wx - c[0], wy - c[1]
        if (abs(math.cos(yaw) * dx + math.sin(yaw) * dy) <= size[0] / 2
                and abs(-math.sin(yaw) * dx + math.cos(yaw) * dy) <= size[1] / 2
                and size[2] < 0.5):
            return f
    return None


_PLANS = {}


def _planned(name):
    """Both halves of the cycle exactly as TaskNode.plan_half() makes them - the pick half
    from the parked pose, the place half from where the lift left the arms back to parked,
    the world as obstacles, fingers as commanded - with the robot facing the object for the
    first and the slot for the second, as it does. Cached, because it is the slow part."""
    if name in _PLANS:
        return _PLANS[name]
    o = OBJECTS[name]
    s_ = _s(o)
    idx = ORDER.index(name)

    def others(h):
        return [c for m in ORDER if m != name for c in _shapes(m, h, ORDER.index(m) < idx)]
    h1 = math.radians(o['pick_azimuth'])
    psi = math.radians(o['axis_yaw']) - h1
    sl = ch.left_sign(psi)
    legs1 = ch.pick_legs(REACH, 0.0, o['surface_z'] + o['cz'] + o['grip_dz'], psi, s_, sl)
    fx1 = list(mo.fixture_obstacles(_fixtures(), _to_base(h1), h1).values())
    fp1, obs1, w1 = ch.plan_inputs(legs1, o['close'], OPEN_POS,
                                   mo.release_closure(o['half_width']),
                                   lying=_shapes(name, h1, False), others=others(h1),
                                   fixtures=fx1)
    flat1 = [w for _, leg in legs1 for w in leg]
    QL1, QR1 = col.solve_path(flat1, finger_pos=fp1, seed_l=ch.TRAVEL_LEFT,
                              seed_r=ch.TRAVEL_RIGHT, obstacles=obs1, pin_start=True, wide=w1)
    h2 = math.radians(o['place_azimuth'])
    psi_to = math.radians(o['place_azimuth'] + 90.0) - h2
    fx2 = mo.fixture_obstacles(_fixtures(), _to_base(h2), h2)
    slot = [fx2.pop(_slot(name))]
    legs2 = ch.place_legs((REACH, 0.0), (REACH, 0.0), o['place_z'] + o['grip_dz'], psi, psi_to,
                          s_, sl, ch.LIFT_Z, ch.PLACE_DROP, o['half_width'])
    fp2, obs2, w2 = ch.plan_inputs(legs2, o['close'], OPEN_POS,
                                   mo.release_closure(o['half_width']),
                                   placed=_shapes(name, h2, True), others=others(h2),
                                   fixtures=list(fx2.values()), slot=slot)
    flat2 = [w for _, leg in legs2 for w in leg]
    QL2, QR2 = col.solve_path(flat2, finger_pos=fp2, seed_l=QL1[-1], seed_r=QR1[-1],
                              obstacles=obs2, pin_start=True,
                              pin_end=(ch.TRAVEL_LEFT, ch.TRAVEL_RIGHT), wide=w2)

    def at(legs):
        out, k = {}, 0
        for n, leg in legs:
            out[n] = list(range(k, k + len(leg)))
            k += len(leg)
        return out
    _PLANS[name] = (
        dict(label='pick', legs=legs1, QL=QL1, QR=QR1, fp=fp1, at=at(legs1), others=others(h1),
             fixtures=dict(zip(mo.fixture_obstacles(_fixtures(), _to_base(h1), h1), fx1)),
             handled=_shapes(name, h1, False), slot=None),
        dict(label='place', legs=legs2, QL=QL2, QR=QR2, fp=fp2, at=at(legs2), others=others(h2),
             fixtures=fx2, handled=_shapes(name, h2, True), slot=slot[0]))
    return _PLANS[name]


def test_shared_files_match_the_manifest():
    """The two sections are ONE robot. The files that make it so must be the same bytes.

    Until 2026-09-28 this section carried its own copies of the IK, the planner, the wall
    fit and the trajectory sender - copies of the reorient section's from before its fixes -
    and its own task primitives, which had drifted: wall-clock waits, no settle before
    closing, no flat/round gate. SHARED_FILES.sha256 at the section root lists the shared
    files and their hashes, and is identical in both sections. If this fails, a shared file
    was edited here without being copied to the other section (or the reverse).
    """
    import hashlib
    man = os.path.join(REPO, 'SHARED_FILES.sha256')
    assert os.path.exists(man), 'SHARED_FILES.sha256 is missing from the section root'
    pkg = os.path.join(os.path.dirname(__file__), '..', 'pnp_behavior')
    # this section belongs to these groups; pick-place has its own choreography, so it is
    # not in the 'flip' group that reorient and perceive share
    mine = ('all',)
    listed = []
    for line in open(man):
        if not line.strip() or line.startswith('#'):
            continue
        digest, fname, group = line.split()
        if group not in mine:
            continue
        listed.append(fname)
        got = hashlib.sha256(open(os.path.join(pkg, fname), 'rb').read()).hexdigest()
        assert got == digest, (
            f'{fname} differs from the shared copy - edit it in one section, copy it to the '
            f'other, and regenerate the manifest in both')
    for f in ('ik.py', 'collision.py', 'arm_commander.py', 'wall_ref.py', 'motion.py',
              'node_base.py'):
        assert f in listed, f'{f} is not in the shared manifest'


def test_the_box_is_gripped_round_its_sides_not_at_its_top():
    """THE GAZEBO FAULT: the fingers closed at the box's top and held nothing. Two causes,
    both guarded here.

    1. The node never read which objects are flat, so the distal phalanx curled on the box
       too, drove into its side and held the pads off it. It now sets curl_this_object from
       the config's round flag, parsed with _as_bool.
    2. The descent was one trajectory from parked, timed on the wall clock with a 2 s grace,
       and nothing waited for the arms to arrive - on a laptop simulating below real time the
       fingers closed while the hands were still coming down.
    """
    tn = open(TASK_NODE).read()
    base = open(NODE_BASE).read()
    assert "self.curl_this_object = bool(o['round'])" in tn, 'the curl is not shape-gated'
    assert re.search(r"'round':\s*_as_bool\(", base), 'round is not parsed through _as_bool'
    cfg = open(CFG).read()
    for name, o in OBJECTS.items():
        blk = re.search(rf'^    {name}:\n((?:      .+\n|\s*\n)+)', cfg, re.M).group(1)
        m = re.search(r'^\s*round:\s*(\w+)', blk, re.M)
        assert m and (m.group(1) == 'true') == o['round'], f'{name}: round flag missing/wrong'
    body = tn[tn.index('    def cycle(self, name):'):]
    order = [body.index(k) for k in ("self.move(leg['unpark']",
                                     "self.settle_arms(name, 'the hover')",
                                     "self.move(leg['approach']",
                                     "self.settle_arms(name, 'the grasp pose')",
                                     "self.curl_this_object = bool(o['round'])",
                                     "self.grippers(o['close']")]
    assert order == sorted(order), (
        'the arms must settle at the hover and at the grasp, and the jaw be gated, before '
        'the fingers close')
    m = re.search(r'^SETTLE_TOL\s*=\s*([0-9.]+)', base, re.M)
    assert m and float(m.group(1)) <= 0.005
    # and the fingers really do reach down its sides: the pad spans the box's upper sides
    o = OBJECTS['box']
    grasp = o['cz'] + o['grip_dz']
    lowest = grasp - (col.finger_reach(0.0) - col.LG)
    assert lowest < o['cz'] - 0.030 and lowest > 0.005, (
        f'the straight fingertips reach {lowest:.3f} m, not down the box sides')


def test_every_wait_is_on_simulation_time():
    """The controllers run on /clock; a wait on the wall clock expires while the arm moves."""
    src = _node_src()
    for fn in ('send_pair', 'settle_arms', 'grippers', 'set_grasp', 'check_grip', 'drive'):
        body = re.search(r'    def %s\(.*?(?=\n    def |\n    @)' % fn, src, re.S).group(0)
        assert 'time.time()' not in body, f'{fn}() still times itself on the wall clock'
        assert 'self._now()' in body or 'self._sleep(' in body, f'{fn}() is not on sim time'
    assert 'get_clock().now()' in src


def test_the_node_assigns_every_attribute_it_reads():
    """E22/E23: an attribute read and never assigned is an AttributeError waiting mid-run."""
    src = _node_src()
    api = {'create_publisher', 'create_subscription', 'create_timer', 'get_logger',
           'get_parameter', 'declare_parameter', 'destroy_node', 'get_clock'}
    assigned = set(re.findall(r'self\.(\w+)\s*(?:=|\[)', src))
    assigned |= set(re.findall(r'^    ([A-Z_]+)\s*=', src, re.M))
    methods = set(re.findall(r'def (\w+)\(', src))
    used = set(re.findall(r'self\.(\w+)', src))
    missing = sorted(used - assigned - methods - api)
    assert not missing, f'read but never assigned: {missing}'


def test_the_cycle_is_built_only_from_its_legs():
    """task_node builds its sequence ONLY through pick_legs() and place_legs(), and runs
    every leg - two copies of a sequence is how a removed phase keeps running."""
    src = open(TASK_NODE).read()
    for fn in ('approach_waypoints', 'lift_waypoints', 'yaw_waypoints', 'place_waypoints',
               'settle_waypoints', 'retreat_waypoints', 'transit_waypoints', 'full_cycle'):
        assert f'ch.{fn}(' not in src, f'task_node calls ch.{fn}() directly'
    assert src.count('ch.pick_legs(') == 1 and src.count('ch.place_legs(') == 1
    for leg in ('unpark', 'approach', 'lift'):
        assert f"leg['{leg}']" in src, f'the {leg} leg is never executed'
    for leg in ('carry', 'yaw', 'lower', 'settle', 'rise', 'park'):
        assert f"leg2['{leg}']" in src, f'the {leg} leg is never executed'


def test_the_planner_is_given_the_world():
    """Every fixture and every known object reaches the planner, pinned at parked."""
    src = open(TASK_NODE).read()
    body = src[src.index('    def plan_half('):src.index('    def cycle(')]
    for need in ('ch.plan_inputs(', 'obstacles=obs', 'pin_start=True', 'wide=wide',
                 'n != name', 'self.fixture_boxes()', '(ch.TRAVEL_LEFT, ch.TRAVEL_RIGHT)'):
        assert need in body, f'plan_half() is missing {need!r}'


def test_every_height_is_measured_from_a_surface():
    """The grasp from the surface it lies on, the set-down from the surface it is put on, the
    rise derived from the gripper - and the shelf top in the config IS the shelf's."""
    for name, o in OBJECTS.items():
        pick, place = ch.cycle_halves(REACH, o['surface_z'] + o['cz'], o['place_z'],
                                      _psi(name), math.radians(PLACE_PSI), o['length'])
        d1, d2 = dict(pick), dict(place)
        (pl, _), _r = d1['approach'][-1]
        assert abs(pl[2] - (o['surface_z'] + o['cz'] + o['grip_dz'])) < 1e-9
        (pl, _), _r = d2['settle'][-1]
        assert abs(pl[2] - (o['place_surface_z'] + o['cz'] + o['grip_dz'])) < 1e-9
        (pr, _), _r = d2['rise'][-1]
        rise = pr[2] - pl[2]
        assert abs(rise - mo.rise_clear(o['half_width'], o['grip_dz'])) < 1e-9
        tips = pr[2] - (col.finger_reach(0.0) - col.LG)
        assert abs(tips - (o['place_z'] + o['half_width'] + mo.TIP_CLEAR)) < 1e-9, (
            f'{name}: after the rise the tips are not TIP_CLEAR above the placed object')
    peds = [b for n, b in _shelf_boxes().items() if n.startswith('ped')]
    tops = {round(pose[2] + size[2] / 2.0, 6) for pose, size in peds}
    assert tops == {SHELF_TOP}, f'the shelf model tops are {tops}, not {SHELF_TOP}'


def test_the_fixtures_are_the_world():
    """Every fixture the planner avoids must be a collision box in the shelf model."""
    boxes = _shelf_boxes()
    sdf_name = {'back_wall': 'backwall', 'side_wall': 'sidewall'}
    fx = _fixtures()
    for f, (centre, yaw, size) in fx.items():
        pose, sz = boxes[sdf_name.get(f, f)]
        assert max(abs(a - b) for a, b in zip(centre, pose[:3])) < 1e-9, f'{f}: centre differs'
        assert max(abs(a - b) for a, b in zip(size, sz)) < 1e-9, f'{f}: size differs'
        assert abs(math.remainder(yaw - pose[5], 2 * math.pi)) < 1e-4, f'{f}: yaw differs'
    assert {'back_wall', 'side_wall', 'ped1', 'ped2', 'ped3'} <= set(fx)


def test_the_wall_constants_are_the_faces_in_the_world():
    """wall_ref's planes must be the wall FACES in the shelf model: -0.45, not the -0.46
    centres - a 10 mm bias in every pose fix, and so in every grasp and placement."""
    boxes = _shelf_boxes()
    (ax, *_), (sx, *_) = boxes['backwall']
    (_bx, by, *_), (_tx, ty, *_) = boxes['sidewall']
    assert abs(wr.WALL_A_X - (-(ax + sx / 2.0))) < 1e-9
    assert abs(wr.WALL_B_Y - (-(by + ty / 2.0))) < 1e-9
    hs = open(os.path.join(os.path.dirname(__file__), 'harness.py')).read()
    assert 'world_wall_faces()' in hs and 'synth_scan(\n            self.heading, WALL_A' in hs


def test_no_planned_joint_move_spins_the_long_way():
    """Raw joint steps within the cap, and the halves start and end at parked."""
    for name in OBJECTS:
        pick, place = _planned(name)
        for Q, park in ((pick['QL'], ch.TRAVEL_LEFT), (pick['QR'], ch.TRAVEL_RIGHT)):
            assert max(abs(a - b) for a, b in zip(Q[0], park)) < 1e-9, (
                f'{name}: the pick half does not start at the parked pose')
        for Q, park in ((place['QL'], ch.TRAVEL_LEFT), (place['QR'], ch.TRAVEL_RIGHT)):
            assert max(abs(a - b) for a, b in zip(Q[-1], park)) < 1e-9, (
                f'{name}: the place half does not end at the parked pose')
        for half in (pick, place):
            for Q in (half['QL'], half['QR']):
                for i in range(1, len(Q)):
                    step = max(abs(a - b) for a, b in zip(Q[i], Q[i - 1]))
                    assert step <= col.JOINT_MAX_STEP + 1e-9, (
                        f'{name} ({half["label"]}): waypoint {i - 1} -> {i} moves a joint '
                        f'{math.degrees(step):.0f} deg')
        # and the place half starts EXACTLY where the pick half ended
        for a, b in ((pick['QL'][-1], place['QL'][0]), (pick['QR'][-1], place['QR'][0])):
            assert max(abs(x - y) for x, y in zip(a, b)) < 1e-9


def test_the_arms_keep_clear_of_the_world_along_the_real_path():
    """Along the path the controller really takes, joint-space interpolation included:
      * every OTHER object, lying or already on the shelf, and every fixture: all links
      * the pedestal being placed on: the arms while the hands work into it, else all links
      * the object just placed: the arms while rising off it, all links on the way home
    """
    for name in OBJECTS:
        for half in _planned(name):
            QL, QR, fp, at = half['QL'], half['QR'], half['fp'], half['at']
            into = {i for n in ch.SLOT_LEGS if n in at for i in at[n]}
            leave = {i for n in ch.LEAVE_LEGS if n in at for i in at[n]}
            home = {i for n in ch.HOME_LEGS if n in at for i in at[n]}
            worst = {}
            for i in range(len(QL) - 1):
                for Q, sd in ((QL, +1), (QR, -1)):
                    for f in (0.0, 0.25, 0.5, 0.75):
                        q = [a + f * (b - a) for a, b in zip(Q[i], Q[i + 1])]
                        caps = col.arm_capsules_fine(q, sd, fp[i])
                        terms = [('other objects', c, None) for c in half['others']]
                        terms += [(fx, b, None) for fx, b in half['fixtures'].items()]
                        if half['slot'] is not None:
                            terms.append(('the slot', half['slot'],
                                          col.ARM_LINKS if i in into else None))
                        if half['label'] == 'place' and (i in leave or i in home):
                            terms += [('the placed object', c,
                                       col.ARM_LINKS if i in leave else None)
                                      for c in half['handled']]
                        for what, c, links in terms:
                            g = col.obstacle_clearance(caps, c, links)
                            if g < worst.get(what, (9.9,))[0]:
                                worst[what] = (g, i)
            for what, (g, i) in worst.items():
                assert g > 0.010, (
                    f'{name} ({half["label"]}): an arm comes {g * 1000:+.1f} mm from '
                    f'{what} near waypoint {i}')


def test_the_arm_never_intersects_itself_on_the_plan():
    """Self-clearance on the plan the node makes, fingers as commanded."""
    for name in OBJECTS:
        for half in _planned(name):
            for i, (ql, qr) in enumerate(zip(half['QL'], half['QR'])):
                for q, sd in ((ql, +1), (qr, -1)):
                    g, who = col.self_clearance(col.arm_capsules_fine(q, sd, half['fp'][i]),
                                                optimistic=False)
                    assert g > -0.015, (
                        f'{name} ({half["label"]}): the arm passes {-g * 1000:.1f} mm into '
                        f'itself at waypoint {i} ({who})')


def test_the_suction_carry_keeps_the_fingers_out_of_the_object():
    """Carrying on the cup, the pads sit 0-1 mm outside the surface; they squeeze again
    before the release so the settle is held by friction. The same rule as reorient's."""
    base = open(NODE_BASE).read()
    ease = float(re.search(r'^CARRY_EASE\s*=\s*([0-9.]+)', base, re.M).group(1))
    for name, o in OBJECTS.items():
        gap = (o['close'] + ease) - mo.PAD_HALF - o['half_width']
        assert -0.0005 <= gap <= 0.0015, f'{name}: carrying, the pad sits {gap * 1000:+.1f} mm off'
    assert re.search(r'^\s*grasp_mode:\s*suction', open(CFG).read(), re.M), (
        'pick-place must use the same grasp as the reorient section')


def test_config_masses_are_the_models():
    """GRASP QUALITY and SUCTION lines are computed from these; they must be the real ones."""
    for name, o in OBJECTS.items():
        sdf = open(os.path.join(REPO, 'src', 'pnp_gazebo', 'models', name, 'model.sdf')).read()
        m = re.search(r'<mass>([0-9.]+)</mass>', sdf)
        assert m and abs(float(m.group(1)) - o['mass']) < 1e-9, (
            f'{name}: config mass {o["mass"]} vs model {m.group(1) if m else None}')


def test_the_pad_half_thickness_is_the_urdf():
    """motion.PAD_HALF sets the release opening and the carry ease; it must be the pad."""
    xac = open(os.path.join(REPO, 'src', 'pnp_description', 'urdf', 'arm.xacro')).read()
    th = {float(v) for v in re.findall(r'<box size="0\.080 ([0-9.]+) [0-9.]+"/>', xac)}
    assert th == {2 * mo.PAD_HALF}, f'pads are {th}, motion.PAD_HALF is {mo.PAD_HALF}'


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
    for name in OBJECTS:
        for half in _planned(name):
            reps = [col.check_pose(a, b, half['fp'][i])
                    for i, (a, b) in enumerate(zip(half['QL'], half['QR']))]
            d = min(reps, key=lambda r: r['worst'])
            print('%-7s %-5s the plan the node makes: worst %+.4f m (%s) | arm-arm %+.3f '
                  'floor %+.3f base %+.3f' % (name, half['label'], d['worst'], d['worst_name'],
                                              min(r['arm_arm'] for r in reps),
                                              min(r['floor'] for r in reps),
                                              min(r['base'] for r in reps)))
    print('fingertips reach %.3f m past the grasp point, open'
          % (col.finger_reach(0.0) - col.LG))
    sys.exit(1 if fails else 0)
