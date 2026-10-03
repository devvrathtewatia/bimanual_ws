"""Offline validation for section B. No ROS, no Gazebo.

    python3 test/test_flip.py

THREE FAULTS THIS SUITE NOW CATCHES THAT IT PREVIOUSLY MISSED
------------------------------------------------------------
1. **A hand underneath the object.** Standing an elongated object up stacks its two
   ends vertically, so the two hands finish one above the other. The old sequence
   lowered both, putting the lower hand between the object and the floor — measured
   at 15–30 mm above the ground, holding the thing directly above it. Perfectly
   reachable, so a reachability test saw nothing wrong.

2. **The arms swapping targets mid-flip.** Left/right used to be decided by
   whichever hand had the larger y, re-evaluated at every step. At the top of the
   flip both ends sit on the same vertical line, the comparison tipped over, and the
   two arms traded places in a single waypoint — a 170 mm instantaneous jump.

3. **The load-bearing hand trapped below.** Only the left palm carries the weld, so
   the flip must leave the LEFT hand on top or the object cannot be set down at all.

Each is now an explicit invariant below. The clearance checks run against
``collision.py``, which models links as bodies rather than points.
"""
import math

# THE CONTACT SURFACE IS THE RIDGE TIP, NOT THE PAD FACE. The trapezoidal jaw puts
# two ridges 6 mm proud of the 6 mm pad half-thickness, so the surface that meets
# the object sits 12 mm inside the joint position. Was 0.006 with flat pads.
def _pad_reach():
    """How far the pad's inner face stands proud of the commanded finger position.

    READ FROM THE URDF, NEVER HARDCODED. This constant was 0.012 - a 6 mm plate plus a
    6 mm ridge - and when the ridge was removed the URDF went to 0.006 while this stayed
    at 0.012. The suite then computed a face position 6 mm deeper than the robot actually
    has, so a gripper that closed 3 mm SHORT OF THE OBJECT passed all 32 checks and mimed
    an entire cycle in Gazebo over a bottle that never left the floor.
    """
    import pathlib as _pl, re as _re
    t = (_pl.Path(__file__).resolve().parents[2] / 'reorient_description' / 'urdf'
         / 'arm.xacro').read_text()
    th = {float(v) for v in _re.findall(r'<box size="0\.080 ([0-9.]+) [0-9.]+"/>', t)}
    assert len(th) == 1, 'finger pads disagree on thickness: %s' % th
    reach = th.pop() / 2.0
    # RIDGES STAND PROUD OF THE PAD, so they, not the pad face, make contact. Forgetting
    # this is what shipped a gripper that closed 3 mm short of the object.
    m = _re.search(r'name="RIDGE_H"\s+value="([0-9.]+)"', t)
    if m:
        reach += float(m.group(1))
    return reach


PAD_REACH = _pad_reach()
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reorient_behavior import choreography as ch
from reorient_behavior import collision as col
from reorient_behavior import ik

CFG = os.path.join(os.path.dirname(__file__), '..', 'config', 'objects.yaml')

REACH = 0.30
FLIP_Z = 0.24
LATERAL = math.pi / 2
OPEN_POS = 0.065
MIN_CLEARANCE = 0.004
GRIP_DZ = 0.025
PICK_PSI = {'bottle': 75.0, 'box': 105.0}     # object axis in the BASE frame, deg
CANON_PSI = 90.0                             # canonical axis the flip happens at
OBJECTS = {
    'bottle': dict(cz=0.045, upright_cz=0.120, length=0.240, close=0.048,
                   azimuth=0.0, axis_yaw=75.0, half_width=0.045,
                   round=True, mass=0.45),
    'box': dict(cz=0.045, upright_cz=0.110, length=0.220, close=0.048,
                azimuth=-55.0, axis_yaw=50.0, half_width=0.045,
                round=False, mass=0.50),
}


def _s(o):
    return ch.grip_half_separation(o['length'])


def _psi(name):
    return math.radians(PICK_PSI[name])


def _name(o):
    return 'bottle' if o['length'] == 0.240 else 'box'


def _legs(o):
    """The cycle AS THE ROBOT RUNS IT - the one list task_node plans and executes.

    This used to be assembled here from the individual phase functions, and it drifted:
    it still contained the handover phases for a whole session after the robot stopped
    running them, so every clearance in this file was measured on a cycle that was not
    the one being sent.
    """
    return ch.cycle_legs((REACH, 0.0), o['cz'], o['upright_cz'], _psi(_name(o)), _s(o),
                         FLIP_Z, ch.HOVER_DZ, ch.PLACE_DROP, o['half_width'])


def _cycle(o, name=None):
    return [w for _, leg in _legs(o) for w in leg]


def _at(o):
    """leg name -> the waypoint indices it occupies in _cycle(o)."""
    out, k = {}, 0
    for n, leg in _legs(o):
        out[n] = list(range(k, k + len(leg)))
        k += len(leg)
    return out


ORDER = ('bottle', 'box')          # the order the config handles them in


def _scene(name):
    """The world as TaskNode._object_capsules builds it while ``name`` is being handled:
    the robot facing it, the handled object lying and then standing where it lay, every
    object handled earlier standing where it was put, every later one lying where it is."""
    h = math.radians(OBJECTS[name]['azimuth'])

    def caps(n, standing):
        on = OBJECTS[n]
        wx = REACH * math.cos(math.radians(on['azimuth']))
        wy = REACH * math.sin(math.radians(on['azimuth']))
        bx, by = math.cos(h) * wx + math.sin(h) * wy, -math.sin(h) * wx + math.cos(h) * wy
        if standing:
            return col.object_capsules((bx, by, on['upright_cz']), (0.0, 0.0, 1.0),
                                       on['length'], on['half_width'], on['round'])
        a = math.radians(on['axis_yaw']) - h
        return col.object_capsules((bx, by, on['cz']), (math.cos(a), math.sin(a), 0.0),
                                   on['length'], on['half_width'], on['round'])
    others = []
    for n in ORDER:
        if n != name:
            others += caps(n, ORDER.index(n) < ORDER.index(name))
    return caps(name, False), caps(name, True), others


_PLANS = {}


def _planned(o):
    """The whole-cycle plan exactly as TaskNode.plan_cycle() makes it - seeded and pinned to
    the parked pose, the world as obstacles, fingers as commanded - with its per-waypoint
    finger closure. Cached, because it is the slow part."""
    key = _name(o)
    if key not in _PLANS:
        lying, standing, others = _scene(key)
        fp, obs, wide = ch.plan_inputs(_legs(o), o['close'], OPEN_POS, lying, standing,
                                       others)
        QL, QR = col.solve_path(_cycle(o), finger_pos=fp, seed_l=ch.TRAVEL_LEFT,
                                seed_r=ch.TRAVEL_RIGHT, obstacles=obs, pin_start=True,
                                pin_end=(ch.TRAVEL_LEFT, ch.TRAVEL_RIGHT), wide=wide)
        _PLANS[key] = (QL, QR, fp)
    return _PLANS[key]


def _plan(o):
    QL, QR, _fp = _planned(o)
    return QL, QR


def _reachable7(p, R, side):
    """SEVEN-DOF reachability: SOME swivel and either shoulder solution solves it."""
    return any(ik.solve_branches(p, R, side, math.radians(d), True)
               for d in range(0, 360, 5))


def _binding_half_width(o):
    """Widest part of the object the FINGER PLATE can reach.

    Not the width at the grasp height. Same helper as section A: the finger spans about
    10 to 45 mm about the object's axis, so on a cylinder the binding width is at the
    bottom of that span. The relationship is invariant under the flip, because the
    grasp offset and the approach axis rotate together, so one value covers the cycle.
    """
    if not o['round']:
        return o['half_width']
    lo = max(0.0, min(GRIP_DZ - (col.FINGERTIP_X - col.LG), o['half_width']))
    return math.sqrt(max(0.0, o['half_width'] ** 2 - lo ** 2))


def test_palm_clears_the_object_in_both_orientations():
    """The section A palm fault, which section B had too.

    The palm plate sits 0.027 m behind the grasp point, so gripping on the axis of a
    90 mm object put the palm 18 mm INSIDE it and the arm pressed a rigid body until
    the base lifted. The grasp point is offset off the axis instead. Because the offset
    is expressed in the OBJECT's frame it rotates with the object, so it works lying
    down and standing up - both are checked.
    """
    palm_behind = col.LG - 0.032 - 0.011
    for name, o in OBJECTS.items():
        lying = o['cz'] + GRIP_DZ + palm_behind
        assert lying > o['cz'] + o['half_width'] + 0.004, (
            '%s: lying, palm underside %.4f vs top %.4f'
            % (name, lying, o['cz'] + o['half_width']))
        standing = GRIP_DZ + palm_behind + 0.032
        assert standing > o['half_width'] + 0.004, (
            '%s: standing, palm %.4f from the axis vs radius %.4f'
            % (name, standing, o['half_width']))


def test_fingers_do_not_gouge_into_the_object():
    for name, o in OBJECTS.items():
        pen = _binding_half_width(o) - (o['close'] - PAD_REACH)
        assert 0.001 <= pen <= 0.004, (
            '%s: finger penetration %.1f mm (close %.3f, binding %.4f)'
            % (name, pen * 1000, o['close'], _binding_half_width(o)))


def test_grip_holds_the_object_after_the_suction_is_released():
    """During the settle BOTH hands hold it, by friction alone, with the suction off.

    Read from the URDF - both hands' finger efforts and the effective friction - not
    restated. If this fails the object drops the last place_drop onto the floor.
    """
    eff_left, eff_right = _finger_efforts()
    mu = min(_pad_mu(), _object_mu())
    hold = mu * 2.0 * (eff_left + eff_right)
    for name, o in OBJECTS.items():
        need = o['mass'] * 9.81
        assert hold > need * 2.0, (
            '%s: four fingers hold %.2f N by friction but the object weighs %.2f N'
            % (name, hold, need))


def test_the_two_handed_phase_takes_the_object_all_the_way_up():
    """BOTH HANDS CARRY IT TO VERTICAL - there is no handover angle.

    At six DOF the flip had to stop at 70 deg: the upright pose is coaxial, both hands land
    on one vertical line 170 mm apart, and with a mirrored elbow the arms met - -11.4 mm of
    arm-arm clearance and -19.6 mm of the holding arm inside its own forearm. The seventh
    axis routes the two elbows round opposite sides of that column. The angle used to be a
    config value, flip_handover_deg; it is now a constant and the config may not carry it.
    """
    assert abs(ch.FLIP_ANGLE - math.pi / 2) < 1e-12, (
        'the flip goes to %.1f deg, not 90 - part of the rotation would have to be '
        'single-handed again' % math.degrees(ch.FLIP_ANGLE))
    assert not hasattr(ch, 'FLIP_HANDOVER') and not hasattr(ch, 'FULLY_BIMANUAL'), (
        'the handover switch is back in choreography - two copies of the sequence is how '
        'a removed handover kept running in Gazebo for a session')
    R = ik.tool_down_fingers_along(LATERAL)
    for name, o in OBJECTS.items():
        legs = dict(_legs(o))
        last = legs['flip%d' % ch.FLIP_SEGMENTS][-1]
        want = ch._ends([REACH, 0.0, FLIP_Z], LATERAL, _s(o), R,
                        ch.FLIP_DIR * math.radians(90.0))
        assert math.dist(last[0][0], want[0][0]) < 1e-9, (
            '%s: the two-handed flip does not reach vertical' % name)
        lp, rp = last
        assert math.hypot(lp[0][0] - rp[0][0], lp[0][1] - rp[0][1]) < 1e-9, (
            '%s: the two hands are not on one vertical line at the top, so this is not '
            'the coaxial pose the seventh axis was added for' % name)


def _rel(pair):
    """Right hand's position and rotation expressed in the LEFT hand's frame."""
    (pl, Rl), (pr, Rr) = pair
    d = [pr[k] - pl[k] for k in range(3)]
    pos = [sum(Rl[r][k] * d[r] for r in range(3)) for k in range(3)]
    rot = ik.mat_mul(ik.transpose(Rl), Rr)
    return pos, rot


def test_both_hands_stay_on_the_object_from_grasp_to_floor():
    """THE BIMANUAL INVARIANT, over EVERY waypoint from the grasp until the object stands
    on the floor: the right hand keeps exactly the pose it gripped at, relative to the left.

    Two hands holding one rigid object cannot move relative to each other. So if either
    hand leaves - the old 70 deg handover, or the set-down that sent the right hand to a
    parked pose while the left lowered it alone - the relative pose changes. This is
    checked on the cycle the robot actually runs, and the harness checks the same thing on
    the running task node.
    """
    for name, o in OBJECTS.items():
        legs = _legs(o)
        grasp = dict(legs)['approach'][-1]
        ref_p, ref_R = _rel(grasp)
        span = math.dist(grasp[0][0], grasp[1][0])
        assert abs(span - 2.0 * _s(o)) < 1e-9, '%s: the grasp span is wrong' % name
        seen = []
        for leg, seq in legs:
            if leg not in ch.CARRY_LEGS:
                continue
            seen.append(leg)
            for i, pair in enumerate(seq):
                p, Rm = _rel(pair)
                dp = math.dist(p, ref_p)
                dR = max(abs(Rm[r][c] - ref_R[r][c]) for r in range(3) for c in range(3))
                assert dp < 1e-9 and dR < 1e-9, (
                    '%s: in the %s leg, waypoint %d, the right hand has moved %.1f mm '
                    'relative to the left - one hand has let go of the object'
                    % (name, leg, i, dp * 1000))
        assert tuple(seen) == ch.CARRY_LEGS, (
            '%s: carry legs %s, expected %s' % (name, seen, ch.CARRY_LEGS))
        # and the carry ends with the object ON THE FLOOR, upright, still in both hands
        (pl, _), (pr, _) = dict(legs)['settle'][-1]
        assert abs(pl[2] - (o['upright_cz'] + _s(o))) < 1e-9, (
            '%s: the settle does not end with the object standing on the floor' % name)
        assert abs(pr[2] - (o['upright_cz'] - _s(o))) < 1e-9


def test_the_hands_leave_the_standing_object_the_way_they_can():
    """After letting go, NOTHING MOVES SIDEWAYS NEAR THE FLOOR - whatever is lying around.

    The 2026-09-27 run backed the lower hand sideways at floor height into the box beside the
    bottle. The fix is not to know where the box is - in the finished system objects are
    scattered and perceived - but a sequence that cannot do it:
      release   both hands slide straight UP: the upper off the top, the lower to CLUTTER_Z,
                which is above anything that could be lying on the floor
      clear     only then does the lower hand move sideways, straight back along its own
                approach axis, by exactly enough for its fingertips to clear the object
    """
    for name, o in OBJECTS.items():
        d = dict(_legs(o))
        (pl0, _Rl), (pr0, Rr) = d['settle'][-1]
        for (pl, _), (pr, _) in d['release']:
            assert abs(pl[0] - pl0[0]) + abs(pl[1] - pl0[1]) < 1e-9, (
                '%s: the upper hand moves sideways while leaving the top' % name)
            assert abs(pr[0] - pr0[0]) + abs(pr[1] - pr0[1]) < 1e-9, (
                '%s: the lower hand moves sideways before it has climbed clear' % name)
        (pl1, _), (pr1, _) = d['release'][-1]
        assert pl1[2] - pl0[2] > 0.05, '%s: the upper hand barely lifts' % name
        assert pr1[2] >= ch.CLUTTER_Z - 1e-9, (
            '%s: the lower hand only climbs to %.3f before moving sideways, not above floor '
            'clutter at %.3f' % (name, pr1[2], ch.CLUTTER_Z))
        assert (pl1[2] - ch.HAND_HALF_H) - (pr1[2] + ch.HAND_HALF_H) >= ch.HAND_GAP - 1e-9, (
            '%s: the lower hand climbs into the upper one' % name)
        appr = [Rr[i][0] for i in range(3)]
        for (plc, _), (prc, _) in d['clear']:
            assert math.dist(plc, pl1) < 1e-9, '%s: the upper hand moves during clear' % name
            assert prc[2] >= ch.CLUTTER_Z - 1e-9, (
                '%s: the lower hand moves sideways below CLUTTER_Z' % name)
        (_, _), (pr2, _) = d['clear'][-1]
        dvec = [pr2[k] - pr1[k] for k in range(3)]
        along = -sum(dvec[k] * appr[k] for k in range(3))
        assert math.dist(dvec, [-along * a for a in appr]) < 1e-9, (
            '%s: the lower hand does not back straight out along its approach axis' % name)
        # the tips end TIP_CLEAR off the object's near side, derived from the gripper
        tip = col.FINGERTIP_X + col.DISTAL_L - ik.LG
        left_at = (tip - along) - (GRIP_DZ - o['half_width'])
        assert abs(left_at + ch.TIP_CLEAR) < 1e-9, (
            '%s: after backing out the fingertips are %.1f mm from the object, want %.0f'
            % (name, -left_at * 1000, ch.TIP_CLEAR * 1000))


# --------------------------------------------------------------- reachability
TASK_NODE = os.path.join(os.path.dirname(__file__), '..', 'reorient_behavior',
                         'task_node.py')
NODE_BASE = os.path.join(os.path.dirname(__file__), '..', 'reorient_behavior',
                         'node_base.py')
FLIP_TASK = os.path.join(os.path.dirname(__file__), '..', 'reorient_behavior',
                         'flip_task.py')


def _node_src():
    """What the running node is made of: this section's task_node.py, the shared stand-up in
    flip_task.py, and the shared node_base.py every primitive comes from - in that order. A
    check that read task_node.py alone would go blind the moment code moved into a shared
    file."""
    return '\n'.join(open(f).read() for f in (TASK_NODE, FLIP_TASK, NODE_BASE))
NODE_API = {'create_publisher', 'create_subscription', 'create_timer', 'get_logger',
            'get_parameter', 'declare_parameter', 'destroy_node', 'get_clock'}


def test_the_node_assigns_every_attribute_it_reads():
    """The fault that shipped this section with an AttributeError waiting in it.

    ``self.place_drop`` and ``ch.FLIP_HANDOVER`` were both read by the release sequence
    and never assigned, because the patch that added them anchored on an ``open_pos``
    default of 0.065 while this file says 0.062 - and ``str.replace`` reports nothing
    when it matches nothing. The section would have died at the first placement, and the
    offline suite passed throughout because it never executes the node. Third instance of
    this class; see E22/E23 in ERROR_CATALOGUE.txt.
    """
    src = _node_src()
    assigned = set(re.findall(r'self\.(\w+)\s*(?:=|\[)', src))
    # class attributes too - ROBOT_MODEL is set on the class, per section
    assigned |= set(re.findall(r'^    ([A-Z_]+)\s*=', src, re.M))
    methods = set(re.findall(r'def (\w+)\(', src))
    used = set(re.findall(r'self\.(\w+)', src))
    missing = sorted(used - assigned - methods - NODE_API)
    assert not missing, f'read but never assigned: {missing}'


def test_the_runtime_checks_are_actually_wired_in():
    """A check that is defined but never called reads as reassurance."""
    src = _node_src()
    for fn in ('check_grip', 'check_lifted', 'check_yaw', 'check_upright',
               'check_stands', 'check_both_hands_holding', 'check_grasp_centring',
               'check_localisation', 'settle_arms'):
        assert f'def {fn}(' in src, f'{fn} is not defined'
        assert src.count(f'self.{fn}(') >= 1, f'{fn} is defined but never called'
    # the stand check must be taken TWICE - a tippy object can pass at the instant of
    # release and topple a second later
    assert 'at release' in src and 'after settling' in src, (
        'the stand check must sample both at release and after settling')


def test_every_choreography_call_has_the_right_arity():
    """A wrong argument list that only fails at runtime, checked against real signatures."""
    import ast
    import inspect
    src = _node_src()
    checked = 0
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == 'ch'):
            continue
        fn = getattr(ch, node.func.attr, None)
        if not callable(fn):
            continue
        ps = list(inspect.signature(fn).parameters.values())
        req = sum(1 for q in ps if q.default is inspect.Parameter.empty)
        given = len(node.args) + len(node.keywords)
        checked += 1
        assert req <= given <= len(ps), (
            f'line {node.lineno}: ch.{node.func.attr} got {given} args, '
            f'takes {req}..{len(ps)} {tuple(q.name for q in ps)}')
    # The node now builds its whole sequence through ONE call, so there are few left to
    # check - but these two must be among them or this test has gone vacuous.
    names = {n.func.attr for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'ch'}
    assert {'cycle_legs', 'grip_half_separation'} <= names, (
        f'only {checked} choreography calls found: {sorted(names)}')


def test_pick_yaws_bracket_the_canonical_axis():
    """Both objects on the same side would leave one branch of left_sign() untested."""
    offs = [PICK_PSI[n] - CANON_PSI for n in OBJECTS]
    assert min(offs) < 0 < max(offs), f'pick yaws {offs} are all on one side'
    for name in OBJECTS:
        off = abs(PICK_PSI[name] - CANON_PSI)
        assert 5.0 <= off <= 20.0, (
            f'{name} is {off:.0f} deg off canonical - either too close to prove '
            f'anything or outside the measured +/-20 deg grasp envelope')


def test_the_flip_only_ever_runs_at_the_canonical_axis():
    """The flip envelope is ONE-SIDED, so it must not be handed an arbitrary yaw.

    Measured by sweep: axis lines 65..90 deg work and nothing above 90 does, because at
    the top of the flip the welded wrist sits (LG + grip_dz) = 95 mm along the object's
    axis, which pushes it away from the shoulder when cos(psi) < 0 - 0.340 m against a
    0.320 m limit. FLIP_DIR cannot fix it (both representatives of a line give the same
    sign) and no station fixed it either. Hence yaw-align first.
    """
    for name, o in OBJECTS.items():
        legs = _legs(o)
        order = [n for n, _ in legs]
        assert order.index('yaw') < order.index('flip1'), (
            'the yaw alignment must precede the flip')
        d = dict(legs)
        for k in range(1, ch.FLIP_SEGMENTS + 1):
            want = ch.flip_waypoints((REACH, 0.0), ch.CANONICAL_AXIS, _s(o), FLIP_Z,
                                     upto=ch.FLIP_ANGLE * k / ch.FLIP_SEGMENTS,
                                     frm=ch.FLIP_ANGLE * (k - 1) / ch.FLIP_SEGMENTS)
            assert d['flip%d' % k] == want, (
                f'{name}: flip segment {k} is not run at the canonical axis')
        # and the yaw leg ends there
        (pl, _), (pr, _) = d['yaw'][-1]
        got = math.atan2(pl[1] - pr[1], pl[0] - pr[0]) % math.pi
        assert abs(got - ch.CANONICAL_AXIS) < 1e-6, f'{name}: yaw ends off the flip axis'
    assert abs(ch.CANONICAL_AXIS - math.radians(CANON_PSI)) < 1e-12


def test_yaw_alignment_reaches_the_canonical_axis():
    for name, o in OBJECTS.items():
        seq = ch.yaw_waypoints((REACH, 0.0), _psi(name), math.radians(CANON_PSI),
                               _s(o), FLIP_Z)
        assert seq, f'{name}: no yaw waypoints'
        (pl, _), (pr, _) = seq[-1]
        got = math.atan2(pl[1] - pr[1], pl[0] - pr[0]) % math.pi
        want = math.radians(CANON_PSI)
        assert abs(((got - want + math.pi / 2) % math.pi) - math.pi / 2) < 1e-6, (
            f'{name}: ended at {math.degrees(got):.2f} deg, wanted {CANON_PSI}')


def test_flip_direction_matches_the_hand_that_must_end_on_top():
    """FLIP_DIR = -sign_left, or the welded hand ends up UNDERNEATH.

    Hard-coded to -1, which is correct only while sin(psi) >= 0. It is right for the
    canonical axis the flip now always uses, so the fault is dormant - but it is a trap
    for anyone who changes that axis, so it is asserted rather than commented.
    """
    canon = math.radians(CANON_PSI)
    assert ch.FLIP_DIR == -ch.left_sign(canon), (
        f'FLIP_DIR is {ch.FLIP_DIR} but the canonical axis needs '
        f'{-ch.left_sign(canon)} for the welded hand to finish on top')
    for name, o in OBJECTS.items():
        assert ch.left_ends_on_top(canon, _s(o), FLIP_Z), (
            f'{name}: the welded hand does not end on top')


def test_every_waypoint_is_reachable():
    for name, o in OBJECTS.items():
        for i, ((pl, Rl), (pr, Rr)) in enumerate(_cycle(o)):
            assert _reachable7(pl, Rl, +1), f'{name}: left unreachable at waypoint {i}'
            assert _reachable7(pr, Rr, -1), f'{name}: right unreachable at waypoint {i}'


def test_the_flip_needs_the_wrist_pitch_it_was_given():
    """The design justification, kept as a regression test.

    The flip is the motion that made four degrees of freedom insufficient: both
    hands' full orientation has to track the object's rotation. This records how much
    wrist pitch it actually demands, so shrinking that limit fails a test instead of
    quietly breaking the section.
    """
    worst = 0.0
    for name, o in OBJECTS.items():
        for (pl, Rl), (pr, Rr) in ch.flip_waypoints((REACH, 0.0), LATERAL,
                                                    _s(o), FLIP_Z):
            for p, R, side in ((pl, Rl, +1), (pr, Rr, -1)):
                worst = max(worst, abs(ik.solve(p, R, side)[4]))
    assert worst < ik.W_PITCH_LIM, 'the flip exceeds the wrist pitch limit'
    assert worst > 1.0, ('the flip barely uses the wrist - suspicious, the '
                         'geometry may not be what is intended')


# ------------------------------------------------------- the three invariants
def test_the_welded_hand_ends_up_on_top():
    """Invariant 3: only the left palm holds the object, so it must be the upper one."""
    for name, o in OBJECTS.items():
        assert ch.left_ends_on_top(LATERAL, _s(o), FLIP_Z), (
            f'{name}: the flip leaves the WELDED hand underneath the object, so it '
            f'cannot be set down')


def test_the_arms_never_swap_targets():
    """Invariant 2: the left/right assignment is pinned, not re-sorted.

    Checked by walking the flip and confirming the left hand rises while the
    right descends. A swap shows up as a sign reversal.

    TOLERANCE, and why it is not a weakening. The grasp point is offset off the
    object's axis and that offset rotates with the object, so each hand's height
    is s*(-sin theta) + grip_dz*cos theta rather than a pure sine. That peaks
    slightly before the end and comes back down a few millimetres - real
    geometry, not a swap. The fault this test exists to catch was an
    instantaneous 170 mm exchange between the arms, so a 6 mm allowance still
    catches it by a factor of thirty.
    """
    dip = 0.006
    for name, o in OBJECTS.items():
        seq = ch.flip_waypoints((REACH, 0.0), LATERAL, _s(o), FLIP_Z)
        zl = [pair[0][0][2] for pair in seq]
        zr = [pair[1][0][2] for pair in seq]
        assert all(b >= a - dip for a, b in zip(zl, zl[1:])), \
            f'{name}: left hand dropped mid-flip - targets swapped'
        assert all(b <= a + dip for a, b in zip(zr, zr[1:])), \
            f'{name}: right hand rose mid-flip - targets swapped'
        assert zl[-1] > zr[-1] + 0.02, f'{name}: left hand not on top'
        assert zl[-1] - zl[0] > 0.4 * _s(o), f'{name}: left hand barely rose'


def test_no_hand_is_ever_under_the_object_when_it_is_set_down():
    """Invariant 1: when the object touches down nothing is beneath it.

    The first version of this section lowered both hands with the lower one UNDER the
    object, 15-30 mm above the floor. Now the lower hand grips the bottom end from the
    SIDE: at touchdown its palm is outside the object's radius and its lowest plate clears
    the floor. Checked on the planned arms, with the fingers closed, not on grasp points.
    """
    for name, o in OBJECTS.items():
        legs = _legs(o)
        idx = sum(len(seq) for n, seq in legs[:[n for n, _ in legs].index('settle') + 1]) - 1
        QL, QR = _plan(o)
        (pl, _), (pr, _) = _cycle(o)[idx]
        assert pl[2] > o['upright_cz'], f'{name}: the upper hand is below the centre'
        assert pr[2] < o['upright_cz'], f'{name}: the lower hand is not on the bottom end'
        caps = col.arm_capsules_fine(QR[idx], -1, o['close'])
        floor = min(min(a[2], b[2]) - r for _n, a, b, r in caps)
        assert floor > 0.010, (
            f'{name}: at touchdown the lower hand is {floor * 1000:.1f} mm off the floor')
        for n, a, b, r in caps:
            if n.startswith('palm'):
                for q in (a, b):
                    rad = math.hypot(q[0] - REACH, q[1]) - r
                    assert rad > o['half_width'], (
                        f'{name}: the lower palm is inside the object footprint at touchdown')


# ------------------------------------------------------------------- geometry
def test_collision_model_agrees_with_the_ik():
    import random
    rng = random.Random(99)
    for _ in range(200):
        # DERIVED, not 6 or 7 written out. The seventh axis arrived by inserting a
        # name into ac.ARM_JOINT_SUFFIX, and this test is the one that proves
        # collision.link_frames and ik.fk walk the SAME chain - so it must follow
        # that list automatically or it silently stops covering the new joint.
        q = [rng.uniform(-1.2, 1.2) for _ in range(ik.N_ARM_JOINTS)]
        for side in (+1, -1):
            p_ik, _ = ik.fk(q, side)
            R, t = col.link_frames(q, side)['palm']
            appr = [R[0][0], R[1][0], R[2][0]]
            p_col = [t[i] + col.LG * appr[i] for i in range(3)]
            assert max(abs(p_ik[i] - p_col[i]) for i in range(3)) < 1e-9


def test_no_body_collides_anywhere_in_the_cycle():
    """Arm-arm, floor, base and mast at every waypoint of the plan the node makes."""
    for name, o in OBJECTS.items():
        QL, QR, fp = _planned(o)
        for i, (ql, qr) in enumerate(zip(QL, QR)):
            rep = col.check_pose(ql, qr, fp[i])
            assert rep['worst'] > MIN_CLEARANCE, (
                f'{name}: {rep["worst_name"]} clearance {rep["worst"]:+.4f} m at waypoint {i} '
                f'(arm-arm {rep["arm_arm"]:+.4f}, floor {rep["floor"]:+.4f}, '
                f'base {rep["base"]:+.4f})')


def test_fingertips_stay_above_the_floor():
    # From the GRASP HEIGHT, which is GRIP_DZ above the object's axis. The fingers
    # now reach well past the grasp point on purpose, so that they straddle the widest
    # part of the body rather than pinching its shoulder; the invariant is that the
    # TIPS stay above the floor, not that the reach is short.
    overhang = col.FINGERTIP_X - col.LG
    for name, o in OBJECTS.items():
        tip = o['cz'] + GRIP_DZ - overhang
        assert tip > 0.015, (
            f'{name}: fingertips at {tip:+.4f} m - underground')


def test_path_is_continuous():
    for name, o in OBJECTS.items():
        seq = _cycle(o)
        step = max(max(math.dist(seq[i][h][0], seq[i + 1][h][0])
                       for h in (0, 1)) for i in range(len(seq) - 1))
        assert step < 0.08, (
            f'{name}: biggest step {step:.3f} m - that is a velocity step, i.e. '
            f'a jerk')


def test_arms_start_on_their_own_sides():
    for name, o in OBJECTS.items():
        (pl, _), (pr, _) = _cycle(o)[0]
        assert pl[1] > 0 > pr[1], f'{name}: hands do not straddle the centreline'


def test_fingers_actually_reach_the_object():
    for name, o in OBJECTS.items():
        face = o['close'] - PAD_REACH
        assert face <= o['half_width'], f'{name}: finger face never makes contact'
        assert o['half_width'] - face <= 0.006, f'{name}: squeezing too hard'
        assert OPEN_POS - PAD_REACH > o['half_width'] + 0.005, \
            f'{name}: open fingers do not clear it'


def test_upright_height_is_half_the_length():
    for name, o in OBJECTS.items():
        assert abs(o['upright_cz'] - o['length'] / 2.0) < 1e-9, (
            f'{name}: upright_cz {o["upright_cz"]} is not half of '
            f'{o["length"]}')


# -------------------------------------------------------------------- config
def test_config_matches_this_file():
    txt = open(CFG).read()
    # compared against the module constants, not against literals repeated here:
    # the previous form hardcoded 0.29/0.30 and had to be edited by hand whenever the
    # station moved, which is exactly the kind of drift this test exists to prevent
    for key, want in (('reach', REACH), ('flip_z', FLIP_Z)):
        m = re.search(rf'^\s*{key}:\s*([-0-9.]+)', txt, re.M)
        assert m, f'{key} missing from the config'
        assert abs(float(m.group(1)) - want) < 1e-9, (
            f'{key}: config {m.group(1)} vs test {want}')
    # open_pos WAS NOT CHECKED HERE, and nothing else checked it either: a config value
    # of 0.070 over-travels the 0.065 finger joint limit and every test still passed.
    # It is read straight into self.open_pos by the node, so it must agree with the
    # constant this suite reasons about.
    m = re.search(r'^\s*open_pos:\s*([-0-9.]+)', txt, re.M)
    assert m, 'open_pos missing from the config'
    assert abs(float(m.group(1)) - OPEN_POS) < 1e-9, (
        f'open_pos: config {m.group(1)} vs test {OPEN_POS}')

    # THE HANDOVER KNOB IS GONE and must not come back: a value below 90 would make part
    # of the rotation single-handed, which is the behaviour this section removed.
    assert not re.search(r'^\s*flip_handover_deg:', txt, re.M), (
        'flip_handover_deg is back in the config - the flip is always fully bimanual')
    for name, o in OBJECTS.items():
        block = re.search(rf'^    {name}:\n((?:      .+\n)+)', txt, re.M)
        assert block, f'{name} missing from the config'
        body = block.group(1)
        for key in ('cz', 'upright_cz', 'length', 'close', 'axis_yaw', 'half_width'):
            m = re.search(rf'{key}:\s*([-0-9.]+)', body)
            assert m, f'{name}.{key} missing'
            assert abs(float(m.group(1)) - o[key]) < 1e-9, (
                f'{name}.{key}: config {m.group(1)} vs test {o[key]}')


def test_the_arms_load_already_parked():
    """Every arm joint's initial_value must equal its travel-pose angle.

    THE 2026-09-20 FAILURE, AS A TEST. All twelve arm joints shared initial_value 0.0,
    which is both arms straight out horizontally with each hand 0.524 m from base
    centre - 0.35 m outside the base's own radius. The first command was then 126 deg of
    error in one step, the arm whipped hard enough to trip the protective stop before
    the first cycle printed, the arms froze extended, and the base rotated with them
    stuck out and swept the objects across the room.

    Nothing downstream can detect this: every waypoint the cycle asks for is reachable
    and collision-free. The fault is entirely in where the arms are BEFORE the first
    waypoint, which no trajectory test looks at.
    """
    import pathlib
    xacro = (pathlib.Path(__file__).resolve().parents[2]
             / 'reorient_description' / 'urdf' / 'ros2_control.xacro')
    src = xacro.read_text()
    # Names come from arm_commander, NOT written out here: j2b is inserted at index 2,
    # so a positional list of j1..j7 would pair left_j3's init with the ROLL's travel
    # angle and report a phantom 94 deg error. Zipping the real name list against the
    # real travel pose keeps the two in step whatever the axis count.
    names = ([f'left_{j}' for j in ik.ARM_JOINT_SUFFIX]
             + [f'right_{j}' for j in ik.ARM_JOINT_SUFFIX])
    want = dict(zip(names, list(ch.TRAVEL_LEFT) + list(ch.TRAVEL_RIGHT)))
    assert len(want) == len(ch.TRAVEL_LEFT) + len(ch.TRAVEL_RIGHT) == 2 * ik.N_ARM_JOINTS, (
        'joint names and travel angles are different lengths - one was changed without '
        'the other')

    found = dict(re.findall(r'<xacro:ctrl_joint name="(\w+)"\s+init="([-0-9.]+)"', src))
    missing = sorted(set(want) - set(found))
    assert not missing, (
        f'these arm joints have no init= and will load at 0.0, i.e. straight out: '
        f'{missing}')

    for j, target in want.items():
        got = float(found[j])
        assert abs(got - target) < 1e-3, (
            f'{j} loads at {got:+.4f} rad but the travel pose wants {target:+.4f} - '
            f'the arm starts {abs(got - target) * 57.3:.0f} deg away from parked, and '
            f'that error is applied as a single step by the first command')

    # THE GAIN AND THE PARKED START ARE ONE DECISION, not two.
    # 6.0 tracks about 3x better than 1.9 (right_j4 off 0.151 rad against 0.448 on the
    # identical flip), and that lag is what lets the un-welded right hand drift off the
    # object. 6.0 is only safe BECAUSE the arms load parked: unparked, the first command
    # is a 126 deg step and 6.0 turns it into a whip that wrecked the 2026-09-20 run.
    # The parked assertion above is the precondition for this one, so they live together.
    gain = re.search(r'position_proportional_gain">([0-9.]+)<', src)
    assert gain, 'no position_proportional_gain found'
    assert abs(float(gain.group(1)) - 1.9) < 1e-6, (
        f'gain is {gain.group(1)}, expected 1.9. This has been changed three times now. '
        f'6.0 tracks better on paper and makes the arm RING - the base shakes and the '
        f'robot walks, observed twice on 2026-09-20. The right hand losing the object is '
        f'a weld problem, not a gain problem; fix it with a second weld on the right '
        f'palm, not here')


def test_the_grasp_frame_is_resolved_after_the_turn():
    """psi_pick and the aim point must be computed AFTER face(), never before.

    THE BOX BUG. psi = axis_yaw - heading cancels the heading only if the heading is
    the one the robot is at WHEN THE HANDS ARE PLACED. Both were computed before
    face(), so they used the pre-turn heading:

        bottle  azimuth   0 -> pre == post, error 0.00 deg, worked for days
        box     azimuth -55 -> psi 51.45 instead of 104.78, off by 53.33 deg,
                               each hand 67 mm from the end it aimed at

    The only object that ever worked is the one whose azimuth is zero, where the fault
    is arithmetically invisible. The harness now checks the resulting alignment in the
    world frame; this checks the ordering directly, because the ordering is the fault.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / 'reorient_behavior' / 'task_node.py').read_text().splitlines()
    at = {}
    for i, ln in enumerate(src, 1):
        if 'if not self.face(' in ln:
            at.setdefault('face', i)
        if 'psi_pick = self.base_frame_yaw' in ln:
            at.setdefault('psi', i)
        if 'self.aim(' in ln and 'def aim' not in ln:
            at.setdefault('aim', i)
        if 'self.stand_up(' in ln:
            at.setdefault('grasp', i)
    for k in ('face', 'psi', 'aim', 'grasp'):
        assert k in at, f'could not locate {k} in task_node.py'
    assert at['face'] < at['psi'], (
        f"psi_pick is computed at line {at['psi']} but the base does not turn until "
        f"line {at['face']} - the grasp angle would use the pre-turn heading, which is "
        f"53 deg wrong for any object whose azimuth is not zero")
    assert at['face'] < at['aim'], (
        f"aim() is called at line {at['aim']} but the turn is at line {at['face']} - "
        f"the aim point would be measured from the wrong place")
    assert at['psi'] < at['grasp'] and at['aim'] < at['grasp'], (
        'the grasp frame must be resolved before the hands are commanded')

    # and the position fix must actually be USED, not measured and discarded
    whole = (pathlib.Path(__file__).resolve().parents[1]
             / 'reorient_behavior' / 'task_node.py').read_text()
    assert 'xy = (self.reach, 0.0)' not in whole, (
        'the grasp centre is hardcoded to (reach, 0) again - that assumes the base sits '
        'exactly on the origin, which an in-place turn on a differential drive does not')


def test_the_arms_keep_clear_of_every_object_along_the_real_path():
    """No arm touches an object it is not handling - checked along the path the controller
    really takes, joint-space interpolation BETWEEN waypoints included.

    THE 2026-09-27 FAULTS: the right hand struck the box lying beside the bottle on its way
    out, and an arm swept the standing bottle over on the way home - the home move was one
    blind joint-space jump the planner never saw. Now the world is an obstacle to the
    planner, the trip home is a planned path, and this samples every step of it:
      * every OTHER object, all links, the whole cycle
      * the object just stood up: the arms while the hands leave it, everything on the way
        home (the hands are beside it by design until then)
    """
    for name, o in OBJECTS.items():
        QL, QR, fp = _planned(o)
        _lying, standing, others = _scene(name)
        at = _at(o)
        leave = {i for n in ch.LEAVE_LEGS for i in at[n]}
        park = {i for n in ch.HOME_LEGS for i in at[n]}
        worst = {'other': (9.9, None), 'placed': (9.9, None)}
        for i in range(len(QL) - 1):
            for Q, sd in ((QL, +1), (QR, -1)):
                for f in (0.0, 0.25, 0.5, 0.75):
                    q = [a + f * (b - a) for a, b in zip(Q[i], Q[i + 1])]
                    caps = col.arm_capsules_fine(q, sd, fp[i])
                    if others:
                        g = min(col.obstacle_clearance(caps, c) for c in others)
                        if g < worst['other'][0]:
                            worst['other'] = (g, i)
                    if i in leave or i in park:
                        links = None if i in park else col.ARM_LINKS
                        g = min(col.obstacle_clearance(caps, c, links) for c in standing)
                        if g < worst['placed'][0]:
                            worst['placed'] = (g, i)
        for what, (g, i) in worst.items():
            assert g > 0.010, (
                f'{name}: an arm comes {g * 1000:+.1f} mm from the {what} object at waypoint '
                f'{i} - it can knock it over')


def test_no_planned_joint_move_spins_the_long_way():
    """Every step of the plan, and both ends, is a RAW joint move inside the cap.

    THE BOTTLE WAS KNOCKED OVER BY THIS. The planner measured joint steps modulo 360 deg,
    but every joint stops short of +/-180, so a step it scored as 39 deg was travelled as
    321 deg the other way: right_j2b spinning almost a full turn on the last retreat
    waypoint (`right_j2b off 4.865` in the log). And the plan now starts and ends AT the
    parked joint vector, so there is no joint-space jump at either end either.
    """
    assert col._joint_step([-3.02], [2.58]) > 5.0, (
        'the step between -3.02 and +2.58 rad reads as small - it is being wrapped again')
    for name, o in OBJECTS.items():
        QL, QR, _fp = _planned(o)
        for Q, park in ((QL, ch.TRAVEL_LEFT), (QR, ch.TRAVEL_RIGHT)):
            assert max(abs(a - b) for a, b in zip(Q[0], park)) < 1e-9, (
                f'{name}: the plan does not start at the parked pose')
            assert max(abs(a - b) for a, b in zip(Q[-1], park)) < 1e-9, (
                f'{name}: the plan does not end at the parked pose')
            for i in range(1, len(Q)):
                step = max(abs(a - b) for a, b in zip(Q[i], Q[i - 1]))
                assert step <= col.JOINT_MAX_STEP + 1e-9, (
                    f'{name}: waypoint {i - 1} -> {i} moves a joint '
                    f'{math.degrees(step):.0f} deg')


def test_the_solver_can_reach_the_parked_pose():
    """The parked pose tips the upper arm back over the shoulder (j2 = -114 deg), which is
    the SECOND shoulder solution; asin() only ever gave the first, and the nearest solution
    it had for the parked hand was 147 deg of joint travel away - which is why the arms used
    to reach parked by a blind jump. With both_shoulders the exact parked vector comes back."""
    for park, side in ((ch.TRAVEL_LEFT, +1), (ch.TRAVEL_RIGHT, -1)):
        p, R = ik.fk(park, side)
        near = min(max(abs(a - b) for a, b in zip(q, park))
                   for d in range(0, 360, 1)
                   for q in ik.solve_branches(p, R, side, math.radians(d), True))
        assert near < math.radians(1.0), (
            'the solver cannot reproduce the parked pose (nearest %.1f deg away)'
            % math.degrees(near))
        first_only = min(max(abs(a - b) for a, b in zip(q, park))
                         for d in range(0, 360, 5)
                         for q in ik.solve_branches(p, R, side, math.radians(d)))
        assert first_only > math.radians(45), (
            'the first shoulder solution alone now reaches parked - the second is no longer '
            'what makes the planned trip home possible, so its note is out of date')


def test_every_wait_is_on_simulation_time():
    """The controllers run on /clock and the test laptop simulates slower than real time,
    so a wait timed on the wall clock expires while the arm is still moving. That is what
    made every leg of the 2026-09-27 run give up exactly at trajectory + 5 s with the joints
    0.25-0.42 rad short, and the next leg start early."""
    src = _node_src()
    for fn in ('send_pair', 'settle_arms', 'grippers', 'set_grasp', 'check_stands',
               'check_grip'):
        body = re.search(r'    def %s\(.*?(?=\n    def |\n    @)' % fn, src, re.S).group(0)
        assert 'time.time()' not in body, f'{fn}() still times itself on the wall clock'
        assert ('self._now()' in body or 'self._sleep(' in body), (
            f'{fn}() does not wait on simulation time')
    assert 'get_clock().now()' in src, '_now() does not read the ROS (simulation) clock'


def test_the_planner_is_given_the_world():
    """The node must hand every known object to the planner, via plan_inputs(), pinned
    parked to parked - a planner that does not know the box is there backs into it."""
    src = _node_src()
    body = src[src.index('    def plan_cycle('):src.index('    def move(')]
    for need in ('ch.plan_inputs(', 'obstacles=obs', 'pin_start=True',
                 'pin_end=(ch.TRAVEL_LEFT, ch.TRAVEL_RIGHT)', 'wide=wide',
                 'n != name'):
        assert need in body, f'plan_cycle() is missing {need!r}'


def test_slip_is_measured_in_a_frame_that_rotates_with_the_object():
    """A held object must read as stationary through the whole flip.

    THE BUG THIS CATCHES, which reached Gazebo and broke both grasp modes. The slip
    monitor compared the object's centre against the MIDPOINT OF THE TWO HANDS, in the
    base frame. But _ends puts both grasp points grip_dz off the object's axis along the
    object's OWN up, so that offset turns with the object and the apparent movement is

        2 * grip_dz * sin(theta / 2)

    = 6.5 mm at 15 degrees and exactly 15.0 mm at 34.9 degrees. The monitor therefore
    aborted every flip at 35 degrees. The weld run settled it beyond argument: an object
    rigidly joined to the palm reported 15.0 mm of slip.

    Measuring in the LEFT HAND'S frame cancels the rotation, because a held object is
    stationary in that frame by construction. This test asserts that invariant, so the
    artefact cannot come back if _ends or grip_dz changes.
    """
    for name, o in OBJECTS.items():
        sep = _s(o)
        R = ik.tool_down_fingers_along(LATERAL)
        centre = [REACH, 0.0, FLIP_Z]

        def rel_at(theta):
            (pl, Rl), _r = ch._ends(centre, LATERAL, sep, R, theta)
            d = [centre[k] - pl[k] for k in range(3)]
            return tuple(sum(Rl[r][k] * d[r] for r in range(3)) for k in range(3))

        ref = rel_at(0.0)
        for deg in (5, 15, 35, 60, 90):
            now = rel_at(ch.FLIP_DIR * math.radians(deg))
            drift = math.sqrt(sum((now[k] - ref[k]) ** 2 for k in range(3)))
            assert drift < 1e-6, (
                f'{name}: a stationary held object appears to move {drift * 1000:.1f} mm '
                f'at {deg} deg of flip. The slip frame is not rotating with the object, '
                f'so the monitor will abort the flip on geometry alone')

        # and a REAL displacement must still be seen, or the metric is inert
        (pl, Rl), _r = ch._ends(centre, LATERAL, sep, R, 0.0)
        moved = [centre[0] + 0.012, centre[1], centre[2]]
        d = [moved[k] - pl[k] for k in range(3)]
        now = tuple(sum(Rl[r][k] * d[r] for r in range(3)) for k in range(3))
        seen = math.sqrt(sum((now[k] - ref[k]) ** 2 for k in range(3)))
        assert abs(seen - 0.012) < 1e-6, (
            f'{name}: a genuine 12 mm shift reads as {seen * 1000:.1f} mm - the metric '
            f'has been made blind rather than correct')




def _urdf_dir():
    import pathlib as _pl
    return _pl.Path(__file__).resolve().parents[2] / 'reorient_description' / 'urdf'


def _finger_efforts():
    """(left, right) finger force, read where the arms are actually instantiated."""
    t = (_urdf_dir() / 'robot.urdf.xacro').read_text()
    out = {}
    for side in ('left', 'right'):
        mm = re.search(r'<xacro:arm_6dof side="%s"\s+reflect="[-0-9]+"\s+'
                       r'finger_effort="([0-9.]+)"' % side, t)
        assert mm, '%s arm_6dof invocation has no finger_effort' % side
        out[side] = float(mm.group(1))
    return out['left'], out['right']


def _pad_half_length():
    """Half the pad's extent ALONG the object axis - the gravity-tipping lever arm."""
    t = (_urdf_dir() / 'arm.xacro').read_text()
    lens = {float(v) for v in re.findall(r'<box size="0\.080 0\.012 ([0-9.]+)"/>', t)}
    assert len(lens) == 1, 'finger pads disagree on length: %s' % lens
    return lens.pop() / 2.0


def _pad_mu():
    t = (_urdf_dir() / 'arm.xacro').read_text()
    mm = re.search(r'<mu1>([0-9.]+)</mu1>', t)
    assert mm, 'pad mu1 not found'
    return float(mm.group(1))


def test_the_support_hand_is_too_weak_to_fight_the_holding_hand():
    """The grip asymmetry is the design, not an oversight.

    Two RIGID grasps on one RIGID body is an over-constrained closed chain. The arms lag
    by different amounts - they carry different shares - and position control has no give,
    so the mismatch becomes force: 19.5 N was measured on a wrist holding a 4.41 N object,
    4.4x its weight. That shear is what pulled the object out, and no pad or gain fixes it.

    A weak SUPPORT hand makes the chain self-relieving: it slides along the object once
    shear exceeds mu*N, so the internal force is capped BY CONSTRUCTION rather than by how
    well the controller tracks. Both bounds matter - strong enough to carry its share,
    weak enough to yield before it fights - so both are asserted.
    """
    eff_left, eff_right = _finger_efforts()
    assert eff_right < eff_left, (
        'the support hand (%s N) must be weaker than the holding hand (%s N), or the '
        'closed chain has nothing to relieve it' % (eff_right, eff_left))
    cap = 2 * eff_right * _pad_mu()
    MEASURED_FIGHT = 19.5
    assert cap < 0.5 * MEASURED_FIGHT, (
        'the support hand can transmit %.1f N, not far enough below the %.1f N fight '
        'this design exists to prevent' % (cap, MEASURED_FIGHT))
    for name, o in OBJECTS.items():
        share = o.get('mass', 0.45) * 9.81 * 0.5
        assert cap >= 2.0 * share, (
            '%s: the support hand can transmit only %.1f N but must carry %.2f N - under '
            '2x, so it would slide instead of supporting' % (name, cap, share))


def test_the_reported_constants_match_the_hardware():
    """Every number the run PRINTS must be the number the robot actually has.

    The GRASP QUALITY and ONE-HAND TORQUE lines are the evidence a run is judged on, and
    they are computed in task_node from its own constants. If those drift from the URDF
    the log becomes confident fiction - which has happened twice in this project, once
    where the config said one handover angle and the code another, and once where the
    reported finger force was not the force being applied.
    """
    src = _node_src()

    def const(name):
        mm = re.search(r'^%s\s*=\s*([-0-9.]+)' % name, src, re.M)
        assert mm, '%s missing from task_node.py' % name
        return float(mm.group(1))

    eff_left, eff_right = _finger_efforts()
    assert const('FINGER_EFFORT_LEFT_N') == eff_left, (
        'task_node reports the holding hand at %s N, the URDF drives it at %s'
        % (const('FINGER_EFFORT_LEFT_N'), eff_left))
    assert const('FINGER_EFFORT_RIGHT_N') == eff_right, (
        'task_node reports the support hand at %s N, the URDF drives it at %s'
        % (const('FINGER_EFFORT_RIGHT_N'), eff_right))
    assert abs(const('PAD_HALF_LEN') - _pad_half_length()) < 1e-9, (
        'task_node uses a %.0f mm pad half-length, the URDF builds %.0f mm'
        % (const('PAD_HALF_LEN') * 1000, _pad_half_length() * 1000))
    # the EFFECTIVE coefficient is the minimum of the two surfaces, not the pad's
    obj_mu = _object_mu()
    eff = min(_pad_mu(), obj_mu)
    assert abs(const('PAD_MU') - _pad_mu()) < 1e-9, (
        'task_node reports pad mu %.2f, the pads are built at %.2f'
        % (const('PAD_MU'), _pad_mu()))
    assert abs(const('OBJECT_MU') - obj_mu) < 1e-9, (
        'task_node reports object mu %.2f, the models are built at %.2f'
        % (const('OBJECT_MU'), obj_mu))
    assert abs(min(const('PAD_MU'), const('OBJECT_MU')) - eff) < 1e-9, (
        'the reported effective mu must be the MINIMUM of the two surfaces (%.2f), '
        'because that is how Gazebo combines them' % eff)


def _object_mu():
    import pathlib as _pl
    root = _pl.Path(__file__).resolve().parents[2] / 'reorient_gazebo' / 'models'
    mus = set()
    for f in root.glob('*/model.sdf'):
        for v in re.findall(r'<mu>([0-9.]+)</mu>', f.read_text()):
            mus.add(float(v))
    assert len(mus) == 1, 'objects disagree on friction: %s' % mus
    return mus.pop()


def test_the_distal_phalanx_wraps_below_the_object_equator():
    """The fingertips must reach UNDER the widest part of the object.

    THIS IS THE DEFECT EVERY EARLIER JAW SHARED. The object's equator sits grip_dz + LG
    down the finger - 95 mm, and independent of the object's radius, because it lies on
    the floor so r cancels. The rigid plate ends at 110 mm. So it reaches 15 mm past the
    widest point and then runs out, and no amount of ridge, pad length, friction or force
    changes that: a flat pinch on a 90 mm cylinder has about 3 mm of lateral capture and
    cannot get underneath. The 22 Sep run showed the consequence - "left 1/2 touching,
    right 1/2 touching", one finger per hand, the object merely cradled between two palms
    and dropped the moment one opened.

    A hinged second segment curls inward as the hand closes, so the tip reaches below the
    equator WITHOUT extending the rigid plate into the floor, which is the constraint that
    capped every previous attempt at 13.8 mm of clearance. This test asserts the wrap is
    geometrically real: at full curl the tip must reach inside the object's surface at its
    own depth, and still clear the floor.
    """
    xac = (_urdf_dir() / 'arm.xacro').read_text()
    m = re.search(r'name="(\w+)_finger_\$\{fprefix\}_tip_link"', xac)
    assert 'tip_joint' in xac and 'tip_link' in xac, (
        'there is no distal phalanx - the jaw is a flat pinch again, which cannot reach '
        'below the object equator. See this docstring.')

    mlen = re.search(r'<box size="([0-9.]+) 0\.012 [0-9.]+"/>\s*</geometry>\s*'
                     r'</collision>\s*<inertial>\s*<mass value="0\.012"', xac, re.S)
    assert mlen, 'distal segment length not found'
    L = float(mlen.group(1))
    mcurl = re.search(r'<limit lower="0\.0" upper="([0-9.]+)" effort="\$\{finger_effort\}"',
                      xac)
    assert mcurl, 'distal joint limit not found'
    joint_limit = float(mcurl.group(1))

    # THE CURL THAT MATTERS IS THE ONE COMMANDED, not the joint's travel limit. Checking
    # the limit instead evaluated a pose the robot is never asked to reach.
    import pathlib as _pl2
    ac_src = (_pl2.Path(__file__).resolve().parents[1] / 'reorient_behavior'
              / 'arm_commander.py').read_text()
    mcc = re.search(r'^CURL_CLOSED\s*=\s*([0-9.]+)', ac_src, re.M)
    assert mcc, 'CURL_CLOSED missing from arm_commander.py'
    curl_max = float(mcc.group(1))
    assert curl_max <= joint_limit + 1e-9, (
        'the commanded curl %.3f exceeds the joint limit %.3f, so the tips would stall '
        'short of the commanded wrap' % (curl_max, joint_limit))

    R = 0.045
    equator = GRIP_DZ + ik.LG
    wrist_z = R + GRIP_DZ + ik.LG
    face = R - 0.003                      # pad face, pressing 3 mm in
    best = None
    for frac in (0.6, 0.8, 1.0):
        th = curl_max * frac
        below = (col.FINGERTIP_X + L * math.cos(th)) - equator
        inward = L * math.sin(th)
        floor = wrist_z - (col.FINGERTIP_X + L * math.cos(th))
        if below >= R:
            continue
        need = face - math.sqrt(max(0.0, R * R - below * below))
        if inward >= need and floor > 0.004:
            best = (math.degrees(th), below, inward, need, floor)
            break
    assert best, (
        'a %.0f mm distal segment curling to %.0f deg never reaches inside the object '
        'surface while clearing the floor - the wrap is not real'
        % (L * 1000, math.degrees(curl_max)))
    deg, below, inward, need, floor = best
    # more than half the radius below the equator, or it is not meaningfully underneath:
    # a short stub satisfies the reach test trivially because the object is still wide
    # there, which is exactly how a 8 mm stub passed an earlier version of this check
    # AND THE TIP MUST NOT OVER-REACH. This bound was missing, and its absence is what
    # put the tips 5.9 mm inside the object's surface: they pushed it outward until it
    # lifted clear of the flat pads and balanced on the two flaps alone. A wrap that
    # displaces the object is worse than no wrap - it removes the pad contact that carries
    # the load and leaves two narrow lines that cannot resist a tilt.
    surf = math.sqrt(max(0.0, R * R - below * below))
    tip_at = face - inward
    over = surf - tip_at
    assert over <= 0.004, (
        'at %.0f deg of curl the tip sits %.1f mm INSIDE the object surface, which levers '
        'it off the flat pad - the object then rests on the two tips alone'
        % (deg, over * 1000))
    assert over >= 0.0005, (
        'at %.0f deg of curl the tip is %.1f mm short of the surface - it never touches'
        % (deg, -over * 1000))

    assert below > 0.025, (
        'the tip only gets %.0f mm below the equator at %.0f deg of curl - the object is '
        'still %.0f mm wide there, so that is a pinch, not a wrap'
        % (below * 1000, deg, math.sqrt(max(0.0, 0.045 ** 2 - below ** 2)) * 2000))

def test_the_pad_does_not_overhang_the_end_of_the_object():
    """With margin for the tracking error that actually occurs, not a nominal 5 mm.

    The 60 mm pad left 5 mm of object beyond it, and 0.19 rad of wrist error moves the
    hand about 21 mm along the object, so it overhung on every cycle - three quarters on
    the cylinder and a quarter in the air.
    """
    a = _pad_half_length()
    WRIST_ERR = 0.010          # m of axial slop to tolerate at the fingertip
    for name, o in OBJECTS.items():
        s = o['length'] / 2.0 - ch.END_CLEAR
        spare = o['length'] / 2.0 - (s + a)
        assert spare >= WRIST_ERR, (
            '%s: only %.0f mm of object beyond the pad, want at least %.0f mm to absorb '
            'tracking error' % (name, spare * 1000, WRIST_ERR * 1000))


def test_the_distal_curl_is_actually_commanded():
    """A second phalanx that is never driven is decoration.

    This project has shipped dead code twice - regrasp() and weld_rescue_now() were both
    defined, never called, and their absence reported as success. The distal joints are
    the same risk: the URDF can declare them, the controller can list them, and if
    grippers() keeps publishing four values the tips simply never move and the jaw is a
    flat pinch again with extra links.
    """
    import pathlib as _pl
    src = _node_src()
    ac_src = (_pl.Path(__file__).resolve().parents[1]
              / 'reorient_behavior' / 'arm_commander.py').read_text()

    # every gripper publish must carry eight values, not four
    # every gripper publish must carry the curl as well as the closure. Matched by
    # splitting on the call rather than by regex, because the calls span lines.
    chunks = src.split('grip_pub.publish(')[1:]
    assert chunks, 'no gripper publish found'
    for c in chunks:
        head = c[:c.index('time.sleep')] if 'time.sleep' in c else c[:200]
        assert '_curl_for' in head, (
            'a gripper publish sends only the closure, so the distal joints would never '
            'be driven: %s' % ' '.join(head.split())[:90])

    # and the curl must be non-zero when the hand is shut
    mc = re.search(r'^CURL_CLOSED\s*=\s*([0-9.]+)', ac_src, re.M)
    assert mc and float(mc.group(1)) > 0.3, (
        'CURL_CLOSED is %s - the tips would barely move'
        % (mc.group(1) if mc else 'missing'))
    assert 'return 0.0' not in re.search(
        r'def _curl_for.*?(?=\n    def )', src, re.S).group(0).split('span')[-1], (
        '_curl_for() short-circuits to zero, so the curl is never applied')

    # the controller must list all eight joints
    ctl = (_pl.Path(__file__).resolve().parents[2] / 'reorient_description' / 'config'
           / 'controllers.yaml').read_text()
    n = ctl.count('_tip_joint')
    assert n == 4, 'the gripper controller lists %d distal joints, want 4' % n


def test_the_flip_is_impossible_at_4_and_5_dof_and_solvable_at_6():
    """The repeatable proof that the sixth axis is a requirement, not a luxury.

    PROJECT_STATUS.md has always claimed this sweep existed. It did not, until now - the
    document cited a proof that was never written, which is exactly the kind of unbacked
    claim this project has been bitten by elsewhere.

    THE ARGUMENT. Holding an object rigidly with two hands means both position AND
    orientation follow the object:
        hand position(t) = centre + R(t) . (hand position(0) - centre)
        hand rotation(t) = R(t) . hand rotation(0)
    The second line is what kills 4 DOF. Every hand POSITION in the flip is reachable -
    positions were never the problem. The ORIENTATIONS are not: to keep a rigid grip each
    hand's approach axis has to rotate with the object, and a 4-DOF wrist has no axis to
    do it with.

    HOW IT IS TESTED. Solve the real flip waypoints with the full 6-DOF IK, then ask
    whether the answer would still be available with axes removed. A 4-DOF arm has no
    wrist roll (j4) and no tool roll (j6), so it can only reach a pose whose solution
    needs both at zero. A 5-DOF arm adds tool roll, freeing the gripper's spin - which is
    why 5 DOF fixed GRASPING - but the approach direction stays trapped in the arm's
    plane, so j4 must still be zero.
    """
    TOL = math.radians(2.0)          # a missing axis may sit within 2 deg of zero
    rows = []
    for name, o in OBJECTS.items():
        s_end = _s(o)
        R = ik.tool_down_fingers_along(LATERAL)
        for deg in range(0, 91, 10):
            th = ch.FLIP_DIR * math.radians(deg)
            pair = ch._ends([REACH, 0.0, FLIP_Z], LATERAL, s_end, R, theta=th)
            for (p, Rm), sgn in zip(pair, (+1, -1)):
                try:
                    rows.append(ik.solve(p, Rm, sgn))
                except Exception:                                   # noqa: BLE001
                    rows.append(None)
    total = len(rows)
    assert total >= 36, 'the sweep must cover at least 36 configurations, got %d' % total

    solved6 = [q for q in rows if q is not None]
    ok5 = [q for q in solved6 if abs(q[3]) <= TOL]
    ok4 = [q for q in solved6 if abs(q[3]) <= TOL and abs(q[5]) <= TOL]

    assert len(solved6) == total, (
        '6 DOF failed %d of %d flip configurations - the design does not close'
        % (total - len(solved6), total))
    assert len(ok4) == 0, (
        '%d of %d configurations came out reachable with j4 and j6 at zero, so this sweep '
        'no longer demonstrates that 4 DOF is insufficient' % (len(ok4), total))
    assert len(ok5) < total * 0.2, (
        '%d of %d configurations are reachable without wrist roll - the 5-DOF argument '
        'has stopped holding' % (len(ok5), total))

    # and record HOW FAR the missing axes have to travel, so the claim is quantitative
    worst_j4 = max(abs(q[3]) for q in solved6)
    worst_j6 = max(abs(q[5]) for q in solved6)
    assert worst_j4 > math.radians(45), (
        'wrist roll only reaches %.0f deg; if it were genuinely small a 5-DOF arm would '
        'nearly suffice' % math.degrees(worst_j4))
    assert worst_j6 > math.radians(45), (
        'tool roll only reaches %.0f deg' % math.degrees(worst_j6))


def test_nothing_on_the_pad_stands_proud_of_its_face():
    """Pad, suction cup and curled tip must all bear at once.

    THIS IS THE FAULT PATTERN THAT HAS COST THE MOST IN THIS PROJECT, three times in
    different clothing. Anything that reaches further than the flat pad holds the pad OFF
    the object, so the load ends up on whatever protrudes:

      * ridges 6 mm proud, centred wrongly -> one line contact on the upper curve, which
        wedges a cylinder downward out of the grip
      * curled tips reaching 5.9 mm inside the surface -> the bottle lifted clear of the
        pads and balanced on two flaps, then fell as soon as it tilted
      * a suction cup 1 mm proud -> the same thing again, in miniature

    A flush face means the pad carries the load, the cup seals against it, and the tip
    wraps under it, all at the same time.
    """
    xac = (_urdf_dir() / 'arm.xacro').read_text()
    pad_th = {float(v) for v in re.findall(r'<box size="0\.080 ([0-9.]+) [0-9.]+"/>', xac)}
    assert len(pad_th) == 1, 'pads disagree on thickness: %s' % pad_th
    pad_face = pad_th.pop() / 2.0

    cups = re.findall(r'<origin xyz="\$\{RIDGE_X_CUP\} \$\{-faxis \* ([0-9.]+)\} 0"'
                      r'[^>]*/>\s*<geometry><cylinder radius="[0-9.]+" length="([0-9.]+)"',
                      xac)
    assert cups, 'no suction cup found on the finger pads'
    for off, ln in cups:
        face = float(off) + float(ln) / 2.0
        assert face <= pad_face + 1e-9, (
            'the suction cup face sits %.1f mm out against a pad face at %.1f mm - it '
            'stands proud and would hold the flat pad off the object'
            % (face * 1000, pad_face * 1000))


def test_the_fingertips_clear_the_floor_with_the_hand_OPEN():
    """The finger is at its LONGEST when open, and that is when it descends.

    THE FAULT THIS CATCHES, which shipped once. The distal phalanx only curls as the hand
    CLOSES, so while descending onto a floor-lying object the segment is straight and the
    finger is at full extension. A 30 mm segment on a plate ending at 110 mm put the tip at
    exactly 140 mm below the wrist - and the wrist sits 140 mm above the floor, so the tip
    was at ground level with zero clearance. The suite reported 13.8 mm throughout, because
    collision.py only modelled the proximal plate and knew nothing about the segment.

    This is fault E2 from ERROR_CATALOGUE.txt returning by a new route: fingers driven into
    the floor, which reads as failure to converge and as a jerk at pickup.
    """
    R = 0.045
    wrist_z = R + GRIP_DZ + ik.LG
    for lbl, curl in (('open, distal straight', 0.0),
                      ('closed, distal curled', _commanded_curl())):
        gap = wrist_z - col.finger_reach(curl)
        assert gap > 0.005, (
            'fingertips sit %.1f mm from the floor with the hand %s - the descent would '
            'drive them into the ground' % (gap * 1000, lbl))

    # and collision.py must actually model the segment, or every floor number is fiction
    assert hasattr(col, 'DISTAL_L') and col.DISTAL_L > 0, (
        'collision.py has no DISTAL_L, so its floor clearances ignore the distal phalanx')
    xac = (_urdf_dir() / 'arm.xacro').read_text()
    mm = re.search(r'<box size="([0-9.]+) 0\.012 [0-9.]+"/>\s*</geometry>\s*</collision>'
                   r'\s*<inertial>\s*<mass value="0\.012"', xac, re.S)
    assert mm, 'distal segment not found in the URDF'
    assert abs(float(mm.group(1)) - col.DISTAL_L) < 1e-9, (
        'collision.py models a %.0f mm distal segment but the URDF builds %.0f mm'
        % (col.DISTAL_L * 1000, float(mm.group(1)) * 1000))


def _commanded_curl():
    import pathlib as _pl
    t = (_pl.Path(__file__).resolve().parents[1] / 'reorient_behavior'
         / 'arm_commander.py').read_text()
    return float(re.search(r'^CURL_CLOSED\s*=\s*([0-9.]+)', t, re.M).group(1))


def test_the_gripper_parts_do_not_interfere_or_exceed_their_limits():
    """Dimensional cross-check of the gripper as a whole.

    Written after five separate faults of one kind: something added to the finger whose
    downstream consequence was not re-derived - ridges proud of the pad, tips over-reaching,
    a cup 1 mm proud, `close` left sized for a ridge that had been removed, and a distal
    segment that lengthened the finger while collision.py still measured the plate alone.
    Each looked fine because the arithmetic was done on the geometry intended rather than
    the geometry in the file. So this reads the files.
    """
    D = _urdf_dir()
    xac = (D / 'arm.xacro').read_text()
    robot = (D / 'robot.urdf.xacro').read_text()
    ac_src = (_urdf_dir().parents[1] / 'reorient_behavior' / 'reorient_behavior'
              / 'arm_commander.py').read_text()

    def num(pat, src):
        mm = re.search(pat, src, re.M)
        assert mm, 'not found: %s' % pat
        return float(mm.group(1))

    travel = num(r'name="finger_travel"\s+value="([0-9.]+)"', robot)
    lower = num(r'<limit lower="([0-9.]+)" upper="\$\{finger_travel\}"', xac)
    tip_lim = num(r'<limit lower="0\.0" upper="([0-9.]+)" effort="\$\{finger_effort\}"', xac)
    curl = num(r'^CURL_CLOSED\s*=\s*([0-9.]+)', ac_src)
    close = min(o['close'] for o in OBJECTS.values())

    # commands must stay inside the joints' travel
    assert close >= lower, 'close %.4f is below the finger lower limit %.4f' % (close, lower)
    assert OPEN_POS <= travel + 1e-9, (
        'open_pos %.4f OVER-travels the finger limit %.4f' % (OPEN_POS, travel))
    assert curl <= tip_lim + 1e-9, (
        'commanded curl %.3f exceeds the tip joint limit %.3f' % (curl, tip_lim))

    # the suction cup must sit on the pad and not reach the distal hinge
    CUP_X, CUP_R, HINGE_X, PAD_Z = 0.065, 0.015, 0.080, 0.040
    assert CUP_X + CUP_R <= HINGE_X + 1e-9, (
        'the cup reaches x=%.3f, past the distal hinge at %.3f' % (CUP_X + CUP_R, HINGE_X))
    assert CUP_X - CUP_R >= 0.0, 'the cup hangs off the inboard end of the pad'
    assert CUP_R <= PAD_Z / 2 + 1e-9, (
        'the cup is wider than the pad it sits on (%.0f mm vs %.0f mm)'
        % (CUP_R * 2000, PAD_Z * 1000))

    # with the object absent the two curled tips must not close on each other
    import math as _m
    tip_lat = (close - 0.006) - col.DISTAL_L * _m.sin(curl)
    assert 2 * tip_lat > 0.010, (
        'the two curled tips come within %.1f mm of each other with nothing between them'
        % (2 * tip_lat * 1000))


def test_the_curl_is_disabled_for_flat_sided_objects():
    """A flat face must not be wrapped. The pad already bears over its full height.

    THE FAULT THIS CATCHES, measured on the box in Gazebo. The distal curl is sized for a
    ROUND cross-section, whose surface narrows with depth: at 32.7 mm below the mid-height
    a 45 mm cylinder has narrowed to 30.9 mm, so a tip reaching 13.1 mm inward bites 2.0 mm.
    A cuboid does NOT narrow - its face is vertical at 45 mm all the way down - so the same
    tip drives **16.1 mm into the face**. That is the `LOAD CHECK ... wrists 19.0 N ->
    PRESSING ON SOMETHING` and the `0/2 touching` on the box: the flaps held the pads off
    the object and suction carried it in spite of the grasp.

    A flat face needs no wrap anyway. The wrap exists because a flat pad on a CYLINDER is a
    line contact; against a flat face the pad is already an area contact, which is better
    than anything the tip could add.

    Also guards the parse: bool('false') is True, so a config-declared `round: false`
    arriving as a STRING would still have curled. The harness caught that by reporting a
    cuboid as round.
    """
    import pathlib as _pl
    tn = _node_src()
    assert 'curl_this_object' in tn, 'the curl is not shape-gated at all'
    # CHECK THE CALL SITE, not merely that the helper exists. An earlier version of this
    # assertion looked for the string '_as_bool' anywhere in the file, so replacing the
    # call with bool() while leaving the helper defined passed the test - and the harness
    # was what actually caught it, by reporting a cuboid as round.
    assert re.search(r"'round':\s*_as_bool\(", tn), (
        "the round flag is not parsed through _as_bool at its call site, so a config "
        "value arriving as the string 'false' would read as True - bool('false') is True "
        "- and a flat-sided object would have its tips driven into the face")

    cfg = (_pl.Path(__file__).resolve().parents[1] / 'config' / 'objects.yaml').read_text()
    for name, o in OBJECTS.items():
        blk = re.search(rf'^    {name}:\n((?:      .+\n)+)', cfg, re.M)
        assert blk, f'{name} missing from the config'
        mr = re.search(r'^\s*round:\s*(\w+)', blk.group(1), re.M)
        assert mr, f'{name} does not declare its cross-section in the config'
        declared = mr.group(1).strip().lower() == 'true'
        assert declared == bool(o['round']), (
            f'{name}: config says round={mr.group(1)} but this suite says {o["round"]}')

    # and the arithmetic: a flat face at full curl would be gouged
    import math as _m
    R = 0.045
    below = col.finger_reach(_commanded_curl()) - (GRIP_DZ + ik.LG)
    inward = col.DISTAL_L * _m.sin(_commanded_curl())
    flat_bite = R - (0.042 - inward)
    round_bite = _m.sqrt(max(0.0, R * R - below * below)) - (0.042 - inward)
    assert round_bite <= 0.004, 'the round bite has drifted out of band'
    assert flat_bite > 0.010, (
        'a flat face would only be bitten %.1f mm at full curl, so this test is no longer '
        'demonstrating why flat objects must be excluded' % (flat_bite * 1000))


# Self-collision now lives in collision.py as the SINGLE implementation. Four
# ad-hoc copies existed at one point, three of them wrong in different ways -
# see the note above collision.self_clearance().

def test_the_arm_never_intersects_itself_anywhere_in_the_cycle():
    """From the parked pose to the object standing on the floor, the arm never meets itself.

    RESOLVED BY THE SEVENTH AXIS AND THE 20 mm FOREARM, and pinned so it stays resolved:
    at six DOF the holding hand pressed into its own forearm from ~60 deg of flip onward,
    -19.6 mm at the worst. Measured on the plan the node makes, fingers as commanded.

    AFTER RELEASE, ONE KNOWN BRUSH, bounded rather than hidden. Unrolling the left wrist from
    the flip orientation to the parked one passes its open finger ~9-12 mm past its own
    forearm, for every elbow and wrist solution available. It was always there - inside the
    old blind joint-space jump home, which no check looked at - and planning that trip made it
    visible. The model has no self_collide, so Gazebo does not simulate it; it is visual.
    """
    for name, o in OBJECTS.items():
        QL, QR, fp = _planned(o)
        at = _at(o)
        strict = {i for n in ('unpark', 'approach') + ch.CARRY_LEGS for i in at[n]}
        for i, (ql, qr) in enumerate(zip(QL, QR)):
            for q, side in ((ql, +1), (qr, -1)):
                g, who = col.self_clearance(col.arm_capsules_fine(q, side, fp[i]),
                                            optimistic=False)
                if i in strict:
                    assert g > 0.0015, (
                        '%s: the arm intersects itself by %.1f mm at waypoint %d (%s vs %s) '
                        'while handling the object - this was fixed by the seventh axis plus '
                        'a 20 mm forearm; something has undone one of them'
                        % (name, -g * 1000, i, who[0], who[1]))
                else:
                    assert g > -0.015, (
                        '%s: after release the arm passes %.1f mm into itself at waypoint %d '
                        '(%s vs %s) - the known wrist-unroll brush is ~12 mm; this is worse'
                        % (name, -g * 1000, i, who[0], who[1]))


def test_the_seventh_axis_is_necessary_and_so_is_the_thinner_forearm():
    """ATTRIBUTION. Two changes made the bimanual flip work and BOTH are load-bearing.
    This test removes each in turn and requires the cycle to fail.

    It is easy to add a joint and assume it earned its keep. A fair comparison has to give
    the six-DOF arm everything else the seven-DOF one gets - the same 20 mm forearm, the
    same path planner, both wrist branches - and pin ONLY the swivel. Done that way:

      swivel pinned at 0 (six DOF)   the cycle is INFEASIBLE: no continuous path exists
                                     from waypoint 21 to 22 inside the step cap, and at
                                     the coaxial pose arm-arm is +5.6 mm against +38.7
      forearm back to 28 mm          self-collision -2.0 mm: the hand presses into its own
                                     forearm through the whole descent

    So the seventh axis buys FEASIBILITY and the thinner forearm buys the last two
    millimetres of self-clearance. Neither substitutes for the other.
    """
    o = OBJECTS['bottle']
    # THE OBJECT-HANDLING PART of the cycle, where the claim is made - the parked pose at
    # either end is not what the seventh axis was added for, and including it would make
    # the pinned-swivel case fail for a reason that has nothing to do with the flip
    pairs = [w for n, leg in _legs(o) if n in ('approach',) + ch.CARRY_LEGS for w in leg]
    orig_opts, orig_fore = col._arm_options, col.BOX_LINKS['forearm']

    def pinned(p, R, side, other, fp, fz, obstacles=(), both_shoulders=False):
        return [(q, col.pose_clearance(col.arm_capsules_fine(q, side, fp), other, fz))
                for q in ik.solve_branches(p, R, side, 0.0, both_shoulders)]
    try:
        col._arm_options = pinned
        failed = False
        try:
            QL, QR = col.solve_path(pairs, finger_pos=o['close'])
            worst = min(
                col.self_clearance(col.arm_capsules_fine(q, sd, o['close']), False)[0]
                for ql, qr in zip(QL, QR) for q, sd in ((ql, +1), (qr, -1)))
            failed = worst <= 0.0
        except ik.IKError:
            failed = True                      # no continuous path at all
        assert failed, (
            'with the swivel pinned at zero the six-DOF arm completes the fully bimanual '
            'cycle cleanly - the seventh axis is then unjustified and the design record '
            'which credits it is wrong')
    finally:
        col._arm_options = orig_opts
    try:
        col.BOX_LINKS['forearm'] = (0.028, 0.038, 'y')
        QL, QR = col.solve_path(pairs, finger_pos=o['close'])
        worst = min(
            col.self_clearance(col.arm_capsules_fine(q, sd, o['close']), False)[0]
            for ql, qr in zip(QL, QR) for q, sd in ((ql, +1), (qr, -1)))
        assert worst <= 0.0, (
            'a 28 mm forearm no longer self-collides (%.1f mm) - the slimming to 20 mm is '
            'then unjustified and should be reverted, since a thicker link is stiffer'
            % (worst * 1000))
    finally:
        col.BOX_LINKS['forearm'] = orig_fore


def test_the_wall_constants_are_the_faces_in_the_world():
    """wall_ref's reference planes must be the wall FACES in reorient.sdf, derived here.

    THE OFFSET FAULT. Both walls are 20 mm boxes centred at -0.46, so the surfaces the LiDAR
    hits are at -0.45 - and WALL_A_X / WALL_B_Y said 0.46, the centres. Every pose fix was
    then 10 mm off in x and in y, every grasp was aimed at (+10, +10) mm in the world, and
    for the bottle that is 7.1 mm across the closing axis against a ~+/-3 mm pad window:
    one finger buried in the object, the other flush, in both of the operator's photos.
    Nothing caught it because the harness scanned walls placed at the same constants.
    """
    from reorient_behavior import wall_ref as wr
    import pathlib as _pl
    sdf = (_pl.Path(__file__).resolve().parents[2] / 'reorient_gazebo' / 'worlds'
           / 'reorient.sdf').read_text()

    def box(name):
        m = re.search(r'<collision name="%s"><pose>([^<]+)</pose>\s*<geometry><box><size>'
                      r'([^<]+)</size>' % name, sdf)
        assert m, f'{name} not found in reorient.sdf'
        return [float(v) for v in m.group(1).split()], [float(v) for v in m.group(2).split()]

    (ax, *_), (sx, *_) = box('backwall')
    (_bx, by, *_), (_tx, ty, *_) = box('sidewall')
    face_a, face_b = -(ax + sx / 2.0), -(by + ty / 2.0)
    assert abs(wr.WALL_A_X - face_a) < 1e-9, (
        f'wall A: wall_ref says the surface is at x = -{wr.WALL_A_X:.3f} but the wall face '
        f'in the world is at x = -{face_a:.3f} - every pose fix, and so every grasp, is '
        f'off by {abs(wr.WALL_A_X - face_a) * 1000:.0f} mm')
    assert abs(wr.WALL_B_Y - face_b) < 1e-9, (
        f'wall B: wall_ref says y = -{wr.WALL_B_Y:.3f}, the face is at y = -{face_b:.3f}')
    # and the harness must scan the WORLD's walls, not wall_ref's, or it cannot see this
    hs = (_pl.Path(__file__).resolve().parent / 'harness.py').read_text()
    assert 'world_wall_faces()' in hs and 'synth_scan(\n            self.heading, WALL_A' in hs, (
        'the harness scans walls placed at wall_ref\'s own constants again, so a wrong '
        'constant can never show up in it')


def test_the_arms_settle_before_the_fingers_close():
    """The fingers must not start closing while the hands are still arriving.

    send_pair() returns at 0.08 rad of joint error - 34 mm of hand position at the
    shoulder's lever - and at gain 1.9 the arm is a 0.53 s lag, so the first finger used
    to meet the object while the hand was still moving and push it (a lying bottle rolls)
    before the suction froze the offset in.
    """
    src = _node_src()
    body = src[src.index('    def cycle(self, name):'):]
    i_hover = body.index("self.move(leg['unpark']")
    i_set_h = body.index("self.settle_arms(name, 'the hover')")
    i_app = body.index("self.move(leg['approach']")
    i_set = body.index("self.settle_arms(name, 'the grasp pose')")
    i_close = body.index("self.grippers(o['close']")
    assert i_hover < i_set_h < i_app, (
        'the arms must settle at the hover before descending - moving straight on let the '
        'hands come down while still swinging into line')
    assert i_app < i_set < i_close, (
        'settle_arms() must run between the descent and the finger closure')
    m = re.search(r'^SETTLE_TOL\s*=\s*([0-9.]+)', src, re.M)
    assert m and float(m.group(1)) <= 0.005, (
        'SETTLE_TOL %s rad is too loose - at 0.43 m it leaves more than 2 mm at the hand'
        % (m.group(1) if m else 'missing'))


def test_the_suction_carry_keeps_the_fingers_out_of_the_object():
    """In suction mode the fingers must not sit inside the object while it is carried.

    `close` presses each pad 3 mm into the surface on purpose - that is the stall GRIP CHECK
    measures and the friction that holds the object during the settle. But once the cup
    attaches Gazebo stops computing finger-object contact, so the fingers reach the full
    command and sit 3 mm inside the object for the whole carry. They back off to just
    outside the surface before the attach, and squeeze again before the release.
    """
    src = _node_src()
    m = re.search(r'^CARRY_EASE\s*=\s*([0-9.]+)', src, re.M)
    assert m, 'CARRY_EASE missing from task_node.py'
    ease = float(m.group(1))
    for name, o in OBJECTS.items():
        face = (o['close'] + ease) - PAD_REACH
        gap = face - o['half_width']
        assert 0.0 <= gap <= 0.001, (
            f'{name}: carrying, the pad face sits {gap * 1000:+.1f} mm from the surface - '
            f'want 0 to 1 mm outside it, so it neither sinks in nor visibly lets go')
        assert o['close'] + ease <= OPEN_POS, f'{name}: the carry closure opens past open'
    eng = src[src.index('    def engage_grasp(self, name):'):src.index('    def carry_closure(')]
    assert eng.index('self.grippers(carry') < eng.index('self.set_grasp(name, True)'), (
        'the fingers must back off BEFORE the suction attaches, or they sink in first')
    rel = src[src.index('    def release_grasp(self, name):'):
              src.index('    def weld_rescue_now(')]
    assert rel.index("self.grippers(self.obj[name]['close']") < rel.index(
        'self.set_grasp(name, False)'), (
        'the fingers must squeeze again BEFORE the suction releases, or the object drops')


def test_the_cycle_has_one_definition():
    """task_node must build its sequence ONLY through ch.cycle_legs().

    Two copies of the sequence is how the old handover kept running in Gazebo for a whole
    session while the suite validated a cycle without it. So no phase function may be
    called directly from the node.
    """
    src = open(FLIP_TASK).read()
    tn = open(TASK_NODE).read()
    assert 'ch.' not in tn and 'self.stand_up(' in tn, (
        'task_node must hand the object to the shared stand-up, not build a sequence itself')
    for fn in ('approach_waypoints', 'lift_waypoints', 'yaw_waypoints', 'flip_waypoints',
               'lower_waypoints', 'settle_waypoints', 'release_waypoints',
               'clear_waypoints', 'retreat_waypoints', 'withdraw_waypoints',
               'transit_waypoints'):
        assert f'ch.{fn}(' not in src, (
            f'flip_task calls ch.{fn}() directly - the cycle must come from cycle_legs()')
    assert src.count('ch.cycle_legs(') == 1, 'cycle_legs must be called exactly once'
    for leg in ('unpark', 'approach', 'lift', 'yaw', 'lower', 'settle', 'release',
                'clear', 'retreat', 'withdraw', 'park'):
        assert f"leg['{leg}']" in src, f'the {leg} leg is never executed'
    assert "leg['flip%d' % k]" in src, 'the flip legs are never executed'



# ---------------------------------------------------------------- the surface
RAISED = 0.10        # a table under each object - the height the harness also runs at


def _raised(o, h=RAISED):
    """One object's cycle, plan and scene on a table of height ``h`` - config changes only."""
    name = _name(o)
    psi = _psi(name)
    legs = ch.cycle_legs((REACH, 0.0), o['cz'], o['upright_cz'], psi, _s(o), FLIP_Z,
                         ch.HOVER_DZ, ch.PLACE_DROP, o['half_width'], h)
    table = col.box_obstacle((REACH, 0.0, h / 2.0), 0.0, (0.20, 0.34, h))
    lying = col.object_capsules((REACH, 0.0, h + o['cz']), (math.cos(psi), math.sin(psi), 0.0),
                                o['length'], o['half_width'], o['round'])
    standing = col.object_capsules((REACH, 0.0, h + o['upright_cz']), (0.0, 0.0, 1.0),
                                   o['length'], o['half_width'], o['round'])
    fp, obs, wide = ch.plan_inputs(legs, o['close'], OPEN_POS, lying, standing, [], [table])
    flat = [w for _, leg in legs for w in leg]
    QL, QR = col.solve_path(flat, finger_pos=fp, seed_l=ch.TRAVEL_LEFT, seed_r=ch.TRAVEL_RIGHT,
                            obstacles=obs, pin_start=True,
                            pin_end=(ch.TRAVEL_LEFT, ch.TRAVEL_RIGHT), wide=wide)
    return legs, QL, QR, fp, table, standing


def test_every_height_is_measured_from_the_surface():
    """Put the object on a surface h up and every height that belongs to the OBJECT moves up h.

    The grasp, the set-down, the release and the climb above clutter are all about the
    object and what it stands on, so they must follow the surface exactly. What may NOT simply
    follow it is the flip station - that is where the ARM works best - which is raised only as
    far as keeps the turning object clear of its surface.
    """
    for name, o in OBJECTS.items():
        for h in (0.0, 0.05, RAISED):
            d = dict(ch.cycle_legs((REACH, 0.0), o['cz'], o['upright_cz'], _psi(name), _s(o),
                                   FLIP_Z, ch.HOVER_DZ, ch.PLACE_DROP, o['half_width'], h))
            (pl, _), (pr, _) = d['approach'][-1]
            assert abs(pl[2] - (h + o['cz'] + GRIP_DZ)) < 1e-9, f'{name}: grasp not from the surface'
            (pl, _), (pr, _) = d['settle'][-1]
            assert abs(pl[2] - (h + o['upright_cz'] + _s(o))) < 1e-9 and \
                abs(pr[2] - (h + o['upright_cz'] - _s(o))) < 1e-9, (
                f'{name}: at h={h} the object is not set down ON its surface')
            (_, _), (pr, _) = d['release'][-1]
            assert abs(pr[2] - (h + ch.CLUTTER_H)) < 1e-9, (
                f'{name}: the lower hand climbs to {pr[2]:.3f}, not CLUTTER_H above the surface')
            flip = d['lift'][-1][0][0][2] - GRIP_DZ
            low = flip - math.hypot(o['upright_cz'], o['half_width'])
            assert flip >= FLIP_Z - 1e-9 and low >= h + ch.FLIP_CLEAR - 1e-9, (
                f'{name}: at h={h} the flip runs at {flip:.3f}, so the turning object comes '
                f'within {(low - h) * 1000:.0f} mm of its surface')
        assert ch.flip_height(0.0, o['upright_cz'], o['half_width']) == FLIP_Z, (
            f'{name}: on the floor the flip station must be the swept FLIP_Z')


def test_the_cycle_plans_on_a_raised_surface():
    """THE SHELF, before there is one: the whole cycle on a 0.10 m table, planned the way the
    node plans it, with the table as a solid obstacle - and clear of it and of the object.

    Two things had to change for this to pass, and both were model or motion faults the floor
    hid: the objects were capsules 45 mm too long at each end, and the wrists turned for home
    while a finger still reached across the object's top.
    """
    for name, o in OBJECTS.items():
        legs, QL, QR, fp, table, standing = _raised(o)
        at = {}
        k = 0
        for n, seq in legs:
            at[n] = range(k, k + len(seq))
            k += len(seq)
        home = [i for n in ch.HOME_LEGS for i in at[n]]
        for i, (ql, qr) in enumerate(zip(QL, QR)):
            rep = col.check_pose(ql, qr, fp[i])
            assert rep['worst'] > MIN_CLEARANCE, (
                f'{name}: on a table, {rep["worst_name"]} {rep["worst"] * 1000:+.1f} mm at {i}')
            for q, sd in ((ql, +1), (qr, -1)):
                g = col.obstacle_clearance(col.arm_capsules_fine(q, sd, fp[i]), table)
                assert g > 0.010, f'{name}: an arm comes {g * 1000:+.1f} mm from the table at {i}'
        for i in [home[0] - 1] + home[:-1]:
            for Q, sd in ((QL, +1), (QR, -1)):
                for f in (0.0, 0.25, 0.5, 0.75):
                    q = [a + f * (b - a) for a, b in zip(Q[i], Q[i + 1])]
                    g = min(col.obstacle_clearance(col.arm_capsules_fine(q, sd, OPEN_POS), c)
                            for c in standing)
                    assert g > 0.010, (
                        f'{name}: on the way home an arm passes {g * 1000:+.1f} mm from the '
                        f'object it just stood on the table')


def test_the_upper_hand_slides_off_the_top_before_the_wrists_turn():
    """The fix for the finger that dipped across a standing object's top on the way home.

    At the end of the retreat the upper hand's wrist is near full stretch, so it cannot back
    out; it slides on along its own fingers until the tips are TIP_CLEAR past the far side,
    orientation unchanged, and only the park leg turns the wrists.
    """
    for name, o in OBJECTS.items():
        d = dict(_legs(o))
        (pl0, Rl0), (pr0, _) = d['retreat'][-1]
        (pl1, Rl1), (pr1, _) = d['withdraw'][-1]
        appr = [Rl0[k][0] for k in range(3)]
        moved = [pl1[k] - pl0[k] for k in range(3)]
        assert math.dist(moved, [ch.slide_off_distance(o['half_width']) * a for a in appr]) < 1e-9, (
            f'{name}: the upper hand does not slide straight along its own fingers')
        assert Rl1 == Rl0 and math.dist(pr0, pr1) < 1e-12, (
            f'{name}: the wrists turn, or the lower hand moves, during the slide')
        tips = ch.TIP_PAST_GRASP + ch.slide_off_distance(o['half_width'])
        assert abs(tips - (GRIP_DZ + o['half_width'] + ch.TIP_CLEAR)) < 1e-9


def test_solid_obstacles_are_exact():
    """Boxes (fixtures, cuboids) and cylinders (bottles, cans) against brute force.

    The planner used to know only capsules. A capsule round a bottle is 45 mm too long at each
    end, and round a shelf it is not even the right shape; these are exact, so this proves it.
    """
    import random
    rng = random.Random(11)
    for _ in range(300):
        c = [rng.uniform(-0.3, 0.3) for _ in range(3)]
        shapes = [col.box_obstacle(c, rng.uniform(-3, 3), [rng.uniform(0.02, 0.4) for _ in range(3)]),
                  col.cylinder_obstacle(c, [rng.uniform(-1, 1) for _ in range(3)],
                                        rng.uniform(0.05, 0.4), rng.uniform(0.01, 0.1))]
        p0 = [rng.uniform(-0.6, 0.6) for _ in range(3)]
        p1 = [rng.uniform(-0.6, 0.6) for _ in range(3)]
        for sh in shapes:
            got = col.segment_solid_distance(p0, p1, sh)
            best = min(col.solid_sdf([p0[k] + t / 2000.0 * (p1[k] - p0[k]) for k in range(3)], sh)
                       for t in range(2001))
            assert got <= best + 1e-6 and abs(got - best) < 3e-4, (sh[0], got, best)
    # and the object model uses them: a bottle's top is flat, not a hemisphere 45 mm higher
    (cyl,) = col.object_capsules((0.3, 0.0, 0.12), (0, 0, 1), 0.24, 0.045, True)
    assert col.is_cylinder(cyl) and abs(col.solid_sdf((0.3, 0.0, 0.25), cyl) - 0.010) < 1e-9
    (box,) = col.object_capsules((0.3, 0.0, 0.11), (0, 0, 1), 0.22, 0.045, False)
    assert col.is_box(box) and abs(col.solid_sdf((0.3, 0.0, 0.23), box) - 0.010) < 1e-9


def test_the_fixtures_are_the_world():
    """Every fixture in the config must be a collision box in reorient.sdf, same pose and size.

    The planner keeps the arms off what the config says is there; if the config and the world
    disagree the planner is avoiding a wall that is not where the real one is.
    """
    import pathlib as _pl
    sdf = (_pl.Path(__file__).resolve().parents[2] / 'reorient_gazebo' / 'worlds'
           / 'reorient.sdf').read_text()
    boxes = {}
    for m in re.finditer(r'<collision name="(\w+)"><pose>([^<]+)</pose>\s*<geometry><box><size>'
                         r'([^<]+)</size>', sdf):
        boxes[m.group(1)] = ([float(v) for v in m.group(2).split()],
                             [float(v) for v in m.group(3).split()])
    txt = open(CFG).read()
    names = re.search(r'^\s*fixtures:\s*\[([^\]]*)\]', txt, re.M)
    assert names, 'no fixtures declared in the config - the planner would not know the walls'
    sdf_name = {'back_wall': 'backwall', 'side_wall': 'sidewall'}
    for f in [v.strip() for v in names.group(1).split(',') if v.strip()]:
        blk = re.search(rf'^    {f}:\n((?:      .+\n)+)', txt, re.M)
        assert blk, f'fixture {f} is listed but not defined'
        body = blk.group(1)
        centre = [float(v) for v in re.search(r'centre:\s*\[([^\]]+)\]', body).group(1).split(',')]
        size = [float(v) for v in re.search(r'size:\s*\[([^\]]+)\]', body).group(1).split(',')]
        yaw = float(re.search(r'yaw:\s*([-0-9.]+)', body).group(1))
        pose, sz = boxes[sdf_name.get(f, f)]
        assert max(abs(a - b) for a, b in zip(centre, pose[:3])) < 1e-9, f'{f}: centre differs'
        assert max(abs(a - b) for a, b in zip(size, sz)) < 1e-9, f'{f}: size differs'
        assert abs(math.radians(yaw) - pose[5]) < 1e-6, f'{f}: yaw differs'



def test_shared_files_match_the_manifest():
    """The two sections are ONE robot, so the files that make it so must be the same bytes.

    ik, collision (the planner and the world model), arm_commander, wall_ref, motion (the
    parked pose, the trips to and from it, the grip) and node_base (every task primitive)
    are shared with the pick-place section. SHARED_FILES.sha256 at the section root lists
    them with their hashes and is identical in both sections; if this fails, a shared file
    was edited here without being copied to the other section, or the reverse.
    """
    import hashlib
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    man = os.path.join(root, 'SHARED_FILES.sha256')
    assert os.path.exists(man), 'SHARED_FILES.sha256 is missing from the section root'
    pkg = os.path.join(os.path.dirname(__file__), '..', 'reorient_behavior')
    # this section belongs to these groups; pick-place has its own choreography, so it is
    # not in the 'flip' group that reorient and perceive share
    mine = ('all', 'flip')
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
              'node_base.py', 'choreography.py', 'flip_task.py', 'verify_node.py'):
        assert f in listed, f'{f} is not in the shared manifest'


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
    for name, o in OBJECTS.items():
        QL, QR, fp = _planned(o)
        reps = [col.check_pose(a, b, fp[i]) for i, (a, b) in enumerate(zip(QL, QR))]
        d = min(reps, key=lambda r: r['worst'])
        print('%-7s the plan the node makes, parked to parked: worst %+.4f m (%s) | '
              'arm-arm %+.3f floor %+.3f' % (name, d['worst'], d['worst_name'],
                                             min(r['arm_arm'] for r in reps),
                                             min(r['floor'] for r in reps)))
    print('flip station: reach %.2f, height %.2f, welded hand on top: %s'
          % (REACH, FLIP_Z, all(ch.left_ends_on_top(LATERAL, _s(o), FLIP_Z)
                                for o in OBJECTS.values())))
    sys.exit(1 if fails else 0)