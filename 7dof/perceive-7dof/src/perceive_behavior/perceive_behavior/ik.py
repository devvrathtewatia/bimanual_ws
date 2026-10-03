"""Analytic FK/IK for the 6-DOF arms (3-DOF positioning + spherical wrist).

Pure Python + math. No ROS, no Gazebo - so the entire motion plan can be proven
feasible offline. See HOW_CONFIG_SOLVING_WORKS.txt.

    J1 yaw (z)   J2 pitch (y)   J3 elbow pitch (y)
    J4 roll (x)  J5 pitch (y)   J6 roll (x)        <- axes meet at the wrist centre

WHY SIX AXES
------------
A 4-DOF arm (yaw + 3 coplanar pitches) can place the hand anywhere in its
workspace but can only orient it within ONE vertical plane - the plane is fixed
by where the hand has to be. A 5-DOF arm adds roll about the approach axis,
which frees the gripper's spin but still leaves the approach direction trapped
in that plane.

Holding an object rigidly with two hands through a 90 deg flip requires each
hand's FULL orientation to follow the object. That needs three wrist axes. With
five it fails: the wrist pitch would have to reach 128 deg at the 70 deg point
of the flip while the approach direction is simultaneously constrained. With six
- and a wrist pitch range of +/-140 deg - all 36 tested configurations solve
through the entire rotation.

BECAUSE THE LAST THREE AXES INTERSECT, IK STAYS CLOSED FORM
-----------------------------------------------------------
    1. wrist centre = grasp point - LG * approach direction
    2. J1..J3 from that position   (yaw + cosine rule on a two-link chain)
    3. J4..J6 from the residual rotation, as X-Y-X Euler angles
No iteration, no solver, no tolerances: a failure is a true geometric
impossibility.

FRAMES
------
base_link: origin on the ground under the base centre, x forward, y left, z up.
Every z therefore reads as height above ground.
Tool frame columns are (approach, finger-closing, third) axes. "Approach" is the
direction the hand points; the fingers close along the second axis.
"""
import math

# ---------------------------------------------------------------- geometry
# MUST match the robot description xacro.
SHOULDER_X = 0.080
SHOULDER_Y = 0.120     # separation 0.24 m; mount radius 0.144 m, inside the deck
SHOULDER_Z = 0.240
L1 = 0.180          # see the note in robot.urdf.xacro
L2 = 0.180
LG = 0.070             # wrist centre -> grasp point

# ---------------------------------------------------------------- limits
YAW_LIM = math.radians(150)
PITCH2_LIM = math.radians(115)     # shoulder
PITCH3_LIM = math.radians(145)     # elbow: raised to shrink the dead zone
W_ROLL_LIM = math.radians(180)     # J4 and J6
W_PITCH_LIM = math.radians(140)    # J5: 120 was too small - see module docstring
ROLL2_LIM = math.radians(175)       # J2b shoulder roll - the SEVENTH axis, see solve()

# Closest the elbow can fold. 0.096 m at a 145 deg elbow limit, versus 0.174 m
# at 115 deg - the dead sphere in front of each shoulder is roughly halved.
D_MIN = math.sqrt(2 * L1 * L1 * (1 - math.cos(math.pi - PITCH3_LIM)))

DOWN = -math.pi / 2

# THE JOINT ORDER, and the single place it is written down.
# j2b is the shoulder ROLL and sits between j2 and j3, so this is NOT j1..j7. Naming it
# j2b instead of renumbering keeps every existing joint name, log line and recorded pose
# valid. It lives HERE, in the ROS-free module, because the order IS the kinematic chain -
# and because the offline tests cannot import arm_commander (it pulls in ROS messages),
# so a copy there would be unreachable to the tests that need to follow it.
ARM_JOINT_SUFFIX = ('j1', 'j2', 'j2b', 'j3', 'j4', 'j5', 'j6')
N_ARM_JOINTS = len(ARM_JOINT_SUFFIX)


class IKError(Exception):
    pass


# ---------------------------------------------------------------- 3x3 helpers
def mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def mat_vec(A, v):
    return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]


def transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def rot_x(t):
    c, s = math.cos(t), math.sin(t)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def rot_y(t):
    c, s = math.cos(t), math.sin(t)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def frame(approach, finger):
    """Build a tool rotation matrix from an approach axis and a finger axis.
    The finger axis is orthogonalised against the approach axis."""
    ax = _unit(approach)
    fy = [finger[i] - sum(finger[k] * ax[k] for k in range(3)) * ax[i]
          for i in range(3)]
    fy = _unit(fy)
    return [[ax[i], fy[i], cross(ax, fy)[i]] for i in range(3)]


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-12:
        raise IKError('degenerate axis')
    return [c / n for c in v]


def tool_down_fingers_along(axis_angle):
    """Hand pointing straight down, fingers closing across a horizontal object
    whose long axis lies at `axis_angle`. The finger axis is perpendicular to
    the object axis, which is what makes the gripper straddle it."""
    perp = [-math.sin(axis_angle), math.cos(axis_angle), 0.0]
    return frame([0.0, 0.0, -1.0], perp)


# ---------------------------------------------------------------- kinematics
def shoulder(side):
    """side: +1 left arm, -1 right arm."""
    return (SHOULDER_X, side * SHOULDER_Y, SHOULDER_Z)


def solve6(p, R, side, prefer_elbow_up=True):
    """Inverse kinematics.

    p: grasp point (x, y, z) in base_link
    R: desired TOOL rotation matrix (columns: approach, finger, third)
    Returns [q1..q6]. Raises IKError if impossible within the joint limits.
    SIX-DOF. Kept because the tests that prove six axes are insufficient need it;
    production code calls solve().
    """
    S = shoulder(side)
    approach = [R[0][0], R[1][0], R[2][0]]
    W = [p[i] - LG * approach[i] for i in range(3)]      # wrist centre

    dx, dy = W[0] - S[0], W[1] - S[1]
    q1 = math.atan2(dy, dx)
    if abs(q1) > YAW_LIM:
        raise IKError(f'yaw {math.degrees(q1):.0f} deg beyond limit')

    r = math.hypot(dx, dy)
    dz = W[2] - S[2]
    d = math.hypot(r, dz)
    if d > L1 + L2 - 1e-9:
        raise IKError(f'wrist centre {d:.3f} m beyond reach {L1 + L2:.3f}')
    if d < D_MIN:
        raise IKError(f'wrist centre {d:.3f} m inside the elbow dead zone '
                      f'{D_MIN:.3f}')

    gamma = math.atan2(dz, r)
    ca = max(-1.0, min(1.0, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)))
    cb = max(-1.0, min(1.0, (L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2)))
    alpha, beta = math.acos(ca), math.acos(cb)

    signs = (+1, -1) if prefer_elbow_up else (-1, +1)
    last = 'no solution'
    for sgn in signs:
        e2 = gamma + sgn * alpha
        e3 = e2 - sgn * (math.pi - beta)
        q2, q3 = -e2, e2 - e3
        if abs(q2) > PITCH2_LIM or abs(q3) > PITCH3_LIM:
            last = 'shoulder/elbow limit'
            continue

        # orientation of the forearm frame at the wrist centre
        xa = mat_vec(rot_z(q1), [math.cos(e3), 0.0, math.sin(e3)])
        ya = mat_vec(rot_z(q1), [0.0, 1.0, 0.0])
        Ra = [[xa[i], ya[i], cross(xa, ya)[i]] for i in range(3)]

        Rw = mat_mul(transpose(Ra), R)          # residual wrist rotation
        q5 = math.acos(max(-1.0, min(1.0, Rw[0][0])))     # X-Y-X extraction
        if abs(math.sin(q5)) < 1e-6:
            cands = [(0.0, q5, math.atan2(Rw[2][1], Rw[1][1]))]
        else:
            q6 = math.atan2(Rw[0][1], Rw[0][2])
            q4 = math.atan2(Rw[1][0], -Rw[2][0])
            cands = [(q4, q5, q6),
                     (q4 + math.pi, -q5, q6 + math.pi)]    # equivalent branch
        for c in cands:
            a4, a5, a6 = [math.remainder(v, 2 * math.pi) for v in c]
            if (abs(a4) <= W_ROLL_LIM and abs(a5) <= W_PITCH_LIM
                    and abs(a6) <= W_ROLL_LIM):
                return [q1, q2, q3, a4, a5, a6]
        last = f'wrist limit (pitch needs {math.degrees(q5):.0f} deg)'
    raise IKError(f'{last} at ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) side {side}')


def fk6(q, side):
    """Forward kinematics. Returns (grasp point, tool rotation)."""
    q1, q2, q3, q4, q5, q6 = q
    S = shoulder(side)
    e2 = -q2
    e3 = -(q2 + q3)
    r = L1 * math.cos(e2) + L2 * math.cos(e3)
    dz = L1 * math.sin(e2) + L2 * math.sin(e3)
    W = [S[0] + r * math.cos(q1), S[1] + r * math.sin(q1), S[2] + dz]

    xa = mat_vec(rot_z(q1), [math.cos(e3), 0.0, math.sin(e3)])
    ya = mat_vec(rot_z(q1), [0.0, 1.0, 0.0])
    Ra = [[xa[i], ya[i], cross(xa, ya)[i]] for i in range(3)]
    R = mat_mul(Ra, mat_mul(rot_x(q4), mat_mul(rot_y(q5), rot_x(q6))))
    approach = [R[0][0], R[1][0], R[2][0]]
    p = [W[i] + LG * approach[i] for i in range(3)]
    return p, R


# =========================================================================
# SEVEN DOF - the shoulder roll, and why it exists
# =========================================================================
# For a given hand pose the elbow is NOT determined: it may lie anywhere on a CIRCLE
# about the shoulder-wrist line. Six axes sample that circle at exactly TWO points
# (elbow up, elbow down) and each fails somewhere in this task -
#
#   elbow up   collides arm-against-arm at the coaxial 90 deg pose
#   elbow down enters the base on the descent
#   switching between them is a 207 deg jump in j3
#
# and, measured with the box-decomposed collision model, six DOF puts the finger
# INSIDE its own forearm on 23 of 94 arm-poses across the cycle, worst -19.6 mm.
#
# A roll about the upper arm's own axis exposes the whole circle continuously. It does
# NOT move the elbow - the elbow position is fixed by j1 and j2 - it rotates the PLANE
# the forearm swings in. That is the classic 3-1-3 arrangement (Franka, iiwa, Baxter).
#
# THE SOLVE STAYS CLOSED FORM because psi is an INPUT, not something optimised inside.
# Given psi the elbow is a point, so the arm is fully determined and every step below is
# direct. No iteration, no tolerances: a failure is still a true geometric impossibility.
# Choosing psi is a separate, explicit concern - see collision.best_psi().
#
#   psi = 0 reproduces the six-DOF ELBOW-UP solution, because u is defined as the
#   component of world +z perpendicular to the shoulder-wrist line. So solve(psi=0) and
#   solve6(prefer_elbow_up=True) agree, which is what test_psi_zero_matches_six_dof pins.
# =========================================================================
def elbow_circle(p, R, side):
    """Centre, radius and basis of the circle the elbow may lie on.

    Returns (C, rad, u, v, W, S). u is the 'elbow up' direction, so psi=0 is elbow-up.
    """
    S = shoulder(side)
    approach = [R[0][0], R[1][0], R[2][0]]
    W = [p[i] - LG * approach[i] for i in range(3)]
    D = [W[i] - S[i] for i in range(3)]
    d = math.sqrt(sum(c * c for c in D))
    if d > L1 + L2 - 1e-9:
        raise IKError(f'wrist centre {d:.3f} m beyond reach {L1 + L2:.3f}')
    if d < D_MIN:
        raise IKError(f'wrist centre {d:.3f} m inside the elbow dead zone {D_MIN:.3f}')
    n = [c / d for c in D]
    a = (L1 * L1 - L2 * L2 + d * d) / (2 * d)
    rad = math.sqrt(max(0.0, L1 * L1 - a * a))
    C = [S[i] + a * n[i] for i in range(3)]
    # u = world +z with the along-axis part removed, so psi=0 lifts the elbow.
    ref = [0.0, 0.0, 1.0] if abs(n[2]) < 0.99 else [1.0, 0.0, 0.0]
    u = _unit([ref[i] - sum(ref[k] * n[k] for k in range(3)) * n[i] for i in range(3)])
    v = cross(n, u)
    return C, rad, u, v, W, S


def solve(p, R, side, psi=0.0, seed_q=None, _all_branches=False, both_shoulders=False):
    """SEVEN-DOF inverse kinematics. psi is the elbow swivel in radians.

    p:   grasp point (x, y, z) in base_link
    R:   desired TOOL rotation (columns: approach, finger, third)
    psi: where on the elbow circle to sit. 0 = elbow up, matching solve6().
    seed_q: the previous waypoint's solution, if there is one. The X-Y-X wrist has TWO
        valid solutions for the same hand pose - (q4, q5, q6) and (q4+pi, -q5, q6+pi) -
        and both are usually inside the limits. Taking whichever comes first makes the
        wrist flip 180 deg mid-path for no kinematic reason; it put a 163 deg step in j4
        between two waypoints 4 mm apart. Given a seed, the nearer branch is taken.
        Without one the first valid branch is used, which keeps psi=0 identical to
        solve6() and is what test_psi_zero_matches_six_dof pins.
    both_shoulders: also offer the SECOND shoulder solution - see below.
    Returns [q1, q2, q2b, q3, q4, q5, q6].

    THE SHOULDER HAS TWO SOLUTIONS TOO. The upper arm's direction fixes (j1, j2) only up
    to (j1 + pi, pi - j2): yaw round the other way and pitch back over the top. asin()
    returns |j2| <= 90 deg, so the second one was never offered - yet the PARKED pose is
    exactly that kind (j2 = -114 deg, the upper arm tipped back over the shoulder), and the
    nearest solution this function could give for the parked hand was 147 deg of joint
    travel away. That is why the arms reached and left parked by blind joint-space jumps, one
    of which swept a bottle over. Offered on request, so the default solution set - and
    every plan that does not ask for it - is unchanged.
    """
    C, rad, u, v, W, S = elbow_circle(p, R, side)
    E = [C[i] + rad * (math.cos(psi) * u[i] + math.sin(psi) * v[i]) for i in range(3)]

    # --- j1, j2: the upper arm aims at the elbow -------------------------------
    ua = _unit([E[i] - S[i] for i in range(3)])
    q1a = math.atan2(ua[1], ua[0])
    q2a = -math.asin(max(-1.0, min(1.0, ua[2])))
    shoulders = [(q1a, q2a)]
    if both_shoulders:
        shoulders.append((math.remainder(q1a + math.pi, 2 * math.pi),
                          math.remainder(math.pi - q2a, 2 * math.pi)))
    ok, last = [], None
    for q1, q2 in shoulders:
        if abs(q1) > YAW_LIM:
            last = f'yaw {math.degrees(q1):.0f} deg beyond limit'
            continue
        if abs(q2) > PITCH2_LIM:
            last = f'shoulder pitch {math.degrees(q2):.0f} deg beyond limit'
            continue

        # --- j2b, j3: the forearm, in the upper arm's own frame --------------------
        # After the roll and the elbow bend the forearm points, in that frame, along
        #     [cos q3, sin(q2b) sin q3, -cos(q2b) sin q3]
        # so q3 comes from the x component and q2b from the other two. q3 is taken
        # non-negative: a negative bend is the same arm as +q3 with the roll turned
        # through pi, and the roll is now free to express it.
        Ru = mat_mul(rot_z(q1), rot_y(q2))
        f = mat_vec(transpose(Ru), [W[i] - E[i] for i in range(3)])
        q3 = math.acos(max(-1.0, min(1.0, f[0] / L2)))
        if abs(q3) > PITCH3_LIM:
            last = f'elbow {math.degrees(q3):.0f} deg beyond limit'
            continue
        q2b = 0.0 if abs(math.sin(q3)) < 1e-9 else math.atan2(f[1], -f[2])
        # THE ROLL AND THE ELBOW HAVE TWO REPRESENTATIONS, exactly as the wrist does:
        # rolling the arm plane through pi and bending the elbow the other way reaches the
        # same forearm direction. q3 comes from acos() so it is always non-negative, which
        # forces the roll to carry the sign - and when the arm is PLANAR (f[1] = 0, the
        # ordinary case) that puts q2b at exactly +/-180 deg whenever the elbow bends the far
        # way. A 175 deg stop then rejects an ordinary, reachable pose: 'shoulder roll -180
        # deg beyond limit'. So offer both and take whichever is inside the stops.
        if abs(q2b) > ROLL2_LIM:
            alt_b = math.remainder(q2b - math.copysign(math.pi, q2b), 2 * math.pi)
            if abs(alt_b) <= ROLL2_LIM and abs(-q3) <= PITCH3_LIM:
                q2b, q3 = alt_b, -q3
            else:
                last = f'shoulder roll {math.degrees(q2b):.0f} deg beyond limit'
                continue

        # --- j4, j5, j6: residual rotation, X-Y-X as before -----------------------
        Ra = mat_mul(Ru, mat_mul(rot_x(q2b), rot_y(q3)))
        Rw = mat_mul(transpose(Ra), R)
        q5 = math.acos(max(-1.0, min(1.0, Rw[0][0])))
        if abs(math.sin(q5)) < 1e-6:
            cands = [(0.0, q5, math.atan2(Rw[2][1], Rw[1][1]))]
        else:
            cands = [(math.atan2(Rw[1][0], -Rw[2][0]), q5, math.atan2(Rw[0][1], Rw[0][2])),
                     (math.atan2(Rw[1][0], -Rw[2][0]) + math.pi, -q5,
                      math.atan2(Rw[0][1], Rw[0][2]) + math.pi)]
        found = False
        for c in cands:
            a4, a5, a6 = [math.remainder(x, 2 * math.pi) for x in c]
            if (abs(a4) <= W_ROLL_LIM and abs(a5) <= W_PITCH_LIM
                    and abs(a6) <= W_ROLL_LIM):
                ok.append([q1, q2, q2b, q3, a4, a5, a6])
                found = True
        if not found:
            last = (f'wrist limit (pitch needs {math.degrees(q5):.0f} deg) at '
                    f'({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) side {side} psi '
                    f'{math.degrees(psi):.0f}')
    if ok:
        if _all_branches:
            return ok
        if seed_q is not None and len(ok) > 1:
            # RAW travel, not the angle modulo 360: every joint stops short of +/-180 deg,
            # so a solution that is "near" the seed only across the wrap is the far one.
            ok.sort(key=lambda w: max(abs(w[i] - seed_q[i]) for i in range(len(w))))
        return ok[0]
    raise IKError(last or 'no solution')


def solve_branches(p, R, side, psi, both_shoulders=False):
    """Every valid arm for this hand pose and swivel - one entry per wrist branch (and per
    shoulder solution, when asked).

    THE TWO ARE NOT INTERCHANGEABLE. The X-Y-X wrist's second solution flips the palm
    over (q4+pi, -q5, q6+pi), which points the fingers and the suction cups the other way
    round the object, so the two branches have genuinely DIFFERENT clearances. Picking a
    branch for continuity alone and only then measuring clearance therefore hides half the
    options, and cost 8 self-collisions that a joint search over both finds a way round.
    Exposing them lets the caller score psi and branch together.
    """
    try:
        return solve(p, R, side, psi, None, _all_branches=True, both_shoulders=both_shoulders)
    except IKError:
        return []


def fk(q, side):
    """SEVEN-DOF forward kinematics. Returns (grasp point, tool rotation)."""
    q1, q2, q2b, q3, q4, q5, q6 = q
    S = shoulder(side)
    Ru = mat_mul(rot_z(q1), rot_y(q2))
    E = [S[i] + L1 * Ru[i][0] for i in range(3)]
    Ra = mat_mul(Ru, mat_mul(rot_x(q2b), rot_y(q3)))
    W = [E[i] + L2 * Ra[i][0] for i in range(3)]
    R = mat_mul(Ra, mat_mul(rot_x(q4), mat_mul(rot_y(q5), rot_x(q6))))
    approach = [R[0][0], R[1][0], R[2][0]]
    return [W[i] + LG * approach[i] for i in range(3)], R


def reachable(p, R, side, psi=0.0):
    try:
        solve(p, R, side, psi)
        return True
    except IKError:
        return False
