"""Section B choreography - the fully bimanual flip, from surface to surface.

WHAT "BIMANUAL" HAS TO MEAN HERE
--------------------------------
Both hands take hold of the lying object, both lift it, both yaw it to the flip axis,
both rotate it 90 deg to upright, both lower it to the surface it lay on (the floor, or a
shelf) and both let go. There is no point in the cycle where one hand carries the object
while the other is parked.

That was NOT true until 2026-09-27, in two stages worth remembering:

  * at 6 DOF the flip handed over at 70 deg - the coaxial upright pose made the two arms
    collide, so the right hand withdrew and the left finished alone;
  * at 7 DOF the flip itself went to 90 deg with both hands, but the SET-DOWN still sent
    the right hand straight to a parked pose at the first lowering waypoint - fingers
    still closed round the bottom of the object - and the left lowered it alone. On
    screen that is the same "left hand carries the rest", just after the flip.

THE ORDER, and why each part is where it is
-------------------------------------------
    approach -> lift -> yaw -> flip x3 -> lower -> settle -> release -> retreat

**The lower hand is not trapped.** Stood up, the object's two ends are on one vertical
line and the right hand holds the BOTTOM end - 35 mm above the surface once the object is
down, its lowest plate 15 mm clear. That is tight but real, and the path planner treats
the floor and every fixture as obstacles throughout. So both hands can carry it all the
way down.

**Both hands release together, then leave in the only directions they can.** The right
hand backs straight out along its own approach axis - its fingers straddle the object
sideways, so once open they slide off it - and the left lifts straight up off the top.
Lifting the right instead would drive it into the left, which is directly above it.

**The welded hand ends on top.** Only ``left_palm_link`` carries the ``DetachableJoint``
(suction), so the flip rotates the way that takes the LEFT hand UP (``FLIP_DIR = -1``).

**One definition.** :func:`cycle_legs` is the ONLY place the sequence is written down.
The task node plans and executes it leg by leg and the offline suite validates the same
list. The fault that let a removed handover keep running for a whole session was two
copies of the sequence - one the tests checked and one the robot ran.
"""
import math

from . import collision as _col
from . import ik
# THE SHARED GEOMETRY - the parked pose, the planned trips to and from it, the grip. Same
# file in the pick-place section; see motion.py.
from .motion import (TRAVEL_LEFT, TRAVEL_RIGHT, HOVER_DZ, GRIP_DZ, END_CLEAR,  # noqa: F401
                     TIP_PAST_GRASP, TIP_CLEAR, HAND_HALF_H, HAND_GAP, PARK_SAMPLES,
                     grip_half_separation, left_sign, parked_pair, transit_waypoints,
                     rot_about as _rot_about, slerp as _slerp)

FLIP_REACH = 0.30        # base-frame x of the flip station (re-swept, see config)
FLIP_Z = 0.24            # object centre height during the rotation (re-swept)
PLACE_DROP = 0.010       # weld released this far above the surface, see settle_waypoints
LOWER_CLEAR = 0.08       # how far the released hand lifts clear
CLEAR_OUT = 0.15         # how far the lower hand withdraws sideways
SAMPLES_PER_LEG = 4
FLIP_SAMPLES = 13

# The rotation sense of the flip. -1 takes the LEFT hand (the welded one) UP.
FLIP_DIR = -1.0

# THE FLIP GOES THE WHOLE 90 DEG WITH BOTH HANDS. At 6 DOF it had to stop at 70: the
# coaxial upright pose puts both hands on one vertical line, and with a mirrored elbow the
# two arms met - -11.4 mm of arm-arm clearance and -19.6 mm of the holding arm inside its
# own forearm. The seventh axis routes the elbows round opposite sides of that column, so
# there is no handover angle any more and this is not a tunable. It used to be read from
# the config as flip_handover_deg; that knob is gone so it cannot be turned back down.
FLIP_ANGLE = math.radians(90.0)
FLIP_SEGMENTS = 3        # driven in segments so the object's real tilt is sampled en route

# The axis every flip happens at, in the base frame: lateral, i.e. across the reach.
CANONICAL_AXIS = math.pi / 2

# THE PARKED POSE (TRAVEL_LEFT / TRAVEL_RIGHT), the grip separation and which hand takes
# which end are in motion.py, shared with the pick-place section - one robot, one parked pose.


def _ends(centre, axis_angle, s, R_tool, theta=0.0, sign_left=None,
          grip_dz=None):
    """The two hand poses gripping the object's ends, with the object rotated by
    ``theta`` about the horizontal axis perpendicular to its long axis.

    Returns ``((p_left, R_left), (p_right, R_right))``.

    THE ASSIGNMENT IS FIXED, NOT SORTED. An earlier version chose the left hand as
    whichever end had the larger y, re-deciding at every ``theta``. That works while
    the object is horizontal, but standing it up brings both ends onto the same
    vertical line, so at the end of the flip the two y values become equal and the
    comparison tipped the other way. The two arms then swapped targets in a single
    waypoint - a 170 mm instantaneous jump, which is precisely the violent motion
    seen just before the object was set down. Deciding once at ``theta = 0`` and
    holding it makes the swap impossible rather than unlikely.

    THE GRASP POINT IS OFF THE OBJECT'S AXIS BY ``grip_dz``, in the object's own
    frame, so the offset ROTATES WITH the object. This is the section A palm fault:
    the palm plate sits 0.027 m behind the grasp point, so gripping on the axis of a
    90 mm object put the palm 18 mm INSIDE it, and the arm pressed a rigid body until
    the base lifted. Offsetting along the object's local "up" fixes it in both
    orientations at once, because that direction rotates too:

        theta = 0    offset is +z          -> 25 mm above the lying object's axis
        theta = -90  offset is -axis       -> 25 mm to the side of the standing one

    and in both cases the palm ends up 63 mm from the axis of a 45 mm-radius body.
    The finger plate's span relative to the object's axis is likewise invariant
    (10 to 45 mm), which is why a single ``close`` value serves the whole cycle.
    """
    if sign_left is None:
        sign_left = left_sign(axis_angle)
    if grip_dz is None:
        grip_dz = GRIP_DZ      # resolved at CALL time, so sweeps can vary it
    ax = [math.cos(axis_angle), math.sin(axis_angle), 0.0]
    n = [-math.sin(axis_angle), math.cos(axis_angle), 0.0]
    # the object's local "up" is ax x n, which is exactly +z while it lies down
    up = [0.0, 0.0, 1.0]
    Rt = _rot_about(n, theta)

    # s MAY BE A PAIR (s_left, s_right). The left hand grips the object's CENTRE
    # (s_left = 0) and the right supports an END, so the two offsets differ. A scalar
    # still means the old symmetric +/- s, which the geometry tests rely on.
    s_pair = tuple(s) if isinstance(s, (tuple, list)) else (s, s)
    assert len(s_pair) == 2, 's must be a scalar or a (left, right) pair'

    out = []
    for hand, sign in enumerate((sign_left, -sign_left)):
        off = [sign * s_pair[hand] * ax[i] + grip_dz * up[i] for i in range(3)]
        p = [centre[i] + ik.mat_vec(Rt, off)[i] for i in range(3)]
        R = ik.mat_mul(Rt, R_tool)
        out.append((p, R))
    return (out[0], out[1])


def _leg(start, end, n=SAMPLES_PER_LEG):
    """Interpolate positions between two station pairs, excluding the start.

    Orientation is held at the END pair's value. Every leg in this section is a
    pure translation, so there is no rotation to interpolate.
    """
    out = []
    for i in range(1, n + 1):
        f = i / n
        pair = []
        for (p0, _), (p1, R1) in zip(start, end):
            pair.append(([p0[k] + f * (p1[k] - p0[k]) for k in range(3)], R1))
        out.append((pair[0], pair[1]))
    return out


# --------------------------------------------------------------- the sequence
def approach_waypoints(centre_xy, cz, axis_angle, s, hover_dz=HOVER_DZ):
    """Hover above the lying object, then descend onto its two ends."""
    R = ik.tool_down_fingers_along(axis_angle)
    hover = _ends([centre_xy[0], centre_xy[1], cz + hover_dz], axis_angle, s, R)
    grasp = _ends([centre_xy[0], centre_xy[1], cz], axis_angle, s, R)
    return [hover] + _leg(hover, grasp)


def lift_waypoints(centre_xy, cz, axis_angle, s, flip_z=FLIP_Z):
    """Lift the still-horizontal object to the flip station, from the grasp pose."""
    R = ik.tool_down_fingers_along(axis_angle)
    grasp = _ends([centre_xy[0], centre_xy[1], cz], axis_angle, s, R)
    high = _ends([centre_xy[0], centre_xy[1], flip_z], axis_angle, s, R)
    return _leg(grasp, high)


def yaw_waypoints(centre_xy, psi_from, psi_to, s, flip_z=FLIP_Z,
                  step=math.radians(4.0)):
    """Rotate the held object about the VERTICAL, to a canonical axis for the flip.

    WHY THE FLIP CANNOT JUST TAKE AN ARBITRARY YAW. Sweeping it showed the flip envelope
    is ONE-SIDED: axis lines from about 65 to 90 degrees work and nothing above 90 does.
    At the top of the flip the welded hand's wrist is displaced from the object's centre
    by (LG + grip_dz) = 95 mm ALONG the object's axis, which pulls it toward the robot
    when cos(psi) > 0 and pushes it away when cos(psi) < 0 - 0.340 m against a 0.320 m
    limit. Choosing FLIP_DIR per object does not help: the grip offset and the tool
    approach reverse together, so both representatives of an axis line give the same
    sign. No station fixed it either (reach 0.26-0.32 x height 0.22-0.30, nothing
    reached even +/-10 degrees).

    The GRASP, by contrast, is symmetric over +/-20 degrees. So the variability is
    absorbed here: pick the object at whatever angle it lies, turn it to the canonical
    axis, and flip in the frame where the flip is already verified. Both hands orbit the
    object's centre to do this, which is a genuinely two-armed motion.
    """
    sl = left_sign(psi_from)
    n = max(2, int(math.ceil(abs(psi_to - psi_from) / step)))
    R_at = ik.tool_down_fingers_along
    out = []
    for i in range(1, n + 1):
        psi = psi_from + (psi_to - psi_from) * i / n
        out.append(_ends([centre_xy[0], centre_xy[1], flip_z], psi, s,
                         R_at(psi), 0.0, sl))
    return out


def flip_waypoints(centre_xy, axis_angle, s, flip_z=FLIP_Z, n=FLIP_SAMPLES,
                   upto=None, frm=0.0):
    """THE COLLABORATIVE FLIP: both hands rotate the object together.

    Rotates from ``frm`` to ``upto`` (default :data:`FLIP_ANGLE`, the full 90 deg).
    ``frm`` exists so the rotation can be commanded in SEGMENTS with the object's ACTUAL
    tilt sampled part-way through. Without it the only instrument running during the
    rotation was the slip monitor, which measures the GRIP; nothing measured the TASK
    until the very end, so a failed flip reported "90 deg from vertical, NOT UPRIGHT" -
    true, and useless for working out when it stopped tracking.

    Sampled finely because this is the phase where both hands' full orientation has to
    track the object's rotation - the motion that needs a spherical wrist on each arm.
    """
    if upto is None:
        upto = FLIP_ANGLE
    R = ik.tool_down_fingers_along(axis_angle)
    centre = [centre_xy[0], centre_xy[1], flip_z]
    seq = []
    for i in range(1, n + 1):
        theta = FLIP_DIR * (frm + (upto - frm) * i / n)
        seq.append(_ends(centre, axis_angle, s, R, theta))
    return seq


def _upright(centre_xy, axis_angle, s, cz):
    """Both hands on the STANDING object, its centre at height ``cz``."""
    R = ik.tool_down_fingers_along(axis_angle)
    return _ends([centre_xy[0], centre_xy[1], cz], axis_angle, s, R,
                 FLIP_DIR * math.pi / 2)


LOWER_SAMPLES = 6        # the set-down is ~110 mm, so ~18 mm a step


def lower_waypoints(centre_xy, axis_angle, s, upright_cz, flip_z=FLIP_Z,
                    n=LOWER_SAMPLES):
    """BOTH hands lower the upright object from the flip station towards the surface.

    ``upright_cz`` is where its centre stops - the caller passes ``upright_cz + drop`` so
    the suction can be released just above the surface (see :func:`settle_waypoints`).

    THIS USED TO BE ONE HAND. The right hand was sent to a parked pose at the very first
    waypoint - 150 mm sideways and re-levelled in one step, fingers still closed round the
    bottom of the object - and the left lowered it alone. The two hands now move together,
    so the grasp span between them is the same here as it is in the flip; the suite and the
    harness both assert that span from the grasp to the release.
    """
    return _leg(_upright(centre_xy, axis_angle, s, flip_z),
                _upright(centre_xy, axis_angle, s, upright_cz), n)


def settle_waypoints(centre_xy, axis_angle, s, upright_cz, drop=PLACE_DROP, n=2):
    """The last ``drop`` onto the surface, with BOTH hands, after the suction lets go.

    WHY THIS PHASE EXISTS. The arms are position controlled, so lowering an ATTACHED object
    until its base is exactly on the surface drives a rigid body into a rigid surface: the
    controller keeps pushing and the reaction lifts the base. So :func:`lower_waypoints`
    stops ``drop`` short, the suction releases there, and this phase closes the gap with all
    four fingers still shut - the object is held by friction alone, so it settles instead of
    being forced. The fingers only open once it is standing.
    """
    return _leg(_upright(centre_xy, axis_angle, s, upright_cz + drop),
                _upright(centre_xy, axis_angle, s, upright_cz), n)


# ---------------------------------------------------------------------------
# LEAVING THE STANDING OBJECT - planned for ANY surroundings, not for this layout.
#
# The Gazebo run of 2026-09-27 showed the two ways a retreat goes wrong: the lower hand
# backed out sideways at floor height into the box lying beside the bottle, and an arm
# swung through the bottle it had just stood up. Neither is fixed by knowing where the box
# is - in the finished system objects are scattered across a room and their positions come
# from perception. So the hands leave by a sequence that is safe BY CONSTRUCTION, with every
# distance derived from the held object and the gripper, and the planner then checks the
# whole thing against whatever objects are known:
#
#   release   both hands, open, slide straight UP along the object - the upper one off its
#             top, the lower one CLUTTER_H above the surface, clear of anything lying nearby
#   clear     only then does the lower hand move sideways: straight back along its own
#             approach axis, just far enough for its fingertips to be TIP_CLEAR off the object
#   retreat   both rise
#   park      a PLANNED path to the parked pose - not a joint-space jump, which knows
#             nothing about the object and was what swept it over
# ---------------------------------------------------------------------------

# HOW HIGH THE LOWER HAND CLIMBS BEFORE IT MOVES SIDEWAYS: above anything lying on the
# SURFACE the object stands on. The widest object in the survey (STORY.md, section 7) is a 2 L
# soda bottle at 110 mm across, so nothing lying down stands taller than that; the hand's
# plates reach 20 mm below its grasp point, and 10 mm is left spare. Measured from the surface,
# so on a shelf it is the shelf's clutter it climbs above, not the floor's.
CLUTTER_H = 0.110 + 0.020 + 0.010
CLUTTER_Z = CLUTTER_H          # the same height on the floor, where the surface is at 0

# (TIP_CLEAR, TIP_PAST_GRASP, HAND_HALF_H and HAND_GAP are in motion.py, shared.)


def backout_distance(half_width):
    """How far the lower hand must back out for its fingertips to clear the object.

    Along the approach axis, measured from the grasp point, the object's near surface is at
    GRIP_DZ - half_width (the axis sits GRIP_DZ beyond the grasp point) and the open
    fingertips at TIP_PAST_GRASP. Derived, so a different object or finger changes it.
    """
    return TIP_PAST_GRASP - (GRIP_DZ - half_width) + TIP_CLEAR


def release_waypoints(centre_xy, axis_angle, s, upright_cz, clear=LOWER_CLEAR,
                      n=SAMPLES_PER_LEG, surface_z=0.0):
    """Both hands, now open, slide straight up - the upper off the top, the lower along the side.

    Neither moves sideways yet. The fingers straddle the object across the closing axis, so
    once open they slide freely along it. The upper hand lifts ``clear`` off the top; the
    lower one rises CLUTTER_H above the surface, but never to within HAND_GAP of the upper.
    ``upright_cz`` is the standing object's centre height, absolute.
    """
    start = _upright(centre_xy, axis_angle, s, upright_cz)
    (pl, Rl), (pr, Rr) = start
    top_l = pl[2] + clear
    z_r = max(pr[2], min(surface_z + CLUTTER_H, top_l - 2 * HAND_HALF_H - HAND_GAP))
    end = (([pl[0], pl[1], top_l], Rl), ([pr[0], pr[1], z_r], Rr))
    return _leg(start, end, n)


def clear_waypoints(centre_xy, axis_angle, s, upright_cz, half_width=0.045,
                    n=SAMPLES_PER_LEG, surface_z=0.0):
    """The lower hand, now above the surface's clutter, backs straight out along its approach axis.

    That is the only way it can leave: the upper hand is directly above it. It goes just far
    enough for the fingertips to end TIP_CLEAR off the object - see backout_distance().
    """
    (pl, Rl), (pr, Rr) = release_waypoints(centre_xy, axis_angle, s, upright_cz,
                                           surface_z=surface_z)[-1]
    appr = [Rr[i][0] for i in range(3)]
    out = backout_distance(half_width)
    end = ((pl, Rl), ([pr[k] - out * appr[k] for k in range(3)], Rr))
    return _leg(((pl, Rl), (pr, Rr)), end, n)


# HOW HIGH TO LIFT BEFORE GOING HOME. 0.140 above the object's top puts the left wrist at
# 0.38 for the bottle and 0.36 for the box. RETREAT_Z_MAX caps it because a rise to 0.46 is
# not reachable at this radius - that one is the ARM's limit, so it is absolute. The lower
# hand rises RETREAT_RIGHT_H above the surface, staying HAND_GAP below the upper one so the
# two do not meet over the object.
RETREAT_UP = 0.140
RETREAT_Z_MAX = 0.40
RETREAT_RISE = 4          # samples for the lift, so no single step is large
RETREAT_RIGHT_H = 0.25
RETREAT_RIGHT_Z = RETREAT_RIGHT_H       # the same height on the floor


def _tool_at(axis_angle, theta):
    """The tool rotation at flip angle ``theta``, position handled separately."""
    n = [-math.sin(axis_angle), math.cos(axis_angle), 0.0]
    return ik.mat_mul(_rot_about(n, theta), ik.tool_down_fingers_along(axis_angle))


def retreat_waypoints(centre_xy, axis_angle, s, upright_cz, half_width=0.045,
                      surface_z=0.0):
    """Both hands rise clear of the standing object, wrists still rolled, before going home."""
    (pl, Rl), (pr, Rr) = clear_waypoints(centre_xy, axis_angle, s, upright_cz,
                                         half_width, surface_z=surface_z)[-1]
    top = 2.0 * upright_cz - surface_z          # the standing object's top, absolute
    z_hi = min(RETREAT_Z_MAX, max(top + RETREAT_UP, pl[2] + 0.02))
    z_r = max(pr[2], min(surface_z + RETREAT_RIGHT_H, z_hi - 2 * HAND_HALF_H - HAND_GAP))
    end = (([pl[0], pl[1], z_hi], Rl), ([pr[0], pr[1], z_r], Rr))
    return _leg(((pl, Rl), (pr, Rr)), end, RETREAT_RISE)


# TO AND FROM THE PARKED POSE - PLANNED, NOT JUMPED: parked_pair() and transit_waypoints() are
# in motion.py, shared with the pick-place section.


# SLIDE OFF THE TOP BEFORE TURNING THE WRISTS. After the retreat the upper hand's fingers
# still reach across the top of the standing object, and the trip home rotates both wrists
# from the flip orientation to the parked one. Done at once, that rotation swings a fingertip
# down across the object's top - harmless on the floor, where the hand is 140 mm up, but on
# a 0.10 m table the retreat is capped by the arm's reach and the finger passed 4 mm from the
# bottle. Backing the hand out instead is not possible: at the end of the retreat its wrist is
# 0.338 m from the shoulder against a 0.36 m reach. So the upper hand slides ON, along its own
# fingers - towards its own side of the robot - until its fingertips are TIP_CLEAR past the
# object's far side, and only then do the wrists turn for home. Measured: 0.10 m table, bottle
# +4 -> +21 mm, box +23 -> +24 mm; floor +52 -> +59 mm.
WITHDRAW_SAMPLES = 4


def slide_off_distance(half_width):
    """How far the upper hand slides along its fingers to take them off the object's top.

    Along the approach axis the object's far side is GRIP_DZ + half_width beyond the grasp
    point and the open fingertips reach TIP_PAST_GRASP; the tips end TIP_CLEAR past it.
    """
    return max(0.0, GRIP_DZ + half_width + TIP_CLEAR - TIP_PAST_GRASP)


def withdraw_waypoints(start, half_width=0.045):
    """The upper (left) hand slides along its own approach axis off the top; the lower stays."""
    (pl, Rl), right = start
    d = slide_off_distance(half_width)
    appr = [Rl[k][0] for k in range(3)]
    end = (([pl[k] + d * appr[k] for k in range(3)], Rl), right)
    return _leg(start, end, WITHDRAW_SAMPLES)


# THE FLIP HEIGHT IS TIED TO THE SHOULDER, NOT TO THE SURFACE - unless the surface is high
# enough that the object would hit it. FLIP_Z was chosen by sweeping reach against the arm,
# so on a raised surface it stays where the arm works best; but the object sweeps down by up
# to sqrt(half_length^2 + half_width^2) below its centre while it turns, so on a shelf the
# station is raised until that lowest point is FLIP_CLEAR above the surface.
FLIP_CLEAR = 0.030


def flip_height(surface_z, upright_cz, half_width, flip_z=FLIP_Z):
    """Object centre height for the flip, above a surface at ``surface_z``.

    ``upright_cz`` is the object's centre height ABOVE THE SURFACE when standing, i.e. half
    its length.
    """
    return max(flip_z, surface_z + math.hypot(upright_cz, half_width) + FLIP_CLEAR)


def cycle_legs(centre_xy, cz, upright_cz, psi_pick, s, flip_z=None,
               hover_dz=HOVER_DZ, drop=None, half_width=0.045, surface_z=0.0):
    """THE CYCLE, parked to parked, as ``[(leg_name, waypoints), ...]`` in execution order.

    The ONLY definition of the sequence. task_node plans all of it as one path and then
    sends it leg by leg; the offline suite and the harness validate the same list. Every
    leg is a move of BOTH arms, and from the end of ``approach`` to the end of ``settle``
    both hands are on the object.

    ``unpark`` starts AT the parked pose and ends at the hover; ``withdraw`` slides the upper
    hand off the standing object's top before ``park`` turns the wrists and ends parked. The hover is its own leg so the arms can settle there before descending - moving
    from parked to hover and straight into the descent let the hand come down while it was
    still swinging into line (``right_j4 off 0.788`` at the end of the descent).

    ``psi_pick`` is the object's axis in the BASE frame as it lies. The flip always runs at
    :data:`CANONICAL_AXIS` - the envelope is one-sided - so ``yaw`` turns it there first.
    ``half_width`` is the object's half-width across the closing axis; it sets how far the
    lower hand backs out.

    EVERY HEIGHT IS FROM THE SURFACE. ``cz`` (lying) and ``upright_cz`` (standing) are the
    object's centre heights ABOVE the surface it lies on, which is at ``surface_z`` in the
    base frame - 0 for the floor, a shelf's top otherwise. The flip station is kept clear of
    that surface by flip_height(); the leaving sequence climbs above ITS clutter.
    """
    if flip_z is None:
        flip_z = FLIP_Z
    if drop is None:
        drop = PLACE_DROP
    canon = CANONICAL_AXIS
    flip_z = flip_height(surface_z, upright_cz, half_width, flip_z)
    cz = surface_z + cz                  # absolute from here on
    upright_cz = surface_z + upright_cz
    reach = approach_waypoints(centre_xy, cz, psi_pick, s, hover_dz)
    parked = parked_pair()
    legs = [('unpark', [parked] + transit_waypoints(parked, reach[0], PARK_SAMPLES)),
            ('approach', reach[1:]),
            ('lift', lift_waypoints(centre_xy, cz, psi_pick, s, flip_z)),
            ('yaw', yaw_waypoints(centre_xy, psi_pick, canon, s, flip_z))]
    for k in range(1, FLIP_SEGMENTS + 1):
        legs.append(('flip%d' % k,
                     flip_waypoints(centre_xy, canon, s, flip_z,
                                    upto=FLIP_ANGLE * k / FLIP_SEGMENTS,
                                    frm=FLIP_ANGLE * (k - 1) / FLIP_SEGMENTS)))
    retreat = retreat_waypoints(centre_xy, canon, s, upright_cz, half_width, surface_z)
    withdraw = withdraw_waypoints(retreat[-1], half_width)
    legs += [('lower', lower_waypoints(centre_xy, canon, s, upright_cz + drop, flip_z)),
             ('settle', settle_waypoints(centre_xy, canon, s, upright_cz, drop)),
             ('release', release_waypoints(centre_xy, canon, s, upright_cz,
                                           surface_z=surface_z)),
             ('clear', clear_waypoints(centre_xy, canon, s, upright_cz, half_width,
                                       surface_z=surface_z)),
             ('retreat', retreat),
             ('withdraw', withdraw),
             ('park', transit_waypoints(withdraw[-1], parked, PARK_SAMPLES))]
    return legs


# Legs during which BOTH hands hold the object. Asserted by the suite and the harness.
CARRY_LEGS = ('lift', 'yaw', 'flip1', 'flip2', 'flip3', 'lower', 'settle')
# Legs after the hands open, while they are still next to the object they just stood up.
LEAVE_LEGS = ('release', 'clear', 'retreat')
# Legs on the way home, clear of it by construction: the WHOLE arm must keep off it.
HOME_LEGS = ('withdraw', 'park')


def full_cycle(centre_xy, cz, upright_cz, psi_pick, s, flip_z=None, drop=None,
               hover_dz=HOVER_DZ, surface_z=0.0):
    """Every waypoint of one cycle, flattened - exactly what the robot is sent."""
    return [w for _, leg in cycle_legs(centre_xy, cz, upright_cz, psi_pick, s, flip_z,
                                       hover_dz, drop, surface_z=surface_z) for w in leg]


def left_ends_on_top(axis_angle, s, flip_z=FLIP_Z):
    """True if the flip leaves the WELDED (left) hand above the object.

    Asserted by the test suite: if this is ever false, the load-bearing hand is
    trapped under the object and the place phase cannot work.
    """
    R = ik.tool_down_fingers_along(axis_angle)
    left, right = _ends([FLIP_REACH, 0.0, flip_z], axis_angle, s, R,
                        FLIP_DIR * math.pi / 2)
    return left[0][2] > right[0][2] + 1e-6


def is_continuous(pairs, max_step=0.12):
    """True if no consecutive waypoints jump - a jump reads as a lurch in sim."""
    for (a_l, a_r), (b_l, b_r) in zip(pairs, pairs[1:]):
        for (pa, _), (pb, _) in ((a_l, b_l), (a_r, b_r)):
            if math.sqrt(sum((pa[k] - pb[k]) ** 2 for k in range(3))) > max_step:
                return False
    return True


def plan_inputs(legs, close, open_pos, lying, standing, others=(), fixtures=()):
    """Per-waypoint finger closure and obstacles for planning a cycle, from its leg names.

    The one statement of what the planner must keep clear of, and when:

      * ``fixtures`` - walls, shelves, tables, as boxes: all links, the whole cycle, with no
        margin - like the floor, a surface is where the hands are meant to go close
      * ``others`` - every OTHER known object, as capsules: all links, the whole cycle
      * the handled object while it LIES (``lying``): all links on the way to the hover;
        from then on the hands go onto it by design
      * the handled object once it STANDS (``standing``): only the ARMS while the hands are
        leaving it (the hands' paths are fixed to clear it), then all links on the way home

    The hands are OPEN except while carrying, which is what the collision model is given.
    ``wide`` marks the legs to and from parked, where the second shoulder solution is needed.
    Object positions come from wherever the caller has them - the config now, perception
    later; nothing here depends on where any particular object happens to be.
    """
    fingers, obs, wide = [], [], []
    base = [(c, None) for c in others] + [(b, None, 0.0) for b in fixtures]
    for name, seq in legs:
        for _ in seq:
            fingers.append(close if name in CARRY_LEGS else open_pos)
            wide.append(name in ('unpark', 'park'))
            lst = list(base)
            if name == 'unpark':
                lst += [(c, None) for c in lying]
            elif name in LEAVE_LEGS:
                lst += [(c, _col.ARM_LINKS) for c in standing]
            elif name in HOME_LEGS:
                lst += [(c, None) for c in standing]
            obs.append(lst)
    return fingers, obs, wide
