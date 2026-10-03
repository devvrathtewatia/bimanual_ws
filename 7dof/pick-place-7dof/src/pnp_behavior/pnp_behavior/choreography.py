"""Waypoint generation for section A: pick an object lying at an ARBITRARY yaw and
place it on a shelf slot, yaw-aligned to that slot.

WHAT CHANGED AND WHY
--------------------
The old version could only grasp an object whose axis was exactly tangential, because
``face()`` rotated the base until it was. That made ``face()`` load-bearing for
correctness: every degree of heading error rotated the grasp line, so one hand met the
object and the other closed on air. It also meant a 6-DOF wrist was exercised at exactly
one orientation, which is a poor demonstration of why it needs six.

Now the object's axis angle is a PARAMETER. The task node measures the true heading off
the walls and computes the axis angle in the base frame as ``axis_yaw - heading``, so a
heading error cancels instead of corrupting the grasp. The shelf cradles still require
the object tangential, so the object is rotated about the vertical - by both arms
together, at carry height, over the slot - before it is lowered. That yaw alignment is
what the wrist is for.

THE GRASP IS NEAR THE ENDS. ``s = L/2 - end_clear`` puts each hand about 15% in from its
end, which gives the pair a long moment arm on the object so it cannot pivot in the grip.
``end_clear`` must exceed the palm's 0.020 m half-extent along the axis or the palm hangs
off the end; at 0.035 there is 15 mm of object beyond each palm.

WHAT MUST NOT BE BROKEN (see ERROR_CATALOGUE.txt)
  E19  left/right is PINNED once, never re-sorted. The old code ended _stations with
       ``return (a, b) if a[0][1] >= b[0][1] else (b, a)``, re-deciding by y at every
       call. With a fixed tangential axis that always gave the same answer. With
       arbitrary yaw and a yaw rotation it flips mid-motion and the arms trade ends -
       a 170 mm instantaneous jump. Section B already suffered exactly this.
  E6   the grasp sits grip_dz ABOVE the object's axis so the palm clears its top. In
       this section the object stays horizontal and only yaws, so "above" is always +z
       and palm clearance is yaw-independent. That is why grip_dz is a scalar here and
       had to be a rotating vector in section B.
  E7   phases are continuous and sampled by DISTANCE, not by a fixed count. A fixed
       4 samples put 67.5 mm between waypoints on the 0.27 m lift, which the controller
       renders as a lurch.

THE CYCLE AS LEGS, THE SAME WAY AS THE REORIENT SECTION (2026-09-28)
--------------------------------------------------------------------
    pick_legs():   unpark -> approach -> lift
    place_legs():  carry -> yaw -> lower -> settle -> rise -> park

The base turns between the two halves, so each half is planned by the node as ONE
continuous path: the first from the exact parked joint vector, the second from wherever
the lift left the arms back to the exact parked joint vector. The trips to and from parked
are planned legs (motion.transit_waypoints), not the blind joint-space jumps that once swept
an arm through an object; the parked pose, the grip and those trips are in motion.py,
shared with the reorient section byte for byte.

EVERY HEIGHT IS FROM A SURFACE: ``gz`` is the grasp height on the surface the object lies
on (the floor), ``pz`` the grasp height on the surface it is put on (the shelf top) - the
caller adds ``surface + centre height + grip_dz``. The rise off a placed object is derived
from the gripper and the object (motion.rise_clear), not a fixed 60 mm.
"""
import math

from . import collision as _col
from . import ik
# THE SHARED GEOMETRY - see motion.py
from .motion import (TRAVEL_LEFT, TRAVEL_RIGHT, HOVER_DZ, GRIP_DZ, END_CLEAR,  # noqa: F401
                     PARK_SAMPLES, grip_half_separation, left_sign, parked_pair,
                     transit_waypoints, rise_clear)

# ---------------------------------------------------------------- stations
# CARRY HEIGHT IS THE ARM'S, NOT A SURFACE'S: it only has to clear what the carried object
# sweeps over - objects lying on the floor and objects already on the shelf (tops at 0.19 on
# the 0.10 shelf) - and it is checked against both by the suite.
LIFT_Z = 0.34            # object centre height while carried
PLACE_DROP = 0.016       # released this far above resting height, see settle_waypoints
MAX_STEP = 0.025         # m between consecutive waypoints, see E7
YAW_STEP = math.radians(4.0)   # rad between waypoints while yaw-rotating
# (HOVER_DZ, GRIP_DZ and END_CLEAR are in motion.py, shared with the reorient section.)

# THE PARKED POSE (TRAVEL_LEFT / TRAVEL_RIGHT), grip_half_separation() and left_sign() are in
# motion.py - one robot, one parked pose, one rule for which hand takes which end (E19).


def _stations(cx, cy, cz, psi, s, sign_left):
    """The two hand poses gripping the ends of an object at (cx, cy, cz), axis ``psi``.

    ``cz`` is the GRASP height, already grip_dz above the object's axis - the caller
    adds that, because only the caller knows the object's resting height.

    Returns ``(left, right)`` with the assignment fixed by ``sign_left``. There is no
    sorting here by design; see E19.
    """
    R = ik.tool_down_fingers_along(psi)
    ax = (math.cos(psi), math.sin(psi), 0.0)
    left = ([cx + sign_left * s * ax[0], cy + sign_left * s * ax[1], cz], R)
    right = ([cx - sign_left * s * ax[0], cy - sign_left * s * ax[1], cz], R)
    return (left, right)


def _n_for(start, end):
    """Sample count for a straight leg, from the larger of the two hands' travel."""
    d = max(math.dist(start[0][0], end[0][0]), math.dist(start[1][0], end[1][0]))
    return max(2, int(math.ceil(d / MAX_STEP)))


def _leg(start, end, n=None):
    """Interpolate between two station pairs, EXCLUDING the start.

    Excluding the start is what keeps consecutive phases continuous: each phase begins
    at the pose the arm is already holding, so there is no jump at a seam.
    """
    if n is None:
        n = _n_for(start, end)
    out = []
    for i in range(1, n + 1):
        f = i / n
        pair = []
        for (p0, _), (p1, r1) in zip(start, end):
            pair.append(([p0[k] + f * (p1[k] - p0[k]) for k in range(3)], r1))
        out.append((pair[0], pair[1]))
    return out


# ---------------------------------------------------------------- the phases
def approach_waypoints(cx, cy, gz, psi, s, sign_left, hover_dz=HOVER_DZ):
    """Hover above the lying object, then descend onto its two ends."""
    hover = _stations(cx, cy, gz + hover_dz, psi, s, sign_left)
    grasp = _stations(cx, cy, gz, psi, s, sign_left)
    return [hover] + _leg(hover, grasp)


def lift_waypoints(cx, cy, gz, psi, s, sign_left, lift_z=LIFT_Z):
    """Straight vertical lift to carry height, starting FROM the grasp pose."""
    grasp = _stations(cx, cy, gz, psi, s, sign_left)
    carry = _stations(cx, cy, lift_z + GRIP_DZ, psi, s, sign_left)
    return _leg(grasp, carry)


def yaw_waypoints(cx, cy, psi_from, psi_to, s, sign_left, lift_z=LIFT_Z):
    """THE YAW ALIGNMENT: both arms rotate the held object about the vertical.

    This is the phase the shelf makes necessary and the wrist makes possible. The
    cradle chamfers run tangentially, so an object picked up at any other angle has to
    be turned before it can be lowered in. Both hands orbit the object's centre while
    their tool frames rotate with it, which is a genuinely two-armed motion: neither
    arm can do it alone without the object slipping.

    Sampled by ANGLE so the step size is bounded however large the rotation is.
    """
    sweep = abs(psi_to - psi_from)
    n = max(2, int(math.ceil(sweep / YAW_STEP)))
    out = []
    for i in range(1, n + 1):
        psi = psi_from + (psi_to - psi_from) * i / n
        out.append(_stations(cx, cy, lift_z + GRIP_DZ, psi, s, sign_left))
    return out


def place_waypoints(cx, cy, pz, psi, s, sign_left, lift_z=LIFT_Z, drop=PLACE_DROP):
    """Lower from carry height to ``drop`` above the resting height.

    Stops short on purpose: the weld is released there, above the barrier crest, so the
    object is a free body before it can touch a chamfer (E12b).
    """
    carry = _stations(cx, cy, lift_z + GRIP_DZ, psi, s, sign_left)
    high = _stations(cx, cy, pz + drop, psi, s, sign_left)
    return _leg(carry, high)


def settle_waypoints(cx, cy, pz, psi, s, sign_left, drop=PLACE_DROP):
    """The last few millimetres, AFTER the weld is released, fingers still closed.

    A chamfer can only steer a body that is free to move. While the weld is active the
    object is rigidly part of the arm and a sideways push from the chamfer just fights
    the position controller - which is the tipping failure. Here the object is held by
    friction alone, so it can slide into the slot instead of being forced.
    """
    high = _stations(cx, cy, pz + drop, psi, s, sign_left)
    rest = _stations(cx, cy, pz, psi, s, sign_left)
    return _leg(high, rest)


def retreat_waypoints(cx, cy, pz, psi, s, sign_left, half_width=0.045, grip_dz=GRIP_DZ):
    """Lift the released hands straight up off the placed object, FROM the release pose.

    Straight up is the only direction that is safe whatever is next to the object: the
    fingers straddle it across the closing axis and the shelf's barriers sit 11 mm outside
    it. It goes until the fingertips are TIP_CLEAR above the object's top - motion.rise_clear,
    derived from the gripper and the object. It used to be a fixed 60 mm, which left the
    (then newly added) distal tips 18 mm above the object for a trip home that was a blind
    joint-space jump.
    """
    rest = _stations(cx, cy, pz, psi, s, sign_left)
    clear = _stations(cx, cy, pz + rise_clear(half_width, grip_dz), psi, s, sign_left)
    return _leg(rest, clear)


# ---------------------------------------------------------------- the legs
def pick_legs(cx, cy, gz, psi, s, sign_left, hover_dz=HOVER_DZ, lift_z=LIFT_Z):
    """The first half of a cycle, from the PARKED pose to the object held at carry height.

    ``unpark`` starts AT the parked pose and ends at the hover, so the arms can settle there
    before descending - moving from parked straight into the descent let the hands come down
    while still swinging into line.
    """
    reach = approach_waypoints(cx, cy, gz, psi, s, sign_left, hover_dz)
    parked = parked_pair()
    return [('unpark', [parked] + transit_waypoints(parked, reach[0], PARK_SAMPLES)),
            ('approach', reach[1:]),
            ('lift', lift_waypoints(cx, cy, gz, psi, s, sign_left, lift_z))]


def place_legs(from_xy, to_xy, pz, psi_from, psi_to, s, sign_left, lift_z=LIFT_Z,
               drop=PLACE_DROP, half_width=0.045):
    """The second half, from where the lift left the object to the PARKED pose.

    ``carry`` starts exactly where the lift ended (``from_xy``, ``psi_from``) - the base has
    turned since, and the object with it, so in the base frame it has not moved - and
    shifts it over the slot the base now faces (``to_xy``), which differs only by the drift
    the aim corrected. Then the yaw alignment, the set-down, the rise off the object and the
    planned trip home.
    """
    carry_z = lift_z + GRIP_DZ
    start = _stations(from_xy[0], from_xy[1], carry_z, psi_from, s, sign_left)
    over = _stations(to_xy[0], to_xy[1], carry_z, psi_from, s, sign_left)
    cx, cy = to_xy
    rise = retreat_waypoints(cx, cy, pz, psi_to, s, sign_left, half_width)
    return [('carry', [start] + _leg(start, over)),
            ('yaw', yaw_waypoints(cx, cy, psi_from, psi_to, s, sign_left, lift_z)),
            ('lower', place_waypoints(cx, cy, pz, psi_to, s, sign_left, lift_z, drop)),
            ('settle', settle_waypoints(cx, cy, pz, psi_to, s, sign_left, drop)),
            ('rise', rise),
            ('park', transit_waypoints(rise[-1], parked_pair(), PARK_SAMPLES))]


# Legs during which BOTH hands hold the object. Asserted by the suite and the harness.
CARRY_LEGS = ('lift', 'carry', 'yaw', 'lower', 'settle')
# After the hands let go, while they are still next to the object they just put down.
LEAVE_LEGS = ('rise',)
# On the way home: the WHOLE arm must keep off it.
HOME_LEGS = ('park',)
# Where the hands work INTO the shelf slot by design - its pedestal is then an obstacle to
# the ARMS only, as the floor is to nothing but the planner's floor check.
SLOT_LEGS = ('lower', 'settle', 'rise')


def plan_inputs(legs, close, open_pos, release=None, lying=(), placed=(), others=(),
                fixtures=(), slot=()):
    """Per-waypoint finger closure, obstacles and search width for planning a half-cycle.

    The one statement of what the planner must keep clear of, and when - the same rules as
    the reorient section's:

      * ``fixtures`` - walls and shelf pedestals, as boxes: all links, the whole plan, no
        margin (like the floor, a surface is where the hands are meant to go close) -
        except ``slot``, the pedestal being placed on, which during SLOT_LEGS is an
        obstacle to the ARMS only, because the hands put the object into it by design
      * ``others`` - every OTHER known object, lying or already placed: all links
      * the handled object while it LIES (``lying``): all links on the way to the hover
      * the handled object once PLACED (``placed``): the ARMS while the hands rise off it,
        all links on the way home

    The hands are CLOSED while carrying, at ``release`` (just off the object) while rising
    off it, and OPEN otherwise. ``wide`` marks the legs to and from parked, which need the
    second shoulder solution.
    """
    if release is None:
        release = open_pos
    fingers, obs, wide = [], [], []
    base = [(c, None) for c in others] + [(b, None, 0.0) for b in fixtures]
    for name, seq in legs:
        for _ in seq:
            fingers.append(close if name in CARRY_LEGS
                           else release if name in LEAVE_LEGS else open_pos)
            wide.append(name in ('unpark', 'park'))
            lst = list(base)
            lst += [(b, _col.ARM_LINKS if name in SLOT_LEGS else None, 0.0) for b in slot]
            if name == 'unpark':
                lst += [(c, None) for c in lying]
            elif name in LEAVE_LEGS:
                lst += [(c, _col.ARM_LINKS) for c in placed]
            elif name in HOME_LEGS:
                lst += [(c, None) for c in placed]
            obs.append(lst)
    return fingers, obs, wide


def cycle_halves(reach, cz, place_z, psi_pick, psi_place, length, end_clear=END_CLEAR,
                 lift_z=LIFT_Z, drop=PLACE_DROP, half_width=0.045):
    """Both halves of one cycle as legs, with the object at the nominal (reach, 0) - exactly
    what the node builds when the aim needs no correction. ``cz`` and ``place_z`` are the
    object's centre heights lying and placed, ABSOLUTE (surface + centre height)."""
    s = grip_half_separation(length, end_clear)
    sl = left_sign(psi_pick)
    pick = pick_legs(reach, 0.0, cz + GRIP_DZ, psi_pick, s, sl, HOVER_DZ, lift_z)
    place = place_legs((reach, 0.0), (reach, 0.0), place_z + GRIP_DZ, psi_pick, psi_place,
                       s, sl, lift_z, drop, half_width)
    return pick, place


def full_cycle(reach, cz, place_z, psi_pick, psi_place, length,
               end_clear=END_CLEAR, lift_z=LIFT_Z, drop=PLACE_DROP):
    """Every waypoint of one cycle, parked to parked, flattened - for offline validation.

    Mirrors the task node exactly, including the single ``sign_left`` pinned from the
    pick yaw and reused by every later phase.
    """
    pick, place = cycle_halves(reach, cz, place_z, psi_pick, psi_place, length, end_clear,
                               lift_z, drop)
    return [w for _, leg in pick + place for w in leg]


def max_step(pairs):
    """Largest distance either hand moves between consecutive waypoints."""
    worst = 0.0
    for (a_l, a_r), (b_l, b_r) in zip(pairs, pairs[1:]):
        for (pa, _), (pb, _) in ((a_l, b_l), (a_r, b_r)):
            worst = max(worst, math.dist(pa, pb))
    return worst


def is_continuous(pairs, max_jump=1.5 * MAX_STEP):
    """True if no hand jumps more than ``max_jump`` between consecutive waypoints.

    This is NOT a test that consecutive poses are equal - the arm is moving, so of
    course they differ. It is a test that the motion is broken into steps the controller
    can follow. A jump is invisible to a reachability check, because both ends are
    perfectly reachable, and shows up in simulation as a lurch. Phase SEAMS are the
    usual offenders: every phase here begins at the pose the arm already holds, so a
    seam contributes no step at all.
    """
    return max_step(pairs) <= max_jump
