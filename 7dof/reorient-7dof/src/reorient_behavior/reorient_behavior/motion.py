"""Motion geometry SHARED by every section: the parked pose, the planned trips to and from it,
and the grip geometry both hands are placed by.

SHARED FILE. It is byte-for-byte the same in reorient_behavior and pnp_behavior, and the
suite in each section fails if it is not (see SHARED_FILES.sha256 at the section root). The
two sections are one robot; the parked pose, the grip and the way the arms leave and return
must not be two similar things that drift apart. Edit it in one section, copy it to the
other, and regenerate the manifest in both.

HEIGHTS ARE MEASURED FROM THE SURFACE an object lies on, not from the floor, wherever that
matters: the floor is simply the surface at 0.
"""
import math

from . import collision as _col
from . import ik

# TRAVEL POSE - the arms' resting pose, and the one they pass through between objects.
#
# THIS POSE USED TO PUT THE TWO PALM PLATES 45.7 mm INSIDE EACH OTHER. The hands sat
# only 64 mm apart while each palm plate is 140 mm across, so the arms visibly
# interpenetrated from the moment the model loaded until they reached the first grasp.
# The replacement holds the hands at (0.15, +/-0.07, 0.37) with the palms edge-on to
# each other: +92.8 mm of clearance standing still. Still inside the base circle (radius
# 0.166 m against 0.1775 m) and still with each hand on its own side.
# j2b, the shoulder roll, is inserted at index 2 AT ZERO. Zero roll is the old planar
# arm exactly, so the parked pose is geometrically unchanged - which matters because
# these values are also the URDF initial_value for every joint, and the arms loading
# anywhere other than parked once threw an object 3.15 m across the world.
TRAVEL_LEFT = [-0.6202, -1.9871, 0.0000, 1.6451, 0.0000, 1.9129, -2.1910]
TRAVEL_RIGHT = [0.6202, -1.9871, 0.0000, 1.6451, 0.0000, 1.9129, -0.9505]

HOVER_DZ = 0.10          # above the object before descending
GRIP_DZ = 0.025          # grasp this far off the object's AXIS, so the palm clears its top
END_CLEAR = 0.035        # object left beyond each hand, along its axis

# The fingertip, open and straight, measured past the grasp point along the approach axis.
TIP_PAST_GRASP = _col.FINGERTIP_X + _col.DISTAL_L - ik.LG

# How far a hand's fingertips end from an object it has just let go of, before it moves
# anywhere else. 60 mm is the clearance the retreat guard has always required of a
# fingertip near a placed object.
TIP_CLEAR = 0.060

# The finger's half-thickness: the pad face sits this far inside the commanded finger
# position. Mirrors the pad box in arm.xacro (checked by the suite).
PAD_HALF = 0.006

# The two hands' plates reach this far above and below their grasp points, and one hand must
# never come within HAND_GAP of the other while both slide along an object.
HAND_HALF_H = 0.020
HAND_GAP = 0.030


def rot_about(axis, theta):
    """Rotation matrix about an arbitrary unit axis (Rodrigues)."""
    x, y, z = axis
    c, s = math.cos(theta), math.sin(theta)
    t = 1.0 - c
    return [[t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]


def grip_half_separation(length, end_clear=END_CLEAR):
    """Half the distance between the hands, derived from the object's length.

    ``end_clear`` must exceed the palm's 0.020 m half-extent along the axis or the palm
    hangs off the end; at 0.035 there is 15 mm of object beyond each palm, and the hands
    sit about 15% in from the ends - a long enough moment arm that the object cannot
    pivot in the grip.
    """
    return length / 2.0 - end_clear


def left_sign(axis_angle):
    """Which end of the object the LEFT arm grips, decided ONCE from the pick yaw.

    ``+1`` means the end at ``centre + s * axis``. Chosen so that at the start of the
    motion the left arm takes the end with the larger y, i.e. each arm begins on its own
    side of the base centreline. Then carried unchanged through every phase: re-deciding
    by y mid-motion flips when the two ends pass through equal y, and the arms trade ends
    in a single waypoint - a 170 mm instantaneous jump (E19).
    """
    return 1.0 if math.sin(axis_angle) >= 0.0 else -1.0


def rise_clear(half_width, grip_dz=GRIP_DZ):
    """How far OPEN hands rise off the top of a LYING object before going anywhere else.

    Up to where the fingertips are TIP_CLEAR above its top. Along the (vertical) approach
    axis the top is ``half_width - grip_dz`` ABOVE the grasp point and the open tips reach
    TIP_PAST_GRASP BELOW it, so the tips have both of those to climb before the clearance
    even starts. Derived, so a different object or finger changes it. (The first version
    subtracted the first term, leaving the tips 20 mm over the object instead of 60; the
    suite's test of where the tips end caught it.)
    """
    return TIP_PAST_GRASP + (half_width - grip_dz) + TIP_CLEAR


def release_closure(half_width, gap=0.004):
    """A finger command that stands each pad ``gap`` off an object's surface.

    Letting go by opening FULLY swings the fingertips out 20 mm on each side, into whatever
    is next to the object - a shelf's anti-roll barrier sits 11 mm outside a placed object.
    Opening just clear of the object lets the hands leave it without touching anything else.
    """
    return half_width + PAD_HALF + gap


# ---------------------------------------------------------------------------
# TO AND FROM THE PARKED POSE - PLANNED, NOT JUMPED.
#
# The parked pose used to be reached by ONE joint-space move from wherever the cycle
# ended, and left the same way at the start. A straight line in joint angles knows nothing
# about the world: it swept an arm through the bottle just stood up. Now both trips are
# ordinary legs - hand position interpolated in a straight line, hand orientation by slerp -
# that the path planner solves and collision-checks like everything else, pinned to the
# EXACT parked joint vector at their ends so the arm cannot finish in some other elbow or
# wrist configuration and snap to it.
# ---------------------------------------------------------------------------
PARK_SAMPLES = 8


def parked_pair():
    """Both hands' poses at the parked joint vectors."""
    return (tuple(ik.fk(TRAVEL_LEFT, +1)), tuple(ik.fk(TRAVEL_RIGHT, -1)))


def slerp(R0, R1, f):
    """Rotation a fraction ``f`` of the way from R0 to R1, about the single axis between them."""
    rel = ik.mat_mul(ik.transpose(R0), R1)
    c = max(-1.0, min(1.0, (rel[0][0] + rel[1][1] + rel[2][2] - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return [row[:] for row in R1]
    if math.pi - ang < 1e-6:
        d = [(rel[i][i] + 1.0) / 2.0 for i in range(3)]
        i = max(range(3), key=lambda k: d[k])
        ax = [0.0, 0.0, 0.0]
        ax[i] = math.sqrt(max(0.0, d[i]))
        for j in range(3):
            if j != i:
                ax[j] = (rel[i][j] + rel[j][i]) / (4.0 * ax[i])
    else:
        s2 = 2.0 * math.sin(ang)
        ax = [(rel[2][1] - rel[1][2]) / s2, (rel[0][2] - rel[2][0]) / s2,
              (rel[1][0] - rel[0][1]) / s2]
    nrm = math.sqrt(sum(v * v for v in ax))
    ax = [v / nrm for v in ax]
    return ik.mat_mul(R0, rot_about(ax, f * ang))


def transit_waypoints(start, end, n=PARK_SAMPLES):
    """Both hands from ``start`` to ``end``: straight in position, slerped in orientation.

    Excludes the start pose, like every other leg.
    """
    out = []
    for i in range(1, n + 1):
        f = i / float(n)
        out.append(tuple(([p0[k] + f * (p1[k] - p0[k]) for k in range(3)], slerp(R0, R1, f))
                         for (p0, R0), (p1, R1) in zip(start, end)))
    return out


def fixture_obstacles(fixtures, to_base, heading):
    """World-frame fixture boxes as base-frame planner obstacles.

    ``fixtures`` is ``{name: (centre_xyz, yaw_rad, size_xyz)}`` in the WORLD; ``to_base`` maps
    a world (x, y) to the base frame and ``heading`` is the base's yaw in the world.
    """
    out = {}
    for name, (c, yaw, size) in fixtures.items():
        bx, by = to_base(c[0], c[1])
        out[name] = _col.box_obstacle((bx, by, c[2]), yaw - heading, size)
    return out
