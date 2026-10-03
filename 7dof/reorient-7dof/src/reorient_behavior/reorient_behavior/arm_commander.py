"""Turn (position, rotation) waypoint pairs into JointTrajectory messages.

WHY VELOCITIES ARE SET HERE
---------------------------
``joint_trajectory_controller`` chooses its interpolation from what a point
carries. Positions only, and it interpolates **linearly** between waypoints: speed
is constant within a segment and changes instantaneously at every waypoint, then
drops to zero at the end. That is a velocity step at every point, and a whole
trajectory's worth of them is the jerking seen at pickup and while placing.

Give each point a velocity as well and the controller fits a **cubic spline**,
which is continuous in velocity through the interior and comes to rest smoothly at
the ends. So every trajectory here carries:

  * zero velocity at the first and last point (start and finish at rest), and
  * a central-difference estimate in between.

The estimate is deliberately simple. It is exact for a constant-speed straight leg,
which is what most of these paths are, and any small mismatch is absorbed by the
controller's own tracking rather than fed forward as a discontinuity.
"""
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from . import ik

# SEVEN joints per arm. j2b is the shoulder ROLL, the seventh axis, and it sits between
# j2 and j3 - so the list is NOT j1..j7 but j1, j2, j2b, j3..j6, matching the order
# ik.solve() returns and the order arm.xacro builds the chain. Naming it j2b rather than
# renumbering keeps every existing joint name, log line and recorded pose valid.
ARM_JOINT_SUFFIX = ik.ARM_JOINT_SUFFIX          # defined in ik.py - see the note there
LEFT_JOINTS = ['left_%s' % j for j in ARM_JOINT_SUFFIX]
RIGHT_JOINTS = ['right_%s' % j for j in ARM_JOINT_SUFFIX]
FINGER_JOINTS = ['left_finger_l_joint', 'left_finger_r_joint',
                 'right_finger_l_joint', 'right_finger_r_joint']

# DISTAL PHALANGES, one per finger, in the SAME ORDER as FINGER_JOINTS. Kept as a
# separate list so the contact and stall logic, which slices FINGER_JOINTS [:2] and
# [2:] per hand, is untouched. The gripper controller takes all eight in one array:
# the four closures first, then the four curls.
TIP_JOINTS = [j.replace('_joint', '_tip_joint') for j in FINGER_JOINTS]

# How far the distal segment curls when the hand is shut (rad). A real build would drive
# both segments from one tendon and let a return spring set this; here it is commanded in
# step with the closure - see _curl_for() in task_node.py.
# WAS 0.80, AND THAT WAS A DESIGN ERROR: the tip reach was bounded
# from below (it must get under the equator) but never from above. At 0.80 the tips sat
# 5.9 mm INSIDE the object's surface, so they pushed it outward until it lifted clear of
# the flat pads and balanced on the two flaps alone - narrow line contacts, no pad contact,
# and nothing holding it once it began to tilt. Observed in Gazebo before the arithmetic
# caught it. 0.709 was the next attempt; 0.637 (36.5 deg) is the value that shipped and
# stood both objects up: the tip bites 2.0 mm and is still 32.7 mm below the equator, so it
# wraps AND the pad keeps its contact.
CURL_CLOSED = 0.637

# The closure at which the curl is complete. MUST match `close` in objects.yaml.
CLOSE_REF = 0.048

# SUCTION CUP RATING. One 30 mm bellows cup per finger, four in all, at -60 kPa:
#     F = P . A = 60000 x pi(0.015)^2 = 42.4 N per cup
# Only the two cups on the HOLDING hand are counted, because the supporting hand is
# deliberately weak and may slide. Against a 4.41 N object that is a large margin, and
# unlike a pinch it does not depend on where the cup lands or on pad geometry.
SUCTION_KPA = 60.0
CUP_RADIUS = 0.015
CUPS_PER_HAND = 2


def _duration(t):
    sec = int(t)
    return Duration(sec=sec, nanosec=int((t - sec) * 1e9))


def _velocities(q_list, dt):
    """Central-difference joint velocities, zero at both ends.

    Zero endpoints matter: a trajectory that ends with non-zero velocity leaves the
    controller ramping into whatever comes next, which is how a smooth-looking path
    still produces a lurch at a phase boundary.
    """
    n = len(q_list)
    if n == 1:
        return [[0.0] * len(q_list[0])]
    out = []
    for i, q in enumerate(q_list):
        if i == 0 or i == n - 1:
            out.append([0.0] * len(q))
        else:
            prev, nxt = q_list[i - 1], q_list[i + 1]
            out.append([(nxt[k] - prev[k]) / (2.0 * dt) for k in range(len(q))])
    return out


def joint_traj(names, q_list, dt, t0=1.0, with_velocities=True):
    """Build a JointTrajectory, splining through the waypoints by default."""
    msg = JointTrajectory()
    msg.joint_names = list(names)
    vels = _velocities(q_list, dt) if with_velocities else None
    t = t0
    for i, q in enumerate(q_list):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        if vels is not None:
            pt.velocities = [float(v) for v in vels[i]]
        pt.time_from_start = _duration(t)
        msg.points.append(pt)
        t += dt
    return msg


def pair_trajs(waypoint_pairs, dt, t0=1.0, seed_l=None, seed_r=None,
               finger_pos=0.0, **home):
    """waypoint_pairs: [((p_left, R_left), (p_right, R_right)), ...]

    Solves both arms and returns time-matched trajectories. Raises ik.IKError if
    ANY waypoint is unreachable, so a partially valid motion is never sent - a
    half-executed flip would drop the object.

    PASS THE SEEDS. They are where the arm ALREADY IS - the last joint vector commanded by
    the previous leg of the cycle. Without them the planner is continuous only WITHIN one
    list, and a cycle sends about nine of them, so at every phase boundary it is free to
    pick the other wrist branch: same hand pose, (q4+pi, -q5, q6+pi), 180 deg of wrist.
    Observed in Gazebo as `right_j4 off 1.531` - 87.7 deg of error - with the right hand
    spinning about the wrist and sweeping through the bottle it was holding, which then
    dragged the object out of the grip and tripped the slip stop at 15 mm.
    """
    ql, qr = solve_pairs(waypoint_pairs, seed_l, seed_r, finger_pos, **home)
    return (joint_traj(LEFT_JOINTS, ql, dt, t0),
            joint_traj(RIGHT_JOINTS, qr, dt, t0))


def solve_pairs(waypoint_pairs, seed_l=None, seed_r=None, finger_pos=0.0, **home):
    """Just the joint solutions, for offline checking without building messages.

    THE ELBOW SWIVEL IS RESOLVED HERE, once, for the whole list - and seeded from where
    the arm already is, so continuity holds ACROSS legs of the cycle and not merely within
    one. See the note in pair_trajs(). Each swivel is seeded from the previous waypoint's, so the redundancy is
    spent on staying continuous rather than on chasing a clearance optimum that hops
    around the circle. Solving waypoints independently would reintroduce exactly the
    joint snap the seventh axis was added to remove.

    ``home`` passes the remaining collision.solve_path options through - the obstacles,
    the per-waypoint finger closure and search width, and the pins that tie both ends of a
    cycle to the parked pose.
    """
    from . import collision as _col
    return _col.solve_path(waypoint_pairs, finger_pos=finger_pos,
                           seed_l=seed_l, seed_r=seed_r, **home)


def final_positions(traj):
    return dict(zip(traj.joint_names, traj.points[-1].positions))


def total_time(traj):
    tf = traj.points[-1].time_from_start
    return tf.sec + tf.nanosec * 1e-9

def hold(joints, current, dt=0.25):
    """A one-point trajectory commanding the joints to stay exactly where they are.

    Used by the protective stop. Simply ceasing to publish does NOT stop the arm - the
    controller keeps driving toward the last goal it was given, which is the goal that
    was pressing on something. This replaces that goal with the present position, so the
    arm holds instead of continuing to push.
    """
    return joint_traj(joints, [[float(current.get(j, 0.0)) for j in joints]],
                      dt, t0=0.0)
