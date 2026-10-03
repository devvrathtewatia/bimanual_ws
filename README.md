Two Hands, Seven Joints

A small robot that learns to stand things up.

Somewhere in a quiet room, a bottle lies on its side. It has been lying there a while, the way unfinished things do. A small wheeled robot turns toward it, lifts its two arms, takes the bottle gently by both ends, and turns the world a quarter-turn, until the bottle stands.

That is the whole project. Everything else in this repository is the long, honest story of how that one quiet motion became possible.

What is this?

This is a bimanual mobile manipulator, which is a long phrase for a simple idea:

word

what it means here

mobile

it drives around on a Kobuki wheeled base

bimanual

it has two arms and uses them together, like you do when you carry something awkward

manipulator

the arms and hands that pick things up and move them

The robot's job: find everyday objects lying on their sides in a room, drive to each one, pick it up with both hands, rotate it in mid-air, and stand it upright.

It runs in simulation on ROS 2 Jazzy + Gazebo Harmonic. Everything here works in the simulator. The mass, torque and cost numbers describe a real build that has not been built yet.

The idea in one picture

   LOOK            DRIVE            GRIP             TURN            STAND

    ◉ camera        ┌─┐            ┌─┐ ┌─┐          ┌─┐ ┌─┐            ┌─┐
   ─┴─              │R│ ──►        │ ╲ ╱ │           ╲ │ │ ╱            │ │
  [robot]  ▬▬▬      └─┘   ▬▬▬     [robot]▬▬▬         [robot]  ▌          [robot]  ▌
  object lies      goes to it    both hands       90° in mid-air      set down, 
  on its side                    hold both ends                       standing up

Why two arms, and why seven joints?

This is the heart of the project. If you only read one section, read this one.

The first robot couldn't do it

The first design gave each arm 4 degrees of freedom (DOF). A "degree of freedom" is one way a joint can move: one hinge, one twist. Four was enough to put a hand anywhere in reach. Position was never the problem.

Orientation was. To hold an object with two hands while turning it, each hand must rotate with the object, not only travel with it. A 4-DOF wrist has no spare axis to rotate with. The flip was not hard. It was geometrically impossible, and no amount of tuning could change that.

So the team stopped tuning and proved it. Out of 40 flip configurations:

  6 joints solve   40 / 40
  5 joints solve    4 / 40
  4 joints solve    0 / 40

Six joints is the minimum to place and orient something freely in 3D, which is why almost every industrial robot arm has six.

Then the six hit a wall of their own

With six joints the robot could flip an object, but not cleanly. At the top of the flip, the object stands vertical and its two ends, and the two hands gripping them, line up in one vertical column. During the full 6-DOF flip, some required configurations caused arm-against-arm self-collision, with the worst clearance around −20 mm.

The reason is subtle and a little beautiful. For any hand position, the elbow is free to sit anywhere on a circle. A 6-DOF arm can pick only two points on that circle. Each of the two points fails somewhere in the motion: one collides with the other arm, the other dives into the base.

The seventh joint gives back the whole circle

Adding a shoulder roll (called j2b) lets the elbow swing continuously around that circle. Then the planner can choose, moment by moment, where the elbow should be to stay clear.



6-DOF

7-DOF

arm self-collision

−19.6 mm (touching)

+2.0 mm (clear)

arm-vs-arm clearance at the flip

+5.6 mm

+38.7 mm

the flip

needs an awkward handover

fully two-handed, start to finish

The result is verified by removal: pin the new joint and the cycle becomes infeasible. Restore the old thicker forearm and the hand presses into it. Both changes are necessary, and the tests prove it.

📖 The full narrative, with every dead end, is in final_proj/project_story/README.md.

The four chapters

The project is split into four sections. Each one is a complete, standalone ROS workspace with its own world, its own README and its own offline tests. They are separate on purpose: when something breaks, it should have exactly one possible home. A camera bug must not be mistaken for a navigation bug.

Run them in this order. Each chapter inherits its goal from the one before.

#

Folder

The question it answers

What it adds

A

7dof/pick-place-7dof

Can both hands lift an object and set it on a shelf?

grasping, carrying, placing on a 0.10 m shelf

B

7dof/reorient-7dof

Can both hands stand an object upright?

the 90° two-handed flip, the heart of the project

C

7dof/perceive-7dof

Can it find the object and decide what to do?

cameras: where is it, is it lying or standing?

D

7dof/navigate-7dof

Can it do the whole job in a room?

scanning, route planning, driving, parking, finishing

Section D is the finished system. It scans a 5 m × 5 m room, ignores the object already standing, drives to each lying one, stands it up, backs away, and scans again until all three stand.

One robot, four behaviours

The sections are not four different programs. They are one robot. The inverse kinematics, the path planner, the collision checker, the wall-based localisation and every task primitive live in files that are byte-for-byte identical across all four folders. A checksum manifest (SHARED_FILES.sha256) lives in each, and every test suite fails if a copy drifts.

Why so strict? Because the project learned this the hard way: two copies of working code are one working copy and one that is quietly going stale.

 pick-place ──┐
 reorient   ──┼── ik.py · collision.py · arm_commander.py · wall_ref.py · motion.py · node_base.py
 perceive   ──┤        (the shared body: shared by all four)
 navigate   ──┘
                  reorient + perceive + navigate  →  flip_task.py  (the stand-up itself)
                  perceive + navigate             →  vision.py · detect.py · looking.py  (the eyes)
                  navigate only                   →  room_ref.py · navigation.py  (the legs)

Quick start

You need Ubuntu with ROS 2 Jazzy and Gazebo Harmonic.

Each section works the same way. Here is Section B as the example:

cd 7dof/reorient-7dof

# 1. Offline tests first. No ROS and no simulator needed. Takes seconds.
cd src/reorient_behavior && python3 test/test_flip.py && python3 test/harness.py && cd -

# 2. Build.  --symlink-install is not optional: without it you debug a stale copy.
source /opt/ros/jazzy/setup.bash
rm -rf build install log
colcon build --symlink-install
source install/setup.bash

# 3. Run.
QT_QPA_PLATFORM=xcb ros2 launch reorient_gazebo reorient.launch.py

Launch files for the other sections:

section

launch command

pick-place

ros2 launch pnp_gazebo pnp.launch.py

reorient

ros2 launch reorient_gazebo reorient.launch.py

perceive

ros2 launch perceive_gazebo perceive.launch.py

navigate

ros2 launch navigate_gazebo navigate.launch.py

Success is a number, not a video. Each run ends with a table scored against Gazebo's ground truth:

bottle  PASS  tilt 0.0 deg from vertical, drift 0.003 m, height 0.120 m
box     PASS  tilt 0.0 deg from vertical, drift 0.014 m, height 0.110 m
RESULT: 2/2 objects standing upright

The "harness": a robot that runs without a simulator

Inside each section's test/ folder is a harness.py. It replaces ROS with a stand-in, then runs the real task node against a fake robot that drives, scans and carries objects. It checks what the code does, not just what the arithmetic says it computes. It found bugs that every geometry test had missed.

The robot in numbers





Arms

2, each with 7 DOF (14 actuators in total)

Base

Kobuki, about 2.35 kg

Hands

two-phalanx fingers that wrap objects like a basket, plus 4 suction cups each

Senses

a 2D LiDAR (for walls and position) and two cameras (a mast camera and a workspace camera)

One arm

1.53 kg

Everything mounted on the base

4.55 kg

Kobuki's rated payload

5.00 kg

What is left for the object

0.45 kg at most, 0.35 kg recommended

The hardest joint

the shoulder, so a counterbalance spring is required

Forearm

20 mm metal (a thin plastic one would flex and clash)

Planning power draw

about 84 W

Estimated hardware cost (India)

about ₹1.1 to ₹1.6 lakh, excluding the Kobuki and the computer

Next hardware step

The current design leaves about 0.45 kg of payload on the Kobuki. The next hardware
iteration will focus on reducing the mass of the arms, mounting plate and sensor structure.
A target of about 1 kg useful payload is a future design goal, not a demonstrated result
of the current robot.

The robot never drives while holding a heavy object. At full forward extension with a 0.50 kg load, the calculated static margin is only about 3 mm, so this is a marginal pose and should be avoided. The normal stand-up stays much closer in.

📖 Mass, payload, torque, tipping and a cost breakdown: final_proj/mass_budget/MASS_BUDGET_7DOF.md

Check counts (last recorded in each section)

section

offline checks

headless harness

pick-place-7dof

77

44 / 44

reorient-7dof

56

passes on the floor and on 0.10 m tables

perceive-7dof

26

38 / 38

navigate-7dof

42

166 checks across three rooms

What this project learned

Almost every failure here was a margin that was never checked, or a check that could not fail. Four lessons, in four lines:

Check the margin, not just the sign. 0.98×, 4 mm, 3 mm: nearly every failure was a quantity that was positive but too small, and nobody divided.

A check that cannot fail is worse than no check. It manufactures confidence. Break every guard on purpose to see that it fires.

Derive constants, never restate them. If two files each hold the same number, one of them will eventually be wrong. Read it from the file that builds the robot.

Run the code, not just the arithmetic. The harness caught four bugs that every geometry test waved through.

Where to read next

if you want to...

read

hear the whole story, dead ends included

final_proj/project_story/README.md

see exactly how the 7th joint was designed

final_proj/reorientation/DESIGN_7DOF.md

see the original 6-DOF design

final_proj/reorientation/DESIGN.md

learn how "does this configuration solve?" is proven offline

final_proj/pick_place/HOW_CONFIG_SOLVING_WORKS.txt

check mass, torque, tipping and cost

final_proj/mass_budget/MASS_BUDGET_7DOF.md

work on one section

the README.md inside that section's folder

Repository map

bimanual_ws/
├── 7dof/                      the four runnable sections (each a colcon workspace)
│   ├── pick-place-7dof/       A: pick it up, set it on a shelf
│   ├── reorient-7dof/         B: the two-handed flip to upright
│   ├── perceive-7dof/         C: find objects with cameras and decide
│   └── navigate-7dof/         D: the whole job, in a room
│
├── final_proj/                the thinking behind the code
│   ├── mass_budget/           what it weighs, what it can carry, what it costs
│   ├── reorientation/         design notes for the 6-DOF and 7-DOF flip
│   ├── pick_place/            how configurations are proven solvable offline
│   └── project_story/         the full narrative
│
├── README.md                  you are here
└── .gitignore

Honest limits

Simulation only. Real wheel slip, real LiDAR noise, camera lighting and the contact physics of a real grasp are not modelled offline. Gazebo covers some of that; hardware covers the rest.

Object size. The hands handle objects about 90 mm across. A paperback, a tissue box or a cereal box is too wide.

Short objects. Below roughly 200 mm, two arms on one small base cannot stack tightly enough for a fully two-handed flip.

Payload. 0.35 kg is the practical working limit; 0.45 kg is the absolute rated limit.

Cheap actuators are unproven. The low-cost serial-bus servos in the budget must be tested for continuous torque and heat before anyone calls them final hardware.

Costs are planning values. Recheck prices before buying anything.

The robot has two hands, fourteen joints and one quiet job. It is not finished the way a product is finished. It is finished the way a proof is: every margin measured, every guess either confirmed or thrown out.

Somewhere in the simulated room, the bottle stands