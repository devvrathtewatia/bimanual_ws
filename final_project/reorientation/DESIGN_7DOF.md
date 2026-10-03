# Adding the Seventh Joint

> **In one sentence:** a six-joint arm could flip an object but could not do it *cleanly*, so we added a seventh joint at the shoulder, and the two arms can now take an object from lying down to standing up without ever letting go.

**Status note.** The numbers in this document were recorded by the offline test suites (no simulator needed) between 24 and 28 September. The finished system has since been confirmed working in simulation. For the earlier 6-DOF gripper story, see [`DESIGN.md`](DESIGN.md).

---

## Contents

1. [The problem: two arms, one vertical line](#1-the-problem-two-arms-one-vertical-line)
2. [The idea: let the elbow swing](#2-the-idea-let-the-elbow-swing)
3. [Where the joint goes](#3-where-the-joint-goes)
4. [Keeping the maths exact](#4-keeping-the-maths-exact)
5. [Choosing the elbow angle: four attempts](#5-choosing-the-elbow-angle-four-attempts)
6. [Proof that both changes were needed](#6-proof-that-both-changes-were-needed)
7. [The first Gazebo run: three faults](#7-the-first-gazebo-run-three-faults)
8. [The grip offset: a 10 mm mistake](#8-the-grip-offset-a-10-mm-mistake)
9. [Two-handed, floor to floor](#9-two-handed-floor-to-floor)
10. [The retreat that knocked the bottle over](#10-the-retreat-that-knocked-the-bottle-over)
11. [Heights from the surface](#11-heights-from-the-surface)
12. [One robot, shared code](#12-one-robot-shared-code)
13. [Numbers at a glance](#13-numbers-at-a-glance)
14. [Still open](#14-still-open)
15. [Appendix: files that changed](#15-appendix-files-that-changed)

---

## Words you will meet

| word | plain meaning |
|---|---|
| **DOF** | one independent way a joint can move |
| **IK** | "I want the hand there; what should each joint be?" |
| **Swivel angle (ψ, "psi")** | where the elbow sits on its circle of possible positions |
| **Waypoint** | one step in a planned motion |
| **Clearance** | the gap between two bodies. Positive = clear, negative = touching |
| **Harness** | a test that runs the *real* task node against a fake robot |
| **Bimanual** | using both arms together |

---

## 1. The problem: two arms, one vertical line

To stand a lying object up, the robot rotates it 90° while holding both ends. At the *end* of that rotation the object is vertical, so its two ends, and **the two hands holding them, lie on one vertical line**, 170 mm apart.

Both arms must then reach the same spot on the floor, one hand above the other. With the shoulders bolted only 240 mm apart on the same small base, **the arms collide.**

We measured it with a careful collision model (every flat link modelled as a flat box, not a fat round capsule):

| | 6 DOF | 7 DOF (elbow swivel) |
|---|---|---|
| bottle, worst clearance | **−19.6 mm** (touching), 23 of 94 arm-poses intersecting | 47 waypoints, **0 blocked** |
| box, worst clearance | **−19.7 mm**, 21 of 94 intersecting | 47 waypoints, **0 blocked** |
| margin at the worst point | none | **50°** of elbow angle to spare |

Five constraints were checked at once: arm-arm clearance above 8 mm, floor, base, wrist pitch within 140°, and self-collision.

---

## 2. The idea: let the elbow swing

### Try this

Hold your hand still, out in front of you, palm down. Now lift your elbow up and out to the side, without moving your hand at all. You can. Your elbow is free to sit anywhere on a **circle** while your hand stays put.

### Why six joints can't use it

A 6-DOF arm has only **two** elbow positions for any hand pose: elbow up, or elbow down. Both fail somewhere in the motion:

| elbow choice | where it fails |
|---|---|
| **elbow up** | collides with the other arm at the top of the flip |
| **elbow down** | dives into the base on the way down |
| **switching between them** | a **207°** joint jump, which is violent and unplanned |

### What seven joints gives

A seventh joint gives the **whole circle, continuously**. The planner can slide the elbow smoothly to wherever it stays clear, so a smooth, feasible path exists.

---

## 3. Where the joint goes

A **shoulder roll**, which turns the upper arm about its own length, is inserted between the shoulder pitch (j2) and the elbow (j3). It is called `j2b`.

```
6 DOF:   j1 yaw   j2 pitch              j3 elbow   j4 roll   j5 pitch   j6 roll
7 DOF:   j1 yaw   j2 pitch   j2b roll   j3 elbow   j4 roll   j5 pitch   j6 roll
```

This is the standard shoulder arrangement used by the Franka, KUKA iiwa, PR2 and Baxter arms.

**It does not move the elbow.** The elbow's position is set by j1 and j2 alone. What `j2b` does is rotate the *plane the forearm swings in*, which is exactly the swivel needed.

**It also sits in the best possible place.** At the shoulder, added mass loads nothing downstream. That matters because the shoulder is already the hardest joint (see the [mass budget](../mass_budget/MASS_BUDGET_7DOF.md)).

**The slimmer forearm.** The forearm was also thinned from **28 mm to 20 mm**, to clear the hand against its own forearm. It is part of the same decision.

---

## 4. Keeping the maths exact

The old solver had a lovely property: it was **closed-form**, with no iteration and no "close enough". We kept that by making the swivel angle an **input** to the solver, rather than something it searches for.

```
solve(hand position, hand rotation, side, ψ):

  1. Wrist centre   W = hand position − gripper length × approach direction     (as before)

  2. The elbow circle.  Every elbow position that keeps both links the right
     length lies on a circle around the line from shoulder to W.
     ψ picks one point E on that circle.

  3. j1, j2    from the direction shoulder → E        (the upper arm aims at the elbow)

  4. j2b, j3   from the direction E → W, seen in the upper arm's own frame
               j2b = the roll that brings the forearm's plane onto W
               j3  = the elbow bend, by the cosine rule as before

  5. j4, j5, j6   from the leftover rotation, as before
```

Two properties are pinned by tests:

- it round-trips to machine precision (joints → hand → joints);
- ψ = 0 reproduces the old 6-DOF solver **exactly**.

**A free check on j5:** the wrist pitch equals the bend between the forearm axis and the tool's approach axis. At the upright pose it already reaches **132.7°** against a 140° limit, leaving only 7.3° spare, so any change to the flip station or grasp offset has to respect that.

---

## 5. Choosing the elbow angle: four attempts

Choosing ψ at each waypoint turned out to be the hard part. It is a **path problem**: a good ψ now must also be a good neighbour of the ψ at the next waypoint.

| # | approach | why it failed |
|---|---|---|
| 1 | **Nearest to the previous ψ** among those that clear | never walks toward the region the flip needs, so it arrives at the top pose in the wrong part of the circle. 8 self-collisions. |
| 2 | **Widen the step limit** until something clears | leaps **120°** at the tight waypoint and puts a **163°** lurch in j4. That defeats the purpose. |
| 3 | **Greedy hill-climb** within a hard step cap | gets stuck in a local peak and cannot cross the valley. 5 to 9 self-collisions. No cap value helped, because the real problem was the *horizon*, not the cap. |
| 4 | **Dynamic programming over the whole waypoint list** | **worked first time.** |

### How the winning planner works

- One **layer** per waypoint.
- One **node** per (ψ, wrist branch) at that waypoint.
- An **edge** wherever the joint step between neighbours is inside the cap.
- The value carried forward is the **worst clearance so far**.

It is exact, and cheap at this size.

**The wrist branch has to be searched together with ψ.** The second wrist solution flips the palm over, so the two branches have genuinely different clearances. Picking a branch first and measuring second hid half the options.

---

## 6. Proof that both changes were needed

It is easy to add a joint and credit it falsely. So a test removes each change in turn and requires failure:

| remove... | keep... | result |
|---|---|---|
| the swivel (ψ pinned at 0) | 20 mm forearm, same planner | **infeasible**: no continuous path from waypoint 25 to 26 inside the step cap |
| the 20 mm forearm (back to 28 mm) | swivel free | still **−2.0 mm** of self-collision through the whole descent |

**So the seventh joint buys feasibility, and the thinner forearm buys the last two millimetres.** The test `test_the_seventh_axis_is_necessary_and_so_is_the_thinner_forearm` keeps both honest.

### Result, 24 September

| | 6 DOF (was) | 7 DOF (now) |
|---|---|---|
| self-collision | −19.6 mm | **+2.0 mm** (0 of 74 arm-poses intersecting) |
| arm-arm at 90° | +5.6 mm | **+38.7 mm** |
| floor | | +24.0 mm |
| base | | +50.0 mm |
| worst joint step | | 44.8° |
| waypoints per cycle | 47 | **37** (the handover phases are gone) |
| offline checks | 41 | 43, harness 12/12 |

### Two bugs in the *model*, not the robot

Both were mine, and both had been distorting the numbers this decision rested on:

- **The palm plate was stretched.** Its capsule already ran along one axis, and filling the row along the same axis stretched a 140 mm plate to 258 mm. That invented a phantom −1.8 mm breach against the base.
- **A "thin axis" constant held half the *wide* axis.** For the forearm and upper arm, this made the optimistic bound more pessimistic than the model it was meant to bracket.

---

## 7. The first Gazebo run: three faults

The bottle finished upright (1.4° off, 4 mm drift), but the run was untidy. Three causes, all in our own code.

### Fault 1: the handover was still running

The new "fully bimanual" switch had been added to `full_cycle()`, which is **only what the offline tests call**. The robot's own `cycle()` builds its sequence step by step and still opened the right hand and ran the old handover code. The log showed `one hand alone completes the rotation` on a rotation that was already finished, so the seventh joint was never actually exercised.

**All 43 offline checks and 12 harness checks passed anyway**, because the guards sat on the thing that was changed rather than the thing that runs. Both phases are now gated, and a harness check fails if any handover line is ever logged during a bimanual flip.

### Fault 2: the right wrist spun through the object

```
WARN arms did not converge - right_j4 off 1.531 rad      (that is 87.7°)
```

The planner was started fresh on each of nine legs, so at a leg boundary it was free to take the *other* wrist solution, the same hand pose with the wrist turned 180° the other way. Measured at the boundaries: **179.8°**.

Seeding each leg from the previous one fixed the branch but caused a **−15.7 mm** self-collision, because a leg tied to where the last one ended cannot reach the pose the flip needs.

| strategy | worst joint step | worst self-clearance | |
|---|---|---|---|
| each leg planned alone | 179.8° | +2.0 mm | wrist branch flips |
| each leg seeded from the last | 120.0° | −15.7 mm | cannot reach |
| **one plan for the whole cycle** | **44.7°** | **+2.0 mm** | **both** |

The fix is neither: **plan the whole cycle in one call**, then send it in legs. The node says `PLAN MISS` loudly if a leg was not in the plan.

### Fault 3: only the left hand released

With both hands carrying to the end, the release still opened only the left, because the right used to have let go at the old handover. The harness caught it (`grasp "detached" not confirmed`). The fake robot was also wrong in the same place and was fixed.

---

## 8. The grip offset: a 10 mm mistake

**The symptom (from the operator's photos):** one finger of each hand buried in the object while the other sat flush, on both objects.

**The cause:** the two LiDAR reference walls are 20 mm boxes centred at −0.46 m. The faces the LiDAR actually *sees* are at **−0.45 m**. The code used 0.46, the *centres*. Every position fix was therefore **10 mm off** in x and in y.

| | across the closing axis | along the axis |
|---|---|---|
| bottle (75°) | 7.1 mm | 12.2 mm |
| box (50°) | 1.2 mm | 14.1 mm |

Against a pad window of roughly ±3 mm, that is a large miss. It also explains why fingers were drawn toward the object's *end*: by design they sit 15 mm in from it.

### Why it survived

The harness's fake LiDAR scanned walls placed at the code's **own** constants, so the localisation was only ever checked against itself. *A measurement taken in the frame it is supposed to check cannot see that frame's errors.*

### How it is now impossible to hide

- `WALL_A_X = WALL_B_Y = 0.45`, and a test **derives both faces from the world file**.
- The harness scans the walls read from the world file.
- The robot publishes its **ground-truth pose** (used for checks only). `LOCALISATION CHECK` compares the wall fit with it every cycle, and `CENTRING` measures the grasp against the *true* base pose.

### Two smaller contributors, also fixed

- **The fingers closed while the hands were still arriving.** The arm is a 0.53 s lag, so the first finger met the object mid-motion and pushed it (a lying bottle rolls). `settle_arms()` now waits until the error is under 0.004 rad, which is under 2 mm at the hand.
- **Suction made the designed squeeze visible.** The pads are meant to press 3 mm in; once the cup attaches, Gazebo stops computing finger contact and the fingers sink to it. In suction mode the fingers now back off to 0.5 mm outside the surface for the carry and squeeze again just before release.

---

## 9. Two-handed, floor to floor

The flip already went to 90° with both hands. But the **set-down** sent the right hand away at the first lowering waypoint, and the left lowered the object alone. On screen, that looked like the old handover.

The lower hand is not trapped: once the object is standing, it grips the bottom end from the *side*, 35 mm above the floor, with its lowest plate 15 mm clear.

```
approach → lift → yaw → flip ×3 → lower → settle → release → retreat      (68 waypoints)
           └──────────── both hands on the object ────────────┘
```

| phase | what happens |
|---|---|
| **lower / settle** | both hands carry the object down. The suction releases 10 mm above the floor, both hands settle it by friction, and **both open together** |
| **release** | the left hand lifts straight off the top; the right, directly beneath it, backs straight out along its own approach axis, the only way it can leave |
| **retreat** | both rise (the right to 0.25 m) before the move home |

### One definition of the sequence

`choreography.cycle_legs()` is now the **only** place the sequence exists. The node plans and executes it; the tests and harness validate the same list. The old handover code is deleted, because **two copies of the sequence is what let a removed handover keep running in Gazebo for a session.**

### The move home, which nothing had checked

After the last waypoint the arms go home in one joint-space move, which knows nothing about the object just stood up. Measured, the default end state put the right hand's fingers up to **50 mm inside** the standing object on the way home.

The planner now chooses the end state with that move in view (`HOME_MARGIN` = 30 mm against the object, floor, base, mast and the other arm). Result: **+58 / +70 mm** (bottle), **+44 / +62 mm** (box). A test proves the goal is load-bearing: planned without it, the move home falls below the margin.

### A harness that can fail on what matters

The fake robot used to capture a held object's pose when only the left arm had been updated, so it reported the object 97 mm out of the grip, and many checks had to be excluded. Now it freezes the object's pose in the **left hand's frame** at the grasp (what a real detachable joint does). Nothing is excluded, and three new checks run on the real `cycle()`:

- the right hand never moves relative to the left from grasp to release (`0.0 mm`);
- the object is held two-handed through lift, yaw, flip, lower and settle;
- the hands let go only once it stands upright on the floor.

Each was broken on purpose and fires.

---

## 10. The retreat that knocked the bottle over

The next Gazebo run confirmed the offset fix (`LOCALISATION CHECK dx -0.0 dy -0.0 mm`, `GRIP CHECK 2/2 | 2/2`, `CENTRING -2.8 mm`). Then, on the way out, the right hand struck the box lying beside the bottle, and an arm knocked the standing bottle over.

| fault | what the log showed | cause |
|---|---|---|
| a joint spun almost a full turn | `right_j2b off 4.865` rad | the planner measured joint steps **modulo 360°**, but every joint stops short of ±180°, so a "39°" step was travelled as **321°** the other way |
| moves never finished | every flip segment gave up after exactly 36.55 s, joints 0.25 to 0.42 rad short | every wait used the **wall clock**, while the controllers run on **simulation time**, and the laptop simulates slower than real time. Almost certainly the long-undiagnosed "arms do not converge" fault. |
| hands came down still swinging | `right_j4 off 0.788` | parked → hover (up to 76° of travel) ran straight into the descent in one trajectory |
| the box was hit | reproduced offline: the backing-out hand 32 mm inside the box | the lower hand backed out at floor height, and the planner knew the arms, floor, base and mast but **no objects** |
| the bottle was knocked | the move to parked was one blind jump the planner never saw | the solver could not even *represent* the parked pose |

### What changed

- **Joint steps are real travel**, not modulo 360°. The harness samples every commanded trajectory and fails on any jump over 60°.
- **Every wait is on simulation time.** A test guards this.
- **The whole cycle is planned parked-to-parked.** No blind jump anywhere. This needed the IK's **second shoulder solution**, because the parked pose tips the upper arm back over the shoulder (j2 = −114°), which the normal solution never returns.
- **The world is an obstacle.** Every known object goes to the planner with a 20 mm margin, and the node prints `CLEARANCE [..]` per object every cycle.
- **A retreat safe for any surroundings**, every distance derived from the held object: both hands slide straight up (the lower one to 0.14 m, above anything lying on the floor), the lower hand backs out only as far as the grippers require, and then both go home by a planned path.
- **Hover, settle, descend.** The hover is its own leg, and the arms settle there before descending.
- **The path search counts every waypoint.** It used to maximise only the single worst clearance, which goes blind once one point is unavoidably tight. It now minimises a penalty summed over every waypoint, plus a cost on big joint steps.
- **Planning 4× faster:** about 2.5 s per object, from 10.6 s (43 s on the laptop).
- **The suction cups were never built.** They had been written in URDF syntax inside a Gazebo block, which Gazebo cannot parse. They are now a proper link visual, flush with the pad.

---

## 11. Heights from the surface

Every height is now measured from **the surface the object lies on**, not from the floor. Each object declares a `surface_z` (0 for the floor, a shelf's top otherwise), and its centre heights are *above* that surface. A shelf is added by declaring it as a fixture and setting `surface_z`. Nothing else changes.

| follows the surface exactly | stays tied to the arm |
|---|---|
| the grasp, set-down and release | the flip station height (0.24 m, the swept sweet spot), raised only as far as keeps the turning object 30 mm above its surface |
| the lower hand's climb above clutter (0.14 m above the *surface*) | the retreat ceiling (0.40 m, the arm's reach) |
| the upper hand's retreat height | |

The planner also gained **exact boxes** (for shelves) and **exact cylinders**. A bottle had been modelled as a capsule **45 mm too long at each end**.

### Two faults the floor had hidden

1. **The phantom bottle top.** On a 0.10 m table the retreat is capped by reach, 60 mm above the bottle's real top, and the capsule's rounded end had added 45 mm to that top. Fixed by the exact cylinder.
2. **A fingertip swept across the object's top on the way home.** On the floor it is 140 mm up and harmless; on the table it passed 4 mm from the bottle. Backing the hand out is impossible (its wrist is 0.338 m from the shoulder against a 0.36 m reach), so a new `withdraw` leg slides the upper hand *on* along its own fingers until the tips are clear, and only then do the wrists turn. Table: +4 → +21 mm. Floor: +52 → +59 mm.

**The envelope at the current station:** surfaces up to about **0.14 m** for the bottle and **0.16 m** for the box. Above that, the upper hand's retreat runs out of reach.

---

## 12. One robot, shared code

The IK, collision checker, arm commander and wall localisation are shared **byte for byte** with pick-place. Two more files joined them:

- `motion.py`: the parked pose, the planned trips to and from it, and the grip geometry.
- `node_base.py`: every task primitive (sim-time waits, grippers, grasp and release, every check, localisation, facing, recentring, planning, the protective stop, the slip monitor and the run loop).

`task_node.py` is now only the reorient-specific part. Later, the stand-up itself moved into **`flip_task.py`**, shared with perceive and navigate. That move was a pure refactor: the harness's entire output was byte-identical before and after.

`SHARED_FILES.sha256` lists the shared files with their hashes, and each suite fails if a file drifts.

---

## 13. Numbers at a glance

| | value |
|---|---|
| joints per arm | **7** |
| forearm | 20 mm (was 28 mm) |
| self-collision | **+2.0 mm** (was −19.6 mm) |
| arm-arm clearance at 90° | **+38.7 mm** (was +5.6 mm) |
| waypoints per cycle | 37 for the flip alone (24 Sep); 88 for the full parked-to-parked plan (27 Sep) |
| worst clearance in the full cycle | **+15.0 mm** (floor, lower hand at touchdown) |
| planned move home | ≥ +44 mm |
| offline checks (28 Sep) | **56 pass**, harness 31/31 on the floor, 33/33 on 0.10 m tables |
| highest surface supported | ≈ 0.14 m (bottle), 0.16 m (box) |

---

## 14. Still open

- **+2.0 mm is the tightest clearance in the design** (the hand against its own forearm). It is stable to about 0.2 mm under different model resolutions, so it is a real 2 mm, but it is only 2 mm.
- **The 44.8° worst joint step** sits just inside the 45° cap.
- **A known ~10 to 12 mm brush**: the left wrist, unrolling from the flip orientation to parked, brushes its own forearm mid-swing for every end state available. It is visual only, since the model has no self-collision enabled in Gazebo.
- **Mass and torque** were not re-derived at the time. They have been since: see [`MASS_BUDGET_7DOF.md`](../mass_budget/MASS_BUDGET_7DOF.md). The shoulder needs a counterbalance spring.

---

## 15. Appendix: files that changed

For anyone adding a joint to a similar robot, these are the files the seventh axis touched.

| file | change |
|---|---|
| `reorient_description/urdf/arm.xacro` | new `${side}_j2b` revolute joint and link, roll about x, between j2 and j3 |
| `reorient_description/urdf/ros2_control.xacro` | a control entry for j2b on both sides: 14 arm joints |
| `reorient_description/config/controllers.yaml` | j2b added to both arm controllers' joint lists |
| `reorient_behavior/ik.py` | `solve(p, R, side, psi)` with the swivel as an input |
| `reorient_behavior/collision.py` | the link frames gain the j2b frame; one shared `self_clearance()` |
| `reorient_behavior/arm_commander.py` | left and right joint lists grow from 6 to 7 entries |
| `reorient_behavior/task_node.py` | the total joint count 16 → 18 and the "all joints reporting" wait |
| `test/test_flip.py` | every six-joint assumption; a ψ-continuity guard |
| `test/harness.py` | the fake robot echoes 18 joints |

---

*Related reading: [`DESIGN.md`](DESIGN.md) for the 6-DOF gripper history, [`HOW_CONFIG_SOLVING_WORKS.md`](../pick_place/HOW_CONFIG_SOLVING_WORKS.md) for the offline proof method, and [`STORY.md`](../project_story/STORY.md) for the whole narrative.*