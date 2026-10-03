# 7-DOF Story — The Final Bimanual Flip

## Why the seventh joint was added

The 6-DOF arm could reach the required hand poses, but the full bimanual path still caused self-collision.

The key idea was simple:

> For a fixed hand pose, the elbow can sit anywhere on a circle.

The 6-DOF solution could choose only a small number of useful configurations.

The 7-DOF design added:

**`j2b` — a shoulder-roll joint**

This lets the elbow move continuously around that circle.

---

## The other change: a thinner forearm

The new shoulder motion also created a clearance problem near the hand.

The forearm therefore changed from:

```text
28 mm → 20 mm
```

This was not just a cosmetic change.

The final tests showed that both changes were needed:

```text
Remove the shoulder swivel
→ full cycle becomes infeasible

Restore the old 28 mm forearm
→ the hand still presses into the forearm
```

So the improvement did not come from simply adding another motor.

**The extra joint and the slimmer forearm were both required.**

---

## The 7-DOF path

The final planner searches the swivel angle and the wrist branch together while moving through the flip waypoints.

The recorded result was:

```text
7-DOF elbow swivel

47 waypoints
0 blocked
minimum valid swivel window ≈ 50°
```

The final full-cycle measurements were:

```text
Self-collision:
    -19.6 mm  →  +2.0 mm

Arm-to-arm clearance:
    +5.6 mm   →  +38.7 mm
```

The final verification also recorded:

```text
43 offline checks
12 / 12 harness checks
```

The important result is not just that the number became positive.

The full flip became:

> **fully bimanual from the starting grasp to the final upright pose.**

The earlier handover and single-arm finish were removed.

The final cycle contains **37 waypoints**.

---

## How we know the seventh joint really mattered

We used removal tests.

The final design was run with one change at a time:

```text
7th joint removed
→ cycle infeasible

7th joint kept
+ 20 mm forearm
→ cycle feasible

28 mm forearm restored
→ hand/forearm collision returns
```

This is important because it shows that the seventh joint was not added just because "more DOF is better."

It was added because the old design had a specific geometric limitation.

---

## The final 7-DOF result

The project progression can now be summarized as:

```text
4 DOF
→ cannot control the required orientation

5 DOF
→ better grasp orientation
→ still cannot complete the flip

6 DOF
→ full orientation is possible
→ arms collide during the required path

7 DOF
→ shoulder has another motion
→ elbow can move around the circle
→ full bimanual flip becomes feasible
```

This is why the final robot uses:

**2 arms × 7 DOF = 14 actuated joints**

---

## The next problem: mass

Solving the motion problem introduced a new engineering limit.

The 7-DOF arms and the rest of the hardware now use about:

**4.55 kg**

of the Kobuki's 5 kg hard-floor payload rating.

That leaves about:

**0.45 kg**

for the object in the current mass estimate.

This is the current hardware baseline, not the final long-term payload target.

The next design iteration should therefore focus on reducing:

* arm mass
* mounting plate mass
* sensor structure mass
* hand and suction mass

A future target of about **1 kg useful payload** is reasonable to investigate, but it is not a claim about the current robot.

---

## What the 7-DOF design finally solved

The main goal of the project was not "build a 7-DOF arm."

It was:

> **Use two arms to pick up an object, rotate it 90° in the air, and put it down upright.**

The seventh joint was the design change that finally made that full bimanual path feasible.

The project can therefore be described as:

```text
Perception
    ↓
Navigation
    ↓
Approach
    ↓
Dual-arm grasp
    ↓
7-DOF bimanual reorientation
    ↓
Upright placement
```

The next step is no longer to add more DOF.

It is to make the existing 7-DOF robot **lighter, more efficient and more suitable for physical hardware**.

---

## Final takeaway

The 7-DOF arm was not chosen because seven joints sounds impressive.

It was chosen because the earlier designs exposed a real geometric problem, and the extra shoulder freedom removed that problem.

That is the main design story of the project:

> **The robot became more capable because each new DOF was added to solve a measured failure.**
