# 4-DOF Story — The First Design

## What we were trying to do

The first version used **4 degrees of freedom (DOF) per arm**.

The goal was the same as the final robot:

> Find an object, hold it with two arms, turn it upright, and place it.

At first, 4 DOF looked reasonable because the arms could reach the required positions.

The problem was not **where the hand could go**.

The problem was **how the hand was oriented when it got there**.

---

## The problem

When two hands hold an object, the hands must move and rotate with the object.

For a 90° flip, the direction in which each hand approaches the object also has to change.

The 4-DOF arm did not have enough wrist freedom to do that.

Three clear problems appeared:

* The gripper could not freely choose its orientation.
* A direct mid-air flip was not reachable.
* The task started turning into a single-arm or approximate motion instead of true bimanual reorientation.

This was not something that could be fixed by tuning a controller.

**The required motion was outside the arm's available motion.**

---

## We proved it instead of guessing

The real flip was broken into:

**2 objects × 10 flip angles × 2 hands = 40 configurations**

The test used the full 6-DOF solution as the reference, then checked whether the same poses were still possible after removing the missing axes.

### Result

```text
40 tested configurations

6 DOF   → 40 / 40
5 DOF   →  4 / 40
4 DOF   →  0 / 40
```

The missing axes were not small corrections either:

```text
Worst wrist roll   :  96.1°
Worst wrist pitch  : 133.1°
Worst tool roll    : 159.9°
```

The 4-DOF version therefore failed for a simple reason:

> **The arm did not have the joints needed to follow the object's full orientation.**

---

## Why adding only one joint was not enough

The next idea was 5 DOF.

The extra joint gave the gripper a roll about its approach direction, so the fingers could be rotated into a better grasp.

That fixed part of the problem.

But the hand's **approach direction** was still limited by the arm geometry.

A full 90° flip needs that direction to change too.

So:

```text
4 DOF → grasp orientation is limited
5 DOF → grasp orientation improves
5 DOF → full reorientation is still not possible
6 DOF → full position + orientation becomes available
```

---

## What we learned

The important lesson from the 4-DOF version was:

> **Do not keep tuning a design after the geometry has already proved that the motion is impossible.**

The next step was therefore not a better controller.

It was a better arm.

---

## Next step

The design moved to **6 DOF per arm** so that the end-effector could control both:

* position
* orientation

The next story shows what 6 DOF solved — and the new problem it exposed.
