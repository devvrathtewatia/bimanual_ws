# 6-DOF Story — The Motion Was Possible, But the Arms Collided

## What 6 DOF fixed

The 6-DOF arm solved the main problem from the first design.

Now the end-effector had enough freedom to control:

* 3D position
* 3D orientation

The same 40-configuration test gave:

```text
6 DOF → 40 / 40 configurations solved
```

So the robot could now generate the orientations needed for the full object flip.

That was a real improvement.

But then we found a different problem.

---

## The new problem: self-collision

Solving the hand pose does not guarantee that the two arms can physically move there.

During the full bimanual flip, some required 6-DOF configurations caused the arms to collide.

The recorded full-cycle check was:

```text
Bottle:
    worst clearance = -19.6 mm
    23 of 94 arm poses intersected

Box:
    worst clearance = -19.7 mm
    21 of 94 arm poses intersected
```

A negative clearance means the modeled bodies overlap.

So the situation was:

```text
6 DOF
    ↓
hand poses are solvable
    ↓
full bimanual path is NOT collision-free
```

---

## Could better control fix it?

We checked that too.

A leader-follower / impedance-control approach could not fix the problem because the **commanded waypoints themselves already collided**.

The recorded overlap was about:

```text
24.4 mm
```

even with zero tracking error.

That means the problem was not:

> "The controller follows the path badly."

It was:

> **"The path itself is impossible without collision."**

So changing the controller would not solve the root problem.

---

## The elbow had the clue

For one hand pose, the elbow is not limited to one point.

Geometrically, the elbow can move around a **circle** while the hand stays at the same target pose.

But the 6-DOF implementation only gives the planner two main elbow configurations.

Those two choices failed in different ways:

```text
Elbow-up
→ arm-to-arm collision around the 90° flip

Elbow-down
→ enters the base during the descent

Switching between them
→ about 207° of joint motion
```

So there was no clean continuous path through the useful part of the elbow circle.

---

## The collision model also had problems

Before changing the robot, the collision checker itself had to be cleaned up.

Several early calculations were wrong because the model included:

* zero-length wrist capsules
* incorrect body ordering
* a wrong finger-thickness key
* a comparison between bodies that are physically attached

These caused constant or overly pessimistic collision values.

The fix was to put one shared `self_clearance()` function in `collision.py` and make the links match their real flat shapes more closely.

After that, the collision result became a useful number instead of a vague range.

---

## The real conclusion

At this point we had:

```text
4 DOF
→ orientation impossible

6 DOF
→ orientation possible
→ collision-free full cycle still impossible
```

The missing freedom was now clear.

The elbow needed a way to move continuously around its available circle.

That led to the seventh joint.

---

## Next step

The final design added a **shoulder-roll joint called `j2b`**.

Instead of picking only two elbow configurations, the planner could now move around the full circle and choose a safer configuration during the flip.
