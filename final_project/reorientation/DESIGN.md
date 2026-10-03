# The 6-DOF Design: How We Learned to Actually Hold the Object

> **What this document is.** The design history of the **6-DOF** reorientation robot, from 22 to 23 September. It is kept because it explains *why the hands and grip look the way they do*. The final robot has seven joints per arm: see [`DESIGN_7DOF.md`](DESIGN_7DOF.md) for that. The gripper story here carries straight over.

**In one sentence:** for days we kept improving the gripper while the object kept slipping out of it, and then a single measurement showed the gripper was never the real problem.

---

## Contents

1. [The measurement that changed everything](#1-the-measurement-that-changed-everything)
2. [Round 1: five changes aimed at one problem](#2-round-1-five-changes-aimed-at-one-problem)
3. [The weakest point, stated honestly](#3-the-weakest-point-stated-honestly)
4. [Two ideas we decided against](#4-two-ideas-we-decided-against)
5. [Round 2: there was never a pinch](#5-round-2-there-was-never-a-pinch)
6. [Round 3: the flaps were carrying the bottle](#6-round-3-the-flaps-were-carrying-the-bottle)
7. [Round 4: friction mode, shape-aware curl, a hidden fault](#7-round-4-friction-mode-shape-aware-curl-and-a-hidden-fault)
8. [Round 5: suction confirmed](#8-round-5-suction-confirmed-and-two-faults-the-operator-spotted)
9. [Where it stands](#9-where-it-stands)

---

## 1. The measurement that changed everything

Earlier rounds kept improving the **gripper** while the object kept being pulled out of it. Then one number settled it:

> A wrist carrying an object that weighs **4.41 N** reported **19.5 N**. That is **4.4 times** its weight.

Gravity cannot do that. The extra force was **the two arms pushing against each other through the object.**

### Why would two arms fight?

Picture two people each holding one end of a long plank. If one walks slightly faster than the other, the plank gets squeezed or stretched between them, and they feel it in their arms. Robotics calls this an **over-constrained closed chain**: two rigid grips on one rigid body leave nowhere for any mismatch to go, so the mismatch turns into force.

On this robot the mismatch comes from lag:

- There is **no PID controller anywhere**. The simulator turns position error into a velocity using one proportional gain. That makes every joint a "first-order lag", about **0.53 s** at gain 1.9. The arms only move *because* they are behind.
- The two arms carry **different shares** of the load (62% / 38% measured), so they lag by different amounts.
- Position control has no give, so the difference becomes internal force.

Raising the gain to 6.0 was tried. It rang, which is what pure proportional control does.

---

## 2. Round 1: five changes aimed at one problem

| # | change | what it does |
|---|---|---|
| 1 | **Velocity feedforward** | drive the arm from the trajectory instead of from its own error, to attack the lag at its source. *(Reverted in round 2, see below.)* |
| 2 | **Stronger wrist actuators**, 3.5 / 3.5 / 3.0 → 12 N·m | match the shoulders. The three joints that failed to converge in the last run were exactly the three weakest. *(Later review found the wrists were then over-specified; see the mass budget.)* |
| 3 | **Asymmetric grip**, the load-bearing idea | the left hand grips hard (**6 N**); the right hand only supports (**1.5 N**) |
| 4 | **Longer pads**, 30 → 60 mm | resists gravity tipping the far end once one hand is alone |
| 5 | **Silicone pads** and the ridged jaw **removed** | the ridges were solving a problem the pad length solves better |

### The asymmetric grip, explained

The supporting hand can pass on at most **μ × N = 4.5 N** of sideways force before it simply **slides along the object**. So the internal force is *capped by construction* at about a quarter of the 19.5 N seen before. It no longer depends on how well the controller happens to track.

**Sliding is the release valve, not a failure.** Slip is measured against the left hand, which does the holding.

### Why longer pads, and why friction can't help

Once only one hand holds a horizontal object, gravity tries to tip the far end down. A pad pressing with force N over a patch of half-length *a* resists a turning force up to **N × a**.

**Friction does not resist this.** Friction resists *sliding*, not *tipping*. That is why raising the friction coefficient never fixed it, and why the pad got longer instead.

### The margins, as the run reports them

```
GRASP QUALITY       15 N grip × μ 1.5 = 22.5 N     against 4.41 N of weight   →  5.1×
ONE-HAND TORQUE     worst 0.110 N·m at 90°; pad provides 0.360 N·m             →  3.3×
CLEARANCES          arm-arm +0.123 m (bottle) / +0.103 m (box)
                    floor +0.014 m      base +0.020 m
```

> **A note on μ.** The friction coefficient is reported as **1.5**, not the pad's own 2.2. Gazebo combines two surfaces by taking the *lower* one, and the objects are 1.5. Quoting the pad's number would overstate every grasp by 47%.

---

## 3. The weakest point, stated honestly

If the support hand let go completely while the object was still **horizontal**, one hand would need **0.375 N·m** and the pad gives **0.360 N·m**. That is a margin of about **0.96×**, which is less than 1.

This case is not part of the planned sequence. Both hands stay on the object until the handover, and the support hand has 2.5× margin on the load it actually carries. But it was the thinnest number in the design.

> *(A later round found that 0.375 N·m was itself the wrong requirement: it assumes a horizontal object, and the object is only horizontal while both hands hold it. See [`STORY.md`](../project_story/STORY.md), section 3.)*

---

## 4. Two ideas we decided against

### A centre grip for the left hand

Gripping the middle of the object removes the gravity lever arm. It was built before the consequence was measured:

| | end grip | centre grip |
|---|---|---|
| hand separation | 170 mm | 85 mm |
| arm-arm clearance | +122.8 mm | +37.8 mm |
| first interference | 90° | 70° |

Worse everywhere that matters. Longer pads achieve the same goal at no cost.

### A two-phalanx (two-segment) finger

This is genuinely the better gripper, giving an enveloping wrap instead of a pinch. It was not built *at this point*, because it meant four new joints (16 → 20), plus controller, harness and test changes, and nothing yet needed it.

> *It was built later, once a pad-only gripper proved insufficient. See round 3 and `STORY.md`.*

---

## 5. Round 2: there was never a pinch

**22 September, second Gazebo run.** The robot picked the bottle up and flipped it with both hands sharing the load (8.1 N and 4.3 N, a 65/35 split). Then it **dropped the bottle 198 mm** the instant the support hand opened.

### Root cause

The log said `left 1/2 touching, right 1/2 touching`. That means **one finger per hand** touched, and the other closed into thin air.

A flat pad's face sits **42 mm** from the hand's centreline against a **45 mm** object radius. That is only **3 mm of overlap**. Anything more than 3 mm off-centre gets shoved by the near finger while the far finger misses.

So the object was never *pinched*. It was **cradled** between two palms, which lifts and flips perfectly well and collapses the moment one hand leaves.

The positioning error the grip had to survive: about **8 mm** of aim error plus **20 to 60 mm** of wrist tracking error at the fingertip. Against a 3 mm window.

> **Capture range became a first-class requirement.** Nobody had computed it before.

### Three changes

1. **Self-centring jaw.** Two ridges, 6 mm proud, 15 mm either side of the object's *equator*. Capture range goes from ±3 mm to **±15 mm**, and the object rolls back to centre instead of being pushed away. The closing value is now *derived* (6 mm plate + 6 mm ridge + 42.4 mm seat = 0.0544) and the test suite asserts the derivation, because guessing it had broken two builds.
2. **Pad 60 → 40 mm.** The 60 mm pad overhung the object's end every cycle. Torque falls to 0.240 N·m, still 2.2× the 0.110 N·m needed.
3. **Velocity feedforward reverted.** It made tracking **worse**:

   | wrist | before | after feedforward |
   |---|---|---|
   | worst errors (rad) | 0.299 / 0.256 / 0.213 | 0.621 / 0.513 / 0.470 |

   Declaring both a position and a velocity command makes the simulator apply *both targets to the same joint*, and they fight.

### What survived

The asymmetric grip worked. Peak wrist load fell from **19.5 N to 8.1 N**, and both hands visibly carried load. **The closed-chain fight was solved. It was the grasp itself that had never existed.**

**Still the open risk:** ±15 mm of capture against wrist tracking errors that had reached 0.62 rad. If the errors stay that large, the next lever is the *controller*, not the gripper.

---

## 6. Round 3: the flaps were carrying the bottle

**23 September.** The operator watched the screen and spotted what no number had: once lifted, the bottle was resting on the **curled flaps** (the fingertip segments) and touching the flat pads **not at all**. It would have fallen as soon as it tilted.

### Cause

The fingertip reach was limited from **below** (it had to get under the object's equator) but never from **above**. At the original curl the tips sat **5.9 mm inside** the object's surface, pushing it outward until it lifted off the pads and balanced on two narrow lines.

| curl | tip depth below equator | object half-width | tip reaches to | penetration |
|---|---|---|---|---|
| 40° | 38.0 mm | 24.1 mm | 22.7 mm | +1.4 mm |
| 45° | 36.2 mm | 26.7 mm | 20.8 mm | **+5.9 mm** levers off |
| 51° | 33.9 mm | 29.6 mm | 18.7 mm | **+10.9 mm** levers off |

### Fix

The curl was reduced until the tip bites **2.0 mm** into the surface while staying well below the equator. Now it both wraps *and* lets the pad keep its 3 mm of contact. The guard now checks **both** sides, and checks the *commanded* curl rather than the joint's travel limit (the limit was a pose the robot never visits).

> *Note: later notes in `STORY.md` quote a slightly different final curl angle after further tuning. The principle is the same: pad, tip and cup must all bear at once.*

### Suction cups added

One 30 mm bellows cup per finger, **flush** with the pad face.

```
pressure −60 kPa × cup area  =  42.4 N per cup
two cups on the holding hand =  84.8 N     against a 4.41 N object   →   19.2×
```

Suction removes the two things that had dominated the project, because it does **not depend on geometry**:

- the cup seals wherever it lands, within a few millimetres of straight-on, so there is no ±3 mm window to miss;
- it holds over an **area**, so resisting rotation no longer depends on pad length or ridge spacing.

**Flush is the point.** A cup standing proud would hold the pad off the object, which is the same fault as the over-reaching flaps. The first version was 1 mm proud and the new guard caught it.

`grasp_mode: suction` uses a detachable joint, which is an honest model: a suction gripper really is a detachable attachment.

### The caveat, built into the code

Attaching pulls the object into the robot's kinematic tree, so Gazebo stops computing finger-object contacts and the contact sensors go quiet. `BIMANUAL CHECK` and `HANDOVER CHECK` now **say they are not measurable** in suction mode, instead of reporting a failure that is only an artefact. `GRIP CHECK` is still valid because it runs *before* the attach.

---

## 7. Round 4: friction mode, shape-aware curl, and a hidden fault

### Changes

1. **`grasp_mode: friction`.** Suction fully constrains the object, which makes the second hand unnecessary and the "two arms" claim decorative. Under friction, neither hand alone can control orientation, the load share is measurable, and both checks report real numbers. Margins: 4.1× friction, 2.2× torque. Suction stays available as a one-line fallback.
2. **`weld_rescue: false`.** The rescue weld turned a *failed* friction grasp into a *reported pass*: the experiment rescuing itself, and blinding the contact sensors in the process.
3. **Shape-aware curl.** The curl is sized for a *round* cross-section, which narrows with depth. A box face is vertical, so the same tip drove **16.1 mm** into it against the 2.0 mm intended. Flat objects now do not curl at all; the pad already bears over the full face height.

   > **A trap found on the way:** in Python, `bool('false')` is `True`. A config line `round: false` arriving as a *string* would still have curled. The **harness** caught this by reporting a box as round. The offline tests could not.
4. **Withdrawn wrist levelled.** The parked pose inherited the grasp's tool roll (−149°, the biggest angle in the cycle, on a joint with 0.17 rad of tracking error). It now sits at −111° with joint 4 at zero, and the move home clears by +45.6 mm.

### A fault found and *not* fixed yet: finger against its own forearm

`collision.py` checked arm-against-arm, floor, base and mast. It had **never** checked an arm against *itself*. The operator saw the fault; no test could.

| flip angle | pessimistic bound | optimistic bound | verdict |
|---|---|---|---|
| 0° | +7.9 mm | +12.5 mm | clear |
| 30° | −2.2 mm | +2.4 mm | uncertain |
| 60° | −14.4 mm | −9.8 mm | **real collision** |
| 90° | −26.0 mm | −21.4 mm | **real collision** |

The two bounds exist because a round capsule around a flat plate over-reports by up to 20 mm. But even the *optimistic* bound goes negative from 60°, so this is real geometry, not a modelling artefact. It is shallow (the finger presses on its own forearm rather than blocking the motion), which is why runs still completed.

A test pinned the measurement so it could not be forgotten. It fails in **both directions**: if it gets worse, and if it becomes clear (so the bound is tightened instead of left stale).

> **This fault is the direct seed of the 7-DOF design.** Resolving it was the whole point of adding the seventh joint and slimming the forearm. See [`DESIGN_7DOF.md`](DESIGN_7DOF.md).

---

## 8. Round 5: suction confirmed, and two faults the operator spotted

`grasp_mode: suction` is the mode that **actually stood both objects upright in Gazebo**: bottle and box, tilt 0.0°, drift 3 mm and 14 mm, with the right hand releasing at about 70°. Friction stays available and its margins hold on paper, but suction has the confirmed result and nothing should regress it.

**The trade, stated plainly:** suction fully constrains the object, so the second hand is not load-bearing *during the grasp*. The two-arm claim therefore rests on the **lift** and on the **0 to 70° rotation**. The first 36° of that rotation is genuinely impossible single-handed (0.375 N·m needed against 0.240 available). That is the defensible core.

### Fault 1: the fingers sit inside the grey link

The first guess was the palm plate. It was moved (x 0.032 → 0.019). **That was wrong, and was reverted**, for two reasons:

- the operator's photos show the black finger emerging from the *middle of the forearm*, not from the palm;
- the move cost **12.8 mm** of arm-arm clearance at 90° (+9.4 → −3.4 mm), because it moved the plate toward the upper arm, the body it actually collides with.

The real cause was the self-collision already on record, measured in the exact pose in the photos:

| pose | clearance |
|---|---|
| travel pose | −2.2 mm |
| at the grasp | +17.6 mm (clear) |
| 60° of flip | −9.8 mm |
| 90° of flip | −21.4 mm |

The photographs, the travel-pose reading and the flip readings were **one fault, not three**.

> **Lesson:** a body was moved to fix a fault that had not been located, and a different margin got worse. The photos were available the whole time and would have located it in one look.

### Fault 2: the box was engraved on one side only

The obvious suspect was checked first and **ruled out**. A square cross-section is 45.0 mm face-on and 63.6 mm corner-on, so an uncontrolled roll could gouge 21.6 mm. But the box spawns at exactly 90°, and a cuboid on a flat floor can only rest on a *face*, never an edge.

**The real cause** was the fingertip flap: with the curl active, the tip drove 16.1 mm into the flat face. With only ±3 mm of lateral capture, whichever side the object sat closer to took the whole penetration while the other side never reached. One side engraved, one side clear.

Already fixed by the shape-aware curl: flat objects do not curl, so the pads close to 42.0 mm on a 45.0 mm face (3.0 mm, symmetric, both sides).

---

## 9. Where it stands

| topic | outcome |
|---|---|
| the real problem | arms fighting through the object (19.5 N on a 4.41 N load), **not** the gripper |
| the fix | asymmetric grip: the supporting hand slips instead of fighting |
| capture range | a flat pad had ±3 mm; the redesigned hand wraps instead of pinching |
| the confirmed mode | **suction**, which stood both objects upright in Gazebo |
| the open fault | finger against its own forearm, −21.4 mm at 90° |
| what resolved it | the **seventh joint** and a slimmer forearm: see [`DESIGN_7DOF.md`](DESIGN_7DOF.md) |

**The lessons, short version:**

1. **Measure before redesigning.** One number (4.4× weight) saved days of gripper tweaking.
2. **Bound every guard from both sides.** The flap reach was limited from below and never from above, and that caused two separate faults.
3. **Look at the photographs.** The operator's eyes found faults no check had been built to see.
4. **State the weakest number.** The 0.96× margin was written down openly instead of being hidden.