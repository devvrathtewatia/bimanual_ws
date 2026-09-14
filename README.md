# Kobuki Bimanual Manipulation — ROS 2 Jazzy + Gazebo Harmonic

Dual 4-DOF arms on a Kobuki-class base. The robot scans the room, finds
fallen color-coded objects, drives to them, grasps them with both grippers,
rotates them upright mid-air, and places them on a shelf — repeating for
every fallen object it saw.

Pipeline (matches the design sheet):
`Perception -> Approach -> Align -> IK Grasp -> Co-manipulate -> Reorient -> Place`

## Packages

| package | contents |
|---|---|
| `kobuki_bimanual_description` | URDF/Xacro: base, mast + sensors, 2x 4-DOF arms + grippers, ros2_control, Gazebo plugins, `controllers.yaml` |
| `kobuki_bimanual_gazebo` | worlds (`main`, `bottle`, `bowl`, `box`), object + shelf models, `sim.launch.py` |
| `kobuki_bimanual_behavior` | analytic IK, bimanual choreography, perception node, mission state machine, parameters |

## Requirements

- Pop!\_OS **24.04** (Ubuntu Noble base — required for ROS 2 Jazzy debs)
- ROS 2 Jazzy + Gazebo Harmonic (Harmonic is the default Gazebo paired
  with Jazzy's `ros_gz`)

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-xacro \
  python3-opencv python3-numpy \
  python3-colcon-common-extensions python3-rosdep
```

## Build

```bash
cd ~/bimanual_ws            # wherever you copied this workspace
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y   # optional safety net
colcon build --symlink-install
source install/setup.bash
```

## Run

```bash
# full demo: 3 fallen objects, successive pick -> reorient -> place
ros2 launch kobuki_bimanual_gazebo sim.launch.py

# single-object worlds
ros2 launch kobuki_bimanual_gazebo sim.launch.py world:=bottle
ros2 launch kobuki_bimanual_gazebo sim.launch.py world:=bowl
ros2 launch kobuki_bimanual_gazebo sim.launch.py world:=box

# simulation only (drive/test manually, no autonomous mission)
ros2 launch kobuki_bimanual_gazebo sim.launch.py autostart:=false
```

What you should see, in order:

1. Gazebo opens; robot at the origin, shelf behind it, objects lying ahead.
2. Mission node releases the startup grasp joints, opens grippers, tucks
   both arms, then does one full survey rotation logging which objects it saw.
3. Per object (bottle -> bowl -> box): rotate to reacquire, center, drive in,
   final blind creep on odometry, both grippers descend, fingers close,
   grasp attach, two-phase lift + 90° flip to upright, drive to the shelf,
   lower, release, retract, back away, tuck. Then the next object.
4. `MISSION COMPLETE` in the console when done.

## Tuning (config/mission_params.yaml in the behavior package)

| symptom | knob |
|---|---|
| object not detected | `hsv_lo/hsv_hi` per object, `min_area` |
| stops too far / too close before grasping | `grasp_forward`, `approach_stop_range` |
| fingers hit or miss the object | per-object `lying_z`, `s`, `close` |
| object placed too high/low on shelf | per-object `z_place` |
| robot parks badly at the shelf | `shelf_approach` (odom frame; world-dependent) |
| drives too fast/slow | `v_lin`, `w_rot`, `scan_w` |

Arm geometry constants live in `ik.py` + `robot.urdf.xacro` and must stay in
sync (shoulder at x 0.095, y ±0.08875, z 0.18; L1 = L2 = 0.16; grasp point
0.07 beyond the wrist).

## Design assumptions (deliberate, documented for the report)

- **LiDAR at 0.46 m sees structure, not floor objects.** A horizontal 2D
  scan plane physically cannot intersect 5–10 cm tall lying objects from any
  mast height, so the camera does object detection *and* ranging
  (ground-plane back-projection); the LiDAR provides the forward safety
  guard and room structure. Mast height itself is not critical and can be
  changed in `robot.urdf.xacro` (`lidar_z`).
- Objects spawn with their long axis pointing at the robot start area, so
  arriving head-on leaves the axis aligned with the grasp choreography.
  Repositioning around an arbitrarily-oriented object is future work.
- The grasp is a `DetachableJoint` (rigid constraint, per the design sheet)
  rather than friction-only contact; fingers close to a light-touch fit.
  These joints attach at world load — the mission detaches them at startup.
- Shelf pose is a parameter in the odom frame (odom == world at spawn);
  the tall shelf back-board makes it LiDAR-visible for future localization.
- The bowl skips the upright check (a cylinder bowl's silhouette is
  ambiguous); bottle and box classify lying-vs-upright by apparent height.
- Masses/inertias/limits are placeholders consistent with Dynamixel
  XM-class servos (±150° yaw, ±115° pitch, 4–6 N·m efforts).


```bash
cd src/kobuki_bimanual_behavior
python3 test/test_ik.py      # FK/IK round-trip + full choreography check
```
