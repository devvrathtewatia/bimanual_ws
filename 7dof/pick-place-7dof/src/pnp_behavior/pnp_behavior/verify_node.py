"""Machine-checkable success criterion for section A.

Nearly every debugging round on this project was spent judging outcomes from
video and log narration. This node replaces that with a number: it reads the
objects' ACTUAL poses out of Gazebo and prints PASS/FAIL per object against
where they were supposed to end up.

It reads ground truth from TWO independent sources and uses whichever arrives:

  1. /model/<name>/pose (primary) - a PosePublisher system attached to each object
     model, one topic per object. Publishes unconditionally at 20 Hz.
  2. pose_topic, default /world/<world>/pose/info (fallback) - the world-level
     scene-broadcaster stream, bridged as a TFMessage.

There are two because source 2 delivered ZERO messages on every run so far even
with a well-formed bridge spec and the SceneBroadcaster system loaded, and the
cause was not findable by inspection. Rather than guess again, source 1 avoids the
world stream entirely. The report says which source each pose came from, so the
next run diagnoses the bridge as a side effect of verifying the task.

The same node becomes the regression test for sections B, C and D - only the
expected poses change.
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
from tf2_msgs.msg import TFMessage


def axis_from_quat(x, y, z, w):
    """Horizontal angle of the object's long axis, from its orientation quaternion.

    Both bodies here have their long axis along local +z, so the world axis is the
    third column of the rotation matrix. Only its horizontal projection matters: the
    object lies down, so the axis is horizontal, and an axis is a LINE - reversing it
    is the same object - so the result is folded into [0, 180).
    """
    ax = 2.0 * (x * z + w * y)
    ay = 2.0 * (y * z - w * x)
    return math.degrees(math.atan2(ay, ax)) % 180.0


def _matches(frame, name):
    """Does this TF frame belong to the model called ``name``?

    Gazebo scopes entity names, so the same model can arrive as ``bottle``,
    ``bottle::link`` or ``bottle/link`` depending on version and on whether the
    pose is a model or a link pose. Matching only the bare name silently drops all
    of them and reports "NO POSE RECEIVED", which is indistinguishable from the
    bridge being broken. Accept any of the forms instead.
    """
    return (frame == name
            or frame.startswith(name + '/')
            or frame.startswith(name + '::'))


class VerifyNode(Node):

    def __init__(self):
        super().__init__('verify_node',
                         automatically_declare_parameters_from_overrides=True)

        def p(name, default):
            try:
                v = self.get_parameter(name).value
                return default if v is None else v
            except Exception:  # noqa: BLE001
                return default

        self.reach = float(p('reach', 0.30))
        # DIRECTIONAL, not circular. Radially the cradle gives only 11 mm before the
        # object rides a chamfer; tangentially the pedestal is 0.32 long against an
        # object of at most 0.24, so 40 mm is harmless. One circular tolerance cannot
        # say that, and the old 0.09 graded an object hanging 33 mm off the end of its
        # pedestal as PASS.
        self.tol_radial = float(p('tol_radial', 0.015))
        self.tol_tangential = float(p('tol_tangential', 0.040))
        self.tol_z = float(p('tol_z', 0.020))
        # The object must finish TANGENTIAL to its slot. That is the point of the yaw
        # alignment phase, and the chamfers only cradle it if it is.
        self.tol_yaw = float(p('tol_yaw_deg', 12.0))
        self.names = list(p('objects', ['bottle', 'can', 'box']))

        self.expect = {}
        for n in self.names:
            az = math.radians(float(p(f'{n}.place_azimuth', 180.0)))
            self.expect[n] = {
                'x': self.reach * math.cos(az),
                'y': self.reach * math.sin(az),
                # ON THE SHELF, from its surface: the shelf's top plus the object's
                # lying centre height
                'z': (float(p(f'{n}.place_surface_z', 0.10))
                      + float(p(f'{n}.cz', 0.045))),
                'az': az,
                # tangential to this slot, folded to [0, 180) like the measurement
                'axis': (math.degrees(az) + 90.0) % 180.0,
            }

        self.seen_frames = set()
        # The two sources are kept SEPARATE and merged only at report time. An
        # earlier attempt let them write into one dict with a priority rule, and that
        # can freeze a stale value: once the fallback has filled a slot, a
        # "don't overwrite" rule stops it updating, and if the primary never arrives
        # the reported pose is whatever it was seconds ago. Two dicts cannot do that.
        self.seen_primary = {}
        self.seen_fallback = {}
        # The bridge publishes on the Gazebo topic name; remapping it did not work,
        # so the name is passed in as a parameter and subscribed to directly.
        topic = p('pose_topic', '/model_poses')
        self.get_logger().info(f'listening for ground truth on {topic}')
        self.create_subscription(TFMessage, topic, self._on_poses, 20)
        # Primary source: one PoseStamped topic per object, no name matching needed.
        for n in self.names:
            self.create_subscription(
                PoseStamped, f'/model/{n}/pose',
                lambda msg, name=n: self._on_model_pose(name, msg), 20)
            self.get_logger().info(f'listening for ground truth on /model/{n}/pose')
        self.create_subscription(Empty, '/verify/run', self._on_run, 10)
        self._probe = self.create_timer(12.0, self._first_check)
        self.get_logger().info(
            'verification armed for: ' + ', '.join(self.names))

    def _on_model_pose(self, name, msg):
        """Primary source. Unambiguous: the topic name identifies the object."""
        pos, q = msg.pose.position, msg.pose.orientation
        self.seen_primary[name] = ((pos.x, pos.y, pos.z), (q.x, q.y, q.z, q.w))

    def _on_poses(self, msg):
        """Fallback source: the world-level scene-broadcaster stream."""
        for tr in msg.transforms:
            frame = tr.child_frame_id
            self.seen_frames.add(frame)
            for n in self.names:
                if _matches(frame, n):
                    t, r = tr.transform.translation, tr.transform.rotation
                    self.seen_fallback[n] = ((t.x, t.y, t.z),
                                             (r.x, r.y, r.z, r.w))

    def pose_of(self, name):
        """Best available pose, and where it came from. Primary wins."""
        if name in self.seen_primary:
            return self.seen_primary[name], f'/model/{name}/pose'
        if name in self.seen_fallback:
            return self.seen_fallback[name], 'world pose/info'
        return None, None

    def _first_check(self):
        """One-shot: say early whether ground truth is flowing, not at the end.

        A silent bridge used to be discovered only in the final report, after a full
        run had already been spent. This makes it visible in the first few seconds.
        """
        self._probe.cancel()
        n_p, n_f = len(self.seen_primary), len(self.seen_fallback)
        if n_p or n_f:
            self.get_logger().info(
                f'ground truth OK: {n_p}/{len(self.names)} objects via '
                f'/model/<name>/pose, {n_f} via world pose/info')
        else:
            self.get_logger().error(
                'NO GROUND TRUTH ARRIVING on either source. Verification will be '
                'blind. Check "ros2 topic list | grep -e model -e pose/info" and '
                'the Gazebo console for PosePublisher plugin load errors.')

    def _on_run(self, _msg):
        self.report()

    def report(self):
        log = self.get_logger().info
        log('=' * 66)
        log('SECTION A VERIFICATION - actual object poses vs expected')
        log('=' * 66)
        n_pass = 0
        for n in self.names:
            e = self.expect[n]
            got, src = self.pose_of(n)
            if got is None:
                log(f'  {n:<12} NO POSE RECEIVED')
                continue
            (x, y, z), q = got if isinstance(got[0], tuple) else (got, None)
            e = self.expect[n]
            # split the horizontal error along the slot's own radial and tangential
            # directions, because the shelf tolerates them very differently
            dx, dy = x - e['x'], y - e['y']
            ca, sa = math.cos(e['az']), math.sin(e['az'])
            d_rad = dx * ca + dy * sa
            d_tan = -dx * sa + dy * ca
            dz = z - e['z']
            ok = (abs(d_rad) <= self.tol_radial
                  and abs(d_tan) <= self.tol_tangential
                  and abs(dz) <= self.tol_z)
            yaw_txt = ''
            if q is not None:
                got_axis = axis_from_quat(*q)
                d_yaw = abs((got_axis - e['axis'] + 90.0) % 180.0 - 90.0)
                ok = ok and d_yaw <= self.tol_yaw
                yaw_txt = (f' | axis {got_axis:.0f} deg, want {e["axis"]:.0f}, '
                           f'off {d_yaw:.0f} (tol {self.tol_yaw:.0f})')
            n_pass += 1 if ok else 0
            log(f'  {n:<8} {"PASS" if ok else "FAIL"} at '
                f'({x:+.3f}, {y:+.3f}, {z:.3f}) | radial {d_rad * 1000:+.0f} mm '
                f'(tol {self.tol_radial * 1000:.0f}) | tangential {d_tan * 1000:+.0f} '
                f'mm (tol {self.tol_tangential * 1000:.0f}) | vertical '
                f'{dz * 1000:+.0f} mm{yaw_txt} [via {src}]')
        if any(self.pose_of(n)[0] is None for n in self.names):
            missing = [n for n in self.names if self.pose_of(n)[0] is None]
            seen = sorted(self.seen_frames)
            log('  NO GROUND TRUTH for: %s' % ', '.join(missing))
            log('  world pose/info delivered %d frame name(s): %s'
                % (len(seen), ', '.join(seen[:12]) or 'NONE AT ALL'))
            log('  diagnosis: if BOTH sources are empty, check that the bridge')
            log('    process started - "ros2 topic list | grep model" should show')
            log('    /model/<name>/pose. If /model/<name>/pose exists but is silent,')
            log('    the PosePublisher plugin did not load: check the Gazebo console')
            log('    for "Unable to load plugin gz-sim-pose-publisher-system".')
        log('-' * 66)
        log(f'  RESULT: {n_pass}/{len(self.names)} objects placed correctly')
        log('=' * 66)


def main(args=None):
    rclpy.init(args=args)
    node = VerifyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.report()          # report even if the run was interrupted
    except RuntimeError:
        # Ctrl-C under ros2 launch shuts the context down while a message is being taken,
        # and rclpy raises "Unable to convert call argument" from take_message(). Harmless at
        # shutdown - re-raised if it happens while ROS is still running.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
