"""Machine-checkable success criterion for every section that stands objects upright.

SHARED FILE, between reorient_behavior and perceive_behavior (the 'flip' group in
SHARED_FILES.sha256): both are judged by the same question, so by the same code.

Section A asked "is the object on the shelf?" - a position question. Standing objects up
asks "is the object UPRIGHT?" - an orientation question. So this node reads each object's
actual quaternion out of Gazebo, computes the angle between its long axis and vertical, and
prints PASS/FAIL - and checks it is on its surface and was not flung across the floor, since
standing an object up by throwing it would satisfy an orientation-only test.

WHERE IT STARTED is where Gazebo first reports it, not where a config says: the perceive
section's config holds no object positions at all - that is the point of perceiving them -
and a measured start is the honest reference in every section.

Listens on /model_poses (bridged from the Gazebo pose stream) and evaluates when
the task node publishes /verify/run, and on shutdown so an interrupted run still
reports.
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
from tf2_msgs.msg import TFMessage


def axis_from_quat(x, y, z, w):
    """The object's long axis in world coordinates.

    Every object model is authored with its long axis along its own local Z, so
    the world direction of that axis is the third column of the rotation matrix.
    """
    return (2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y))


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

        self.tilt_tol = math.radians(float(p('tilt_tol_deg', 20.0)))
        self.move_tol = float(p('move_tol', 0.10))
        self.z_tol = float(p('z_tol', 0.020))
        self.names = list(p('objects', ['bottle', 'can', 'box']))

        self.title = str(p('title', 'VERIFICATION'))
        self.expect = {}
        for n in self.names:
            self.expect[n] = {
                # STANDING ON ITS SURFACE: the surface's height plus half its length. On a
                # shelf, an object that fell off and landed upright on the floor is a fail.
                'z': (float(p(f'{n}.surface_z', 0.0))
                      + float(p(f'{n}.upright_cz', 0.10))),
            }

        self.seen_frames = set()
        # TWO INDEPENDENT SOURCES, merged only at report time.
        #
        #   1. /model/<name>/pose (primary) - a PosePublisher on each object model,
        #      one topic per object, published unconditionally at 20 Hz.
        #   2. pose_topic, default /world/<world>/pose/info (fallback) - the
        #      scene-broadcaster stream, bridged as a TFMessage.
        #
        # There are two because source 2 delivered ZERO messages on every run so far
        # even with a well-formed bridge spec, and the cause was not findable by
        # inspection. This node used to listen on /model_poses, which NOTHING ever
        # published: the launch file remapped the bridge output to that name and the
        # remap silently had no effect, because ros_gz_bridge names its ROS side after
        # the Gazebo topic. So every section reported "NO POSE RECEIVED" and every
        # "TASK COMPLETE: ok" was unverified.
        #
        # They are kept in separate dicts on purpose. An earlier attempt merged them
        # with a priority rule, which can freeze a stale value: once the fallback has
        # filled a slot a "do not overwrite" rule stops it updating, and if the primary
        # never arrives the reported pose is seconds old. Two dicts cannot do that.
        self.seen_primary = {}
        self.seen_fallback = {}
        self.start = {}
        topic = p('pose_topic', '/world/reorient/pose/info')
        self.create_subscription(TFMessage, topic, self._on_poses, 20)
        for n in self.names:
            self.create_subscription(
                PoseStamped, f'/model/{n}/pose',
                lambda msg, name=n: self._on_model_pose(name, msg), 20)
        self.create_subscription(Empty, '/verify/run', lambda m: self.report(),
                                 10)
        self._probe = self.create_timer(12.0, self._first_check)
        self.get_logger().info(f'ground truth: /model/<name>/pose (primary), '
                               f'{topic} (fallback)')
        self.get_logger().info('verification armed for: '
                               + ', '.join(self.names))

    def _on_model_pose(self, name, msg):
        """Primary source. Unambiguous: the topic name identifies the object."""
        q = msg.pose.orientation
        pos = msg.pose.position
        self.seen_primary[name] = ((pos.x, pos.y, pos.z), (q.x, q.y, q.z, q.w))
        # the first report is where it started - before the robot has touched anything
        self.start.setdefault(name, (pos.x, pos.y))

    def pose_of(self, name):
        """Best available pose, and where it came from. Primary wins."""
        if name in self.seen_primary:
            return self.seen_primary[name], f'/model/{name}/pose'
        if name in self.seen_fallback:
            return self.seen_fallback[name], 'world pose/info'
        return None, None

    def _first_check(self):
        """Say early whether ground truth is flowing, rather than at the end.

        A silent bridge used to be discovered only in the final report, after a whole
        run had already been spent on it.
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
                'blind. Check "ros2 topic list | grep -e model -e pose/info" and the '
                'Gazebo console for PosePublisher plugin load errors.')

    def _on_poses(self, msg):
        for tr in msg.transforms:
            f = tr.child_frame_id
            self.seen_frames.add(f)
            for n in self.names:
                if _matches(f, n):
                    t = tr.transform.translation
                    r = tr.transform.rotation
                    self.seen_fallback[n] = ((t.x, t.y, t.z),
                                            (r.x, r.y, r.z, r.w))
                    self.start.setdefault(n, (t.x, t.y))

    def report(self):
        log = self.get_logger().info
        log('=' * 70)
        log(f'{self.title} - is each object UPRIGHT, on its surface, where it started?')
        log('=' * 70)
        n_pass = 0
        for n in self.names:
            e = self.expect[n]
            got, src = self.pose_of(n)
            if got is None:
                log(f'  {n:<10} NO POSE RECEIVED'
                f'   [via {src}]')
                continue
            (x, y, z), q = got
            ax, ay, az_ = axis_from_quat(*q)
            # angle between the object's long axis and vertical, folded to 0-90
            tilt = math.acos(max(-1.0, min(1.0, abs(az_))))
            sx, sy = self.start.get(n, (x, y))
            drift = math.hypot(x - sx, y - sy)
            upright = tilt <= self.tilt_tol
            in_place = drift <= self.move_tol
            on_surface = abs(z - e['z']) <= self.z_tol
            ok = upright and in_place and on_surface
            n_pass += 1 if ok else 0
            why = ''
            if not upright:
                why += f' tilt {math.degrees(tilt):.0f} deg > ' \
                       f'{math.degrees(self.tilt_tol):.0f}'
            if not in_place:
                why += f' drifted {drift:.3f} m > {self.move_tol:.2f}'
            if not on_surface:
                why += f' at height {z:.3f} m, not on its surface ({e["z"]:.3f})'
            log(f'  {n:<10} {"PASS" if ok else "FAIL"}  '
                f'tilt {math.degrees(tilt):5.1f} deg from vertical, '
                f'drift {drift:.3f} m, height {z:.3f} m{why}')
        if any(self.pose_of(n)[0] is None for n in self.names):
            seen = sorted(self.seen_frames)
            log('  frames actually received on /model_poses (%d): %s'
                % (len(seen), ', '.join(seen[:12]) or 'NONE AT ALL'))
            log('  if that list is empty the bridge is not running; if it is'
                ' non-empty the name match is wrong')
        log('-' * 70)
        log(f'  RESULT: {n_pass}/{len(self.names)} objects standing upright')
        log('=' * 70)


def main(args=None):
    rclpy.init(args=args)
    node = VerifyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.report()
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
