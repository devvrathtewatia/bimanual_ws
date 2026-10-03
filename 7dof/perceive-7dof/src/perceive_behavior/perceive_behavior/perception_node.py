"""Perception: estimate each object's pose from the robot's cameras.

SHARED FILE, between perceive_behavior and navigate_behavior (the 'sense' group in
SHARED_FILES.sha256). This node fills the GRASP TARGET CONTRACT that the reorient and
pick-place sections read from a config file:

    centre (x, y) in base_link   ...  where the object is
    axis angle                   ...  which way its long axis points
    upright / lying              ...  whether it needs standing up

published as JSON:  {"stamp": <image time, s>, "objects": {name: detection}}, on /targets for
the WORKSPACE camera (always) and on /targets_far for the MAST camera (only where a section
configures ``far_camera: true`` - navigate, which looks across the room).

WHY THE ROBOT "COULD NOT SEE" ANYTHING
--------------------------------------
The camera was fine. This node read ``lying_cz`` and ``upright_cz`` for each object and never
loaded them, so the first image with an object in it raised KeyError, the exception left
rclpy.spin(), the node died - and /targets was never published again. The task node then
timed out on every object and logged it as NOT SEEN. Now: every key is loaded in
detect.object_specs() and a test checks it; one bad frame is logged and skipped rather than
fatal; and the measurement itself was rebuilt (vision.py) because behind the crash it read a
standing bottle as lying, 59 mm off, pointing at the robot.

The STAMP is the image's own time, from the simulation clock, so the task node can refuse any
frame taken before the base stopped turning: a pose measured mid-turn is a pose in a frame
that no longer exists.
"""
import json

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from . import detect as dt
from . import vision as vz


class _Stream:
    """One camera: its model, its detection settings, its output topic and its frame count."""

    def __init__(self, label, camera, min_area, v_max, pub):
        self.label, self.camera, self.min_area, self.v_max = label, camera, min_area, v_max
        self.pub = pub
        self.frames = 0
        self.info_checked = False


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node',
                         automatically_declare_parameters_from_overrides=True)

        def p(name, default):
            try:
                v = self.get_parameter(name).value
                return default if v is None else v
            except Exception:  # noqa: BLE001
                return default

        self.names = list(p('objects', ['bottle', 'can', 'box']))
        self.specs = dt.object_specs(self.names, p)
        # THE WORKSPACE CAMERA, always: pose in base_link and intrinsics; must match
        # mast_sensors.xacro, which test_the_camera_config_is_the_urdf checks
        near = vz.CameraModel.from_fov(
            (float(p('cam_x', 0.05)), float(p('cam_y', 0.0)), float(p('cam_z', 0.62))),
            float(p('cam_pitch', 1.244)), float(p('horizontal_fov', 1.204)),
            int(p('image_width', 640)), int(p('image_height', 480)))
        self.streams = [self._stream('workspace camera', near, int(p('min_area', 250)), None,
                                     '/workspace_cam', '/workspace_cam_info', '/targets')]
        # THE MAST CAMERA, where a section looks across the room (navigate). Its image is cut
        # off from below by the arms in the look pose, at row far_v_max.
        if str(p('far_camera', False)).strip().lower() in ('true', '1', 'yes'):
            far = vz.CameraModel.from_fov(
                (float(p('far_cam_x', -0.075)), float(p('far_cam_y', 0.0)),
                 float(p('far_cam_z', 0.42))),
                float(p('far_cam_pitch', 0.2618)), float(p('far_horizontal_fov', 1.204)),
                int(p('far_image_width', 640)), int(p('far_image_height', 480)))
            self.streams.append(self._stream(
                'mast camera', far, int(p('far_min_area', 120)), int(p('far_v_max', 479)),
                '/camera', '/camera_info', '/targets_far'))
        self.get_logger().info('perception up (' + ', '.join(s.label for s in self.streams)
                               + '), objects: ' + ', '.join(self.specs))

    def _stream(self, label, camera, min_area, v_max, image_topic, info_topic, out_topic):
        st = _Stream(label, camera, min_area, v_max,
                     self.create_publisher(String, out_topic, 10))
        self.create_subscription(Image, image_topic, lambda m: self.on_image(m, st), 5)
        self.create_subscription(CameraInfo, info_topic, lambda m: self.on_info(m, st), 5)
        return st

    def on_info(self, msg, st):
        """Take Gazebo's intrinsics, and say so if they are not the configured camera."""
        cam = st.camera
        if msg.k[0] <= 1.0:
            return
        if not st.info_checked:
            st.info_checked = True
            if (abs(msg.k[0] - cam.fx) > 0.01 * cam.fx
                    or msg.width != cam.width or msg.height != cam.height):
                self.get_logger().warn(
                    f'{st.label}: camera_info fx {msg.k[0]:.1f} at {msg.width}x{msg.height} '
                    f'differs from the configured {cam.fx:.1f} at '
                    f'{cam.width}x{cam.height} - using camera_info')
        cam.fx, cam.fy = msg.k[0], msg.k[4]
        cam.cu, cam.cv = msg.k[2], msg.k[5]
        cam.width, cam.height = msg.width, msg.height

    def on_image(self, msg, st):
        if msg.encoding not in ('rgb8', 'bgr8'):
            self.get_logger().warn(f'unsupported encoding {msg.encoding}',
                                   throttle_duration_sec=10.0)
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        try:
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            out = dt.detect(img, msg.encoding, self.specs, st.camera, st.min_area,
                            v_max=st.v_max)
        except Exception as exc:  # noqa: BLE001
            # ONE BAD FRAME MUST NOT KILL PERCEPTION - that is how the node died last time
            self.get_logger().error(f'{st.label}: frame skipped: {exc!r}',
                                    throttle_duration_sec=5.0)
            return
        st.frames += 1
        if st.frames == 1:
            self.get_logger().info(f'first frame processed ({st.label}): ' + ', '.join(
                f'{n} {"seen" if d.get("found") else "not in view"}' for n, d in out.items()))
        st.pub.publish(String(data=json.dumps({'stamp': stamp, 'objects': out})))


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # Ctrl-C under ros2 launch can interrupt a take; harmless at shutdown
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
