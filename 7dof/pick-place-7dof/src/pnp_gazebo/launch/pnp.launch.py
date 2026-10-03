"""Bring up section A: fixed-base bimanual pick and place.

    ros2 launch pnp_gazebo pnp.launch.py
    ros2 launch pnp_gazebo pnp.launch.py autostart:=false   # sim only

The controller spawners run STRICTLY ONE AFTER ANOTHER. Running them in
parallel makes them contend for the controller_manager lock, and the task can
then start while an arm controller is still inactive - which silently leaves one
arm out of the grasp.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, RegisterEventHandler,
                            TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

WORLD = 'pick_place'


def generate_launch_description():
    pkg_gz = get_package_share_directory('pnp_gazebo')
    pkg_desc = get_package_share_directory('pnp_description')
    pkg_beh = get_package_share_directory('pnp_behavior')

    autostart = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Run the pick-and-place task automatically')

    set_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', os.path.join(pkg_gz, 'models'))

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'),
            'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': '-r -v3 ' + os.path.join(pkg_gz, 'worlds',
                                                WORLD + '.sdf')}.items())

    robot_description = ParameterValue(
        Command(['xacro ', os.path.join(pkg_desc, 'urdf',
                                        'robot.urdf.xacro')]),
        value_type=str)

    rsp = Node(package='robot_state_publisher',
               executable='robot_state_publisher', output='screen',
               parameters=[{'robot_description': robot_description,
                            'use_sim_time': True}])

    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-topic', 'robot_description',
                            '-name', 'pnp_bimanual',
                            '-x', '0', '-y', '0', '-z', '0.005'])

    grasp_topics = []
    # must match 'order' in targets.yaml and 'objects' for the verify node
    for obj in ('bottle', 'can', 'box'):
        grasp_topics += [
            f'/grasp/{obj}/attach@std_msgs/msg/Empty]gz.msgs.Empty',
            f'/grasp/{obj}/detach@std_msgs/msg/Empty]gz.msgs.Empty',
            f'/grasp/{obj}/state@std_msgs/msg/String[gz.msgs.StringMsg',
            # GROUND TRUTH, per object. See the PosePublisher comment in each
            # object's model.sdf: the world-level pose stream below is bridged with a
            # well-formed spec and a loaded SceneBroadcaster and still delivered zero
            # messages, so this is an independent second path. One topic per object
            # means the verify node needs no entity-name matching at all, which is
            # where the earlier "NO POSE RECEIVED" ambiguity came from.
            f'/model/{obj}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose',
        ]

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/camera@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            # FORCE SENSING, which a position controller cannot do without. Every
            # lift-off in this project was the arm pressing on something rigid: the
            # wrist wrench sees the pressing and the base IMU sees the resulting tilt.
            # Together they drive the protective stop in send_pair.
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/ft/left@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '/ft/right@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            # FINGER CONTACT SENSORS. The grasp check used to
            # infer contact from how far short a finger stalled;
            # these report it directly, one per finger, so 'one
            # hand gripping and the other on air' is visible
            # rather than averaged away.
            # GAZEBO IGNORES <topic> FOR CONTACT SENSORS. It publishes on its own
            # generated name, which the console prints at startup:
            #   world/<world>/model/<model>/link/<link>/sensor/<sensor>/contact
            # The first attempt bridged /contact/<finger>, which nothing published,
            # so every finger read "0/2 touching" while the joint stall said +4.3 mm.
            # parameter_bridge names its ROS side after the Gazebo topic, so these are
            # remapped below to the short names the task node subscribes to.
            'world/pick_place/model/pnp_bimanual/link/left_finger_l_link/sensor/left_finger_l_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            'world/pick_place/model/pnp_bimanual/link/left_finger_r_link/sensor/left_finger_r_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            'world/pick_place/model/pnp_bimanual/link/right_finger_l_link/sensor/right_finger_l_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            'world/pick_place/model/pnp_bimanual/link/right_finger_r_link/sensor/right_finger_r_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            '/tf_gz@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            # THE ROBOT'S OWN GROUND TRUTH - checks only, see gazebo_plugins.xacro. It is
            # what LOCALISATION CHECK and CENTRING compare the wall fit against.
            '/model/pnp_bimanual/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose',
            # Ground-truth model poses -> the verification node scores the run
            # from these instead of anyone eyeballing the video.
            #
            # NOTE the topic: pose/info, NOT dynamic_pose/info. dynamic_pose only
            # carries entities Gazebo considers to be MOVING, so an object sitting
            # still - which is exactly the state we want to measure at the end of a
            # run - never appears on it. Every section reported "NO POSE RECEIVED"
            # for that reason. pose/info publishes every entity each update.
            f'/world/{WORLD}/pose/info@tf2_msgs/msg/TFMessage'
            '[gz.msgs.Pose_V',
        ] + grasp_topics,
        # NOTE: no remap for the pose topic. ros_gz_bridge names its ROS side
        # after the Gazebo topic and the remap did not take effect - the log shows
        # it publishing on /world/<world>/pose/info regardless, which is why the
        # verify node saw nothing on /model_poses for every run so far. The verify
        # node now subscribes to the real name instead.
        remappings=[('/tf_gz', '/tf'),
                ('/world/pick_place/model/pnp_bimanual/link/left_finger_l_link/sensor/left_finger_l_contact/contact', '/contact/left_finger_l'),
                ('/world/pick_place/model/pnp_bimanual/link/left_finger_r_link/sensor/left_finger_r_contact/contact', '/contact/left_finger_r'),
                ('/world/pick_place/model/pnp_bimanual/link/right_finger_l_link/sensor/right_finger_l_contact/contact', '/contact/right_finger_l'),
                ('/world/pick_place/model/pnp_bimanual/link/right_finger_r_link/sensor/right_finger_r_contact/contact', '/contact/right_finger_r')],
        parameters=[{'use_sim_time': True}])

    def spawner(name):
        return Node(package='controller_manager', executable='spawner',
                    output='screen', arguments=[name])

    jsb = spawner('joint_state_broadcaster')
    left = spawner('left_arm_controller')
    right = spawner('right_arm_controller')
    grip = spawner('gripper_controller')

    params = os.path.join(pkg_beh, 'config', 'targets.yaml')

    task = Node(package='pnp_behavior', executable='task_node',
                output='screen',
                parameters=[params, {'use_sim_time': True,
                                     'autostart': LaunchConfiguration(
                                         'autostart')}])
    verify = Node(package='pnp_behavior', executable='verify_node',
                  output='screen',
                  parameters=[params, {'use_sim_time': True,
                                     'pose_topic':
                                     f'/world/{WORLD}/pose/info'}])
    # verify_node listens on BOTH /model/<name>/pose (per-object, primary) and
    # pose_topic (world-level, fallback) and reports which one actually delivered.

    chain = [
        RegisterEventHandler(OnProcessExit(target_action=spawn,
                                           on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb,
                                           on_exit=[left])),
        RegisterEventHandler(OnProcessExit(target_action=left,
                                           on_exit=[right])),
        RegisterEventHandler(OnProcessExit(target_action=right,
                                           on_exit=[grip])),
        RegisterEventHandler(OnProcessExit(
            target_action=grip,
            on_exit=[TimerAction(period=3.0, actions=[verify, task])])),
    ]

    return LaunchDescription([autostart, set_path, gz_sim, rsp, spawn, bridge]
                             + chain)
