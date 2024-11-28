import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import TextSubstitution


def generate_launch_description():
    pkg_bmpc = get_package_share_directory('bound_planner')

    # Arguments
    robot_name = DeclareLaunchArgument(
        "robot_name", default_value=TextSubstitution(text="r1")
    )
    prefix = DeclareLaunchArgument(
        "prefix", default_value=TextSubstitution(text="r1/")
    )
    position = DeclareLaunchArgument(
        "position", default_value=TextSubstitution(text="-x 0.0 -y 0.0 -z 0.1")
    )

    urdf_file_name = 'main.xacro'
    urdf = os.path.join(
        pkg_bmpc,
        'urdf',
        urdf_file_name)
    # Convert to urdf
    urdf = xacro.process_file(urdf)
    robot_desc = urdf.toxml()

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[{
            'robot_description': robot_desc,
            'publish_frequency': 30.0,
            'frame_prefix': LaunchConfiguration('prefix')
        }],
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='both',
        parameters=[{'rate': 30,
                     'source_list': ['/set_joint_states']}],
    )

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory('bound_planner'),
                                      'config', 'config.rviz')],
    )

    tf2_opti2base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        # arguments=['0', '0.31', '0', '0', '0', '0', 'map', 'r1/world']
        arguments=['0.13', '-0.205', '0.01', '0', '0', '0', 'map', 'r1/world']
    )

    boundmpc = Node(
        package='bound_planner',
        executable='bound_mpc',
    )

    experimentrunner = Node(
        package='bound_planner',
        executable='run_experiment1',
    )

    return LaunchDescription([
        robot_name,
        prefix,
        position,
        robot_state_publisher,
        joint_state_publisher,
        rviz,
        tf2_opti2base,
        boundmpc,
        experimentrunner,
    ])

