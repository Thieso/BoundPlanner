import os
from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import TextSubstitution


def generate_launch_description():
    # Arguments
    robot_name = DeclareLaunchArgument(
        "robot_name", default_value=TextSubstitution(text="")
    )

    # urdf_file_name = "kinova_rviz.urdf"
    urdf_file_name = "iiwa_rviz.urdf"
    urdf = os.path.join("bound_planner/RobotModel", urdf_file_name)
    # Convert to urdf
    urdf = xacro.process_file(urdf)
    robot_desc = urdf.toxml()

    # Robot state publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            {
                "robot_description": robot_desc,
                "publish_frequency": 30.0,
            }
        ],
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="both",
        parameters=[{"rate": 30, "source_list": ["/set_joint_states"]}],
    )

    # RViz
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=[
            "-d",
            os.path.join("launch", "config.rviz"),
        ],
    )

    return LaunchDescription(
        [
            robot_name,
            robot_state_publisher,
            joint_state_publisher,
            rviz,
        ]
    )
