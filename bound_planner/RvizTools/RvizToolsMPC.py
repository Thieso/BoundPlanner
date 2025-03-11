import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from ..RobotModel import RobotModel


class RvizToolsMPC(Node):
    """Class to publish some useful RViz messages, the planned path and the
    reference path."""

    def __init__(self):
        super().__init__("rviz_tools_mpc")
        self.ee_path_pub = self.create_publisher(Path, "/boundmpc/ee_path", 1)
        self.ee_passed_path_pub = self.create_publisher(
            Path, "/boundmpc/ee_passed_path", 1
        )
        self.ee_path_ref_pub = self.create_publisher(Path, "/boundmpc/ee_ref_path", 1)
        self.ee_passed_path_ref_pub = self.create_publisher(
            Path, "/boundmpc/ee_passed_ref_path", 1
        )
        self.ee_pose_pub = self.create_publisher(PoseArray, "/boundmpc/ee_poses", 1)
        self.collision_sphere_pub = self.create_publisher(
            MarkerArray, "/boundmpc/collision_spheres", 1
        )
        self.robot_pub = self.create_publisher(JointState, "/set_joint_states", 0)
        self.robot_model = RobotModel()

        # Path messages
        self.path_msg = Path()
        self.passed_path_msg = Path()
        self.path_msg.header.frame_id = "world"
        self.passed_path_msg.header.frame_id = "world"

        # Reference path messages
        self.ref_path_msg = Path()
        self.passed_ref_path_msg = Path()
        self.ref_path_msg.header.frame_id = "world"
        self.passed_ref_path_msg.header.frame_id = "world"

        # Full reference path message
        self.via_points_msg = Path()
        self.via_points_msg.header.frame_id = "world"

        # Current poses pub
        self.poses_msg = PoseArray()
        self.poses_msg.header.frame_id = "world"

    def publish_poses(self, p_lie, p_ref):
        self.poses_msg.poses = []

        pose = Pose()
        pose.position.x = p_lie[0]
        pose.position.y = p_lie[1]
        pose.position.z = p_lie[2]
        quat = R.from_rotvec(p_lie[3:]).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        self.poses_msg.poses.append(pose)

        pose = Pose()
        pose.position.x = p_ref[0]
        pose.position.y = p_ref[1]
        pose.position.z = p_ref[2]
        quat = R.from_rotvec(p_ref[3:]).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        self.poses_msg.poses.append(pose)

        self.ee_pose_pub.publish(self.poses_msg)

    def publish_path(self, t_current, p_traj, p_ref_traj):
        """Visualize the planned path in Rviz by publishing a path message
        containing the cartesian end effector path.
        """
        self.path_msg.header.stamp = rclpy.time.Time(seconds=t_current).to_msg()
        self.path_msg.poses = []
        self.ref_path_msg.poses = []
        for j in range(p_traj.shape[1]):
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.pose.position.x = p_traj[0, j]
            pose.pose.position.y = p_traj[1, j]
            pose.pose.position.z = p_traj[2, j]
            quat = R.from_rotvec(p_traj[3:, j]).as_quat()
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            self.path_msg.poses.append(pose)
            if j == 0:
                self.passed_path_msg.poses.append(pose)

            pose_ref = PoseStamped()
            pose_ref.header.frame_id = "world"
            pose_ref.pose.position.x = p_ref_traj[0, j]
            pose_ref.pose.position.y = p_ref_traj[1, j]
            pose_ref.pose.position.z = p_ref_traj[2, j]
            ref_quat = R.from_rotvec(p_ref_traj[3:, j]).as_quat()
            pose_ref.pose.orientation.x = ref_quat[0]
            pose_ref.pose.orientation.y = ref_quat[1]
            pose_ref.pose.orientation.z = ref_quat[2]
            pose_ref.pose.orientation.w = ref_quat[3]
            self.ref_path_msg.poses.append(pose_ref)
            if j == 0:
                self.passed_ref_path_msg.poses.append(pose_ref)
        self.ee_path_pub.publish(self.path_msg)
        self.ee_passed_path_pub.publish(self.passed_path_msg)
        self.ee_path_ref_pub.publish(self.ref_path_msg)
        self.ee_passed_path_ref_pub.publish(self.passed_ref_path_msg)

    def move_robot_kinematic(self, t_ros, q_new):
        """Move the robot kinematically by just publishing the new joint state
        for Rviz (Only for visualization purposes).
        """
        robot_state_msg = JointState()
        robot_link_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]
        robot_state_msg.name = robot_link_names
        robot_state_msg.header.frame_id = "body"
        robot_state_msg.header.stamp = rclpy.time.Time(seconds=t_ros).to_msg()
        robot_state_msg.position = q_new.tolist()
        robot_state_msg.velocity = np.zeros((7,)).tolist()
        self.robot_pub.publish(robot_state_msg)

    def publish_coll_spheres(self, q):
        obj_centers = [self.robot_model.fk_pos_col(q, i) for i in range(7)]
        obj_radii = self.robot_model.col_joint_sizes
        msg = self.create_obj_spheres_msg(
            self.get_clock().now().to_msg(), obj_centers, obj_radii, ns="robot"
        )
        self.collision_sphere_pub.publish(msg)

    def create_obj_spheres_msg(self, t, centers, radii, ns="sphere", alpha=0.5):
        msg = MarkerArray()
        for i, (c, r) in enumerate(zip(centers, radii)):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = t
            d = rclpy.time.Duration(seconds=10000).to_msg()
            marker.lifetime = d
            marker.ns = f"{ns} {i}"
            marker.id = i
            marker.action = marker.ADD
            marker.type = marker.SPHERE
            marker.pose.position.x = c[0]
            marker.pose.position.y = c[1]
            marker.pose.position.z = c[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = r * 2
            marker.scale.y = r * 2
            marker.scale.z = r * 2
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = alpha
            msg.markers.append(marker)
        return msg
