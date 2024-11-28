import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray

from ..RobotModel import RobotModel
from ..utils import create_obj_spheres_msg


class RvizTools(Node):
    """Class to publish some useful RViz messages, the planned path and the
    reference path."""

    def __init__(self, n, t0):
        super().__init__("rviz_tools")
        self.t0 = t0
        # Path publisher for the cartesian end effector path
        self.ee_path_pub = self.create_publisher(Path, "/r1/ee_path", 1)
        self.ee_passed_path_pub = self.create_publisher(Path, "/r1/ee_passed_path", 1)
        self.ee_path_ref_pub = self.create_publisher(Path, "/r1/ee_ref_path", 1)
        self.ee_passed_path_ref_pub = self.create_publisher(
            Path, "/r1/ee_passed_ref_path", 1
        )
        self.error_pub = self.create_publisher(Marker, "/r1/error_plane", 1)
        self.via_point_pub = self.create_publisher(Path, "/r1/ee_full_ref_path", 1)
        self.ee_pose_pub = self.create_publisher(PoseArray, "/r1/ee_poses", 1)
        self.obj_pub = self.create_publisher(MarkerArray, "/r1/collision_obj", 1)

        # Marker publisher
        self.marker_pub = self.create_publisher(MarkerArray, "/r1/marker", 1)
        self.error_plane = Marker()
        self.table_marker = Marker()
        self.table2_marker = Marker()
        self.table7_marker = Marker()
        self.table8_marker = Marker()
        self.obstacle_marker = Marker()
        self.upper_wall_marker = Marker()
        self.lower_wall_marker = Marker()
        self.left_wall_marker = Marker()
        self.right_wall_marker = Marker()
        self.stick_marker = Marker()
        self.sponge_marker = Marker()
        self.cup_marker = Marker()
        self.cup_marker2 = Marker()
        self.cabinet = Marker()
        self.drawer = Marker()
        self.chandle = Marker()
        self.error_plane.header.frame_id = "r1/world"
        self.table_marker.header.frame_id = "r1/world"
        self.table2_marker.header.frame_id = "r1/world"
        self.table7_marker.header.frame_id = "r1/world"
        self.table8_marker.header.frame_id = "r1/world"
        self.obstacle_marker.header.frame_id = "r1/world"
        self.cup_marker.header.frame_id = "r1/world"
        self.cup_marker2.header.frame_id = "r1/world"
        self.upper_wall_marker.header.frame_id = "r1/world"
        self.lower_wall_marker.header.frame_id = "r1/world"
        self.left_wall_marker.header.frame_id = "r1/world"
        self.right_wall_marker.header.frame_id = "r1/world"
        self.stick_marker.header.frame_id = "r1/world"
        self.sponge_marker.header.frame_id = "r1/world"
        self.cabinet.header.frame_id = "r1/world"
        self.drawer.header.frame_id = "r1/world"
        self.chandle.header.frame_id = "r1/world"

        # Path messages
        self.path_msg = Path()
        self.passed_path_msg = Path()
        self.path_msg.header.frame_id = "r1/world"
        self.passed_path_msg.header.frame_id = "r1/world"

        # Reference path messages
        self.ref_path_msg = Path()
        self.passed_ref_path_msg = Path()
        self.ref_path_msg.header.frame_id = "r1/world"
        self.passed_ref_path_msg.header.frame_id = "r1/world"

        # Full reference path message
        self.via_points_msg = Path()
        self.via_points_msg.header.frame_id = "r1/world"

        # Via and current poses pub
        self.poses_msg = PoseArray()
        self.poses_msg.header.frame_id = "r1/world"

    def publish_coll_spheres(self, q):
        robot_model = RobotModel()
        obj_centers = [
            robot_model.fk_pos_j3(q),
            robot_model.fk_pos_j5(q),
            robot_model.fk_pos_j6(q),
            robot_model.fk_pos_j67(q),
            robot_model.fk_pos_j67(q, scal=2.5),
            robot_model.fk_pos_elbow(q),
            robot_model.fk_pos(q),
        ]
        obj_radii = [0.09, 0.09, 0.09, 0.06, 0.06, 0.11, 0.05]
        msg = create_obj_spheres_msg(
            self.get_clock().now().to_msg(), obj_centers, obj_radii, ns="robot"
        )
        self.obj_pub.publish(msg)

    def publish_error_plane(self, e_off, p_ref, dp_ref, p_lower, p_upper, b1, b2):
        self.error_plane.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.error_plane.lifetime = d

        # Namespace and ID
        self.error_plane.ns = "error_plane"
        self.error_plane.id = 5

        # Action and type
        self.error_plane.action = self.error_plane.ADD
        self.error_plane.type = self.error_plane.CUBE

        # Position
        p_mid = np.array(p_ref[:3] + e_off[0] * b1 + e_off[1] * b2).flatten()
        r = np.eye(3)
        r[:, 0] = np.array(dp_ref[:3]).flatten()
        r[:, 1] = np.array(b1).flatten()
        r[:, 2] = np.array(b2).flatten()
        r = R.from_matrix(r)
        orientation = r.as_quat()
        self.error_plane.pose.position.x = p_mid[0]
        self.error_plane.pose.position.y = p_mid[1]
        self.error_plane.pose.position.z = p_mid[2]
        self.error_plane.pose.orientation.x = orientation[0]
        self.error_plane.pose.orientation.y = orientation[1]
        self.error_plane.pose.orientation.z = orientation[2]
        self.error_plane.pose.orientation.w = orientation[3]

        # Size
        e_size = p_upper - p_lower
        e_size3 = e_size[0] * b1 + e_size[1] * b2
        e_size3 = r.as_matrix().T @ e_size3
        self.error_plane.scale.x = float(max(0.001, e_size3[0]))
        self.error_plane.scale.y = float(max(0.001, e_size3[1]))
        self.error_plane.scale.z = float(max(0.001, e_size3[2]))

        # Color
        self.error_plane.color.r = 186 / 255
        self.error_plane.color.g = 18 / 255
        self.error_plane.color.b = 43 / 255
        self.error_plane.color.a = 0.5
        self.error_pub.publish(self.error_plane)

    def publish_via_points(self, p_via, r_via):
        self.via_points_msg.poses = []
        for i in range(len(p_via)):
            pose = PoseStamped()
            pose.header.frame_id = "r1/world"
            pose.pose.position.x = p_via[i][0]
            pose.pose.position.y = p_via[i][1]
            pose.pose.position.z = p_via[i][2]
            quat = R.from_matrix(r_via[i]).as_quat()
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            self.via_points_msg.poses.append(pose)
        self.via_point_pub.publish(self.via_points_msg)

    def publish_poses(self, p_via, r_via, p_lie, p_ref):
        self.poses_msg.poses = []
        for i in range(len(p_via)):
            pose = Pose()
            pose.position.x = p_via[i][0]
            pose.position.y = p_via[i][1]
            pose.position.z = p_via[i][2]
            quat = R.from_matrix(r_via[i]).as_quat()
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            self.poses_msg.poses.append(pose)

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
            pose.header.frame_id = "r1/world"
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
            pose_ref.header.frame_id = "r1/world"
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

    def publish_marker(self, p, goal):
        self.create_table_box()
        self.create_table2_box()
        self.create_table7_box()
        self.create_table8_box()
        self.create_cup_box(goal)
        self.create_cup_box2()
        self.create_lower_wall_box()
        self.create_stick_box(p)
        self.create_sponge_box(p)
        self.create_upper_wall_box()
        self.create_left_wall_box()
        self.create_right_wall_box()
        self.create_obstacle_box()
        self.create_cabinet()
        self.create_drawer()
        self.create_handle()
        marker_list = MarkerArray()
        marker_list.markers.append(self.table_marker)
        marker_list.markers.append(self.table2_marker)
        marker_list.markers.append(self.table7_marker)
        marker_list.markers.append(self.table8_marker)
        marker_list.markers.append(self.cup_marker)
        marker_list.markers.append(self.cup_marker2)
        marker_list.markers.append(self.obstacle_marker)
        marker_list.markers.append(self.lower_wall_marker)
        marker_list.markers.append(self.upper_wall_marker)
        marker_list.markers.append(self.left_wall_marker)
        marker_list.markers.append(self.right_wall_marker)
        marker_list.markers.append(self.stick_marker)
        marker_list.markers.append(self.sponge_marker)
        marker_list.markers.append(self.cabinet)
        marker_list.markers.append(self.drawer)
        marker_list.markers.append(self.chandle)
        self.marker_pub.publish(marker_list)

    def create_table_box(self):
        self.table_marker.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.table_marker.lifetime = d

        # Namespace and ID
        self.table_marker.ns = "table"
        self.table_marker.id = 1

        # Action and type
        self.table_marker.action = self.table_marker.ADD
        self.table_marker.type = self.table_marker.CUBE

        # Position
        self.table_marker.pose.position.x = 0.65
        self.table_marker.pose.position.y = 0.0
        self.table_marker.pose.position.z = -0.025
        self.table_marker.pose.orientation.w = 1.0

        # Size
        self.table_marker.scale.x = 0.4
        self.table_marker.scale.y = 0.7
        self.table_marker.scale.z = 0.1

        # Color
        self.table_marker.color.r = 0.0
        self.table_marker.color.g = 190 / 255
        self.table_marker.color.b = 65 / 255
        self.table_marker.color.a = 1.0

    def create_handle(self):
        self.chandle.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.chandle.lifetime = d

        # Namespace and ID
        self.chandle.ns = "chandle"
        self.chandle.id = 760

        # Action and type
        self.chandle.action = self.chandle.ADD
        self.chandle.type = self.chandle.CUBE

        # Position
        self.chandle.pose.position.x = 0.63
        self.chandle.pose.position.y = 0.26
        self.chandle.pose.position.z = 0.44
        # self.chandle.pose.position.x = 0.48
        # self.chandle.pose.position.y = -0.03
        # self.chandle.pose.position.z = 0.44
        r_goal = R.from_euler("xyz", [0, 0, -2.14])
        r_goal = r_goal.as_quat()
        self.chandle.pose.orientation.x = r_goal[0]
        self.chandle.pose.orientation.y = r_goal[1]
        self.chandle.pose.orientation.z = r_goal[2]
        self.chandle.pose.orientation.w = r_goal[3]

        # Size
        self.chandle.scale.x = 0.02
        self.chandle.scale.y = 0.15
        self.chandle.scale.z = 0.02

        # Color
        self.chandle.color.r = 0 / 255
        self.chandle.color.g = 190 / 255
        self.chandle.color.b = 65 / 255
        self.chandle.color.a = 1.0

    def create_drawer(self):
        self.drawer.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.drawer.lifetime = d

        # Namespace and ID
        self.drawer.ns = "drawer"
        self.drawer.id = 760

        # Action and type
        self.drawer.action = self.drawer.ADD
        self.drawer.type = self.drawer.CUBE

        # Position
        self.drawer.pose.position.x = 0.8
        self.drawer.pose.position.y = 0.5
        self.drawer.pose.position.z = 0.45
        # self.drawer.pose.position.x = 0.61
        # self.drawer.pose.position.y = 0.21
        # self.drawer.pose.position.z = 0.45
        r_goal = R.from_euler("xyz", [0, 0, -2.14])
        r_goal = r_goal.as_quat()
        self.drawer.pose.orientation.x = r_goal[0]
        self.drawer.pose.orientation.y = r_goal[1]
        self.drawer.pose.orientation.z = r_goal[2]
        self.drawer.pose.orientation.w = r_goal[3]

        # Size
        self.drawer.scale.x = 0.41
        self.drawer.scale.y = 0.39
        self.drawer.scale.z = 0.2

        # Color
        self.drawer.color.r = 186 / 255
        self.drawer.color.g = 18 / 255
        self.drawer.color.b = 43 / 255
        self.drawer.color.a = 0.8

    def create_cabinet(self):
        self.cabinet.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.cabinet.lifetime = d

        # Namespace and ID
        self.cabinet.ns = "cabinet"
        self.cabinet.id = 761

        # Action and type
        self.cabinet.action = self.cabinet.ADD
        self.cabinet.type = self.cabinet.CUBE

        # Position
        self.cabinet.pose.position.x = 0.8
        self.cabinet.pose.position.y = 0.5
        self.cabinet.pose.position.z = 0.28
        r_goal = R.from_euler("xyz", [0, 0, -2.14])
        r_goal = r_goal.as_quat()
        self.cabinet.pose.orientation.x = r_goal[0]
        self.cabinet.pose.orientation.y = r_goal[1]
        self.cabinet.pose.orientation.z = r_goal[2]
        self.cabinet.pose.orientation.w = r_goal[3]

        # Size
        self.cabinet.scale.x = 0.4
        self.cabinet.scale.y = 0.4
        self.cabinet.scale.z = 0.56

        # Color
        self.cabinet.color.r = 0.0
        self.cabinet.color.g = 190 / 255
        self.cabinet.color.b = 65 / 255
        self.cabinet.color.a = 1.0

    def create_table2_box(self):
        self.table2_marker.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.table2_marker.lifetime = d

        # Namespace and ID
        self.table2_marker.ns = "table2"
        self.table2_marker.id = 364

        # Action and type
        self.table2_marker.action = self.table2_marker.ADD
        self.table2_marker.type = self.table2_marker.CUBE

        # Position
        self.table2_marker.pose.position.x = -0.105
        self.table2_marker.pose.position.y = -0.5
        self.table2_marker.pose.position.z = 0.09
        self.table2_marker.pose.orientation.w = 1.0

        # Size
        self.table2_marker.scale.x = 0.22
        self.table2_marker.scale.y = 0.25
        self.table2_marker.scale.z = 0.18

        # Color
        self.table2_marker.color.r = 0.0
        self.table2_marker.color.g = 190 / 255
        self.table2_marker.color.b = 65 / 255
        self.table2_marker.color.a = 1.0

    def create_table8_box(self):
        self.table8_marker.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.table8_marker.lifetime = d

        # Namespace and ID
        self.table8_marker.ns = "table8"
        self.table8_marker.id = 568

        # Action and type
        self.table8_marker.action = self.table8_marker.ADD
        self.table8_marker.type = self.table8_marker.CUBE

        # Position
        self.table8_marker.pose.position.x = 0.55
        self.table8_marker.pose.position.y = 0.0
        self.table8_marker.pose.position.z = 0.02
        self.table8_marker.pose.orientation.w = 1.0

        # Size
        self.table8_marker.scale.x = 0.3
        self.table8_marker.scale.y = 1.1
        self.table8_marker.scale.z = 0.1

        # Color
        self.table8_marker.color.r = 0.0
        self.table8_marker.color.g = 190 / 255
        self.table8_marker.color.b = 65 / 255
        self.table8_marker.color.a = 1.0

    def create_table7_box(self):
        self.table7_marker.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.table7_marker.lifetime = d

        # Namespace and ID
        self.table7_marker.ns = "table7"
        self.table7_marker.id = 368

        # Action and type
        self.table7_marker.action = self.table7_marker.ADD
        self.table7_marker.type = self.table7_marker.CUBE

        # Position
        self.table7_marker.pose.position.x = -1.0
        self.table7_marker.pose.position.y = 0.0
        self.table7_marker.pose.position.z = 0.2
        self.table7_marker.pose.orientation.w = 1.0

        # Size
        self.table7_marker.scale.x = 0.25
        self.table7_marker.scale.y = 0.25
        self.table7_marker.scale.z = 0.4

        # Color
        self.table7_marker.color.r = 0.0
        self.table7_marker.color.g = 190 / 255
        self.table7_marker.color.b = 65 / 255
        self.table7_marker.color.a = 1.0

    def create_obstacle_box(self):
        self.obstacle_marker.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.obstacle_marker.lifetime = d

        # Namespace and ID
        self.obstacle_marker.ns = "obstacle"
        self.obstacle_marker.id = 4

        # Action and type
        self.obstacle_marker.action = self.obstacle_marker.ADD
        self.obstacle_marker.type = self.obstacle_marker.CUBE

        # Position
        self.obstacle_marker.pose.position.x = 0.65
        self.obstacle_marker.pose.position.y = 0.0
        self.obstacle_marker.pose.position.z = 0.175
        self.obstacle_marker.pose.orientation.w = 1.0

        # Size
        self.obstacle_marker.scale.x = 0.4
        self.obstacle_marker.scale.y = 0.08
        self.obstacle_marker.scale.z = 0.3

        # Color
        self.obstacle_marker.color.r = 186 / 255
        self.obstacle_marker.color.g = 18 / 255
        self.obstacle_marker.color.b = 43 / 255
        self.obstacle_marker.color.a = 1.0

    def create_cup_box(self, goal):
        self.cup_marker.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.cup_marker.lifetime = d

        # Namespace and ID
        self.cup_marker.ns = "cup"
        self.cup_marker.id = 2

        # Action and type
        self.cup_marker.action = self.cup_marker.ADD
        self.cup_marker.type = self.cup_marker.CUBE

        # Position
        self.cup_marker.pose.position.x = goal[0]
        self.cup_marker.pose.position.y = goal[1]
        self.cup_marker.pose.position.z = goal[2] - 0.025
        r_adapt = R.from_euler("xyz", [0, -90, 0], degrees=True)
        r_goal = R.from_rotvec(goal[3:])
        r_goal = r_goal * r_adapt
        r_goal = r_goal.as_quat()
        self.cup_marker.pose.orientation.x = r_goal[0]
        self.cup_marker.pose.orientation.y = r_goal[1]
        self.cup_marker.pose.orientation.z = r_goal[2]
        self.cup_marker.pose.orientation.w = r_goal[3]

        # Size
        self.cup_marker.scale.x = 0.1
        self.cup_marker.scale.y = 0.05
        self.cup_marker.scale.z = 0.15

        # Color
        self.cup_marker.color.r = 0.0
        self.cup_marker.color.g = 102 / 255
        self.cup_marker.color.b = 153 / 255
        self.cup_marker.color.a = 0.8

    def create_cup_box2(self):
        self.cup_marker2.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.cup_marker2.lifetime = d

        # Namespace and ID
        self.cup_marker2.ns = "cup2"
        self.cup_marker2.id = 3

        # Action and type
        self.cup_marker2.action = self.cup_marker2.ADD
        self.cup_marker2.type = self.cup_marker2.CYLINDER

        # Position
        self.cup_marker2.pose.position.x = 0.5
        self.cup_marker2.pose.position.y = 0.4
        self.cup_marker2.pose.position.z = 0.45
        self.cup_marker2.pose.orientation.w = 1.0

        # Size
        self.cup_marker2.scale.x = 0.05
        self.cup_marker2.scale.y = 0.05
        self.cup_marker2.scale.z = 0.15

        # Color
        self.cup_marker2.color.r = 0.0
        self.cup_marker2.color.g = 102 / 255
        self.cup_marker2.color.b = 153 / 255
        self.cup_marker2.color.a = 1.0

    def create_upper_wall_box(self):
        self.upper_wall_marker.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.upper_wall_marker.lifetime = d

        # Namespace and ID
        self.upper_wall_marker.ns = "upper wall"
        self.upper_wall_marker.id = 9

        # Action and type
        self.upper_wall_marker.action = self.upper_wall_marker.ADD
        self.upper_wall_marker.type = self.upper_wall_marker.CUBE

        self.upper_wall_marker.pose.position.x = 0.3
        self.upper_wall_marker.pose.position.y = -0.4
        self.upper_wall_marker.pose.position.z = 1.03
        orientation = R.from_euler("XYZ", [0, 0, np.pi / 3]).as_quat()
        self.upper_wall_marker.pose.orientation.x = orientation[0]
        self.upper_wall_marker.pose.orientation.y = orientation[1]
        self.upper_wall_marker.pose.orientation.z = orientation[2]
        self.upper_wall_marker.pose.orientation.w = orientation[3]

        # Size
        self.upper_wall_marker.scale.x = 0.05
        self.upper_wall_marker.scale.y = 0.25
        self.upper_wall_marker.scale.z = 0.25

        # Color
        self.upper_wall_marker.color.r = 186 / 255
        self.upper_wall_marker.color.g = 18 / 255
        self.upper_wall_marker.color.b = 43 / 255
        self.upper_wall_marker.color.a = 0.8

    def create_lower_wall_box(self):
        self.lower_wall_marker.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.lower_wall_marker.lifetime = d

        # Namespace and ID
        self.lower_wall_marker.ns = "lower wall"
        self.lower_wall_marker.id = 10

        # Action and type
        self.lower_wall_marker.action = self.lower_wall_marker.ADD
        self.lower_wall_marker.type = self.lower_wall_marker.CUBE

        # Position
        self.lower_wall_marker.pose.position.x = 0.30
        self.lower_wall_marker.pose.position.y = -0.40
        self.lower_wall_marker.pose.position.z = 0.375
        orientation = R.from_euler("XYZ", [0, 0, np.pi / 3]).as_quat()
        self.lower_wall_marker.pose.orientation.x = orientation[0]
        self.lower_wall_marker.pose.orientation.y = orientation[1]
        self.lower_wall_marker.pose.orientation.z = orientation[2]
        self.lower_wall_marker.pose.orientation.w = orientation[3]

        # Size
        self.lower_wall_marker.scale.x = 0.05
        self.lower_wall_marker.scale.y = 0.25
        self.lower_wall_marker.scale.z = 0.75

        # Color
        self.lower_wall_marker.color.r = 186 / 255
        self.lower_wall_marker.color.g = 18 / 255
        self.lower_wall_marker.color.b = 43 / 255
        self.lower_wall_marker.color.a = 0.8

    def create_left_wall_box(self):
        self.left_wall_marker.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.left_wall_marker.lifetime = d

        # Namespace and ID
        self.left_wall_marker.ns = "left wall"
        self.left_wall_marker.id = 115

        # Action and type
        self.left_wall_marker.action = self.left_wall_marker.ADD
        self.left_wall_marker.type = self.left_wall_marker.CUBE

        # Position
        self.left_wall_marker.pose.position.x = 0.03
        self.left_wall_marker.pose.position.y = -0.5
        self.left_wall_marker.pose.position.z = 0.2
        self.left_wall_marker.pose.orientation.w = 1.0

        # Size
        self.left_wall_marker.scale.x = 0.05
        self.left_wall_marker.scale.y = 0.25
        self.left_wall_marker.scale.z = 0.4

        # Color
        self.left_wall_marker.color.r = 186 / 255
        self.left_wall_marker.color.g = 18 / 255
        self.left_wall_marker.color.b = 43 / 255
        self.left_wall_marker.color.a = 0.8

    def create_right_wall_box(self):
        self.right_wall_marker.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.right_wall_marker.lifetime = d

        # Namespace and ID
        self.right_wall_marker.ns = "right wall"
        self.right_wall_marker.id = 222

        # Action and type
        self.right_wall_marker.action = self.right_wall_marker.ADD
        self.right_wall_marker.type = self.right_wall_marker.CUBE

        # Position
        self.right_wall_marker.pose.position.x = -0.24
        self.right_wall_marker.pose.position.y = -0.5
        self.right_wall_marker.pose.position.z = 0.2
        self.right_wall_marker.pose.orientation.w = 1.0

        # Size
        self.right_wall_marker.scale.x = 0.05
        self.right_wall_marker.scale.y = 0.25
        self.right_wall_marker.scale.z = 0.4

        # Color
        self.right_wall_marker.color.r = 186 / 255
        self.right_wall_marker.color.g = 18 / 255
        self.right_wall_marker.color.b = 43 / 255
        self.right_wall_marker.color.a = 0.8

    def create_stick_box(self, p):
        self.stick_marker.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.stick_marker.lifetime = d

        # Namespace and ID
        self.stick_marker.ns = "stick"
        self.stick_marker.id = 11

        # Action and type
        self.stick_marker.action = self.stick_marker.ADD
        self.stick_marker.type = self.stick_marker.CYLINDER

        # Position
        self.stick_marker.pose.position.x = p[0]
        self.stick_marker.pose.position.y = p[1]
        self.stick_marker.pose.position.z = p[2]
        orientation = R.from_rotvec(p[3:])
        orientation = orientation * R.from_euler("XYZ", [0, np.pi / 2, 0])
        orientation = orientation.as_quat()
        self.stick_marker.pose.orientation.x = orientation[0]
        self.stick_marker.pose.orientation.y = orientation[1]
        self.stick_marker.pose.orientation.z = orientation[2]
        self.stick_marker.pose.orientation.w = orientation[3]

        # Size
        self.stick_marker.scale.x = 0.05
        self.stick_marker.scale.y = 0.05
        self.stick_marker.scale.z = 0.35
        # self.stick_marker.scale.x = 0.07
        # self.stick_marker.scale.y = 0.07
        # self.stick_marker.scale.z = 0.12

        # Color
        self.stick_marker.color.r = 0.0
        self.stick_marker.color.g = 102 / 255
        self.stick_marker.color.b = 153 / 255
        self.stick_marker.color.a = 1.0

    def create_sponge_box(self, p):
        self.sponge_marker.header.stamp = self.get_clock().now().to_msg()
        d = rclpy.time.Duration(seconds=10000).to_msg()
        self.sponge_marker.lifetime = d

        # Namespace and ID
        self.sponge_marker.ns = "sponge"
        self.sponge_marker.id = 843

        # Action and type
        self.sponge_marker.action = self.sponge_marker.ADD
        self.sponge_marker.type = self.sponge_marker.CUBE

        # Position
        self.sponge_marker.pose.position.x = p[0]
        self.sponge_marker.pose.position.y = p[1]
        self.sponge_marker.pose.position.z = p[2] - 0.01
        orientation = R.from_rotvec(p[3:])
        orientation = orientation * R.from_euler("XYZ", [0, np.pi / 2, 0])
        orientation = orientation.as_quat()
        self.sponge_marker.pose.orientation.x = orientation[0]
        self.sponge_marker.pose.orientation.y = orientation[1]
        self.sponge_marker.pose.orientation.z = orientation[2]
        self.sponge_marker.pose.orientation.w = orientation[3]

        # Size
        self.sponge_marker.scale.x = 0.05
        self.sponge_marker.scale.y = 0.15
        self.sponge_marker.scale.z = 0.1
        # self.sponge_marker.scale.x = 0.07
        # self.sponge_marker.scale.y = 0.07
        # self.sponge_marker.scale.z = 0.12

        # Color
        self.sponge_marker.color.r = 0.0
        self.sponge_marker.color.g = 102 / 255
        self.sponge_marker.color.b = 153 / 255
        self.sponge_marker.color.a = 1.0


def main():
    rclpy.init()
    node = RvizTools()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
