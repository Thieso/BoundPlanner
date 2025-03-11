import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray

from ..utils import compute_polytope_vertices


class RvizTools(Node):
    """Class to publish some useful RViz messages, the planned path and the
    reference path."""

    def __init__(self):
        super().__init__("rviz_tools")
        # Publisher of reference path
        self.via_point_pub = self.create_publisher(Path, "/bound_planner/ref_path", 1)
        self.set_pub = self.create_publisher(MarkerArray, "/bound_planner/sets", 1)
        self.rviz_marker_msg = MarkerArray()

        # Set message
        self.set_msg = MarkerArray()

        # Path messages
        self.path_msg = Path()
        self.passed_path_msg = Path()
        self.path_msg.header.frame_id = "world"
        self.passed_path_msg.header.frame_id = "world"

        # Full reference path message
        self.via_points_msg = Path()
        self.via_points_msg.header.frame_id = "world"

    def publish_via_points(self, p_via, r_via):
        self.via_points_msg.poses = []
        for i in range(len(p_via)):
            pose = PoseStamped()
            pose.header.frame_id = "world"
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

    def publish_sets(self):
        self.set_pub.publish(self.rviz_marker_msg)

    def delete_sets(self):
        self.rviz_marker_msg = MarkerArray()
        msg = Marker()
        msg.header.frame_id = "world"
        msg.action = Marker.DELETEALL
        self.rviz_marker_msg.markers.append(msg)
        self.set_pub.publish(self.rviz_marker_msg)

    def add_sets(self, sets, name="Set", color=[0, 1, 0], alpha=0.1):
        for i, set_via in enumerate(sets):
            self.rviz_marker_msg.markers.append(
                self.create_marker_msg(set_via, f"{name} {i}", color, alpha=alpha)
            )
        self.set_pub.publish(self.rviz_marker_msg)

    def create_marker_msg(self, convex_set, i, color, alpha):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.action = Marker.ADD
        marker.type = Marker.TRIANGLE_LIST
        marker.ns = f"Set {i}"
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(alpha)

        try:
            points = compute_polytope_vertices(convex_set[0], convex_set[1])
            hull = ConvexHull(points)
            faces = hull.simplices
            for face in faces:
                p1, p2, p3 = np.array(points)[face]
                marker.points.append(self.create_point(p1))
                marker.points.append(self.create_point(p2))
                marker.points.append(self.create_point(p3))
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        except RuntimeError:
            print("(Visualization) Could not compute polytope for RViz")
        return marker

    def create_point(self, point):
        p = Point()
        p.x, p.y, p.z = point
        return p


def main():
    rclpy.init()
    node = RvizTools()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
