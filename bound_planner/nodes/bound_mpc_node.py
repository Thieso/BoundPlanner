import threading
import time

import numpy as np
import rclpy
from bound_mpc_msg.msg import MPCData, Vector
from bound_mpc_msg.srv import MPCParams, Trajectory
from geometry_msgs.msg import Point
from pypoman import compute_polytope_vertices
from rclpy.node import Node
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from bound_planner.BoundMPC.BoundMPC import BoundMPC
from bound_planner.RobotModel import RobotModel
from bound_planner.Rviz_Tools import RvizTools
from bound_planner.utils import integrate_joint, move_robot_kinematic

lock = threading.Lock()


class MPCNode(Node):
    def __init__(self):
        super().__init__("mpc_node")
        self.mpc_srv = self.create_service(
            MPCParams, "/mpc/set_params", self.callback_params
        )
        self.traj_srv = self.create_service(
            Trajectory, "/mpc/set_trajectory", self.callback_trajectory
        )
        self.data_pub = self.create_publisher(MPCData, "/mpc/mpc_data", 1)
        self.rviz_pub = self.create_publisher(MarkerArray, "/r1/obs_marker", 1)

        self.fails = []
        self.t_mpc = 0.0
        self.t_overhead = 0.0
        self.t_switch = [0.0]
        self.phi_switch = [0.0]
        self.phi_bias = 0.0
        self.t_bias = 0.5
        self.t_overtime = 0.0
        self.log = True
        self.robot_model = RobotModel()

        # Robot state publisher
        self.robot_pub = self.create_publisher(JointState, "/set_joint_states", 0)

        self.init = False
        self.params_received = False
        print("(MPCNode) Waiting for initial data ...")
        while True:
            rclpy.spin_once(self)
            if self.init and self.params_received:
                break
        if self.params.experiment:
            robot_ip = "192.168.1.3"
        else:
            robot_ip = "127.0.0.1"

        self.reset()

    def reset(self):
        lock.acquire()
        print("(MPCNode) Resetting MPC")
        self.params_received = False
        self.simulate = self.params.simulate
        self.p = self.p0
        # Init MPC
        self.mpc = BoundMPC(
            self.p_via,
            self.r_via,
            self.bp1,
            self.br1,
            self.e_p_start,
            self.e_p_end,
            self.e_p_mid,
            self.e_r_start,
            self.e_r_end,
            self.e_r_mid,
            self.a_sets,
            self.b_sets,
            self.obstacles,
            p0=self.p0,
            params=self.params,
        )
        start_ik = time.time()
        qd = self.robot_model.inverse_kinematics(
            self.p_via[-1], self.r_via[-1], self.q0
        )
        stop_ik = time.time()
        print(f"(MPCNode) IK took {stop_ik - start_ik:.3f}s")
        self.mpc.set_desired_joint_config(qd)

        self.mpc.set_collision_objects(np.array(self.obj_centers), self.obj_radii)

        t_rviz = 0.0
        self.rviz_tools = RvizTools(self.mpc.N, t_rviz)

        # Initial values
        self.q = self.q0
        self.qf = self.q0
        self.dq = np.zeros(7)
        self.ddq = np.zeros(7)
        self.jerk = np.zeros(7)
        self.p_lie = self.p0
        self.v = np.zeros(6)
        self.t_current = 0.0
        self.t0 = np.copy(self.t_current)
        lock.release()

    def callback_params(self, params, response):
        print("(MPCNode) Received Parameters")
        self.learning_based = params.learning_based
        self.params = params
        self.params_received = True
        return response

    def callback_trajectory(self, msg, response):
        print("(MPCNode) Received Trajectory")
        self.p_via = []
        self.r_via = []
        self.bp1 = []
        self.br1 = []
        self.e_p_start = []
        self.e_p_end = []
        self.e_p_mid = []
        self.e_r_start = []
        self.e_r_end = []
        self.e_r_mid = []
        self.a_sets = []
        self.b_sets = []
        for i in range(len(msg.p_via)):
            self.p_via.append(np.array(msg.p_via[i].x))
            r_rotvec = np.array(msg.r_via[i].x)
            self.r_via.append(R.from_rotvec(r_rotvec).as_matrix())
        for i in range(len(msg.p_via) - 1):
            self.bp1.append(np.array(msg.bp1[i].x))
            self.br1.append(np.array(msg.br1[i].x))
            self.e_p_start.append(np.array(msg.e_p_start[i].x))
            self.e_p_end.append(np.array(msg.e_p_end[i].x))
            self.e_p_mid.append(np.array(msg.e_p_mid[i].x))
            self.e_r_start.append(np.array(msg.e_r_start[i].x))
            self.e_r_end.append(np.array(msg.e_r_end[i].x))
            self.e_r_mid.append(np.array(msg.e_r_mid[i].x))
        for i in range(len(msg.p_via) - 1):
            self.a_sets.append(np.array(msg.a_set[i].x).reshape((-1, 3)))
            self.b_sets.append(np.array(msg.b_set[i].x))
        self.obj_centers = []
        self.obj_radii = np.array(msg.obj_radii.x)
        for i in range(len(msg.obj_centers)):
            self.obj_centers.append(np.array(msg.obj_centers[i].x))
        self.obstacles = []
        for i in range(len(msg.obstacles)):
            self.obstacles.append(np.array(msg.obstacles[i].x))
        self.s = msg.s.x
        self.p0 = np.array(msg.p0.x)
        self.q0 = np.array(msg.q0.x)
        if not self.init:
            self.init = True
        elif msg.update:
            lock.acquire()
            # if self.t_current > 3:
            #     asds
            # asd
            # Init MPC
            self.r_via[0] = R.from_rotvec(self.p_lie[3:]).as_matrix()
            self.p0 = np.copy(self.p_lie)
            self.q0 = np.copy(self.q)
            self.qf = self.q0
            self.p = self.p0

            # Reset the reference path in Rviz
            self.rviz_tools.passed_path_msg.poses = []
            self.rviz_tools.passed_ref_path_msg.poses = []
            dist0 = self.a_sets[0] @ self.p[:3] - self.b_sets[0]
            dist0 = np.max(dist0[np.abs(dist0 - 0.0) > 1e-6])
            print(f"(Replanning) Initial point in set 0: {dist0:.3f}")

            # Update the MPC
            self.mpc.update(
                self.p_via,
                self.r_via,
                self.bp1,
                self.br1,
                self.e_p_start,
                self.e_p_end,
                self.e_p_mid,
                self.e_r_start,
                self.e_r_end,
                self.e_r_mid,
                self.a_sets,
                self.b_sets,
                self.v,
                self.a,
                self.jerk,
                p0=self.p0,
                params=self.params,
            )
            lock.release()

        return response

    def publish_mpc_data(
        self,
        t_current,
        traj_data,
        ref_data,
        err_data,
        time_elapsed,
        iters,
        t_loop,
        t_overhead,
        t_switch,
        phi_switch,
        fails,
    ):
        t_switch_full = t_switch.copy()
        t_switch_full.append(t_current)
        phi_switch_full = phi_switch.copy()
        phi_switch_full.append(self.mpc.phi_current[0])
        msg = MPCData()
        msg.header.stamp = rclpy.time.Time(seconds=t_current).to_msg()

        msg.sector = int(self.mpc.ref_path.sector)
        msg.phi_switch_vector.x = (
            self.mpc.ref_path.phi_switch + self.phi_bias
        ).tolist()

        msg.t_comp = time_elapsed
        msg.t_loop = t_loop
        msg.t_overhead = t_overhead
        msg.iterations = iters
        msg.t_switch = t_switch_full
        msg.phi_switch = (np.array(phi_switch_full) + self.phi_bias).tolist()
        msg.fails = fails

        msg.phi.x = (traj_data["phi"] + self.phi_bias).tolist()
        msg.dphi.x = traj_data["dphi"].tolist()
        msg.ddphi.x = traj_data["ddphi"].tolist()
        msg.dddphi.x = traj_data["dddphi"].tolist()
        msg.phi_max = self.mpc.phi_max[0] + self.phi_bias

        if self.log:
            msg.a_set.x = ref_data["a_current"].tolist()
            msg.a_set_next.x = ref_data["a_next"].tolist()
            msg.a_set_j3.x = ref_data["a_j3"].tolist()
            msg.a_set_j5.x = ref_data["a_j5"].tolist()
            msg.a_set_j6.x = ref_data["a_j6"].tolist()
            msg.a_set_j67.x = ref_data["a_j67"].tolist()
            msg.a_set_elbow.x = ref_data["a_elbow"].tolist()

            msg.b_set.x = ref_data["b_current"].tolist()
            msg.b_set_next.x = ref_data["b_next"].tolist()
            msg.b_set_j3.x = ref_data["b_j3"].tolist()
            msg.b_set_j5.x = ref_data["b_j5"].tolist()
            msg.b_set_j6.x = ref_data["b_j6"].tolist()
            msg.b_set_j67.x = ref_data["b_j67"].tolist()
            msg.b_set_elbow.x = ref_data["b_elbow"].tolist()

        for i in range(traj_data["p"].shape[1]):
            vec = Vector()
            vec.x = traj_data["p"][:, i].tolist()
            msg.p.append(vec)
            vec = Vector()
            vec.x = traj_data["v"][:, i].tolist()
            msg.v.append(vec)
            vec = Vector()
            vec.x = traj_data["a"][:, i].tolist()
            msg.a.append(vec)
            vec = Vector()
            vec.x = traj_data["q"][:, i].tolist()
            msg.q.append(vec)
            vec = Vector()
            vec.x = traj_data["dq"][:, i].tolist()
            msg.dq.append(vec)
            vec = Vector()
            vec.x = traj_data["ddq"][:, i].tolist()
            msg.ddq.append(vec)
            vec = Vector()
            vec.x = traj_data["dddq"][:, i].tolist()
            msg.dddq.append(vec)

            if self.log:
                vec = Vector()
                vec.x = err_data["e_p"][i].tolist()
                msg.e_p.append(vec)
                vec = Vector()
                vec.x = err_data["de_p"][i].tolist()
                msg.de_p.append(vec)
                vec = Vector()
                vec.x = err_data["e_p_par"][i].tolist()
                msg.e_p_par.append(vec)
                vec = Vector()
                vec.x = err_data["e_p_orth"][i].tolist()
                msg.e_p_orth.append(vec)
                vec = Vector()
                vec.x = err_data["de_p_par"][i].tolist()
                msg.de_p_par.append(vec)
                vec = Vector()
                vec.x = err_data["de_p_orth"][i].tolist()
                msg.de_p_orth.append(vec)
                vec = Vector()
                vec.x = err_data["e_r"][i].tolist()
                msg.e_r.append(vec)
                vec = Vector()
                vec.x = err_data["de_r"][i].tolist()
                msg.de_r.append(vec)
                vec = Vector()
                vec.x = err_data["e_r_par"][i].tolist()
                msg.e_r_par.append(vec)
                vec = Vector()
                vec.x = err_data["e_r_orth1"][i].tolist()
                msg.e_r_orth1.append(vec)
                vec = Vector()
                vec.x = err_data["e_r_orth2"][i].tolist()
                msg.e_r_orth2.append(vec)

                vec = Vector()
                vec.x = ref_data["p"][i].tolist()
                msg.p_ref.append(vec)
                vec = Vector()
                vec.x = ref_data["dp"][i].tolist()
                msg.dp_ref.append(vec)
                vec = Vector()
                vec.x = ref_data["dp_normed"][i].tolist()
                msg.dp_normed_ref.append(vec)
                vec = Vector()
                vec.x = ref_data["bound_lower"][i].tolist()
                msg.p_lower.append(vec)
                vec = Vector()
                vec.x = ref_data["bound_upper"][i].tolist()
                msg.p_upper.append(vec)
                vec = Vector()
                vec.x = ref_data["e_p_off"][i].tolist()
                msg.e_p_off.append(vec)
                vec = Vector()
                vec.x = ref_data["e_r_off"][i].tolist()
                msg.e_r_off.append(vec)
                vec = Vector()
                vec.x = np.array(ref_data["bp1"][i]).flatten().tolist()
                msg.bp1.append(vec)
                vec = Vector()
                vec.x = np.array(ref_data["bp2"][i]).flatten().tolist()
                msg.bp2.append(vec)
                vec = Vector()
                vec.x = np.array(ref_data["br1"][i]).flatten().tolist()
                msg.br1.append(vec)
                vec = Vector()
                vec.x = np.array(ref_data["br2"][i]).flatten().tolist()
                msg.br2.append(vec)
        self.data_pub.publish(msg)

    def step(self):
        start_step = time.time()
        print_str = f"(MPCNode) Time: {self.t_current - self.t0:.1f}s, "
        print_str += f"Phi: {self.mpc.phi_current[0]:.3f}/{self.mpc.phi_max[0]:.3f}, "
        print_str += f"t_comp: {self.t_mpc*1000:.0f}ms, "
        print_str += f"t_overhead: {self.t_overhead*1000:.0f}ms"
        print(print_str)

        # Compute forward kinematics
        self.p_lie, jac_fk, _ = self.robot_model.forward_kinematics(self.q, self.dq)

        # Optimization step
        traj_data, ref_data, err_data, self.t_mpc, iters = self.mpc.step(
            self.q, self.dq, self.ddq, self.p_lie, self.v, self.jerk, self.qf
        )
        if ref_data is None:
            self.log = False

        # Save some unregular data
        self.fails.append(1.0 if self.mpc.error_count > 0 else 0.0)
        if self.mpc.ref_path.switched:
            self.t_switch.append(self.t_current - self.mpc.dt)
            self.phi_switch.append(self.mpc.ref_path.phi_switch[0])

        # Increase time by the sampling time
        self.t_current += self.mpc.dt

        traj = traj_data["p"]
        jerk_traj = traj_data["dddq"]

        new_state = integrate_joint(
            self.robot_model, jerk_traj, self.q, self.dq, self.ddq, self.mpc.dt
        )
        self.q = new_state[0]
        self.qf = traj_data["q"][:, -1]
        self.dq = new_state[1]
        self.ddq = new_state[2]
        self.p_lie = new_state[3]
        self.v = new_state[4]
        self.a = new_state[5]
        self.j_cart = new_state[6]

        # Move robot in Rviz
        move_robot_kinematic(
            self.robot_pub, rclpy.time.Time(seconds=self.t_current).to_msg(), self.q
        )

        # Publish in Rviz
        if self.log:
            r_goal = R.from_matrix(self.r_via[-1]).as_rotvec()
            goal = np.concatenate((self.p_via[-1], r_goal))
            p_via = self.mpc.ref_path.p
            r_via = self.mpc.ref_path.r
            self.rviz_tools.publish_path(
                self.t_current, traj, np.array(ref_data["p"]).T
            )
            self.rviz_tools.publish_marker(self.p_lie, goal)
            self.rviz_tools.publish_poses(
                p_via, r_via, self.p_lie, np.array(ref_data["p"][1])
            )
            self.rviz_tools.publish_via_points(p_via, r_via)
            self.rviz_tools.publish_coll_spheres(self.q)
            self.rviz_tools.publish_error_plane(
                ref_data["e_p_off"][0],
                ref_data["p"][0],
                ref_data["dp"][0],
                ref_data["bound_lower"][0][:2],
                ref_data["bound_upper"][0][:2],
                ref_data["bp1"][0].T,
                ref_data["bp2"][0].T,
            )

            # rviz_marker_msg = MarkerArray()
            # a_set = self.mpc.a_set_ca[4]
            # b_set = self.mpc.b_set_ca[4, :]
            # msg = self.create_marker_msg(a_set, b_set, "CA", 0.0, 0.0, 1.0)
            # rviz_marker_msg.markers.append(msg)
            # self.rviz_pub.publish(rviz_marker_msg)

        # if self.mpc.phi_current > 0.001:
        #     asf
        self.p = self.p_lie

        # Update current jerk
        self.jerk = jerk_traj[:, 1]
        stop_step = time.time()
        t_loop = stop_step - start_step
        self.t_overhead = t_loop - self.t_mpc
        # time.sleep(0.5)

        # Publish MPC data
        self.publish_mpc_data(
            self.t_current,
            traj_data,
            ref_data,
            err_data,
            self.t_mpc,
            iters,
            t_loop,
            self.t_overhead,
            self.t_switch,
            self.phi_switch,
            self.fails,
        )

    def create_marker_msg(self, a_set, b_set, i, r, g, b, a=0.1):
        marker = Marker()
        marker.header.frame_id = "r1/world"
        marker.action = Marker.ADD
        marker.type = Marker.TRIANGLE_LIST
        marker.ns = f"Set {i}"
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)

        try:
            points = compute_polytope_vertices(a_set, b_set)
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
    mpc_node = MPCNode()
    rate = mpc_node.create_rate(10, mpc_node.get_clock())

    # Spin in a separate thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(mpc_node,), daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    mpc_node.t_current = mpc_node.iiwa.time.to_sec() if mpc_node.simulate else 0.0
    time_max = 0.0
    try:
        while rclpy.ok():
            # One optimization step
            lock.acquire()
            mpc_node.step()
            lock.release()
            last_time = rate._timer.time_since_last_call() * 1e-9
            if last_time > time_max:
                time_max = last_time
            rate.sleep()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    spin_thread.join()


if __name__ == "__main__":
    main()
