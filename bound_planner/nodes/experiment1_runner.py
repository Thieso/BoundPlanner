import copy
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from bound_mpc_msg.srv import MPCParams, Trajectory
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import MarkerArray

from bound_planner.ConvexSetPlanner import ConvexSetPlanner
from bound_planner.Logger import Logger
from bound_planner.Plotter import Plotter
from bound_planner.RobotModel import RobotModel
from bound_planner.utils import (
    create_obj_spheres_msg,
    create_traj_msg,
    get_default_path,
    get_default_weights,
)


class ExperimentRunner(Node):
    def __init__(self):
        super().__init__("experiment_runner")

        # Initial joint configuration
        scen = 1
        variant = 0
        q0 = np.zeros((7, 1))
        q0[1] = -np.pi / 6
        q0[3] = -np.pi / 2
        q0[5] = np.pi / 3
        if scen == 0:
            q0[0] = -np.pi / 2
        q0 = q0.flatten()
        dq0 = np.zeros(7)

        # Initial Cartesian pose
        robot_model = RobotModel()
        p0fk, _, _ = robot_model.forward_kinematics(q0, dq0)
        p0 = p0fk[:3]
        r0 = R.from_rotvec(p0fk[3:]).as_matrix()

        if scen == 0:
            # Final Cartesian pose
            p1 = np.array([-0.4, -0.6, 0.6])
            r1 = (
                R.from_euler("xyz", [0, 0, -np.pi / 2]).as_matrix()
                @ R.from_euler("xyz", [0, np.pi / 2, 0]).as_matrix()
            )
            po = np.array([p1[0], -0.8, p1[2]])

            # Define obstacles as boxes by limits (-x, -y, -z, x, y, z)
            obstacles = [
                [
                    po[0] - 0.3,
                    po[1] - 0.7,
                    po[2] - 0.3,
                    po[0] + 0.3,
                    po[1] + 0.6,
                    po[2] - 0.1,
                ],
                [
                    po[0] - 0.3,
                    po[1] - 0.7,
                    po[2] + 0.1,
                    po[0] + 0.3,
                    po[1] + 0.6,
                    po[2] + 0.3,
                ],
                [
                    po[0] - 0.3,
                    po[1] - 0.7,
                    po[2] - 0.3,
                    po[0] - 0.1,
                    po[1] + 0.6,
                    po[2] + 0.3,
                ],
                [
                    po[0] + 0.1,
                    po[1] - 0.7,
                    po[2] - 0.3,
                    po[0] + 0.3,
                    po[1] + 0.6,
                    po[2] + 0.3,
                ],
                [
                    po[0] - 0.3,
                    po[1] - 0.7,
                    po[2] - 0.3,
                    po[0] + 0.3,
                    po[1] - 0.5,
                    po[2] + 0.3,
                ],
            ]
            # Set collision objects
            obj_centers = [
                po + np.array([-0.2, 0.55, -0.2]),
                po + np.array([0.2, 0.55, -0.2]),
                po + np.array([-0.2, 0.55, -0.05]),
                po + np.array([0.2, 0.55, -0.05]),
                po + np.array([-0.2, 0.55, 0.1]),
                po + np.array([0.2, 0.55, 0.1]),
                po + np.array([-0.2, 0.55, 0.25]),
                po + np.array([0.2, 0.55, 0.25]),
                po + np.array([-0.1, 0.55, 0.25]),
                po + np.array([0.05, 0.55, 0.25]),
                po + np.array([-0.1, 0.55, -0.2]),
                po + np.array([0.05, 0.55, -0.2]),
            ]
            obj_radii = [
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
            ]
        else:
            # Final Cartesian pose
            p1s = [
                np.array([-0.65, 0.0, 0.3]),
                np.array([-0.55, -0.4, 0.3]),
                np.array([-0.55, 0.4, 0.3]),
                np.array([0.55, -0.4, 0.3]),
                np.array([0.55, 0.4, 0.3]),
                np.array([-0.0, 0.65, 0.3]),
                np.array([-0.0, -0.65, 0.3]),
                np.array([0.75, 0.0, 0.3]),
            ]
            p1 = p1s[variant]
            angle = np.arctan2(p1[1], p1[0])
            r1 = (
                R.from_euler("xyz", [0, np.pi / 2, angle]).as_matrix()
                @ R.from_euler("xyz", [0, np.pi / 2, 0]).as_matrix()
            )
            # r1 = R.from_euler('xyz', [0, np.pi/2, np.pi/2]).as_matrix() @ R.from_euler('xyz', [0, np.pi/2, 0]).as_matrix()
            po = np.array([p1[0], p1[1], 0.0])
            obstacles = [
                [
                    po[0] - 0.225,
                    po[1] - 0.25,
                    po[2],
                    po[0] + 0.225,
                    po[1] - 0.15,
                    po[2] + 0.5,
                ],
                [
                    po[0] - 0.225,
                    po[1] + 0.15,
                    po[2],
                    po[0] + 0.225,
                    po[1] + 0.25,
                    po[2] + 0.5,
                ],
                [
                    po[0] - 0.225,
                    po[1] - 0.25,
                    po[2],
                    po[0] - 0.125,
                    po[1] + 0.25,
                    po[2] + 0.5,
                ],
                [
                    po[0] + 0.125,
                    po[1] - 0.25,
                    po[2],
                    po[0] + 0.225,
                    po[1] + 0.25,
                    po[2] + 0.5,
                ],
                # [-0.55, -0.05, 0.6, -0.35, 0.15, 1.3],
                [po[0] - 0.225, po[1] - 0.25, 0.8, po[0] + 0.225, po[1] + 0.25, 1.1],
            ]
            # Set collision objects
            obj_centers = [
                # np.array([-0.45, 0.05, 0.7]),
                po + np.array([0.2, -0.25, 0.4]),
                po + np.array([0.2, -0.1, 0.4]),
                po + np.array([0.2, -0.0, 0.4]),
                po + np.array([0.2, 0.1, 0.4]),
                po + np.array([-0.2, -0.25, 0.4]),
                po + np.array([-0.2, -0.1, 0.4]),
                po + np.array([-0.2, -0.0, 0.4]),
                po + np.array([-0.2, 0.1, 0.4]),
                po + np.array([0.2, 0.25, 0.4]),
                po + np.array([0.1, 0.25, 0.4]),
                po + np.array([0.0, 0.25, 0.4]),
                po + np.array([-0.1, 0.25, 0.4]),
                po + np.array([-0.2, 0.25, 0.4]),
                po + np.array([0.1, -0.25, 0.4]),
                po + np.array([0.0, -0.25, 0.4]),
                po + np.array([-0.1, -0.25, 0.4]),
                po + np.array([0.175, 0, 0.9]),
                po + np.array([0.175, 0.175, 0.9]),
                po + np.array([0.175, -0.175, 0.9]),
                po + np.array([0.0, 0.175, 0.9]),
                po + np.array([0.0, 0.0, 0.9]),
                po + np.array([0.0, -0.175, 0.9]),
                po + np.array([-0.175, 0, 0.9]),
                po + np.array([-0.175, 0.175, 0.9]),
                po + np.array([-0.175, -0.175, 0.9]),
            ]
            obj_radii = [
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.11,
                0.11,
                0.11,
                0.11,
                0.11,
                0.11,
                0.11,
                0.11,
                0.11,
            ]

        # Plan a path to the final pose
        planner = ConvexSetPlanner(e_p_max=0.5, obstacles=copy.deepcopy(obstacles))
        start = time.time()
        (
            p_via,
            r_via,
            bp1_list,
            sets_via,
        ) = planner.plan_convex_set_path(p0, p1, r0, r1)
        (
            _,
            _,
            _,
            _,
            _,
            br1_list,
            s,
            e_p_start,
            e_p_end,
            e_p_mid,
            e_r_start,
            e_r_end,
            e_r_mid,
        ) = get_default_path(p0, r0, len(p_via))
        # r_via, br1_list, e_r_start, e_r_end, e_r_mid = planner.adapt_rotation_ref(
        #     r0, r1, p_via, bp1_list, p_samples, omega_via)
        stop = time.time()
        print(f"Path planning took {stop - start:.2f}s")

        weights = get_default_weights()

        save_data = False
        show_plots = False
        path = "/ros2_ws/src/bound_planner/data/"
        tail = f"scen_{scen}_{variant}"
        t0 = 0

        params = MPCParams.Request()
        params.n = 15
        params.dt = 0.1
        params.weights = weights.tolist()
        params.build = True
        params.simulate = False
        params.experiment = False
        params.use_acados = False
        params.real_time = False
        params.nr_segs = 4

        self.obj_pub = self.create_publisher(MarkerArray, "/r1/collision_obj", 1)
        msg = create_obj_spheres_msg(
            self.get_clock().now().to_msg(), obj_centers, obj_radii
        )
        self.obj_pub.publish(msg)
        self.obj_pub.publish(msg)
        self.obj_pub.publish(msg)

        # Send the parameters to the MPC
        srv = self.create_client(MPCParams, "/mpc/set_params")
        while not srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Parameter service not available, waiting again...")
        future = srv.call_async(params)
        rclpy.spin_until_future_complete(self, future)

        # Create Logger
        logger = Logger(t0, params.n)

        try:
            # Publish Trajectory
            msg = create_traj_msg(
                p_via,
                r_via,
                e_p_start,
                e_p_end,
                e_p_mid,
                bp1_list,
                br1_list,
                s,
                e_r_start,
                e_r_end,
                e_r_mid,
                p0fk,
                q0,
                obj_centers,
                obj_radii,
                obstacles,
                sets_via,
            )
            traj_srv = self.create_client(Trajectory, "/mpc/set_trajectory")
            traj_srv.wait_for_service()
            future = traj_srv.call_async(msg)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()

            # Start the logging
            logger.start()

            # Wait for logger to receive first message
            while logger.phi < 1e-6:
                rclpy.spin_once(logger)
                time.sleep(0.01)

            # Wait for the robot to finish
            while logger.phi_max - logger.phi > 0.01:
                rclpy.spin_once(logger)
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt")
            pass

        # Stop the logging thread
        logger.stop()
        listener = None

        # Plot stuff
        plotter = Plotter(
            logger,
            params,
            t0,
            p_via,
            r_via,
            listener,
            obstacles=copy.deepcopy(obstacles),
            path=path,
            tail=tail,
            save_data_flag=save_data,
            x_values="path",
        )

        if show_plots:
            plt.show()


def main():
    rclpy.init()
    node = ExperimentRunner()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
