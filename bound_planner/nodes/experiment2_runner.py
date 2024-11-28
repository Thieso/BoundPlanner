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
    normalize_set_size,
)


class ExperimentRunner(Node):
    def __init__(self):
        super().__init__("experiment_runner")

        example = True

        # Initial joint configuration
        q0 = np.zeros((7, 1))
        q0[1] = -np.pi / 6
        q0[3] = -np.pi / 2
        q0[5] = np.pi / 3
        q0 = q0.flatten()
        dq0 = np.zeros(7)

        # Initial Cartesian pose
        robot_model = RobotModel()
        p0fk, _, _ = robot_model.forward_kinematics(q0, dq0)
        p0 = p0fk[:3]
        r0 = R.from_rotvec(p0fk[3:]).as_matrix()

        # Final Cartesian pose
        # p1 = np.array([0.8, -0.3, 0.55])
        p1 = np.array([0.8, 0.3, 0.55])
        r1 = R.from_euler("xyz", [0, np.pi / 2, 0]).as_matrix()

        # Define obstacles as boxes by limits (-x, -y, -z, x, y, z)
        obstacles = [
            [0.5, -0.4, 0.0, 1.0, 0.4, 0.215],
            [0.5, -0.4, 0.45, 1.0, 0.4, 0.47],
            [0.5, -0.4, 0.75, 1.0, 0.4, 0.77],
            [0.5, -0.42, 0.0, 1.0, -0.4, 0.77],
            [0.5, 0.4, 0.0, 1.0, 0.42, 0.77],
            # [0.4, -0.05, 0.0, 1.0, 0.05, 0.65],
            # [-0.15, -0.15, 0.0, 0.15, 0.15, 0.4],
            # [-0.0, -0.15, 0.0, 0.95, 0.15, 0.2],
            # [0.45, -0.15, 0.5, 0.65, 0.15, 1.2],
        ]
        # Set collision objects
        obj_centers = [
            np.array([0.725, 0.0, 0.21]),
            np.array([0.725, 0.1, 0.21]),
            np.array([0.725, 0.2, 0.21]),
            np.array([0.725, 0.3, 0.21]),
            np.array([0.725, -0.1, 0.21]),
            np.array([0.725, -0.2, 0.21]),
            np.array([0.725, -0.3, 0.21]),
            np.array([0.725, 0.0, 0.46]),
            np.array([0.725, 0.1, 0.46]),
            np.array([0.725, 0.2, 0.46]),
            np.array([0.725, 0.3, 0.46]),
            np.array([0.725, -0.1, 0.46]),
            np.array([0.725, -0.2, 0.46]),
            np.array([0.725, -0.3, 0.46]),
            np.array([0.725, 0.0, 0.71]),
            np.array([0.725, 0.1, 0.71]),
            np.array([0.725, 0.2, 0.71]),
            np.array([0.725, 0.3, 0.71]),
            np.array([0.725, -0.1, 0.71]),
            np.array([0.725, -0.2, 0.71]),
            np.array([0.725, -0.3, 0.71]),
        ]
        obj_radii = [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ]

        workspace_max = [1.0, 1.0, 1.2]
        workspace_min = [-0.3, -1.0, 0.0]

        # p0 = np.array([0.50989913, 0.03270671, 0.54449693])
        # p1 = np.array([0.79627805, -0.2519156, 0.3])
        # p_horizon = np.array([0.5856082, 0.05014791, 0.4121142])
        # Evaluating sample np.array([-0.832, 0.004, 0.597])
        # planner = ConvexSetPlanner(
        #     e_p_max=0.5,
        #     obstacles=copy.deepcopy(obstacles),
        #     workspace_min=workspace_min,
        #     workspace_max=workspace_max,
        # )
        # p0 = np.array([0.74668241, 0.02175294, 0.76031163])
        # p1 = np.array([0.71391736, 0.11200469, 0.3])
        # p_horizon = np.array([0.31156919, 0.04000345, 0.81428509])
        # (PosPath) Evaluating sample -0.924, -0.135, 0.297
        # planner.sets_via_prev = [[np.array([1, 0, 0.0]), np.array([3.0])]]
        # (
        #     p_via,
        #     r_via,
        #     bp1_list,
        #     sets_via,
        # ) = planner.plan_convex_set_path(
        #     p0, p1, r0, r1, replanning=True, p_horizon=[p_horizon] * 10
        # )
        # for p in p_via:
        #     print(p)
        # p0 = np.array([0.59159238, 0.05084962, 0.57139768])
        # p1 = np.array([0.77236084, -0.17973018, 0.3])
        # p_horizon = np.array([0.65035887, -0.0772901, 0.52017114])
        # (PosPath) Adding random point [ 0.48780586 -0.14946985  0.88294894] to graph

        # Plan a path to the final pose
        planner = ConvexSetPlanner(
            e_p_max=0.5,
            obstacles=copy.deepcopy(obstacles),
            workspace_min=workspace_min,
            workspace_max=workspace_max,
        )
        # planner.sets_via_prev = [[[1.0, 0, 0], [100.0]]]
        start = time.time()
        (
            p_via,
            r_via,
            bp1_list,
            sets_via,
        ) = planner.plan_convex_set_path(p0, p1, r0, r1)
        normed_set = normalize_set_size(
            [copy.deepcopy(sets_via[-2])], planner.max_set_size
        )
        params_proj = np.concatenate(
            (
                normed_set[0][0].T.flatten(),
                normed_set[0][1],
                p_via[-1] - np.array([0.05, 0, 0]),
            )
        )
        sol = planner.proj_solver(
            x0=p_via[-1] - np.array([0.05, 0, 0]),
            lbx=planner.proj_lbu,
            ubx=planner.proj_ubu,
            lbg=planner.proj_lbg,
            ubg=planner.proj_ubg,
            p=params_proj,
        )
        p_via[-2] = sol["x"].full().flatten()
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
        path = "/ros2_ws/src/bound_mpc/data/"
        tail = "scen_2_0"
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
            replan_ex = [20, 40]
            p_example = [np.array([0.8, -0.25, 0.84]), np.array([0.85, -0.1, 0.295])]
            # replan_ex = [20]
            # p_example = [np.array([0.8, -0.25, 0.8])]
            idx_ex = 0
            replan_i = 0
            nr_replannings = 0
            max_nr_replannings = 20
            rng = np.random.default_rng()
            t_plan = []
            if not example:
                rand_is = 5 * np.ones(max_nr_replannings)
                path = "/home/ubuntu/ros2_ws/src/bound_mpc/"
                # rand_ds = rng.uniform(0.05, 0.3, max_nr_replannings)
                # np.savetxt(path + "replanning_triggers.txt", rand_ds)
                # p_replans = []
                # p1c = np.copy(p1)
                # for _ in range(max_nr_replannings):
                #     p1_new = np.copy(p1c)
                #     while np.linalg.norm(p1_new[1:] - p1c[1:]) < 0.2:
                #         xy = rng.uniform([0.7, -0.3], [0.9, 0.3], 2)
                #         z = rng.integers(0, 3, 1)
                #         z_new = [0.295, 0.55, 0.84]
                #         p1_new = np.array([xy[0], xy[1], z_new[int(z[0])]])
                #     p1c = p1_new
                #     p_replans.append(p1_new)
                # np.savetxt(path + "replanning_positions.txt", np.array(p_replans))
                rand_ds = np.loadtxt(path + "replanning_triggers.txt")
                p_replans = np.loadtxt(path + "replanning_positions.txt").tolist()
                rand_i = rand_is[nr_replannings]
                rand_d = rand_ds[nr_replannings]
            else:
                rand_i = replan_ex[idx_ex]
                rand_d = np.inf
            while logger.phi_max - logger.phi > 0.01:
                if (
                    logger.i > replan_i + rand_i
                    and np.linalg.norm(logger.p_traj[-1][:3] - p1) < rand_d
                ):
                    nr_replannings += 1
                    # p0 = logger.p_traj[-1][:3]
                    p_horizon = [x[-1][:3] for x in logger.p_traj_horizon[1:]]
                    p0 = p_horizon[0]
                    r0 = R.from_rotvec(logger.p_traj[-1][3:]).as_matrix()
                    if not example:
                        p1 = np.array(p_replans[nr_replannings - 1])
                    else:
                        p1 = p_example[idx_ex]

                    print(
                        f"(Replanning) Number {nr_replannings} at phi = {logger.phi:.3f}m"
                    )
                    print(f"(Replanning) p0 = {p0}")
                    print(f"(Replanning) p1 = {p1}")
                    r1 = R.from_euler("xyz", [0, np.pi / 2, 0]).as_matrix()
                    (
                        p_via,
                        r_via,
                        bp1_list,
                        sets_via,
                    ) = planner.plan_convex_set_path(
                        p0, p1, r0, r1, replanning=True, p_horizon=p_horizon
                    )
                    t_plan.append(planner.comp_time_total)
                    print("-------------")
                    print(f"Max planning time {np.max(t_plan):.3f}s")
                    print(f"Mean planning time {np.mean(t_plan):.3f}s")
                    print(f"Min planning time {np.min(t_plan):.3f}s")
                    print("-------------")
                    normed_set = normalize_set_size(
                        [copy.deepcopy(sets_via[-2])], planner.max_set_size
                    )
                    params_proj = np.concatenate(
                        (
                            normed_set[0][0].T.flatten(),
                            normed_set[0][1],
                            p_via[-1] - np.array([0.05, 0, 0]),
                        )
                    )
                    sol = planner.proj_solver(
                        x0=p_via[-1] - np.array([0.05, 0, 0]),
                        lbx=planner.proj_lbu,
                        ubx=planner.proj_ubu,
                        lbg=planner.proj_lbg,
                        ubg=planner.proj_ubg,
                        p=params_proj,
                    )
                    p_via[-2] = sol["x"].full().flatten()
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
                    e_r_start[0] = e_r_end[0]
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
                        update=True,
                    )
                    future = traj_srv.call_async(msg)
                    rclpy.spin_until_future_complete(self, future)
                    if not example:
                        if nr_replannings >= max_nr_replannings - 1:
                            rand_i = np.inf
                        else:
                            rand_i = rand_is[nr_replannings - 1]
                            rand_d = rand_ds[nr_replannings - 1]
                    else:
                        idx_ex += 1
                        if idx_ex >= len(replan_ex):
                            rand_i = np.inf
                        else:
                            rand_i = replan_ex[idx_ex]
                    replan_i = logger.i
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
            obstacles=obstacles,
            path=path,
            tail=tail,
            save_data_flag=save_data,
            x_values="time",
        )

        if show_plots:
            plt.show()


def main():
    rclpy.init()
    node = ExperimentRunner()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
