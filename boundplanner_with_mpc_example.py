import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from bound_planner.BoundMPC.MPCNode import MPCNode
from bound_planner.BoundPlanner import BoundPlanner
from bound_planner.utils.visualization import plot_via_path

USE_RVIZ = False
if USE_RVIZ:
    import rclpy

    from bound_planner.RvizTools import RvizTools, RvizToolsMPC


def main():
    q0 = np.zeros(7)
    # q0[3] = np.pi / 2
    # q0[5] = np.pi / 3
    # q0[6] = np.pi / 2
    q0[1] = 0.0
    q0[3] = -np.pi / 2
    q0[5] = np.pi / 2

    mpc_node = MPCNode(q0)
    mpc_node.step()

    # Initial and desired pose
    p0fk, _, _ = mpc_node.robot_model.forward_kinematics(q0, 0 * q0)
    p0 = p0fk[:3]
    r0 = R.from_rotvec(p0fk[3:]).as_matrix()
    p1 = np.array([0.45, -0.5, 0.2])
    r1 = R.from_euler("XYZ", [0, 90, 0], degrees=True).as_matrix()

    # Obstacles
    size = 0.04
    s_box = 0.12
    w_boxx = 0.02
    w_boxy = 0.02
    p_box = [0.45, -0.48, 0.05]
    h_box = 0.18
    obstacles = [
        [
            p_box[0] + s_box - w_boxx,
            p_box[1] - s_box,
            0.0,
            p_box[0] + s_box,
            p_box[1] + s_box,
            p_box[2] + h_box,
        ],
        [
            p_box[0] - s_box,
            p_box[1] - s_box,
            0.0,
            p_box[0] - s_box + w_boxx,
            p_box[1] + s_box,
            p_box[2] + h_box,
        ],
        [
            p_box[0] - s_box,
            p_box[1] - s_box - w_boxy,
            0.0,
            p_box[0] + s_box,
            p_box[1] - s_box,
            p_box[2] + h_box,
        ],
        [
            p_box[0] - s_box,
            p_box[1] + s_box,
            0.0,
            p_box[0] + s_box,
            p_box[1] + s_box + w_boxy,
            p_box[2] + h_box,
        ],
        [0.2, -1.0, -0.1, 1.0, 1.0, 0.0],
        [-0.3, -1.0, 0.53, 0.2, -0.35, 1.0],
        [-0.2, -1.0, 0.0, -0.14, 1.0, 1.0],
        [-1.0, 0.38, 0.0, 1.0, 0.5, 1.0],
        [0.4, -0.05, 0.0, 0.5, 0.05, 0.15],
        [0.1, -0.55, 0.0, 0.3, -0.35, 0.07],
        [
            0.5 - size,
            -0.2 - size,
            0.03 - size,
            0.5 + size,
            -0.2 + size,
            0.03 + size,
        ],
        [
            0.4 - size,
            0.3 - size,
            0.03 - size,
            0.4 + size,
            0.3 + size,
            0.03 + size,
        ],
    ]

    planner = BoundPlanner(
        e_p_max=0.5,
        obstacles=copy.deepcopy(obstacles),
        workspace_max=[1.0, 0.38, 1.0],
        workspace_min=[-0.14, -1.0, 0.0],
    )

    start = time.time()
    (
        p_via,
        r_via,
        bp1_list,
        sets_via,
    ) = planner.plan_convex_set_path(p0, p1, r0, r1)
    stop = time.time()
    print(f"Path planning took {stop - start:.2f}s")

    if USE_RVIZ:
        rclpy.init()
        rviz_tools = RvizTools()
        rviz_tools_mpc = RvizToolsMPC()
        rviz_tools.publish_via_points(p_via, r_via)
        rviz_tools.delete_sets()
        rviz_tools.publish_sets()
        rviz_tools.add_sets(sets_via, name="Via Set")
        rviz_tools.add_sets(planner.obs_sets_orig, color=[1, 0, 0], name="Obs Set")
        rviz_tools.publish_sets()

    a_sets = [x[0] for x in sets_via]
    b_sets = [x[1] for x in sets_via]
    br1_list = [np.array([0, 0, 1.0])] * len(bp1_list)
    e_r_bound = [np.array([90, 90, 90, -90, -90, -90]) * np.pi / 180] * len(bp1_list)
    mpc_node.update_reference(
        p_via, r_via, bp1_list, br1_list, e_r_bound, a_sets, b_sets, obstacles
    )

    try:
        p = []
        while mpc_node.mpc.phi_current < mpc_node.mpc.phi_max - 0.001:
            # Do one step
            mpc_node.step()
            # Publish in Rviz
            if USE_RVIZ:
                if mpc_node.traj is not None:
                    rviz_tools_mpc.publish_path(
                        mpc_node.t_current,
                        mpc_node.traj,
                        np.array(mpc_node.ref_data["p"]).T,
                    )
                rviz_tools_mpc.publish_poses(mpc_node.p_lie, np.array(mpc_node.p_ref))
                rviz_tools_mpc.publish_coll_spheres(mpc_node.q)

                # Move robot in Rviz
                rviz_tools_mpc.move_robot_kinematic(mpc_node.t_current, mpc_node.q)
            else:
                p.append(mpc_node.traj[:3, 1])

    except KeyboardInterrupt:
        mpc_node.stop_thread()

    if not USE_RVIZ:
        p = np.array(p)
        plot_via_path(p_via, r_via, sets_via, planner.obs_sets)
        plt.plot(p[:, 0], p[:, 1], p[:, 2], linewidth=2, color="black")
        plt.show()


if __name__ == "__main__":
    main()
