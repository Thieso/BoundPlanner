import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from bound_planner.BoundPlanner import BoundPlanner
from bound_planner.utils.visualization import plot_graph, plot_via_path

USE_RVIZ = False
if USE_RVIZ:
    import rclpy

    from bound_planner.RvizTools import RvizTools


def main():
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

    p0 = np.array([0.3, 0.0, 0.7])
    p1 = np.array([0.45, -0.5, 0.2])
    r0 = R.from_euler("XYZ", [0, 90, 0], degrees=True).as_matrix()
    r1 = R.from_euler("XYZ", [0, 90, 0], degrees=True).as_matrix()

    start = time.time()
    (
        p_via,
        r_via,
        bp1_list,
        sets_via,
    ) = planner.plan_convex_set_path(p0, p1, r0, r1)
    stop = time.time()
    print(f"Path planning took {stop - start:.2f}s")

    plot_via_path(p_via, r_via, sets_via, planner.obs_sets)
    plot_graph(p0, p1, planner.graph, planner.inter_graph, planner.obs_sets)

    if USE_RVIZ:
        rclpy.init()
        rviz_tools = RvizTools()
        rviz_tools.publish_via_points(p_via, r_via)
        rviz_tools.delete_sets()
        rviz_tools.publish_sets()
        rviz_tools.add_sets(sets_via, name="Via Set")
        rviz_tools.add_sets(planner.obs_sets_orig, color=[1, 0, 0], name="Obs Set")
        rviz_tools.publish_sets()

    plt.show()


if __name__ == "__main__":
    main()
