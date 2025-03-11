import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..RobotModel import RobotModel
from ..utils import get_default_params, integrate_joint
from .BoundMPC import BoundMPC


class MPCNode:
    def __init__(self, q0):
        self.joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]

        self.fails = []
        self.t_mpc = 0.0
        self.t_overhead = 0.0
        self.t_switch = [0.0]
        self.phi_switch = [0.0]
        self.phi_bias = 0.0
        self.t_bias = 0.0
        self.t_overtime = 0.0
        self.robot_model = RobotModel()
        self.stopped = False

        self.q0 = q0
        self.traj = None
        self.ref_data = None
        self.traj_data = None
        self.p0, _, _ = self.robot_model.forward_kinematics(self.q0, self.q0)
        self.params = get_default_params()
        self.dt = self.params.dt

        self.reset()

    def reset(self):
        print("(MPCNode) Resetting MPC")
        self.p = self.p0
        p_via = [self.p0[:3]] * 2
        r_via = [R.from_rotvec(self.p0[3:]).as_matrix()] * 2
        bp1 = [np.array([1.0, 0.0, 0.0])]
        br1 = [np.array([1.0, 0.0, 0.0])]
        e_r_bound = [np.array([90, 90, 90, -90, -90, -90]) * np.pi / 180]
        a_sets = [np.zeros((15, 3))]
        b_sets = [np.ones(15)]
        obstacles = []
        # Init MPC
        self.mpc = BoundMPC(
            p_via,
            r_via,
            bp1,
            br1,
            e_r_bound,
            a_sets,
            b_sets,
            obstacles,
            p0=self.p0,
            params=self.params,
        )

        # Initial values
        self.q = self.q0
        self.qf = self.q0
        self.dq = np.zeros(7)
        self.ddq = np.zeros(7)
        self.jerk = np.zeros(7)
        self.p_lie = self.p0
        self.p_ref = self.p0
        self.v = np.zeros(6)
        self.t_current = 0.0
        self.k_current = 0
        self.t0 = np.copy(self.t_current)

    def update_reference(
        self, p_via, r_via, bp1, br1, e_r_bound, a_sets, b_sets, obstacles
    ):
        print("(MPCNode) Received Trajectory")
        self.p0 = np.copy(self.p_lie)
        self.q0 = np.copy(self.q)
        self.qf = self.q0
        self.p = self.p0

        # Update the MPC
        self.mpc.update(
            p_via,
            r_via,
            bp1,
            br1,
            e_r_bound,
            a_sets,
            b_sets,
            obstacles,
            self.v,
            p0=self.p0,
            params=self.params,
        )

    def step(self):
        start_step = time.time()
        print_str = f"(MPCNode) Time: {self.t_current - self.t0:.1f}s, "
        print_str += f"Phi: {self.mpc.phi_current[0]:.3f}/{self.mpc.phi_max[0]:.3f}, "
        print_str += f"t_comp: {self.t_mpc*1000:.0f}ms, "
        print_str += f"t_overhead: {self.t_overhead*1000:.0f}ms, "
        print_str += (
            f"sector: {self.mpc.ref_path.sector}/{self.mpc.ref_path.num_sectors}"
        )
        print(print_str)

        # Compute forward kinematics
        self.p_lie, jac_fk, _ = self.robot_model.forward_kinematics(self.q, self.dq)

        # Optimization step
        traj_data, ref_data, err_data, self.t_mpc, iters = self.mpc.step(
            self.q, self.dq, self.ddq, self.p_lie, self.v, self.jerk, self.qf
        )
        self.p_ref = ref_data["p"][1]
        self.traj = traj_data["p"]
        self.traj_data = traj_data
        self.ref_data = ref_data

        # Save some unregular data
        self.fails.append(1.0 if self.mpc.error_count > 0 else 0.0)
        if self.mpc.ref_path.switched:
            self.t_switch.append(self.t_current - self.mpc.dt)
            self.phi_switch.append(self.mpc.ref_path.phi_switch[0])

        # Increase time by the sampling time
        self.t_current += self.mpc.dt
        self.k_current += 1

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

        self.p = self.p_lie

        # Update current jerk
        self.jerk = jerk_traj[:, 1]
        stop_step = time.time()
        t_loop = stop_step - start_step
        self.t_overhead = t_loop - self.t_mpc
        time.sleep(max(0, self.dt - t_loop))
