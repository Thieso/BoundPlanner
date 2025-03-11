import copy as cp
import time
from collections import defaultdict

import casadi
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..BoundPlanner import BoundPlanner
from ..ReferencePath import ReferencePath
from ..RobotModel import RobotModel
from ..utils import (
    normalize_set_size,
)
from ..utils.optimization_functions import jac_SO3_inv_left, jac_SO3_inv_right
from .bound_mpc_functions import (
    compute_initial_rot_errors,
    error_function,
    integrate_rotation_reference,
    reference_function,
)
from .casadi_ocp_formulation import setup_optimization_problem
from .jerk_trajectory_casadi import calcAcceleration, calcAngle, calcVelocity
from .mpc_utils_casadi import compute_bound_params_six


class BoundMPC:
    def __init__(
        self,
        pos_points,
        rot_points,
        bp1,
        br1,
        e_r_bound,
        a_sets,
        b_sets,
        obstacles,
        p0=np.zeros(6),
        params=None,
    ):
        # Prediction horizon
        self.N = params.n

        self.robot_model = RobotModel()

        self.ref_data = defaultdict(list)
        self.err_data = defaultdict(list)
        for i in range(self.N):
            self.ref_data["p"].append([])
            self.ref_data["dp"].append([])
            self.ref_data["ddp"].append([])
            self.ref_data["dp_normed"].append([])
            self.ref_data["dp_normedn"].append([])
            self.ref_data["bp1"].append([])
            self.ref_data["bp2"].append([])
            self.ref_data["br1"].append([])
            self.ref_data["br2"].append([])
            self.ref_data["br1_next"].append([])
            self.ref_data["br2_next"].append([])
            self.ref_data["v1"].append([])
            self.ref_data["v2"].append([])
            self.ref_data["v3"].append([])
            self.ref_data["v1_next"].append([])
            self.ref_data["v2_next"].append([])
            self.ref_data["v3_next"].append([])
            self.ref_data["p_r_omega0"].append([])
            self.ref_data["r_bound_lower"].append([])
            self.ref_data["r_bound_upper"].append([])
            self.ref_data["r_bound_lower_next"].append([])
            self.ref_data["r_bound_upper_next"].append([])

            self.err_data["e_p"].append([])
            self.err_data["de_p"].append([])
            self.err_data["e_p_par"].append([])
            self.err_data["e_p_orth"].append([])
            self.err_data["de_p_par"].append([])
            self.err_data["de_p_orth"].append([])
            self.err_data["e_r"].append([])
            self.err_data["de_r"].append([])
            self.err_data["e_r_par"].append([])
            self.err_data["e_r_orth1"].append([])
            self.err_data["e_r_orth2"].append([])
            self.err_data["e_r_parn"].append([])
            self.err_data["e_r_orth1n"].append([])
            self.err_data["e_r_orth2n"].append([])

        self.updated = False
        self.k_update = 0.0
        self.updated_once = False
        self.nr_slacks = 6 + self.N * 4
        self.slacks0 = np.zeros(6)

        self.obstacles = obstacles

        # Flag whether to compile the OCP problem
        self.build = params.build

        # Initial position
        self.p0 = p0

        # Desired joint config
        self.qd = np.zeros(7)

        # Number of consecutive infeasible solutions
        self.error_count = 0

        # Time steps
        self.dt = params.dt

        # Time horizon
        self.T = self.dt * self.N

        # Create reference trajectory object
        self.nr_segs = params.nr_segs
        self.ref_path = ReferencePath(
            pos_points,
            rot_points,
            bp1,
            br1,
            e_r_bound,
            a_sets,
            b_sets,
            self.nr_segs,
        )
        # Horizon splitting idx
        self.split_idxs = [0] + [self.N] * self.nr_segs
        self.switch = False

        # Initial error
        self.dtau_init = np.empty((3, self.nr_segs))
        self.dtau_init_par = np.empty((3, self.nr_segs))
        self.dtau_init_orth1 = np.empty((3, self.nr_segs))
        self.dtau_init_orth2 = np.empty((3, self.nr_segs))

        # Max path parameter
        self.phi_max = np.array([self.ref_path.phi_max])

        # Objective function weights
        self.weights = np.array(params.weights)

        # Reference integration variables
        self.dp_ref = None
        self.pr_ref = p0[3:]
        self.iw_ref = np.zeros(3)

        # Bounds
        limits = self.robot_model.get_robot_limits()
        self.q_lim_upper = limits[0]
        self.q_lim_lower = limits[1]
        self.dq_lim_upper = limits[2]
        self.dq_lim_lower = limits[3]
        self.tau_lim_upper = limits[4]
        self.tau_lim_lower = limits[5]
        self.u_max = limits[6]
        self.u_min = limits[7]
        self.ut_max = self.u_max
        self.ut_min = self.u_min

        # Path parameter init
        self.phi_current = np.array([0.0])
        self.phi_replan = np.array([0.0])
        self.dphi_current = np.array([0.0])
        self.ddphi_current = np.array([0.0])
        self.dddphi_current = np.array([0.0])

        # Dimension variables
        self.nr_joints = 7
        self.nr_u = self.nr_joints + 1
        self.nr_x = 40

        # Bounds
        self.q_ub = self.robot_model.q_lim_upper
        self.q_lb = self.robot_model.q_lim_lower
        self.q_ub = np.repeat(self.q_ub, self.N)
        self.q_lb = np.repeat(self.q_lb, self.N)

        self.dq_ub = self.robot_model.dq_lim_upper
        self.dq_lb = self.robot_model.dq_lim_lower
        self.dq_ub = np.repeat(self.dq_ub, self.N)
        self.dq_lb = np.repeat(self.dq_lb, self.N)

        self.ddq_ub = 5.0 * np.ones(self.N * self.nr_joints)
        self.ddq_lb = -self.ddq_ub

        self.u_ub = self.robot_model.u_max * np.ones(self.N * self.nr_joints)
        self.u_lb = self.robot_model.u_min * np.ones(self.N * self.nr_joints)

        self.p_ub = np.inf * np.ones(self.N * 6)
        self.p_lb = -np.inf * np.ones(self.N * 6)
        self.v_ub = np.inf * np.ones(self.N * 6)
        self.v_lb = -np.inf * np.ones(self.N * 6)

        # Solution of previous run
        self.prev_solution = None
        self.lam_g0 = 0
        self.lam_x0 = 0
        self.lam = None
        self.pi = None

        # Setup the optimization problem
        path = "."
        ipopt_options = {
            "tol": 10e-6,
            "max_iter": 100,
            "limited_memory_max_history": 6,
            "limited_memory_initialization": "scalar1",
            "limited_memory_max_skipping": 2,
            # "linear_solver": "ma57",
            # "linear_system_scaling": "mc19",
            # "ma57_automatic_scaling": "no",
            # "ma57_pre_alloc": 100,
            "mu_strategy": "adaptive",
            "adaptive_mu_globalization": "kkt-error",
            "print_info_string": "no",
            "fast_step_computation": "yes",
            "warm_start_init_point": "yes",
            "mu_oracle": "loqo",
            # "max_wall_time": self.dt - 0.01,
            "fixed_mu_oracle": "quality-function",
            "line_search_method": "filter",
            "expect_infeasible_problem": "no",
            "print_level": 0,
            # 'dual_inf_tol': 0.9
        }
        fatrop_options = {
            "mu_init": 0.1,
            "tol": 10e-6,
            "acceptable_tol": 10e-4,
            "warm_start_init_point": True,
        }

        self.solver_opts = {
            "verbose": False,
            "verbose_init": False,
            "print_time": False,
            "ipopt": ipopt_options,
            # 'fatrop': fatrop_options,
        }
        # Setup the optimization problem in casadi syntax
        self.solver, self.lbg, self.ubg = setup_optimization_problem(
            self.N,
            self.nr_joints,
            self.nr_segs,
            self.dt,
            self.solver_opts,
        )
        limits_path = ""
        if self.build:
            codegenopt = {"cpp": True}
            limits = {}
            limits["lbg"] = np.array(self.lbg)
            limits["ubg"] = np.array(self.ubg)
            np.savez(limits_path + "limits.npz", **limits)
            self.solver.generate_dependencies("gen_traj_opt_nlp_deps.cpp", codegenopt)
        else:
            limits = np.load(limits_path + "limits.npz")
            self.lbg = limits["lbg"]
            self.ubg = limits["ubg"]
            solver_file = f"{path}/code_generation/mpc{self.N}_segs{self.nr_segs}.so"
            self.solver = casadi.nlpsol(
                "solver", "ipopt", solver_file, self.solver_opts
            )

        # Set planner to have access to set finder
        self.planner = BoundPlanner(obstacles=self.obstacles, obs_size_increase=0.0)

    def set_desired_joint_config(self, qd):
        print(f"(BoundMPC) Setting desired joint config: {qd * 180/np.pi}")
        self.qd = qd

    def update(
        self,
        pos_points,
        rot_points,
        bp1,
        br1,
        e_r_bound,
        a_sets,
        b_sets,
        obstacles,
        v,
        p0=np.zeros(6),
        params=None,
    ):
        self.updated = True
        self.split_idxs = [0] + [self.N] * self.nr_segs
        self.k_update = 0.0
        self.updated_once = True
        self.switch = False
        self.p0 = p0

        self.planner.add_obstacle_reps(obstacles, update=True, reset=True)

        # Create reference path object
        self.ref_path = ReferencePath(
            pos_points,
            rot_points,
            bp1,
            br1,
            e_r_bound,
            a_sets,
            b_sets,
            self.nr_segs,
        )

        # Max path parameter
        self.phi_max = np.array([self.ref_path.phi_max])

        # Objective function weights
        self.weights = np.array(params.weights)

        # Path parameter initialization through projection
        dp0 = self.ref_path.dp[0]
        dp0 /= np.linalg.norm(dp0)
        dp1 = self.ref_path.dp[1]
        dp1 /= np.linalg.norm(dp1)
        self.phi_current = np.array([(p0[:3] - pos_points[0]).T @ dp0])
        self.p_ref = pos_points[0] + self.phi_current * dp0
        self.dp_ref = dp0
        self.phi_replan = self.phi_current
        v_proj = v[:3].T @ self.dp_ref
        self.dphi_current = np.array([v_proj])
        print("(Replanning) Setting new phi state:")
        print(f"(Replanning) phi: {self.phi_current[0]:.3f}")
        print(f"(Replanning) dphi: {self.dphi_current[0]:.3f}")

        # Reference integration variables
        self.pr_ref = integrate_rotation_reference(
            R.from_matrix(rot_points[0]).as_rotvec(),
            self.ref_path.dr[0],
            0.0,
            self.phi_current,
        )
        self.iw_ref = (
            self.ref_path.pd[3:, 0] + self.phi_current * self.ref_path.dpd[3:, 0]
        )

    def compute_orientation_projection_vectors(self, br1, br2, dp_normed_ref):
        # Compute necessary values for the projection vectors
        dp_ref_proj = np.empty_like(dp_normed_ref)
        br1_proj = np.empty_like(br1)
        br2_proj = np.empty_like(br2)
        for i in range(dp_normed_ref.shape[1]):
            dtau_rest1 = (
                R.from_rotvec(self.dtau_init[:, 0]).as_matrix()
                @ R.from_rotvec(self.dtau_init_orth1[:, i]).as_matrix().T
            )
            dtau_rest2 = (
                dtau_rest1 @ R.from_rotvec(self.dtau_init_par[:, i]).as_matrix().T
            )
            jac_dtau_r = jac_SO3_inv_right(self.dtau_init[:, 0])
            jac_dtau_l = jac_SO3_inv_left(self.dtau_init[:, 0])
            jac_r1_r = jac_SO3_inv_right(R.from_matrix(dtau_rest1).as_rotvec())
            jac_r2_r = jac_SO3_inv_right(R.from_matrix(dtau_rest2).as_rotvec())

            dp_ref_proj[:, i] = jac_r1_r @ dp_normed_ref[:, i]
            br1_proj[:, i] = jac_dtau_r @ br1[:, i]
            br2_proj[:, i] = jac_r2_r @ br2[:, i]

        # Projection vectors for the orientation errors
        v_1 = np.empty_like(br1)
        v_2 = np.empty_like(br1)
        v_3 = np.empty_like(br1)
        for j in range(dp_normed_ref.shape[1]):
            v1 = br1_proj[:, j]
            v2 = dp_ref_proj[:, j]
            v3 = br2_proj[:, j]
            a = np.dot(v1, v1)
            b = np.dot(v1, v2)
            c = np.dot(v1, v3)
            d = np.dot(v3, v3)
            e = np.dot(v2, v2)
            f = np.dot(v2, v3)
            g = v1
            h = v2
            i = v3
            v_1[:, j] = (
                -b * d * h + b * f * i - c * e * i + c * f * h + d * e * g - f**2 * g
            ) / (a * d * e - a * f**2 - b**2 * d + 2 * b * c * f - c**2 * e)
            v_2[:, j] = (
                a * d * h - a * f * i + b * c * i - b * d * g - c**2 * h + c * f * g
            ) / (a * d * e - a * f**2 - b**2 * d + 2 * b * c * f - c**2 * e)
            v_3[:, j] = (
                a * e * i - a * f * h - b**2 * i + b * c * h + b * f * g - c * e * g
            ) / (a * d * e - a * f**2 - b**2 * d + 2 * b * c * f - c**2 * e)
        return v_1, v_2, v_3, jac_dtau_l, jac_dtau_r

    def step(self, q0, dq0, ddq0, p0, v0, jerk_current, qf=np.zeros(7)):
        """One optimization step."""
        ts = time.perf_counter()
        # Update the reference trajectory
        (
            p_ref,
            dp_normed_ref,
            dp_ref,
            ddp_ref,
            phi_switch,
        ) = self.ref_path.get_parameters(self.switch)
        if self.switch:
            self.switch = False
        if self.dp_ref is None:
            self.dp_ref = dp_ref[:3, 0]
        bp1, bp2, br1, br2 = self.ref_path.get_basis_vectors()
        (
            e_r_bound,
            a_set,
            b_set,
        ) = self.ref_path.get_bound_params()

        # Set the initial guess based on wheter a previous solution was
        # acquired.
        if self.prev_solution is None:
            w0 = np.zeros(self.nr_x * self.N + self.nr_slacks)
            w0[0 : 7 * self.N] = np.repeat(q0, self.N)
            w0[4 * 7 * self.N : 4 * 7 * self.N + 6 * self.N] = np.repeat(p0, self.N)
            w0 = w0.flatten().tolist()
        else:
            w0 = cp.deepcopy(self.prev_solution)

            # Swap integrated omega if necessary
            p = w0[4 * 7 * self.N : 4 * 7 * self.N + 6 * self.N]
            i_omega = np.reshape(p, (6, self.N))
            if np.linalg.norm(p0[3:] - i_omega[3:, 0]) > 1.5:
                print("(BoundMPC) [INFO] Reversing integrated omega")
                prev_p1 = i_omega[3:, 0]
                i_omega[3:, :-1] = (p0[3:] + (i_omega[3:, 1:].T - prev_p1)).T
                i_omega[3:, -1] = i_omega[3:, -2]
            w0[4 * 7 * self.N : 4 * 7 * self.N + 6 * self.N] = i_omega.flatten()

            # w0 = w0.flatten().tolist()
            # # Shift solution
            # if not self.updated:
            #     s_N = int(len(w0)/self.N)
            #     w0[:(self.N-1)*s_N] = w0[s_N:]

        # Compute initial orientation errors at each via point for iterative
        # computation within the MPC
        prs = [self.pr_ref]
        for i in range(self.nr_segs - 1):
            prs.append(self.ref_path.r_taud[:, i + 1])
        for i in range(dp_ref.shape[1]):
            dtau_inits = compute_initial_rot_errors(
                p0[3:], prs[i], dp_normed_ref[:, i], br1[:, i], br2[:, i]
            )
            self.dtau_init[:, i] = dtau_inits[0]
            self.dtau_init_par[:, i] = dtau_inits[1]
            self.dtau_init_orth1[:, i] = dtau_inits[2]
            self.dtau_init_orth2[:, i] = dtau_inits[3]
        # print(np.linalg.norm(self.pr_ref) * 180 / np.pi)
        # print(np.linalg.norm(self.dtau_init_par[:, 0]) * 180 / np.pi)
        # print(np.linalg.norm(self.dtau_init_par[:, 1]) * 180 / np.pi)
        # print(np.linalg.norm(self.dtau_init_par[:, 2]) * 180 / np.pi)
        # print("---")

        # Compute orientation error projection vectors
        (
            v_1,
            v_2,
            v_3,
            jac_dtau_l,
            jac_dtau_r,
        ) = self.compute_orientation_projection_vectors(br1, br2, dp_normed_ref)

        # Limit desired path parameter to achieve limited speed
        x_phi_d = np.array([self.phi_max[0], 0, 0])
        x_phi_d_current = np.copy(x_phi_d)
        weights_current = np.copy(self.weights)
        if x_phi_d[0] < 1 and self.phi_max[0] > 0.001:
            scaling_factor = 1 / ((self.phi_max[0] - self.phi_current[0]) ** 2)
            scaling_factor = np.min((scaling_factor, 2.0))
            weights_current[4] *= scaling_factor

        # For very long trajectories, this is necessary to avoid numerical
        # issues, performance does not change at all
        phi_max = np.array([np.min((self.phi_current + 5.0, self.phi_max))])
        x_phi_d_current[0] = np.array(
            [np.min((self.phi_current[0] + 5.0, x_phi_d_current[0]))]
        )

        p_list = [self.robot_model.fk_pos_col(q0, i) for i in range(6)]
        p_list_f = [self.robot_model.fk_pos_col(qf, i) for i in range(6)]
        set_joints = []
        joint_sizes = self.robot_model.col_joint_sizes
        start_j = time.perf_counter()
        for i, (pl, pf) in enumerate(zip(p_list, p_list_f)):
            a_c, b_c, _ = self.planner.set_finder.find_set_collision_avoidance(
                pl, pf, limit_space=True, e_max=0.7
            )
            set_joints.append([a_c, b_c - joint_sizes[i]])
            # a_c, b_c = self.convex_sets_cpp.find_set_line(pl, pf, 0.5)
            # set_joints.append([a_c, b_c - joint_sizes[i]])

        sets_normed = normalize_set_size(set_joints, 15)
        a_set_joints = [x[0] for x in sets_normed]
        b_set_joints = np.array([x[1] for x in sets_normed])
        self.a_set_ca = a_set_joints
        self.b_set_ca = b_set_joints
        stop_j = time.perf_counter()
        # print(f"Collision sets took {stop_j - start_j:.3f}s")

        # Create parameter array
        # print(self.iw_ref)
        # print(self.pr_ref)
        # print(self.dtau_init[:, 0])
        # print(self.dtau_init_orth1)
        # print(p_ref[3:, 0], dp_ref[3:, 0])
        params = np.concatenate(
            (
                self.split_idxs,
                self.slacks0,
                self.iw_ref,
                self.dtau_init.T.flatten(),
                self.dtau_init_par.T.flatten(),
                self.dtau_init_orth1.T.flatten(),
                self.dtau_init_orth2.T.flatten(),
                x_phi_d_current,
                phi_switch,
                jac_dtau_r.T.flatten(),
                jac_dtau_l.T.flatten(),
                p_ref.flatten(),
                dp_ref.flatten(),
                dp_normed_ref.flatten(),
                bp1.flatten(),
                bp2.flatten(),
                br1.flatten(),
                br2.flatten(),
                e_r_bound.T.flatten(),
                weights_current,
                phi_max,
                v_1.flatten(),
                v_2.flatten(),
                v_3.flatten(),
                self.qd,
            )
        )

        for i in range(len(a_set)):
            params = np.concatenate((params, a_set[i].T.flatten()))
        params = np.concatenate((params, b_set.T.flatten()))
        for i in range(len(a_set_joints)):
            params = np.concatenate((params, a_set_joints[i].T.flatten()))
        params = np.concatenate((params, b_set_joints.T.flatten()))

        q_lb = self.q_lb
        qp_lb = self.dq_lb
        qpp_lb = self.ddq_lb
        u_lb = self.u_lb
        p_lb = self.p_lb
        v_lb = self.v_lb

        q_lb[0 : -1 : self.N] = q0
        qp_lb[0 : -1 : self.N] = dq0
        qpp_lb[0 : -1 : self.N] = ddq0
        u_lb[0 : -1 : self.N] = jerk_current
        p_lb[0 : -1 : self.N] = p0
        v_lb[0 : -1 : self.N] = v0

        # idx_end = self.N * 6 + 1
        # qp_lb[self.N - 1 : idx_end : self.N] = 0
        # qpp_lb[self.N - 1 : idx_end : self.N] = 0
        # u_lb[self.N - 1 : idx_end : self.N] = 0
        # v_lb[self.N - 1 : self.N * 6 + 1 : self.N] = 0
        # phi_lb[self.N - 1 : self.N * 3 + 1 : self.N] = np.array([-np.inf, 0, 0])

        lbx = np.concatenate([q_lb, qp_lb, qpp_lb, u_lb, p_lb, v_lb])
        lbx = np.concatenate([lbx, np.zeros(self.nr_slacks)])

        q_ub = self.q_ub
        qp_ub = self.dq_ub
        qpp_ub = self.ddq_ub
        u_ub = self.u_ub
        p_ub = self.p_ub
        v_ub = self.v_ub

        q_ub[0 : -1 : self.N] = q0
        qp_ub[0 : -1 : self.N] = dq0
        qpp_ub[0 : -1 : self.N] = ddq0
        u_ub[0 : -1 : self.N] = jerk_current
        p_ub[0 : -1 : self.N] = p0
        v_ub[0 : -1 : self.N] = v0

        # qp_ub[self.N - 1 : idx_end : self.N] = 0
        # qpp_ub[self.N - 1 : idx_end : self.N] = 0
        # u_ub[self.N - 1 : idx_end : self.N] = 0
        # v_ub[self.N - 1 : self.N * 6 + 1 : self.N] = 0
        # phi_ub[self.N - 1 : self.N * 3 + 1 : self.N] = np.array([np.inf, 0, 0])

        ubx = np.concatenate([q_ub, qp_ub, qpp_ub, u_ub, p_ub, v_ub])
        ubx = np.concatenate([ubx, np.inf * np.ones(self.nr_slacks)])

        tst = time.perf_counter()
        # print(f"--Pre step: {tst-ts:.3f}s")
        time_start = time.perf_counter()
        sol = self.solver(
            x0=w0,
            lbx=lbx,
            ubx=ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            # lam_g0=self.lam_g0,
            # lam_x0=self.lam_x0,
            p=params,
        )
        w_curr = sol["x"].full().flatten()
        time_elapsed = time.perf_counter() - time_start
        stats = self.solver.stats()
        iters = stats["iter_count"]
        # print(f"Slacks: {w_curr[-self.nr_slacks:]}")
        # cost = float(sol['f']) / self.N
        self.slacks0 += w_curr[-6:]

        # Check for constraint violations
        g = np.array(sol["g"]).flatten()
        g_viol = -np.sum(g[np.where(g < np.array(self.lbg) - 1e-6)[0]])
        g_viol += np.sum(g[np.where(g > np.array(self.ubg) + 1e-6)[0]])

        success = stats["success"] or g_viol < 1e-4

        using_previous = False
        if not success:
            self.error_count += 1
            print(
                f"(BoundMPC) [ERROR] Could not find feasible solution. Using previous solution. Error count: {self.error_count}"
            )
            print(f"(BoundMPC) Constraint Violation Sum: {g_viol}")
            print(f"(BoundMPC) Casadi status: {stats['return_status']}")
            if self.prev_solution is not None:
                w_opt = np.copy(self.prev_solution)
                using_previous = True
            else:
                print(
                    "(BoundMPC) [WARNING] Previous solution not found, using infeasible solution."
                )
                self.error_count = 0
                w_opt = w_curr
                using_previous = True
                # self.prev_solution = w_opt
                self.lam_g0 = sol["lam_g"]
                self.lam_x0 = sol["lam_x"]
        else:
            self.error_count = 0
            w_opt = w_curr
            self.prev_solution = cp.deepcopy(w_opt)
            self.lam_g0 = sol["lam_g"]
            self.lam_x0 = sol["lam_x"]

        traj_data, ref_data, err_data = self.compute_return_data(
            q0,
            dq0,
            ddq0,
            jerk_current,
            p0,
            w_opt,
            using_previous,
            e_r_bound,
            jac_dtau_l,
            jac_dtau_r,
            p_ref,
            dp_normed_ref,
            dp_ref,
            phi_switch,
            bp1,
            bp2,
            br1,
            br2,
            v_1,
            v_2,
            v_3,
            x_phi_d_current,
            phi_max,
            a_set,
            b_set,
            a_set_joints,
            b_set_joints,
        )
        return traj_data, ref_data, err_data, time_elapsed, iters

    def compute_return_data(
        self,
        q0,
        dq0,
        ddq0,
        jerk_current,
        p0,
        w_opt,
        using_previous,
        e_r_bound,
        jac_dtau_l,
        jac_dtau_r,
        p_ref,
        dp_normed_ref,
        dp_ref,
        phi_switch,
        bp1,
        bp2,
        br1,
        br2,
        v1,
        v2,
        v3,
        x_phi_d,
        phi_max,
        a_set,
        b_set,
        a_set_joints,
        b_set_joints,
    ):
        # w_opt = np.array(w_opt.reshape((self.N, -1))).T
        optimal_q = np.reshape(w_opt[0 : 7 * self.N], (self.N, 7), "F").T
        optimal_dq = np.reshape(w_opt[7 * self.N : 2 * 7 * self.N], (self.N, 7), "F").T
        optimal_ddq = np.reshape(
            w_opt[2 * 7 * self.N : 3 * 7 * self.N], (self.N, 7), "F"
        ).T
        optimal_jerk = np.reshape(
            w_opt[3 * 7 * self.N : 4 * 7 * self.N], (self.N, 7), "F"
        ).T
        optimal_traj = np.reshape(
            w_opt[4 * 7 * self.N : 4 * 7 * self.N + 6 * self.N],
            (self.N, 6),
            "F",
        ).T
        optimal_vel = np.reshape(
            w_opt[4 * 7 * self.N + 6 * self.N : 4 * 7 * self.N + 12 * self.N],
            (self.N, 6),
            "F",
        ).T
        pslacks = w_opt[-2 * self.N : -self.N]
        optimal_q = optimal_q[:, self.error_count :]
        optimal_dq = optimal_dq[:, self.error_count :]
        optimal_ddq = optimal_ddq[:, self.error_count :]
        optimal_jerk = optimal_jerk[:, self.error_count :]
        optimal_traj = optimal_traj[:, self.error_count :]
        optimal_vel = optimal_vel[:, self.error_count :]
        optimal_acc = optimal_vel

        optimal_phi = np.empty(optimal_jerk.shape[1])
        optimal_phi[0] = self.phi_current[0]
        optimal_dphi = np.empty(optimal_jerk.shape[1])
        optimal_dphi[0] = self.dphi_current[0]

        # Create reference trajectory
        iw_ref_copy = np.copy(self.iw_ref)
        # Compute cartesian trajectories
        optimal_i_omega = np.copy(optimal_traj)

        i_omega_0 = p0[3:]
        split_idx_prev = self.split_idxs.copy()
        ref_data = self.ref_data
        err_data = self.err_data
        for i in range(0, optimal_phi.shape[0]):
            reference = reference_function(
                dp_ref=dp_ref.T,
                p_ref=p_ref.T,
                p=np.expand_dims(optimal_traj[:, i], 1),
                v=np.expand_dims(optimal_vel[:, i], 1),
                dp_normed_ref=dp_normed_ref.T,
                phi_switch=np.expand_dims(phi_switch, 1),
                bp1=bp1.T,
                bp2=bp2.T,
                br1=br1.T,
                br2=br2.T,
                v1=v1.T,
                v2=v2.T,
                v3=v3.T,
                e_r_bound=e_r_bound,
                split_idx=split_idx_prev,
                idx=i,
                n_horizon=self.N,
                a_set=a_set,
                b_set=b_set,
            )
            p_d = reference["p_d"]
            p_dr_next = reference["p_dr_next"]
            dp_d = reference["dp_d"]
            ddp_d = reference["ddp_d"]
            dp_normed_d = reference["dp_normed_d"]
            dp_normed_dn = reference["dp_normed_n"]
            phi = reference["phi"]
            dphi = reference["dphi"]
            br1c = reference["br1_current"]
            br2c = reference["br2_current"]
            br1n = reference["br1_next"]
            br2n = reference["br2_next"]
            optimal_phi[i] = phi
            optimal_dphi[i] = dphi

            ref_data["p"][i] = p_d
            ref_data["dp"][i] = dp_d
            ref_data["ddp"][i] = ddp_d
            ref_data["dp_normed"][i] = dp_normed_d
            ref_data["dp_normedn"][i] = dp_normed_dn
            ref_data["bp1"][i] = reference["bp1_current"]
            ref_data["bp2"][i] = reference["bp2_current"]
            ref_data["br1"][i] = reference["br1_current"]
            ref_data["br2"][i] = reference["br2_current"]
            ref_data["br1_next"][i] = reference["br1_next"]
            ref_data["br2_next"][i] = reference["br2_next"]
            ref_data["v1"][i] = reference["v1_current"]
            ref_data["v2"][i] = reference["v2_current"]
            ref_data["v3"][i] = reference["v3_current"]
            ref_data["v1_next"][i] = reference["v1_next"]
            ref_data["v2_next"][i] = reference["v2_next"]
            ref_data["v3_next"][i] = reference["v3_next"]
            ref_data["p_r_omega0"][i] = reference["p_r_omega0"]
            if i == 1:
                ref_data["a_current"] = reference["a_current"].flatten()
                ref_data["b_current"] = reference["b_current"]
                ref_data["a_next"] = reference["a_next"].flatten()
                ref_data["b_next"] = reference["b_next"]
                ref_data["a_j3"] = a_set_joints[0].flatten()
                ref_data["a_j5"] = a_set_joints[1].flatten()
                ref_data["a_j6"] = a_set_joints[2].flatten()
                ref_data["a_j67"] = a_set_joints[3].flatten()
                ref_data["a_elbow"] = a_set_joints[4].flatten()
                ref_data["b_j3"] = b_set_joints[0]
                ref_data["b_j5"] = b_set_joints[1]
                ref_data["b_j6"] = b_set_joints[2]
                ref_data["b_j67"] = b_set_joints[3]
                ref_data["b_elbow"] = b_set_joints[4]

            # Compute errors
            errors = error_function(
                p=optimal_i_omega[:, i],
                pr_next=p_dr_next,
                v=optimal_vel[:, i],
                p_ref=ref_data["p"][i],
                dp_ref=ref_data["dp"][i],
                ddp_ref=ref_data["ddp"][i],
                dp_normed_ref=ref_data["dp_normed"][i],
                dp_normed_refn=ref_data["dp_normedn"][i],
                dphi=dphi,
                i_omega_0=i_omega_0,
                i_omega_ref_0=iw_ref_copy,
                i_omega_ref_seg=ref_data["p_r_omega0"][i].T,
                dtau_init=self.dtau_init.T,
                dtau_init_par=self.dtau_init_par.T,
                dtau_init_orth1=self.dtau_init_orth1.T,
                dtau_init_orth2=self.dtau_init_orth2.T,
                br1=ref_data["br1"][i].T,
                br2=ref_data["br2"][i].T,
                br1n=ref_data["br1_next"][i].T,
                br2n=ref_data["br2_next"][i].T,
                jac_dtau_l=jac_dtau_l,
                jac_dtau_r=jac_dtau_r,
                v1=ref_data["v1"][i].T,
                v2=ref_data["v2"][i].T,
                v3=ref_data["v3"][i].T,
                v1n=ref_data["v1_next"][i].T,
                v2n=ref_data["v2_next"][i].T,
                v3n=ref_data["v3_next"][i].T,
                split_idx=split_idx_prev,
                idx=i,
                n_horizon=self.N,
            )
            e_p_par = errors["e_p_par"]
            e_p_orth = errors["e_p_orth"]
            de_p_par = errors["de_p_par"]
            de_p_orth = errors["de_p_orth"]
            e_p = errors["e_p"]
            de_p = errors["de_p"]
            e_r_par = errors["e_r_par"] @ dp_normed_d
            e_r_orth1 = errors["e_r_orth1"] @ br1c
            e_r_orth2 = errors["e_r_orth2"] @ br2c
            e_r_parn = errors["e_r_parn"] @ dp_normed_dn
            e_r_orth1n = errors["e_r_orth1n"] @ br1n
            e_r_orth2n = errors["e_r_orth2n"] @ br2n
            e_r = errors["e_r"]
            de_r = errors["de_r"]

            r_bound_lower = reference["r_bound_lower"]
            r_bound_upper = reference["r_bound_upper"]
            r_bound_lowern = reference["r_bound_lower_next"]
            r_bound_uppern = reference["r_bound_upper_next"]
            ref_data["r_bound_lower"][i] = r_bound_lower
            ref_data["r_bound_upper"][i] = r_bound_upper
            ref_data["r_bound_lower_next"][i] = r_bound_lowern
            ref_data["r_bound_upper_next"][i] = r_bound_uppern

            err_data["e_p"][i] = e_p
            err_data["de_p"][i] = de_p
            err_data["e_p_par"][i] = e_p_par
            err_data["e_p_orth"][i] = e_p_orth
            err_data["de_p_par"][i] = de_p_par
            err_data["de_p_orth"][i] = de_p_orth
            err_data["e_r"][i] = np.copy(e_r)
            err_data["de_r"][i] = de_r
            err_data["e_r_par"][i] = e_r_par
            err_data["e_r_orth1"][i] = e_r_orth1
            err_data["e_r_orth2"][i] = e_r_orth2
            err_data["e_r_parn"][i] = e_r_parn
            err_data["e_r_orth1n"][i] = e_r_orth1n
            err_data["e_r_orth2n"][i] = e_r_orth2n

        # Integrate the rotation reference
        if self.split_idxs[1] == 1:
            self.pr_ref = R.from_matrix(
                self.ref_path.r[self.ref_path.sector + 1]
            ).as_rotvec()
            self.pr_ref = integrate_rotation_reference(
                self.pr_ref, dp_ref[3:, 1], phi_switch[1], optimal_phi[1]
            )
            self.iw_ref = (
                p_ref[3:, 1] + (optimal_phi[1] - phi_switch[1]) * dp_ref[3:, 1]
            )
        else:
            self.pr_ref = R.from_matrix(
                self.ref_path.r[self.ref_path.sector]
            ).as_rotvec()
            self.pr_ref = integrate_rotation_reference(
                self.pr_ref, dp_ref[3:, 0], phi_switch[0], optimal_phi[1]
            )
            self.iw_ref = (
                p_ref[3:, 0] + (optimal_phi[1] - phi_switch[0]) * dp_ref[3:, 0]
            )

        # Update the splitting idx
        in_set_accuracy = 0.005
        for i in range(1, self.nr_segs - 1):
            # for i in [1]:
            if self.split_idxs[i] < self.N:
                self.split_idxs[i] -= 1
                if self.split_idxs[i] == 0:
                    self.switch = True
                    self.split_idxs[i] = self.N
            elif self.error_count == 0:
                dswitch = optimal_phi > phi_switch[i] - 0.03
                d_in_set0 = np.max(
                    a_set[i - 1] @ optimal_traj[:3, :]
                    - np.expand_dims(b_set[i - 1], 1),
                    axis=0,
                )
                d_in_set1 = np.max(
                    a_set[i] @ optimal_traj[:3, :] - np.expand_dims(b_set[i], 1),
                    axis=0,
                )
                in_set0 = d_in_set0 < in_set_accuracy + pslacks
                in_set1 = d_in_set1 < in_set_accuracy + pslacks
                idx_seg = np.where(np.array(split_idx_prev) < self.N)[0][0] + 1
                pr_ref_end = R.from_matrix(
                    self.ref_path.r[self.ref_path.sector + idx_seg]
                ).as_rotvec()
                pr_ref_end = integrate_rotation_reference(
                    pr_ref_end,
                    dp_ref[3:, idx_seg],
                    phi_switch[idx_seg + 1],
                    optimal_phi[-1],
                )
                e_r_parn = np.array(err_data["e_r_parn"])
                e_r_orth1n = np.array(err_data["e_r_orth1n"])
                e_r_orth2n = np.array(err_data["e_r_orth2n"])
                e_r_par = np.array(err_data["e_r_par"])
                e_r_orth1 = np.array(err_data["e_r_orth1"])
                e_r_orth2 = np.array(err_data["e_r_orth2"])
                e_rs = np.vstack((e_r_orth1, e_r_par, e_r_orth2)).T
                e_rsn = np.array([e_r_orth1n, e_r_parn, e_r_orth2n]).T
                rot_lower = np.array(ref_data["r_bound_lower"])
                rot_upper = np.array(ref_data["r_bound_upper"])
                rot_lowern = np.array(ref_data["r_bound_lower_next"])
                rot_uppern = np.array(ref_data["r_bound_upper_next"])
                in_next_rot_bounds = (e_rs < rot_upper) * (e_rs > rot_lower)
                in_next_rot_bounds = (
                    in_next_rot_bounds
                    * (e_rsn < rot_uppern + 5 * np.pi / 180)
                    * (e_rsn > rot_lowern - 5 * np.pi / 180)
                )
                in_next_rot_bounds = np.min(in_next_rot_bounds, axis=1)
                # Set all values of in_set1 to false that are before the last false one
                last_false = np.where(in_set1 == False)[-1]
                if last_false.shape[0] > 0:
                    in_set1[: last_false[-1]] = False
                # idx_new = np.where(dswitch * in_set0 * in_set1)[0]
                # if i == 1:
                #     print("Current distance from switching")
                #     print(f"Phi {np.min(optimal_phi - phi_switch[i])}")
                #     print(f"Set0 {np.min(d_in_set0 - pslacks)}")
                #     print(f"Set1 {np.min(d_in_set1 - pslacks)}")
                #     print(f"Rot_up {np.min(e_rs - rot_upper)}")
                #     print(f"Rot_low {np.min(-e_rs + rot_lower)}")
                #     print(f"NRot_up {np.min(e_rsn - rot_uppern)}")
                #     print(f"NRot_low {np.min(-e_rsn + rot_lowern)}")
                idx_new = np.where(dswitch * in_set0 * in_set1 * in_next_rot_bounds)[0]
                not_at_end = self.ref_path.sector + (i - 1) < self.ref_path.num_sectors
                # if i == 1:
                #     print(dswitch)
                #     print(in_set0)
                #     print(in_set1)
                #     print(in_next_rot_bounds)
                if idx_new.shape[0] > 0 and not_at_end:
                    if self.split_idxs[i] == self.N:
                        self.split_idxs[i] = idx_new[0] - 1
                        # Update via point
                        print(f"Adapting Path Vias {i} {idx_new[0]}")
                        sec = self.ref_path.sector
                        dp = dp_ref[:3, i]
                        pv = p_ref[:3, i]
                        p_switch = optimal_traj[:3, idx_new[0]]
                        phi_correction = (p_switch - pv) @ dp
                        pv_new = pv + phi_correction * dp

                        # old_phi = np.copy(self.ref_path.phi[sec + i + 1])
                        self.ref_path.pd[:3, i] = pv_new
                        self.ref_path.p[sec + i] = pv_new
                        self.ref_path.phi[sec + i + 1] -= phi_correction
                        self.ref_path.phi_switch[i + 1 :] -= phi_correction
                        self.ref_path.phi_max = (
                            np.array(self.ref_path.phi).cumsum()[
                                self.ref_path.num_sectors + 1
                            ]
                            + self.ref_path.phi_bias
                        )
                        self.phi_max = np.array([self.ref_path.phi_max])
                    if self.split_idxs[i] == 0:
                        self.switch = True
        if self.switch:
            print("--> Switching segment")
            self.split_idxs[1:-1] = self.split_idxs[2:]
            self.split_idxs[-1] = self.N

        for i in range(1, phi_switch.shape[0] - 1):
            if self.split_idxs[i] <= self.split_idxs[i - 1]:
                self.split_idxs[i] = np.min((self.N, self.split_idxs[i - 1] + 1))
        # print(self.split_idxs)

        self.phi_current = np.array([optimal_phi[1]])
        self.dphi_current = np.array([optimal_dphi[1]])

        ref_data["p"][0][3:] = self.pr_ref

        traj_data = {}
        traj_data["p"] = optimal_traj[:, 1:]
        traj_data["v"] = optimal_vel[:, 1:]
        traj_data["a"] = optimal_acc[:, 1:]
        traj_data["q"] = optimal_q[:, 1:]
        traj_data["dq"] = optimal_dq[:, 1:]
        traj_data["ddq"] = optimal_ddq[:, 1:]
        traj_data["dddq"] = optimal_jerk
        traj_data["phi"] = optimal_phi[1:]
        traj_data["dphi"] = optimal_dphi[1:]

        return traj_data, ref_data, err_data
