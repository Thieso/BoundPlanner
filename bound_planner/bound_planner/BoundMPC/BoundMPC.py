import copy as cp
import time
from collections import defaultdict
from importlib.resources import files

import casadi
import numpy as np
# from convexsets import ConvexSets
from scipy.spatial.transform import Rotation as R

from bound_planner.ConvexSetPlanner.ConvexSetPlanner import ConvexSetPlanner
from bound_planner.utils import normalize_set_size

from ..ReferencePath import ReferencePath
from ..RobotModel import RobotModel
from ..utils import compute_initial_rot_errors, integrate_rotation_reference
from ..utils.lie_functions import jac_SO3_inv_left, jac_SO3_inv_right
from .bound_mpc_functions import error_function, reference_function
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
        e_p_start,
        e_p_end,
        e_p_mid,
        e_r_start,
        e_r_end,
        e_r_mid,
        a_sets,
        b_sets,
        obstacles,
        p0=np.zeros(6),
        params=None,
    ):
        # Prediction horizon
        self.N = params.n

        self.robot_model = RobotModel()

        self.updated = False
        self.k_update = 0.0
        self.updated_once = False
        self.nr_slacks = 8

        # Collision objects
        self.max_col_objs = 20
        self.obj_centers = 100 * np.ones((self.max_col_objs, 3))
        self.obj_radii = np.zeros(self.max_col_objs)
        self.obstacles = obstacles

        # Flag whether to compile the OCP problem
        self.build = params.build

        # Flag wether to create logging data
        self.log = not params.real_time

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
            e_p_start,
            e_p_end,
            e_p_mid,
            e_r_start,
            e_r_end,
            e_r_mid,
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
        # self.weights[4] /= self.phi_max[0]
        self.dphi_max = np.array([self.weights[4]])

        # Reference integration variables
        self.p_ref = p0[:3]
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
        self.nr_x = 50

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

        self.uphi_ub = self.robot_model.u_max * np.ones(self.N)
        self.uphi_lb = self.robot_model.u_min * np.ones(self.N)

        self.phi_lb = np.array([0.0, -self.dphi_max[0], -np.inf])
        self.phi_lb = np.repeat(self.phi_lb, self.N)
        self.phi_ub = np.array([np.inf, self.dphi_max[0], np.inf])
        self.phi_ub = np.repeat(self.phi_ub, self.N)

        self.p_ub = np.inf * np.ones(self.N * 6)
        self.p_lb = -np.inf * np.ones(self.N * 6)
        self.v_ub = np.inf * np.ones(self.N * 6)
        self.v_lb = -np.inf * np.ones(self.N * 6)
        self.p_r_ub = np.inf * np.ones(self.N * 3)
        self.p_r_lb = -np.inf * np.ones(self.N * 3)
        self.dp_r_ub = np.inf * np.ones(self.N * 3)
        self.dp_r_lb = -np.inf * np.ones(self.N * 3)

        # Solution of previous run
        self.prev_solution = None
        self.lam_g0 = 0
        self.lam_x0 = 0
        self.lam = None
        self.pi = None

        # Setup the optimization problem
        path = files("bound_planner")
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
            self.max_col_objs,
            self.solver_opts,
        )
        if self.build:
            codegenopt = {"cpp": True}
            self.solver.generate_dependencies("gen_traj_opt_nlp_deps.cpp", codegenopt)
        else:
            solver_file = f"{path}/code_generation/mpc{self.N}_segs{self.nr_segs}.so"
            self.solver = casadi.nlpsol(
                "solver", "ipopt", solver_file, self.solver_opts
            )

        # Set planner to have access to set finder
        self.planner = ConvexSetPlanner(obstacles=self.obstacles, obs_size_increase=0.0)
        # obs_sets_cpp = []
        # for os in self.planner.obs_sets:
        #     obs_sets_cpp.append((os[0], os[1]))
        # self.convex_sets_cpp = ConvexSets(obs_sets_cpp, self.planner.obs_points_sets)

    def set_desired_joint_config(self, qd):
        print(f"(BoundMPC) Setting desired joint config: {qd * 180/np.pi}")
        self.qd = qd

    def update(
        self,
        pos_points,
        rot_points,
        bp1,
        br1,
        e_p_start,
        e_p_end,
        e_p_mid,
        e_r_start,
        e_r_end,
        e_r_mid,
        a_sets,
        b_sets,
        v,
        a,
        jerk,
        p0=np.zeros(6),
        params=None,
    ):
        self.updated = True
        self.split_idxs = [0] + [self.N] * self.nr_segs
        self.k_update = 0.0
        self.updated_once = True
        self.switch = False
        self.p0 = p0

        # Create reference path object
        self.ref_path = ReferencePath(
            pos_points,
            rot_points,
            bp1,
            br1,
            e_p_start,
            e_p_end,
            e_p_mid,
            e_r_start,
            e_r_end,
            e_r_mid,
            a_sets,
            b_sets,
            self.nr_segs,
        )

        # Max path parameter
        self.phi_max = np.array([self.ref_path.phi_max])

        # Objective function weights
        self.weights = np.array(params.weights)

        # Path parameter initalization through projection
        dp0 = self.ref_path.dp[0]
        dp0 /= np.linalg.norm(dp0)
        dp1 = self.ref_path.dp[1]
        dp1 /= np.linalg.norm(dp1)
        self.phi_current = np.array([(p0[:3] - pos_points[0]).T @ dp0])
        euler_start = self.ref_path.phi_switch[1] - self.ref_path.spiral_l[1]
        in_set0 = np.max(a_sets[0] @ p0[:3] - b_sets[0]) < 1e-8
        if self.phi_current > euler_start and in_set0:
            print("(Replanning) Projecting onto euler spiral")
            p_ref = pos_points[0] + euler_start * dp0
            dp_ref = dp0
            phi_spiral = 0.0
            phi_step = 0.001
            while (
                p_ref - p0[:3]
            ).T @ dp_ref < 0 and phi_spiral < self.ref_path.spiral_l[1]:
                print(
                    f"Tangential error at phi = {euler_start + phi_spiral:.3f}: {(p_ref - p0[:3]).T @ dp_ref:.3f}"
                )
                phi_spiral += phi_step
                b = self.ref_path.spiral_a[1]
                rot_2d = self.ref_path.spiral_rot_2d[1]
                dp_ref2d = np.array(
                    [np.cos(b * phi_spiral**2), np.sin(b * phi_spiral**2)]
                )
                dp_ref = rot_2d.T @ dp_ref2d
                p_ref += phi_step * dp_ref
            self.p_ref = p_ref
            self.dp_ref = dp_ref
            self.phi_current = np.array([euler_start + phi_spiral])
        else:
            self.p_ref = pos_points[0] + self.phi_current * dp0
            self.dp_ref = dp0
        self.phi_replan = self.phi_current
        v_proj = v[:3].T @ self.dp_ref
        a_proj = a[:3].T @ self.dp_ref
        j_proj = jerk[:3].T @ self.dp_ref
        self.dphi_current = np.array([v_proj])
        self.ddphi_current = np.array([a_proj])
        self.dddphi_current = np.array([j_proj])
        print("(Replanning) Setting new phi state:")
        print(f"(Replanning) phi: {self.phi_current[0]:.3f}")
        print(f"(Replanning) dphi: {self.dphi_current[0]:.3f}")
        print(f"(Replanning) ddphi: {self.ddphi_current[0]:.3f}")
        print(f"(Replanning) dddphi: {self.dddphi_current[0]:.3f}")

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

    def set_collision_objects(self, obj_centers, obj_radii):
        print("(BoundMPC) Setting collision objects")
        nr_objs = obj_centers.shape[0]
        self.obj_centers = 100 * np.ones((self.max_col_objs, 3))
        self.obj_radii = np.zeros(self.max_col_objs)
        if obj_centers.shape[0] <= self.max_col_objs:
            self.obj_centers[:nr_objs, :] = obj_centers
            self.obj_radii[:nr_objs] = obj_radii
        else:
            print(
                f"(BoundMPC) Amount of collision objects {nr_objs} exceeds maximum {self.max_col_objs}"
            )
            self.obj_centers = obj_centers[: self.max_col_objs, :]
            self.obj_radii = obj_radii[: self.max_col_objs]

    def compute_error_bounds(
        self, phi_switch, e_p_start, e_p_end, e_p_mid, e_r_start, e_r_end, e_r_mid
    ):
        # Compute error bound parameters
        a6 = np.empty((self.nr_segs + 1, 10))
        a5 = np.empty((self.nr_segs + 1, 10))
        a4 = np.empty((self.nr_segs + 1, 10))
        a3 = np.empty((self.nr_segs + 1, 10))
        a2 = np.empty((self.nr_segs + 1, 10))
        a1 = np.empty((self.nr_segs + 1, 10))
        a0 = np.empty((self.nr_segs + 1, 10))
        for i in range(a4.shape[0] - 1):
            phi0 = 0
            phi1 = phi_switch[i + 1] - phi_switch[i]
            e_start = np.concatenate((e_p_start[i], e_r_start[i]))
            e_end = np.concatenate((e_p_end[i], e_r_end[i]))
            e_mid = np.concatenate((e_p_mid[i], e_r_mid[i]))

            (
                a6[i, :],
                a5[i, :],
                a4[i, :],
                a3[i, :],
                a2[i, :],
                a1[i, :],
                a0[i, :],
            ) = compute_bound_params_six(0, phi1 - phi0, e_start, e_end, 0, e_mid)

        return a0, a1, a2, a3, a4, a5, a6

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
        # Update the reference trajectory
        (
            p_ref,
            dp_normed_ref,
            dp_ref,
            ddp_ref,
            phi_switch,
        ) = self.ref_path.get_parameters(self.switch, self.phi_current)
        if self.switch:
            self.switch = False
        if self.dp_ref is None:
            self.dp_ref = dp_ref[:3, 0]
        bp1, bp2, br1, br2 = self.ref_path.get_basis_vectors()
        (
            e_p_start,
            e_p_end,
            e_p_mid,
            e_r_start,
            e_r_end,
            e_r_mid,
            a_set,
            b_set,
        ) = self.ref_path.get_bound_params()
        (
            spiral_a,
            spiral_l,
            spiral_theta,
            spiral_rot2d,
            spiral_rot_bp1,
            spiral_rot_bp1_norm,
            spiral_rot_bp2_norm,
        ) = self.ref_path.get_spiral_params()

        # Set the initial guess based on wheter a previous solution was
        # acquired.
        if self.prev_solution is None:
            w0 = np.zeros(self.nr_x * self.N + self.nr_slacks)
            w0[0 : 7 * self.N] = np.repeat(q0, self.N)
            w0[
                4 * 7 * self.N + 4 * self.N : 4 * 7 * self.N + 4 * self.N + 6 * self.N
            ] = np.repeat(p0, self.N)
            w0 = w0.flatten().tolist()
        else:
            w0 = cp.deepcopy(self.prev_solution)

            # Swap integrated omega if necessary
            p = w0[
                4 * 7 * self.N + 4 * self.N : 4 * 7 * self.N + +4 * self.N + 6 * self.N
            ]
            i_omega = np.reshape(p, (6, self.N))
            if np.linalg.norm(p0[3:] - i_omega[3:, 0]) > 1.5:
                print("(BoundMPC) [INFO] Reversing integrated omega")
                prev_p1 = i_omega[3:, 0]
                i_omega[3:, :-1] = (p0[3:] + (i_omega[3:, 1:].T - prev_p1)).T
                i_omega[3:, -1] = i_omega[3:, -2]
            w0[
                4 * 7 * self.N + 4 * self.N : 4 * 7 * self.N + +4 * self.N + 6 * self.N
            ] = i_omega.flatten()

            if self.updated:
                idx_cur = 0
                dp_new = dp_ref[:3, idx_cur]
                p_ref_new = p_ref[:3, idx_cur]

                # Compute previous Cartesian trajectory
                optimal_q = np.reshape(w0[0 : 7 * self.N], (self.N, 7), "F").T
                optimal_dq = np.reshape(
                    w0[7 * self.N : 2 * 7 * self.N], (self.N, 7), "F"
                ).T
                optimal_ddq = np.reshape(
                    w0[2 * 7 * self.N : 3 * 7 * self.N], (self.N, 7), "F"
                ).T
                optimal_jerk = np.reshape(
                    w0[3 * 7 * self.N : 4 * 7 * self.N], (self.N, 7), "F"
                ).T
                prev_traj = np.empty((6, optimal_q.shape[1]))
                prev_vel = np.empty((6, optimal_q.shape[1]))
                prev_acc = np.empty((6, optimal_q.shape[1]))
                prev_jerk = np.empty((6, optimal_q.shape[1]))
                for i in range(optimal_q.shape[1]):
                    p_c, jac_fk, djac_fk = self.robot_model.forward_kinematics(
                        optimal_q[:, i], optimal_dq[:, i]
                    )
                    ddjac_fk = self.robot_model.ddjacobian_fk(
                        optimal_q[:, i], optimal_dq[:, i], optimal_ddq[:, i]
                    )
                    prev_traj[:, i] = np.copy(p_c)
                    prev_vel[:, i] = jac_fk @ optimal_dq[:, i]
                    prev_acc[:, i] = (
                        jac_fk @ optimal_ddq[:, i] + djac_fk @ optimal_dq[:, i]
                    )
                    prev_jerk[:, i] = (
                        jac_fk @ optimal_jerk[:, i]
                        + 2 * djac_fk @ optimal_ddq[:, i]
                        + ddjac_fk @ optimal_dq[:, i]
                    )

                pk = prev_traj[:3, :]
                vk = prev_vel[:3, :]
                ak = prev_acc[:3, :]
                jk = prev_jerk[:3, :]
                phi_new = phi_switch[idx_cur] + (pk.T - p_ref_new) @ dp_new
                dphi_new = vk.T @ dp_new
                ddphi_new = ak.T @ dp_new
                dddphi_new = jk.T @ dp_new
                # Make sure that the initial phi value never exceeds the
                # segment
                if np.max(phi_new) > phi_switch[idx_cur + 1]:
                    scal = phi_switch[idx_cur + 1] / np.max(phi_new)
                    phi_new *= scal
                    dphi_new *= scal
                    ddphi_new *= scal
                    dddphi_new *= scal

                w0[4 * 7 * self.N + self.N : 4 * 7 * self.N + 2 * self.N] = phi_new
                w0[4 * 7 * self.N + 2 * self.N : 4 * 7 * self.N + 3 * self.N] = dphi_new
                w0[
                    4 * 7 * self.N + 3 * self.N : 4 * 7 * self.N + 4 * self.N
                ] = ddphi_new
                w0[4 * 7 * self.N : 4 * 7 * self.N + self.N] = dddphi_new

            # w0 = w0.flatten().tolist()
            # # Shift solution
            # if not self.updated:
            #     s_N = int(len(w0)/self.N)
            #     w0[:(self.N-1)*s_N] = w0[s_N:]

        # Compute initial orientation errors at each via point for iterative
        # computation within the MPC
        for i in range(dp_ref.shape[1]):
            dtau_inits = compute_initial_rot_errors(
                p0[3:], self.pr_ref, dp_normed_ref[:, i], br1[:, i], br2[:, i]
            )
            self.dtau_init[:, i] = dtau_inits[0]
            self.dtau_init_par[:, i] = dtau_inits[1]
            self.dtau_init_orth1[:, i] = dtau_inits[2]
            self.dtau_init_orth2[:, i] = dtau_inits[3]

        # Compute orientation error projection vectors
        (
            v_1,
            v_2,
            v_3,
            jac_dtau_l,
            jac_dtau_r,
        ) = self.compute_orientation_projection_vectors(br1, br2, dp_normed_ref)

        # Compute polynomial parameters of error bounds
        a0, a1, a2, a3, a4, a5, a6 = self.compute_error_bounds(
            phi_switch, e_p_start, e_p_end, e_p_mid, e_r_start, e_r_end, e_r_mid
        )

        # Limit desired path parameter to achieve limited speed
        x_phi_d = np.array([self.phi_max[0], 0, 0])
        x_phi_d_current = np.copy(x_phi_d)
        weights_current = np.copy(self.weights)
        if x_phi_d[0] < 1:
            scaling_factor = 1 / ((self.phi_max[0] - self.phi_current[0]) ** 2)
            scaling_factor = np.min((scaling_factor, 2.0))
            weights_current[7] *= scaling_factor
        # if self.updated and self.k_update < 1.0:
        #     weights_current[7] = 0.0 + self.k_update * weights_current[7]
        #     self.k_update = np.min((1, self.k_update + 0.1))

        # For very long trajectories, this is necessary to avoid numerical
        # issues, performance does not change at all
        phi_max = np.array([np.min((self.phi_current + 5.0, self.phi_max))])
        x_phi_d_current[0] = np.array(
            [np.min((self.phi_current[0] + 5.0, x_phi_d_current[0]))]
        )

        p_list = [
            self.robot_model.fk_pos_j3(q0),
            self.robot_model.fk_pos_j5(q0),
            self.robot_model.fk_pos_j6(q0),
            self.robot_model.fk_pos_j67(q0),
            self.robot_model.fk_pos_j67(q0, 2.5),
            self.robot_model.fk_pos_elbow(q0),
        ]
        p_list_f = [
            self.robot_model.fk_pos_j3(qf),
            self.robot_model.fk_pos_j5(qf),
            self.robot_model.fk_pos_j6(qf),
            self.robot_model.fk_pos_j67(qf),
            self.robot_model.fk_pos_j67(qf, 2.5),
            self.robot_model.fk_pos_elbow(qf),
        ]
        set_joints = []
        joint_sizes = [0.09, 0.09, 0.09, 0.06, 0.06, 0.11]
        start_j = time.perf_counter()
        for i, (pl, pf) in enumerate(zip(p_list, p_list_f)):
            a_c, b_c = self.planner.set_finder.find_set_collision_avoidance(
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
        print(f"Collision sets took {stop_j - start_j:.3f}s")

        # Create parameter array
        # print(self.iw_ref)
        # print(self.pr_ref)
        # print(self.p_ref)
        # print(self.dp_ref)
        # print(self.dtau_init[:, 0])
        # print(self.dtau_init_orth1)
        params = np.concatenate(
            (
                self.split_idxs,
                self.p_ref,
                self.dp_ref,
                self.iw_ref,
                self.dtau_init[:, 0],
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
                a6.T.flatten(),
                a5.T.flatten(),
                a4.T.flatten(),
                a3.T.flatten(),
                a2.T.flatten(),
                a1.T.flatten(),
                a0.T.flatten(),
                weights_current,
                phi_max,
                self.dphi_max,
                v_1.flatten(),
                v_2.flatten(),
                v_3.flatten(),
                self.qd,
                spiral_rot_bp1.T.flatten(),
                spiral_rot_bp1_norm.flatten(),
                spiral_rot_bp2_norm.flatten(),
                spiral_a,
                spiral_l,
                spiral_theta,
                self.obj_centers.T.flatten(),
                self.obj_radii,
            )
        )

        for i in range(len(spiral_rot2d)):
            params = np.concatenate((params, spiral_rot2d[i, :, :].flatten()))
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
        uphi_lb = self.uphi_lb
        p_lb = self.p_lb
        v_lb = self.v_lb
        p_r_lb = self.p_r_lb
        dp_r_lb = self.dp_r_lb
        phi_lb = self.phi_lb

        q_lb[0 : -1 : self.N] = q0
        qp_lb[0 : -1 : self.N] = dq0
        qpp_lb[0 : -1 : self.N] = ddq0
        u_lb[0 : -1 : self.N] = jerk_current
        uphi_lb[0 : -1 : self.N] = self.dddphi_current
        phi_lb[0 : -1 : self.N] = np.concatenate(
            (self.phi_current, self.dphi_current, self.ddphi_current)
        )
        p_lb[0 : -1 : self.N] = p0
        v_lb[0 : -1 : self.N] = v0
        p_r_lb[0 : -1 : self.N] = self.p_ref
        dp_r_lb[0 : -1 : self.N] = self.dp_ref

        # idx_end = self.N * 7 + 1
        # qp_lb[self.N - 1 : idx_end : self.N] = 0
        # qpp_lb[self.N - 1 : idx_end : self.N] = 0
        # u_lb[self.N - 1 : idx_end : self.N] = 0
        # v_lb[self.N - 1 : self.N * 6 + 1 : self.N] = 0
        # phi_lb[self.N - 1 : self.N * 3 + 1 : self.N] = np.array([-np.inf, 0, 0])

        lbx = np.concatenate(
            [q_lb, qp_lb, qpp_lb, u_lb, uphi_lb, phi_lb, p_lb, v_lb, p_r_lb, dp_r_lb]
        )
        lbx = np.concatenate([lbx, -np.inf * np.ones(self.nr_slacks)])

        q_ub = self.q_ub
        qp_ub = self.dq_ub
        qpp_ub = self.ddq_ub
        u_ub = self.u_ub
        uphi_ub = self.uphi_ub
        p_ub = self.p_ub
        v_ub = self.v_ub
        p_r_ub = self.p_r_ub
        dp_r_ub = self.dp_r_ub
        # phi_ub = np.array([self.phi_max[0] + 0.05, self.dphi_max[0], np.inf])
        phi_ub = self.phi_ub

        q_ub[0 : -1 : self.N] = q0
        qp_ub[0 : -1 : self.N] = dq0
        qpp_ub[0 : -1 : self.N] = ddq0
        u_ub[0 : -1 : self.N] = jerk_current
        uphi_ub[0 : -1 : self.N] = self.dddphi_current
        phi_ub[0 : -1 : self.N] = np.concatenate(
            (self.phi_current, self.dphi_current, self.ddphi_current)
        )
        p_ub[0 : -1 : self.N] = p0
        v_ub[0 : -1 : self.N] = v0
        p_r_ub[0 : -1 : self.N] = self.p_ref
        dp_r_ub[0 : -1 : self.N] = self.dp_ref

        # qp_ub[self.N - 1 : idx_end : self.N] = 0
        # qpp_ub[self.N - 1 : idx_end : self.N] = 0
        # u_ub[self.N - 1 : idx_end : self.N] = 0
        # v_ub[self.N - 1 : self.N * 6 + 1 : self.N] = 0
        # phi_ub[self.N - 1 : self.N * 3 + 1 : self.N] = np.array([np.inf, 0, 0])

        ubx = np.concatenate(
            [q_ub, qp_ub, qpp_ub, u_ub, uphi_ub, phi_ub, p_ub, v_ub, p_r_ub, dp_r_ub]
        )
        ubx = np.concatenate([ubx, np.inf * np.ones(self.nr_slacks)])

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
        print(f"Slacks: {w_curr[-self.nr_slacks:]}")
        # cost = float(sol['f']) / self.N

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

        if self.error_count < self.N:
            traj_data, ref_data, err_data = self.compute_return_data(
                q0,
                dq0,
                ddq0,
                jerk_current,
                p0,
                w_opt,
                using_previous,
                a6,
                a5,
                a4,
                a3,
                a2,
                a1,
                a0,
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
                spiral_a,
                spiral_l,
                spiral_theta,
                spiral_rot2d,
                spiral_rot_bp1,
                spiral_rot_bp1_norm,
                spiral_rot_bp2_norm,
                a_set,
                b_set,
                a_set_joints,
                b_set_joints,
            )
            return traj_data, ref_data, err_data, time_elapsed, iters

        else:
            return None, None, None, None, None

    def compute_return_data(
        self,
        q0,
        dq0,
        ddq0,
        jerk_current,
        p0,
        w_opt,
        using_previous,
        a6,
        a5,
        a4,
        a3,
        a2,
        a1,
        a0,
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
        spiral_a,
        spiral_l,
        spiral_theta,
        spiral_rot2d,
        spiral_rot_bp1,
        spiral_rot_bp1_norm,
        spiral_rot_bp2_norm,
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
        optimal_jerk_phi = np.reshape(
            w_opt[4 * 7 * self.N : 4 * 7 * self.N + self.N], (self.N,), "F"
        ).T
        optimal_phi = np.reshape(
            w_opt[4 * 7 * self.N + self.N : 4 * 7 * self.N + 2 * self.N], (self.N,), "F"
        ).T
        optimal_dphi = np.reshape(
            w_opt[4 * 7 * self.N + 2 * self.N : 4 * 7 * self.N + 3 * self.N],
            (self.N,),
            "F",
        ).T
        optimal_ddphi = np.reshape(
            w_opt[4 * 7 * self.N + 3 * self.N : 4 * 7 * self.N + 4 * self.N],
            (self.N,),
            "F",
        ).T
        optimal_pref = np.reshape(
            w_opt[
                4 * 7 * self.N + 4 * self.N + 12 * self.N : 4 * 7 * self.N
                + 4 * self.N
                + 12 * self.N
                + 3 * self.N
            ],
            (self.N, 3),
            "F",
        ).T
        optimal_dpref = np.reshape(
            w_opt[
                4 * 7 * self.N + 4 * self.N + 12 * self.N + 3 * self.N : 4 * 7 * self.N
                + 4 * self.N
                + 12 * self.N
                + 2 * 3 * self.N
            ],
            (self.N, 3),
            "F",
        ).T
        optimal_q = optimal_q[:, self.error_count :]
        optimal_dq = optimal_dq[:, self.error_count :]
        optimal_ddq = optimal_ddq[:, self.error_count :]
        optimal_phi = optimal_phi[self.error_count :]
        optimal_dphi = optimal_dphi[self.error_count :]
        optimal_ddphi = optimal_ddphi[self.error_count :]
        optimal_jerk = optimal_jerk[:, self.error_count :]
        optimal_jerk_phi = optimal_jerk_phi[self.error_count :]
        optimal_pref = optimal_pref[:, self.error_count :]
        optimal_dpref = optimal_dpref[:, self.error_count :]

        optimal_traj = np.empty((6, optimal_jerk.shape[1]))
        # optimal_phi = np.empty(optimal_jerk.shape[1])
        # optimal_dphi = np.empty(optimal_jerk.shape[1])
        # optimal_ddphi = np.empty(optimal_jerk.shape[1])
        # optimal_phi[0] = np.copy(self.phi_current[0])
        # optimal_dphi[0] = np.copy(self.dphi_current[0])
        # optimal_ddphi[0] = np.copy(self.ddphi_current[0])
        #
        # # Integrate the system. This is necessary because the solver
        # # might not find the optimal solution which makes the dynamic system
        # # constraints go to zero.
        # jerk_phi_mat = np.expand_dims(optimal_jerk_phi, 0)
        # for i in range(1, optimal_jerk.shape[1]):
        #     t = self.dt * i
        #     optimal_phi[i] = calcAngle(
        #         jerk_phi_mat,
        #         t,
        #         self.phi_current,
        #         self.dphi_current,
        #         self.ddphi_current,
        #         self.dt,
        #     )
        #     optimal_dphi[i] = calcVelocity(
        #         jerk_phi_mat, t, self.dphi_current, self.ddphi_current, self.dt
        #     )
        #     optimal_ddphi[i] = calcAcceleration(
        #         jerk_phi_mat, t, self.ddphi_current, self.dt
        #     )
        #     optimal_q[:, i] = calcAngle(optimal_jerk, t, q0, dq0, ddq0, self.dt)
        #     optimal_dq[:, i] = calcVelocity(optimal_jerk, t, dq0, ddq0, self.dt)
        #     optimal_ddq[:, i] = calcAcceleration(optimal_jerk, t, ddq0, self.dt)

        # Compute cartesian trajectories
        optimal_i_omega = np.empty((6, optimal_q.shape[1]))
        optimal_acc = np.empty((6, optimal_q.shape[1]))
        optimal_vel = np.empty((6, optimal_q.shape[1]))
        jac_fk = self.robot_model.jacobian_fk(q0)
        omega_prev = (jac_fk @ dq0)[3:]
        for i in range(optimal_q.shape[1]):
            p_c, jac_fk, djac_fk = self.robot_model.forward_kinematics(
                optimal_q[:, i], optimal_dq[:, i]
            )
            optimal_traj[:, i] = np.copy(p_c)
            optimal_vel[:, i] = jac_fk @ optimal_dq[:, i]
            optimal_acc[:, i] = jac_fk @ optimal_ddq[:, i] + djac_fk @ optimal_dq[:, i]
            k1 = omega_prev
            k2 = optimal_vel[3:, i]
            omega_prev = np.copy(k2)
            if i > 0:
                optimal_i_omega[3:, i] = optimal_i_omega[
                    3:, i - 1
                ] + 1 / 2 * self.dt * (k1 + k2)
            else:
                optimal_i_omega[3:, i] = p0[3:] + 1 / 2 * self.dt * (k1 + k2)
        optimal_i_omega[:3, :] = optimal_traj[:3, :]
        if self.error_count > 0:
            if np.linalg.norm(p0[3:] - optimal_i_omega[3:, 1]) > 3.1:
                optimal_i_omega[3:, :] *= -1

        # Integrate the position reference
        # dphi = optimal_phi[1] - self.phi_current
        # nr_steps = 10
        # dphik = dphi / nr_steps
        # dp_ref_last = np.copy(self.dp_ref)
        # for i in range(nr_steps):
        #     phik = self.phi_current + (i + 1) * dphik
        #     if phik > phi_switch[1] + spiral_l[1]:
        #         dp_ref_current = dp_ref[:3, 1]
        #         # print("Lin: ", dp_ref_current)
        #     elif phik > phi_switch[1]:
        #         phi_spiral = spiral_l[1] - (phik - phi_switch[1])
        #         a = -spiral_a[1]
        #         theta = spiral_theta[1]
        #         r1 = np.array(
        #             [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        #         )
        #         dp_ref_current = (
        #             r1
        #             @ np.array(
        #                 [np.cos(a * phi_spiral**2), np.sin(a * phi_spiral**2)]
        #             ).squeeze()
        #         )
        #         dp_ref_current = spiral_rot2d[1, :, :].T @ dp_ref_current
        #         # print("Start2: ", phi_spiral, dp_ref_current)
        #     elif phik < phi_switch[0] + spiral_l[0]:
        #         phi_spiral = spiral_l[0] - (phik - phi_switch[0])
        #         a = -spiral_a[0]
        #         theta = spiral_theta[0]
        #         r1 = np.array(
        #             [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        #         )
        #         dp_ref_current = (
        #             r1
        #             @ np.array(
        #                 [np.cos(a * phi_spiral**2), np.sin(a * phi_spiral**2)]
        #             ).squeeze()
        #         )
        #         dp_ref_current = spiral_rot2d[0, :, :].T @ dp_ref_current
        #         # print("Start: ", phi_spiral, dp_ref_current)
        #     elif phik > phi_switch[1] - spiral_l[1]:
        #         phi_spiral = phik - (phi_switch[1] - spiral_l[1])
        #         a = spiral_a[1]
        #         dp_ref_current = np.array(
        #             [np.cos(a * phi_spiral**2), np.sin(a * phi_spiral**2)]
        #         ).squeeze()
        #         dp_ref_current = spiral_rot2d[1, :, :].T @ dp_ref_current
        #         # print("End: ", phi_spiral, dp_ref_current)
        #     else:
        #         dp_ref_current = dp_ref[:3, 0]
        #         # print("Lin: ", dp_ref_current)
        #     self.p_ref = self.p_ref + 0.5 * dphik * (dp_ref_last + dp_ref_current)
        #     dp_ref_last = np.copy(dp_ref_current)
        # self.dp_ref = np.copy(dp_ref_current)
        # print(np.linalg.norm(self.dp_ref), self.dp_ref)

        # if optimal_phi[0] > phi_switch[1]:
        #     # self.p_ref = p_ref[:3, 1] + (optimal_phi[0] - phi_switch[1]) * dp_ref[:3, 1]
        #     print(p_ref[:3, 1] + (optimal_phi[0] - phi_switch[1]) * dp_ref[:3, 1])
        # else:
        #     # self.p_ref = p_ref[:3, 0] + (optimal_phi[0] - phi_switch[0]) * dp_ref[:3, 0]
        #     print(p_ref[:3, 0] + (optimal_phi[0] - phi_switch[0]) * dp_ref[:3, 0])
        # print(self.p_ref)
        self.p_ref = optimal_pref[:, 1]
        self.dp_ref = optimal_dpref[:, 1]

        # Integrate the rotation reference
        iw_ref_copy = np.copy(self.iw_ref)
        if optimal_phi[1] > phi_switch[1]:
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
            self.pr_ref = integrate_rotation_reference(
                self.pr_ref, dp_ref[3:, 0], self.phi_current, optimal_phi[1]
            )
            self.iw_ref = (
                p_ref[3:, 0] + (optimal_phi[1] - phi_switch[0]) * dp_ref[3:, 0]
            )

        # Set new path parameter state
        self.phi_current = np.array([optimal_phi[1]])
        self.dphi_current = np.array([optimal_dphi[1]])
        self.ddphi_current = np.array([optimal_ddphi[1]])
        self.dddphi_current = np.array([optimal_jerk_phi[1]])

        # Update the splitting idx
        split_idx_prev = self.split_idxs.copy()
        in_set_accuracy = 0.005
        for i in range(1, phi_switch.shape[0] - 1):
            if self.split_idxs[i] < self.N:
                self.split_idxs[i] -= 1
                if self.split_idxs[i] == 0:
                    self.switch = True
                    self.split_idxs[i] = self.N
            else:
                dswitch = optimal_phi > phi_switch[i] - 0.005
                in_set0 = (
                    np.max(
                        a_set[i - 1] @ optimal_traj[:3, :]
                        - np.expand_dims(b_set[i - 1], 1),
                        axis=0,
                    )
                    < in_set_accuracy
                )
                in_set1 = (
                    np.max(
                        a_set[i] @ optimal_traj[:3, :] - np.expand_dims(b_set[i], 1),
                        axis=0,
                    )
                    < in_set_accuracy
                )
                idx_new = np.where(dswitch * in_set0 * in_set1)[0]
                if idx_new.shape[0] > 0:
                    if self.split_idxs[i] == self.N:
                        self.split_idxs[i] = idx_new[0]
                    else:
                        self.split_idxs[i] -= 1
                    if self.split_idxs[i] == 0:
                        self.switch = True
        if self.switch:
            print("--> Switching segment")
            self.split_idxs[1:-1] = self.split_idxs[2:]
            self.split_idxs[-1] = self.N
        for i in range(1, phi_switch.shape[0] - 1):
            if self.ref_path.sector + (i - 1) >= self.ref_path.num_sectors:
                self.split_idxs[i] = self.N

        for i in range(1, phi_switch.shape[0] - 1):
            if self.split_idxs[i] <= self.split_idxs[i - 1]:
                self.split_idxs[i] = np.min((self.N, self.split_idxs[i - 1] + 1))
        print(self.split_idxs)

        # Create reference trajectory
        if self.log:
            ref_data = defaultdict(list)
            err_data = defaultdict(list)
            i_omega_0 = p0[3:]
            for i in range(1, optimal_phi.shape[0]):
                phi = optimal_phi[i]
                dphi = optimal_dphi[i]
                reference = reference_function(
                    dp_ref=dp_ref.T,
                    p_ref=p_ref.T,
                    dp_ref_p=np.expand_dims(optimal_dpref[:, i], 1),
                    p_ref_p=np.expand_dims(optimal_pref[:, i], 1),
                    dp_normed_ref=dp_normed_ref.T,
                    phi_switch=np.expand_dims(phi_switch, 1),
                    phi=phi,
                    bp1=bp1.T,
                    bp2=bp2.T,
                    br1=br1.T,
                    br2=br2.T,
                    v1=v1.T,
                    v2=v2.T,
                    v3=v3.T,
                    a6=a6,
                    a5=a5,
                    a4=a4,
                    a3=a3,
                    a2=a2,
                    a1=a1,
                    a0=a0,
                    spiral_a=np.expand_dims(spiral_a, axis=1),
                    spiral_l=np.expand_dims(spiral_l, axis=1),
                    spiral_theta=np.expand_dims(spiral_theta, axis=1),
                    spiral_rot2d=spiral_rot2d,
                    spiral_rot_bp1=spiral_rot_bp1,
                    spiral_rot_bp1_norm=np.expand_dims(spiral_rot_bp1_norm, 1),
                    spiral_rot_bp2_norm=np.expand_dims(spiral_rot_bp2_norm, 1),
                    split_idx=split_idx_prev,
                    idx=i,
                    n_horizon=self.N,
                    a_set=a_set,
                    b_set=b_set,
                )
                p_d = np.array(reference["p_d"]).flatten()
                dp_d = np.array(reference["dp_d"]).flatten()
                ddp_d = np.array(reference["ddp_d"]).flatten()
                dp_normed_d = np.array(reference["dp_normed_d"]).flatten()

                ref_data["p"].append(p_d)
                ref_data["dp"].append(dp_d)
                ref_data["ddp"].append(ddp_d)
                ref_data["dp_normed"].append(dp_normed_d)
                ref_data["e_p_off"].append(np.array(reference["e_p_off"]).flatten())
                ref_data["bp1"].append(np.array(reference["bp1_current"]).squeeze())
                ref_data["bp2"].append(np.array(reference["bp2_current"]).squeeze())
                ref_data["br1"].append(reference["br1_current"])
                ref_data["br2"].append(reference["br2_current"])
                ref_data["v1"].append(reference["v1_current"])
                ref_data["v2"].append(reference["v2_current"])
                ref_data["v3"].append(reference["v3_current"])
                if i == 1:
                    ref_data["a_current"] = reference["a_current"].full().flatten()
                    ref_data["b_current"] = reference["b_current"].full().squeeze()
                    ref_data["a_next"] = reference["a_next"].full().flatten()
                    ref_data["b_next"] = reference["b_next"].full().squeeze()
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
                    v=optimal_vel[:, i],
                    p_ref=ref_data["p"][-1],
                    dp_ref=ref_data["dp"][-1],
                    ddp_ref=ref_data["ddp"][-1],
                    dp_normed_ref=ref_data["dp_normed"][-1],
                    dphi=dphi,
                    i_omega_0=i_omega_0,
                    i_omega_ref_0=iw_ref_copy,
                    dtau_init=self.dtau_init[:, 0],
                    dtau_init_par=self.dtau_init_par.T,
                    dtau_init_orth1=self.dtau_init_orth1.T,
                    dtau_init_orth2=self.dtau_init_orth2.T,
                    br1=ref_data["br1"][-1].T,
                    br2=ref_data["br2"][-1].T,
                    jac_dtau_l=jac_dtau_l,
                    jac_dtau_r=jac_dtau_r,
                    phi=phi,
                    phi_switch=phi_switch,
                    v1=ref_data["v1"][-1].T,
                    v2=ref_data["v2"][-1].T,
                    v3=ref_data["v3"][-1].T,
                )
                e_p_par = np.array(errors["e_p_par"]).flatten()
                e_p_orth = np.array(errors["e_p_orth"]).flatten()
                de_p_par = np.array(errors["de_p_par"]).flatten()
                de_p_orth = np.array(errors["de_p_orth"]).flatten()
                e_p = np.array(errors["e_p"]).flatten()
                de_p = np.array(errors["de_p"]).flatten()
                e_r_par = np.array(errors["e_r_par"]).flatten()
                e_r_orth1 = np.array(errors["e_r_orth1"]).flatten()
                e_r_orth2 = np.array(errors["e_r_orth2"]).flatten()
                e_r = np.array(errors["e_r"]).flatten()
                de_r = np.array(errors["de_r"]).flatten()

                bound_lower = np.array(reference["bound_lower"]).flatten()[:2]
                bound_upper = np.array(reference["bound_upper"]).flatten()[:2]
                r_bound_lower = np.array(reference["bound_lower"]).flatten()[2:]
                r_bound_upper = np.array(reference["bound_upper"]).flatten()[2:]
                bound_lower = np.concatenate(
                    (bound_lower, np.array(r_bound_lower).flatten())
                )
                bound_upper = np.concatenate(
                    (bound_upper, np.array(r_bound_upper).flatten())
                )
                e_r_off = np.array((r_bound_lower + r_bound_upper) / 2).flatten()
                ref_data["bound_lower"].append(bound_lower)
                ref_data["bound_upper"].append(bound_upper)
                ref_data["e_r_off"].append(e_r_off)

                err_data["e_p"].append(e_p)
                err_data["de_p"].append(de_p)
                err_data["e_p_par"].append(e_p_par)
                err_data["e_p_orth"].append(e_p_orth)
                err_data["de_p_par"].append(de_p_par)
                err_data["de_p_orth"].append(de_p_orth)
                err_data["e_r"].append(np.copy(e_r))
                err_data["de_r"].append(de_r)
                err_data["e_r_par"].append(e_r_par)
                err_data["e_r_orth1"].append(e_r_orth1)
                err_data["e_r_orth2"].append(e_r_orth2)

            # Update ref data to correct rotation reference
            pr_ref = np.copy(self.pr_ref)
            for i in range(1, self.N - self.error_count - 1):
                ref_data["p"][i][3:] = np.copy(pr_ref)
                tauc = R.from_rotvec(optimal_traj[3:, i]).as_matrix()
                taud = R.from_rotvec(pr_ref).as_matrix()
                dtau_correct = R.from_matrix(tauc @ taud.T).as_rotvec()
                err_data["e_r"][i] = dtau_correct

                phi = optimal_phi[i]
                dphi = optimal_dphi[i]
                phi_next = optimal_phi[i + 1]
                if phi_next > phi_switch[1] and phi < phi_switch[1]:
                    pr_ref = R.from_matrix(
                        self.ref_path.r[self.ref_path.sector + 1]
                    ).as_rotvec()
                    pr_ref = integrate_rotation_reference(
                        pr_ref, dp_ref[3:, 1], phi_switch[1], phi_next
                    )
                elif phi_next > phi_switch[2] and phi < phi_switch[2]:
                    pr_ref = R.from_matrix(
                        self.ref_path.r[self.ref_path.sector + 2]
                    ).as_rotvec()
                    pr_ref = integrate_rotation_reference(
                        pr_ref, dp_ref[3:, 2], phi_switch[2], phi_next
                    )
                elif phi_next > phi_switch[2]:
                    pr_ref = integrate_rotation_reference(
                        pr_ref, dp_ref[3:, 2], phi, phi_next
                    )
                elif phi_next > phi_switch[1]:
                    pr_ref = integrate_rotation_reference(
                        pr_ref, dp_ref[3:, 1], phi, phi_next
                    )
                else:
                    pr_ref = integrate_rotation_reference(
                        pr_ref, dp_ref[3:, 0], phi, phi_next
                    )
            ref_data["p"][-1][3:] = np.copy(pr_ref)
            tauc = R.from_rotvec(optimal_traj[3:, -1]).as_matrix()
            taud = R.from_rotvec(pr_ref).as_matrix()
            dtau_correct = R.from_matrix(tauc @ taud.T).as_rotvec()
            err_data["e_r"][-1] = dtau_correct

            # Use correct integrated position reference
            ref_data["p"][0][:3] = self.p_ref
            ref_data["dp"][0][:3] = self.dp_ref
        else:
            ref_data = None
            err_data = None

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
        traj_data["ddphi"] = optimal_ddphi[1:]
        traj_data["dddphi"] = optimal_jerk_phi[1:]

        return traj_data, ref_data, err_data
