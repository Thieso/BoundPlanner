import time

import casadi as ca
import cvxpy as cp
import mosek.fusion as mf
import numpy as np
from scipy.optimize import linprog

from bound_planner.utils import gram_schmidt, normalize_set_size


def projection_opt_problem():
    max_set_size = 15
    a_set = ca.SX.sym("a set", max_set_size, 3)
    b_set = ca.SX.sym("b set", max_set_size)

    params = ca.vertcat(a_set.reshape((-1, 1)), b_set)

    g = []
    lbg = []
    ubg = []
    u = []
    lbu = []
    ubu = []

    x = ca.SX.sym("x", 3)
    u += [x]
    lbu += [-np.inf] * 3
    ubu += [np.inf] * 3

    J = ca.sumsqr(x)

    g += [a_set @ x - b_set]
    lbg += [-np.inf] * max_set_size
    ubg += [0] * max_set_size

    prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

    # solver = ca.qpsol('solver', 'qpoases', prob, {
    #     "print_time": False,
    #     "printLevel": "none"
    # })
    solver = ca.qpsol(
        "solver",
        "osqp",
        prob,
        {
            "print_time": False,
            "osqp": {"verbose": False, "eps_abs": 1e-6, "eps_rel": 1e-6},
        },
    )
    return solver, lbu, ubu, lbg, ubg


def projection_line_opt_problem():
    max_set_size = 15
    a_set = ca.SX.sym("a set", max_set_size, 3)
    b_set = ca.SX.sym("b set", max_set_size)
    p0 = ca.SX.sym("p0", 3)
    p1 = ca.SX.sym("p1", 3)

    params = ca.vertcat(a_set.reshape((-1, 1)), b_set, p0, p1)

    g = []
    lbg = []
    ubg = []
    u = []
    lbu = []
    ubu = []

    x = ca.SX.sym("x", 3)
    u += [x]
    lbu += [-np.inf] * 3
    ubu += [np.inf] * 3

    phi = ca.SX.sym("phi", 1)
    u += [phi]
    lbu += [0.0]
    ubu += [1.0]

    p_closest = p0 + phi * (p1 - p0)
    J = ca.sumsqr(p_closest - x)

    g += [a_set @ x - b_set]
    lbg += [-np.inf] * max_set_size
    ubg += [0] * max_set_size

    prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

    solver = ca.qpsol(
        "solver", "qpoases", prob, {"print_time": False, "printLevel": "none"}
    )
    # solver = ca.qpsol(
    #     "solver",
    #     "osqp",
    #     prob,
    #     {
    #         "print_time": False,
    #         "osqp": {"verbose": False, "eps_abs": 1e-6, "eps_rel": 1e-6},
    #     },
    # )
    return solver, lbu, ubu, lbg, ubg


class ConvexSetFinder:
    def __init__(self, obs_sets, obs_points, obs_points_sets, e_max, e_min):
        self.region = []
        self.rng = np.random.default_rng(0)
        self.ell_time = 0.0
        self.set_line_time = 0.0
        self.proj_time = 0.0
        self.obs_sets = obs_sets.copy()
        self.obs_points_sets = obs_points_sets.copy()
        (
            self.proj_solver,
            self.proj_lbu,
            self.proj_ubu,
            self.proj_lbg,
            self.proj_ubg,
        ) = projection_opt_problem()
        self.obs_points_sets = obs_points_sets.copy()
        (
            self.projl_solver,
            self.projl_lbu,
            self.projl_ubu,
            self.projl_lbg,
            self.projl_ubg,
        ) = projection_line_opt_problem()
        self.socp_prob, self.socp_params = self.cvx_mvie_socp(20, 12)
        self.socpfm_prob, self.socpfm_params = self.cvx_mvie_socp_fixed_mid(20, 9)
        self.socpfr_prob, self.socpfr_params = self.cvx_mvie_socp_fixed_r(20, 6)
        # self.proj_solver.generate("proj_solver.c")
        # self.proj_solver = ca.external("solver", "./proj_solver.so")
        self.e_max = e_max
        self.e_min = e_min
        self.max_iter = 10

    def recursive_halfspace_computation(self, e_constraint, f_constraint):
        y = np.zeros(e_constraint.shape[1])
        # order = self.rng.permutation(e_constraint.shape[0]).tolist()
        order = np.arange(e_constraint.shape[0]).tolist()
        idx = np.where(np.abs(e_constraint) > 1e-4)[0]
        if e_constraint.shape[1] == 1:
            if e_constraint[idx].shape[0] == 0:
                return np.zeros(e_constraint.shape[1])
            bounds = f_constraint[idx] / e_constraint[idx, :]
            upper_bounds = bounds[np.where(e_constraint[idx] > 0)]
            lower_bounds = bounds[np.where(e_constraint[idx] <= 0)]
            if upper_bounds.shape[0] == 0:
                return np.array([np.max((0, np.max(lower_bounds)))])
            if lower_bounds.shape[0] == 0:
                return np.array([np.min((0, np.min(upper_bounds)))])
            active_upper = np.min(upper_bounds)
            active_lower = np.max(lower_bounds)
            if active_upper < active_lower:
                print("ERROR upper bound smaller than lower bound")
            else:
                if active_upper > 0 and active_lower < 0:
                    return np.zeros(e_constraint.shape[1])
                elif np.abs(active_upper) < np.abs(active_lower):
                    return active_upper
                else:
                    return active_lower

        i_set = []
        for i in order:
            h = (e_constraint[i, :], f_constraint[i])
            if np.dot(h[0], y) > h[1]:
                # Householder projection
                v = h[1] * h[0].T / (h[0] @ h[0].T)
                j = np.argmax(np.abs(v))
                ej = np.zeros(v.shape[0])
                ej[j] = 1
                u = v + np.sign(v[j]) * np.linalg.norm(v) * ej
                h_mat = np.eye(v.shape[0]) - 2 * (
                    np.expand_dims(u, 1) @ np.expand_dims(u, 0)
                ) / (u.T @ u)
                m_mat = np.delete(h_mat.T, j, 1)
                e_constraint_prime = e_constraint[i_set, :] @ m_mat
                f_constraint_prime = f_constraint[i_set] - e_constraint[i_set, :] @ v
                # print(np.linalg.norm(h_mat, axis=1))
                # print(h_mat @ v, ej)
                if i_set == []:
                    y_prime = np.zeros(e_constraint.shape[1] - 1)
                else:
                    y_prime = self.recursive_halfspace_computation(
                        e_constraint_prime, f_constraint_prime
                    )
                y = m_mat @ y_prime + v
            i_set.append(i)
        return y

    def find_set_around_point(self, p_seed, fixed_mid=False, optimize=True):
        p_seed = np.copy(p_seed)
        a = b = c = 1e-4
        q_inv = np.diag((a, b, c))
        q_ellipse = np.diag((1 / a, 1 / b, 1 / c))

        # Create rectangle around the path as inital guess and remove outlying
        # obstacle points
        a_set_init, b_set_init = self.init_halfspaces()

        det_ellipse_old = 1
        det_ellipse = 100
        k = 0
        while np.abs(det_ellipse - det_ellipse_old) / det_ellipse_old > 0.01:
            k += 1
            print(f"(SetAroundPoint) Iteration {k}")
            if k > self.max_iter:
                break
            # Find polyhedron
            a_set, b_set = self.compute_polyhedron(
                q_inv, q_ellipse, p_seed, a_set_init, b_set_init
            )
            a_set_np = np.array(a_set)
            b_set_np = np.array(b_set)
            if not optimize:
                return a_set_np, b_set_np, q_ellipse, p_seed
            # sets = normalize_set_size([[a_set_np, b_set_np]])
            # a_set_np_norm = sets[0][0]
            # b_set_np_norm = sets[0][1]

            det_ellipse_old = np.copy(det_ellipse)
            start = time.perf_counter()
            # Ellipse expansion
            # q_ellipse, p_seed = self.mvie_mosek(a_set_np_norm, b_set_np_norm, p_seed)
            # q_ellipse = self.mvie_mosek_fixed_mid(a_set_np_norm, b_set_np_norm, p_seed)
            # q_ellipse, p_seed = self.mvie_affine_scaling(q_ellipse, p_seed, a_set_np_norm, b_set_np_norm)
            # q_inv, p_seed = self.mvie_affine_scaling_fixed_mid(q_inv, p_seed, a_set_np, b_set_np)
            # q_inv, p_seed = self.mvie_affine_scaling(q_inv, p_seed, a_set_np, b_set_np)
            if fixed_mid:
                q_inv, p_seed = self.mvie_socp_fixed_mid(a_set_np, b_set_np, p_seed)
            else:
                q_inv, p_seed = self.mvie_socp(a_set_np, b_set_np)
            stop = time.perf_counter()
            self.ell_time += stop - start

            svd = np.linalg.svd(q_inv)
            q_ellipse = svd.Vh.T @ np.diag(1 / svd.S) @ svd.U.T
            det_ellipse = np.linalg.det(q_ellipse)

            # Sometimes the ellipsoid converges to too small values due to the fixed mid and the fixed shape
            if np.min(svd.S) < 1e-3:
                break

        if fixed_mid:
            q_inv, p_seed = self.mvie_socp(a_set_np, b_set_np)
            svd = np.linalg.svd(q_inv)
            q_ellipse = svd.Vh.T @ np.diag(1 / svd.S) @ svd.U.T

        return a_set_np, b_set_np, q_ellipse, p_seed

    def find_set_around_line(self, p0, dp1, optimize=True):
        start = time.perf_counter()
        p1 = p0 + dp1
        l_seg = np.linalg.norm(p1 - p0)
        dp_ref = dp1 / l_seg
        p_seed = (p0 + p1) / 2
        # a_lb = l_seg / 2
        a_lb = l_seg**2 / 4
        if np.abs(dp_ref[2]) < 0.99:
            b1d = np.array([0, 0, 1.0])
        else:
            b1d = np.array([0, 1.0, 0])
        b1 = gram_schmidt(dp_ref, b1d)
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(dp_ref, b1)
        b2 /= np.linalg.norm(b2)
        r_ellipse = np.vstack((dp_ref, b1, b2)).T
        np.linalg.det(r_ellipse)
        b = c = 1e-4
        q_inv = r_ellipse @ np.diag((a_lb, b, c)) @ r_ellipse.T
        q_ellipse = r_ellipse @ np.diag((1 / a_lb, 1 / b, 1 / c)) @ r_ellipse.T

        # Create rectangle around the path as inital guess and remove outlying
        # obstacle points
        a_set_init, b_set_init = self.init_halfspaces()

        det_ellipse_old = 1
        det_ellipse = 100
        k = 0
        while np.abs(det_ellipse - det_ellipse_old) / det_ellipse_old > 0.01:
            k += 1
            print(f"(SetAroundLine) Iteration {k}")
            if k > self.max_iter:
                break
            # Find polyhedron
            a_set, b_set = self.compute_polyhedron(
                q_inv, q_ellipse, p_seed, a_set_init, b_set_init
            )
            a_set_np = np.array(a_set)
            b_set_np = np.array(b_set)
            if not optimize:
                q_inv, p_seed = self.mvie_socp(a_set_np, b_set_np)
                svd = np.linalg.svd(q_inv)
                q_ellipse = svd.Vh.T @ np.diag(1 / svd.S) @ svd.U.T
                break

            det_ellipse_old = np.copy(det_ellipse)
            # sets = normalize_set_size([[a_set_np, b_set_np]])
            # a_set_np_norm = sets[0][0]
            # b_set_np_norm = sets[0][1]
            start = time.perf_counter()
            q_inv, q_ellipse, eigs = self.mvie_socp_fixed_r(
                a_set_np, b_set_np, p_seed, r_ellipse, a_lb
            )
            # Sometimes the ellipsoid converges to too small values due to the fixed mid and the fixed shape
            if np.min(eigs) < 1e-3:
                break

            # q_inv, p_seed = self.mvie_socp_fixed_mid(a_set_np, b_set_np, p_seed)
            det_ellipse = np.linalg.det(q_ellipse)

            stop = time.perf_counter()
            self.ell_time += stop - start
            print(f"(SetFinder) MVIE solver total {self.ell_time:.3f}s")

        return a_set, b_set, q_ellipse, p_seed

    def find_set_collision_avoidance(
        self, p0, p1, compute_ellipsoid=False, limit_space=False, e_max=0.3
    ):
        # Create rectangle around the path as inital guess and remove outlying
        # obstacle points
        if limit_space:
            a_set_init, b_set_init = self.init_halfspaces_point(p0, e_max)
        else:
            a_set_init, b_set_init = self.init_halfspaces()

        # Find polyhedron
        obs_remain = self.obs_sets.copy()
        obs_points_remain = self.obs_points_sets.copy()
        a_set = a_set_init.copy()
        b_set = b_set_init.copy()
        obs_points, phi = self.compute_set_projs_line(obs_remain, p0, p1)
        p_closest = np.expand_dims(p0, 1) + phi * np.expand_dims((p1 - p0), 1)
        dists = np.linalg.norm(obs_points.T - p_closest, axis=0).tolist()
        obs_points = obs_points.tolist()
        p_closest = p_closest.T.tolist()
        while len(obs_remain) > 0:
            idx = np.argmin(dists)
            closest_point = np.array(obs_points[idx])

            a_halfspace = closest_point - np.array(p_closest[idx])
            norm_a = np.linalg.norm(a_halfspace)
            if norm_a < 1e-6:
                print("(LineSet) [WARNING] Line is touching an obstacle")
                # TODO choose the face of the obstacle that is closest to p0 as halfspace
                a_halfspace = closest_point - p0
                norm_a = np.linalg.norm(a_halfspace)
                if norm_a < 1e-6:
                    print("(LineSet) [WARNING] P0 is touching an obstacle")
                    a_halfspace = p1 - p0
                    norm_a = np.linalg.norm(a_halfspace)
            a_halfspace /= norm_a
            b_halfspace = a_halfspace @ closest_point - 0.001
            # b_halfspace /= norm_a

            # Delete obstacle points outside the halfspace
            idx_to_del = [idx]
            for i, obs_points_c in enumerate(obs_points_remain):
                if (
                    np.min(a_halfspace @ obs_points_c.T - b_halfspace) >= -1e-4
                    and i != idx
                ):
                    idx_to_del.append(i)
            for i in sorted(np.unique(idx_to_del), reverse=True):
                del obs_remain[i]
                del p_closest[i]
                del obs_points_remain[i]
                del obs_points[i]
                del dists[i]

            # Append to set
            a_set.append(a_halfspace)
            b_set.append(b_halfspace)
        a_set_np = np.array(a_set)
        b_set_np = np.array(b_set)
        if compute_ellipsoid:
            q_inv, p_seed = self.mvie_socp(a_set_np, b_set_np)
            svd = np.linalg.svd(q_inv)
            q_ellipse = svd.Vh.T @ np.diag(1 / svd.S) @ svd.U.T
            return a_set_np, b_set_np, q_ellipse, p_seed
        return a_set_np, b_set_np

    def find_set_dcs(self, p0, p1, limit_space=False, e_max=0.3):
        # Create rectangle around the path as inital guess and remove outlying
        # obstacle points
        if limit_space:
            a_set_init, b_set_init = self.init_halfspaces_point(p0, e_max)
        else:
            a_set_init, b_set_init = self.init_halfspaces()

        # Find polyhedron
        obs_remain = self.obs_sets.copy()
        obs_points_remain = self.obs_points_sets.copy()
        a_set = a_set_init.copy()
        b_set = b_set_init.copy()
        d01 = p1 - p0
        norm_d01 = np.linalg.norm(d01)
        if norm_d01 > 1e-8:
            d01 /= norm_d01
        while len(obs_remain) > 0:
            idx = 0
            obs_points = obs_points_remain[idx]
            # a_ub = -np.array(obs_points)
            # b_ub = np.ones(a_ub.shape[0] + 2)
            # b_ub[: a_ub.shape[0]] *= -1
            # a_ub = np.concatenate((a_ub, np.expand_dims(p0, 0)))
            # a_ub = np.concatenate((a_ub, np.expand_dims(p1, 0)))
            # res = linprog(d01, A_ub=a_ub, b_ub=b_ub, bounds=(-1000, 1000))
            # np.array([1, 0, 0.0]) @ obs_points.T - 0.1
            # a_ub @ np.array([1, 0, 0.0]) - b_ub
            # np.array([0, -6.667, 1.768]) @ obs_points.T - 1
            # np.array([0, -6.667, 1.768]) @ p0 - 1
            # np.array([0, -6.667, 1.768]) @ p1 - 1
            # if res.status == 3:
            #     a_halfspace = d01
            # else:
            #     a_halfspace = res.x
            # norm_a = np.linalg.norm(a_halfspace)
            # a_halfspace /= norm_a
            # b_halfspace = 1 / norm_a
            params = np.concatenate((obs_points.T.flatten(), p0, p1, -d01))
            x_opt = self.dcs_opt(params, obs_points.shape[0])
            a_halfspace = x_opt[:3]
            b_halfspace = x_opt[3]
            # norm_a = np.linalg.norm(a_halfspace)
            # a_halfspace /= norm_a
            # b_halfspace = 1 / norm_a
            # b_halfspace = 1.0
            if a_halfspace @ p0 - b_halfspace > 1e-4:
                raise RuntimeError("p0")
            if a_halfspace @ p1 - b_halfspace > 1e-4:
                raise RuntimeError("p1")
            if np.min(a_halfspace @ obs_points.T - b_halfspace) < -1e-4:
                raise RuntimeError("obs")
            print(np.min(a_halfspace @ obs_points.T - b_halfspace))
            # close_idx = np.argmin(a_halfspace @ obs_points.T - b_halfspace)
            # closest_point = obs_points[close_idx]
            # b_halfspace += a_halfspace @ closest_point - b_halfspace

            # Delete obstacle points outside the halfspace
            idx_to_del = [idx]
            for i, obs_points_c in enumerate(obs_points_remain):
                if (
                    np.min(a_halfspace @ obs_points_c.T - b_halfspace) >= -1e-4
                    and i != idx
                ):
                    idx_to_del.append(i)
            for i in sorted(np.unique(idx_to_del), reverse=True):
                del obs_remain[i]
                del obs_points_remain[i]

            # Append to set
            a_set.append(a_halfspace)
            b_set.append(b_halfspace)
        a_set_np = np.array(a_set)
        b_set_np = np.array(b_set)
        return a_set_np, b_set_np

    def dcs_opt(self, params_num, nr_obs_points=10):
        obs = ca.SX.sym("obs", nr_obs_points, 3)
        x_des = ca.SX.sym("x des", 3)
        p0 = ca.SX.sym("p0", 3)
        p1 = ca.SX.sym("p1", 3)

        params = ca.vertcat(obs.reshape((-1, 1)), p0, p1, x_des)

        g = []
        lbg = []
        ubg = []
        u = []
        lbu = []
        ubu = []

        x = ca.SX.sym("x", 3)
        u += [x]
        lbu += [-np.inf] * 3
        ubu += [np.inf] * 3
        b = ca.SX.sym("b")
        u += [b]
        lbu += [-np.inf]
        ubu += [np.inf]

        J = ca.sumsqr(x - x_des)

        g += [obs @ x - b]
        lbg += [0] * nr_obs_points
        ubg += [np.inf] * nr_obs_points

        g += [p0.T @ x - b]
        lbg += [-np.inf]
        ubg += [0]

        g += [p1.T @ x - b]
        lbg += [-np.inf]
        ubg += [0]

        g += [ca.sumsqr(x)]
        lbg += [1]
        ubg += [1.001]

        prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

        # solver = ca.qpsol('solver', 'qpoases', prob, {
        #     "print_time": False,
        #     "printLevel": "none"
        # })
        # solver = ca.qpsol(
        #     "solver",
        #     "osqp",
        #     prob,
        #     {
        #         "print_time": False,
        #         "osqp": {"verbose": False, "eps_abs": 1e-6, "eps_rel": 1e-6},
        #     },
        # )
        solver = ca.nlpsol("solver", "ipopt", prob)

        sol = solver(
            x0=np.zeros(4),
            lbx=lbu,
            ubx=ubu,
            lbg=lbg,
            ubg=ubg,
            p=params_num,
        )
        print(solver.stats())
        return sol["x"].full().flatten()

    def init_halfspaces(self):
        a_set_init = []
        b_set_init = []
        for i in range(3):
            a_halfspace = np.eye(3)[i, :]
            b_halfspace = self.e_max[i]
            norm_a = np.linalg.norm(a_halfspace)
            if norm_a > 1e-4:
                a_halfspace /= norm_a
                b_halfspace /= norm_a
            a_set_init.append(a_halfspace)
            b_set_init.append(b_halfspace)

            a_halfspace = -np.eye(3)[i, :]
            b_halfspace = -self.e_min[i]
            norm_a = np.linalg.norm(a_halfspace)
            if norm_a > 1e-4:
                a_halfspace /= norm_a
                b_halfspace /= norm_a
            a_set_init.append(a_halfspace)
            b_set_init.append(b_halfspace)
        return a_set_init, b_set_init

    def init_halfspaces_point(self, p, e_max=0.3):
        a_set_init = []
        b_set_init = []
        for i in range(3):
            a_halfspace = np.eye(3)[i, :]
            b_halfspace = p[i] + e_max
            norm_a = np.linalg.norm(a_halfspace)
            if norm_a > 1e-4:
                a_halfspace /= norm_a
                b_halfspace /= norm_a
            a_set_init.append(a_halfspace)
            b_set_init.append(b_halfspace)

            a_halfspace = -np.eye(3)[i, :]
            b_halfspace = -p[i] + e_max
            norm_a = np.linalg.norm(a_halfspace)
            if norm_a > 1e-4:
                a_halfspace /= norm_a
                b_halfspace /= norm_a
            a_set_init.append(a_halfspace)
            b_set_init.append(b_halfspace)
        return a_set_init, b_set_init

    def compute_polyhedron(self, q_inv, q_ellipse, p_seed, a_set_init, b_set_init):
        obs_remain = self.obs_sets.copy()
        obs_points_remain = self.obs_points_sets.copy()
        a_set = a_set_init.copy()
        b_set = b_set_init.copy()
        obs_points = self.compute_set_projs(obs_remain, p_seed, q_inv).tolist()
        dists = np.linalg.norm(q_ellipse @ (obs_points - p_seed).T, axis=0).tolist()
        while len(obs_remain) > 0:
            idx = np.argmin(dists)
            closest_point = obs_points[idx]
            if dists[idx] < 0.99:
                print(
                    "(Polyhedron) ERROR point is inside ellipse but should be outside."
                )
                print(f"(Polyhedron) - Dist is {dists[idx]}")
                raise RuntimeError("Ellipse violates constraints")

            a_halfspace = 2 * (q_ellipse @ q_ellipse.T) @ (closest_point - p_seed)
            b_halfspace = a_halfspace @ closest_point
            norm_a = np.linalg.norm(a_halfspace)
            a_halfspace /= norm_a
            b_halfspace /= norm_a

            # Delete obstacle points outside the halfspace
            idx_to_del = [idx]
            for i, obs_points_c in enumerate(obs_points_remain):
                if (
                    np.min(a_halfspace @ obs_points_c.T - b_halfspace) >= -1e-4
                    and i != idx
                ):
                    idx_to_del.append(i)
            for i in sorted(np.unique(idx_to_del), reverse=True):
                del obs_remain[i]
                del obs_points_remain[i]
                del obs_points[i]
                del dists[i]

            # Append to set
            a_set.append(a_halfspace)
            b_set.append(b_halfspace)
        return a_set, b_set

    def compute_set_projs(self, obs_sets, p0, ellipse_mat):
        obs_points = np.empty((len(obs_sets), 3))
        start = time.perf_counter()
        for i, (a_set, b_set) in enumerate(obs_sets):
            params = np.concatenate(
                ((a_set @ ellipse_mat).T.flatten(), b_set - a_set @ p0)
            )
            # s1 = time.perf_counter()
            sol = self.proj_solver(
                x0=np.zeros(3),
                lbx=self.proj_lbu,
                ubx=self.proj_ubu,
                lbg=self.proj_lbg,
                ubg=self.proj_ubg,
                p=params,
            )
            # s2 = time.perf_counter()
            # self.proj_solver.stats()
            # print(self.proj_solver.stats()["t_wall_solver"])
            # print(s2 - s1)
            # print("--")
            obs_points[i, :] = ellipse_mat @ np.array(sol["x"]).flatten() + p0
        stop = time.perf_counter()
        self.proj_time += stop - start
        return obs_points

    def compute_set_projs_line(self, obs_sets, p0, p1):
        obs_points = np.empty((len(obs_sets), 3))
        phi = np.empty(len(obs_sets))
        start = time.perf_counter()
        for i, (a_set, b_set) in enumerate(obs_sets):
            params = np.concatenate((a_set.T.flatten(), b_set - 0.001, p0, p1))
            # s1 = time.perf_counter()
            sol = self.projl_solver(
                x0=np.zeros(4),
                lbx=self.projl_lbu,
                ubx=self.projl_ubu,
                lbg=self.projl_lbg,
                ubg=self.projl_ubg,
                p=params,
            )
            obs_points[i, :] = np.array(sol["x"][:3]).flatten()
            phi[i] = np.array(sol["x"][3]).flatten()
        stop = time.perf_counter()
        self.proj_time += stop - start
        return obs_points, phi

    def mvie_socp(self, a_set, b_set):
        x_size = 12
        c2 = np.zeros((20, x_size))
        d2 = np.zeros(20)
        d2[: a_set.shape[0]] = b_set
        c2[: a_set.shape[0], 6:9] = -a_set

        a2 = np.zeros((x_size, 3, 20))
        a2[[0, 1, 3], 0, : a_set.shape[0]] = a_set.T
        a2[[2, 4], 1, : a_set.shape[0]] = a_set[:, 1:].T
        a2[5, 2, : a_set.shape[0]] = a_set[:, 2]
        for i in range(20):
            self.socp_params["a"][i].value = a2[:, :, i]

        self.socp_params["c"].value = c2
        self.socp_params["d"].value = d2

        self.socp_prob.solve(solver=cp.CLARABEL)
        # self.socp_prob.solve(eps=1e-4, solver=cp.SCS)

        x = self.socp_params["x"].value
        q_new = np.zeros((3, 3))
        q_new[np.tril_indices(3)] = x[:6]
        q_new = q_new @ q_new.T
        p_mid_new = x[6:9]
        return q_new, p_mid_new

    def mvie_socp_fixed_mid(self, a_set, b_set, p_mid):
        x_size = 9
        c2 = np.zeros((20, x_size))
        d2 = np.zeros(20)
        d2[: a_set.shape[0]] = b_set - a_set @ p_mid

        a2 = np.zeros((x_size, 3, 20))
        a2[[0, 1, 3], 0, : a_set.shape[0]] = a_set.T
        a2[[2, 4], 1, : a_set.shape[0]] = a_set[:, 1:].T
        a2[5, 2, : a_set.shape[0]] = a_set[:, 2]
        for i in range(20):
            self.socpfm_params["a"][i].value = a2[:, :, i]

        self.socpfm_params["c"].value = c2
        self.socpfm_params["d"].value = d2

        self.socpfm_prob.solve(solver=cp.CLARABEL)
        # self.socp_prob.solve(eps=1e-4, solver=cp.SCS)

        x = self.socpfm_params["x"].value
        q_new = np.zeros((3, 3))
        q_new[np.tril_indices(3)] = x[:6]
        q_new = q_new @ q_new.T
        return q_new, p_mid

    def mvie_socp_fixed_r(self, a_set, b_set, p_mid, r_ellipse, a_lb):
        x_size = 6
        c2 = np.zeros((20, x_size))
        d2 = np.zeros(20)
        d2[: a_set.shape[0]] = b_set - a_set @ p_mid

        a2 = np.zeros((x_size, 3, 20))
        a_set_r = a_set @ r_ellipse
        a2[0, 0, : a_set.shape[0]] = a_set_r[:, 0]
        a2[1, 1, : a_set.shape[0]] = a_set_r[:, 1]
        a2[2, 2, : a_set.shape[0]] = a_set_r[:, 2]
        for i in range(20):
            self.socpfr_params["a"][i].value = a2[:, :, i]

        self.socpfr_params["c"].value = c2
        self.socpfr_params["d"].value = d2
        self.socpfr_params["a_lb"].value = [a_lb]

        self.socpfr_prob.solve(solver=cp.CLARABEL)
        # self.socp_prob.solve(eps=1e-4, solver=cp.SCS)

        x = self.socpfr_params["x"].value
        q_new = r_ellipse @ np.diag(x[:3]) ** 2 @ r_ellipse.T
        q_ellipse = r_ellipse @ np.diag(1 / x[:3] ** 2) @ r_ellipse.T
        return q_new, q_ellipse, x[:3]

    def cvx_mvie_socp(self, max_set_size, x_size):
        x_opt = cp.Variable(x_size)

        a1, c1, d1 = self.geometric_mean_constraint(x_size)

        c2 = cp.Parameter((max_set_size, x_size))
        d2 = cp.Parameter((max_set_size,))
        c2.value = np.zeros((max_set_size, x_size))
        d2.value = np.zeros(max_set_size)

        c_kappa = np.zeros(x_size)
        c_kappa[-1] = -1.0

        soc_constraints = []
        for i in range(c1.shape[0]):
            soc_constraints.append(
                cp.SOC(c1[i, :].T @ x_opt + d1[i], a1[:, 2 * i : 2 * (i + 1)].T @ x_opt)
            )
        a2 = []
        for i in range(c2.shape[0]):
            a2c = cp.Parameter((x_size, 3))
            a2c.value = np.zeros((x_size, 3))
            a2.append(a2c)
            soc_constraints.append(cp.SOC(c2[i, :].T @ x_opt + d2[i], a2c.T @ x_opt))
        prob = cp.Problem(cp.Minimize(c_kappa.T @ x_opt), soc_constraints)
        params = {"x": x_opt, "a": a2, "c": c2, "d": d2}
        prob.solve(solver=cp.CLARABEL)
        # prob.solve(eps=1e-4, solver=cp.SCS)
        return prob, params

    def cvx_mvie_socp_fixed_mid(self, max_set_size, x_size):
        x_opt = cp.Variable(x_size)

        a1, c1, d1 = self.geometric_mean_constraint_fixed_mid(x_size)

        c2 = cp.Parameter((max_set_size, x_size))
        d2 = cp.Parameter((max_set_size,))
        c2.value = np.zeros((max_set_size, x_size))
        d2.value = np.zeros(max_set_size)

        c_kappa = np.zeros(x_size)
        c_kappa[-1] = -1.0

        soc_constraints = []
        for i in range(c1.shape[0]):
            soc_constraints.append(
                cp.SOC(c1[i, :].T @ x_opt + d1[i], a1[:, 2 * i : 2 * (i + 1)].T @ x_opt)
            )
        a2 = []
        for i in range(c2.shape[0]):
            a2c = cp.Parameter((x_size, 3))
            a2c.value = np.zeros((x_size, 3))
            a2.append(a2c)
            soc_constraints.append(cp.SOC(c2[i, :].T @ x_opt + d2[i], a2c.T @ x_opt))
        prob = cp.Problem(cp.Minimize(c_kappa.T @ x_opt), soc_constraints)
        params = {"x": x_opt, "a": a2, "c": c2, "d": d2}
        prob.solve(solver=cp.CLARABEL)
        # prob.solve(eps=1e-4, solver=cp.SCS)
        return prob, params

    def cvx_mvie_socp_fixed_r(self, max_set_size, x_size):
        x_opt = cp.Variable(x_size)

        a1, c1, d1 = self.geometric_mean_constraint_fixed_r(x_size)

        c2 = cp.Parameter((max_set_size, x_size))
        d2 = cp.Parameter((max_set_size,))
        a_lb = cp.Parameter((1,))
        c2.value = np.zeros((max_set_size, x_size))
        d2.value = np.zeros(max_set_size)
        a_lb.value = np.zeros((1,))

        soc_constraints = []
        for i in range(c1.shape[0]):
            soc_constraints.append(
                cp.SOC(c1[i, :].T @ x_opt + d1[i], a1[:, 2 * i : 2 * (i + 1)].T @ x_opt)
            )
        c_a_lb = np.zeros((x_size, 1))
        c_a_lb[0] = 1.0
        soc_constraints.append(cp.SOC(c_a_lb.T @ x_opt, a_lb))
        a2 = []
        for i in range(c2.shape[0]):
            a2c = cp.Parameter((x_size, 3))
            a2c.value = np.zeros((x_size, 3))
            a2.append(a2c)
            soc_constraints.append(cp.SOC(c2[i, :].T @ x_opt + d2[i], a2c.T @ x_opt))
        prob = cp.Problem(cp.Maximize(x_opt[-1]), soc_constraints)
        params = {"x": x_opt, "a": a2, "c": c2, "d": d2, "a_lb": a_lb}
        prob.solve(solver=cp.CLARABEL)
        # prob.solve(eps=1e-4, solver=cp.SCS)
        return prob, params

    def eval_f(self, x, c, d, a):
        # viol = c @ x + d > np.linalg.norm(x @ a)
        # if not viol:
        #     print(viol)
        #     pass
        ret = (np.dot(c, x) + d) ** 2 - x.T @ (a @ a.T) @ x
        # if np.abs(ret) < 1e-6:
        #     pass
        return ret

    def eval_df(self, x, c, d, a):
        return 2 * (np.dot(c, x) + d) * c - 2 * a @ a.T @ x

    def eval_ddf(self, x, c, d, a):
        c_ex = np.expand_dims(c, 1)
        return 2 * (np.dot(c_ex, c_ex.T) - a @ a.T)

    def geometric_mean_constraint(self, nr_x):
        a = np.zeros((nr_x, 6))
        c = np.zeros((3, nr_x))
        d = np.zeros(3)
        a[9, 0] = 1.0
        a[0, 1] = 1 / 2
        a[2, 1] = -1 / 2
        c[0, 0] = 1 / 2
        c[0, 2] = 1 / 2

        a[10, 2] = 1.0
        a[2, 3] = 1 / 2
        a[5, 3] = -1 / 2
        c[1, 2] = 1 / 2
        c[1, 5] = 1 / 2

        a[11, 4] = 1.0
        a[9, 5] = 1 / 2
        a[10, 5] = -1 / 2
        c[2, 9] = 1 / 2
        c[2, 10] = 1 / 2
        return a, c, d

    def geometric_mean_constraint_fixed_mid(self, nr_x):
        a = np.zeros((nr_x, 6))
        c = np.zeros((3, nr_x))
        d = np.zeros(3)
        a[6, 0] = 1.0
        a[0, 1] = 1 / 2
        a[2, 1] = -1 / 2
        c[0, 0] = 1 / 2
        c[0, 2] = 1 / 2

        a[7, 2] = 1.0
        a[2, 3] = 1 / 2
        a[5, 3] = -1 / 2
        c[1, 2] = 1 / 2
        c[1, 5] = 1 / 2

        a[8, 4] = 1.0
        a[6, 5] = 1 / 2
        a[7, 5] = -1 / 2
        c[2, 6] = 1 / 2
        c[2, 7] = 1 / 2
        return a, c, d

    def geometric_mean_constraint_fixed_r(self, nr_x):
        a = np.zeros((nr_x, 6))
        c = np.zeros((3, nr_x))
        d = np.zeros(3)
        a[3, 0] = 1.0
        a[0, 1] = 1 / 2
        a[1, 1] = -1 / 2
        c[0, 0] = 1 / 2
        c[0, 1] = 1 / 2

        a[4, 2] = 1.0
        a[1, 3] = 1 / 2
        a[2, 3] = -1 / 2
        c[1, 1] = 1 / 2
        c[1, 2] = 1 / 2

        a[5, 4] = 1.0
        a[3, 5] = 1 / 2
        a[4, 5] = -1 / 2
        c[2, 3] = 1 / 2
        c[2, 4] = 1 / 2
        return a, c, d


if __name__ == "__main__":
    finder = ConvexSetFinder(None, None, None)

    a_set = np.array(
        [[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    )
    b_set = np.array([0.2, 1, 1, 1, 1, 1])
    # finder.mvie_mosek(a_set, b_set)
    # finder.mvie_mosek(a_set, b_set)
    # finder.mvie_mosek(a_set, b_set)
    # finder.mvie_mosek(a_set, b_set)
    # finder.mvie_mosek(a_set, b_set)
    # finder.mvie_mosek(a_set, b_set)
    # finder.mvie_mosek(a_set, b_set)

    q_ellipse = 0.01 * np.eye(3)
    p_mid = np.zeros(3)
    q_ellipse_as, p_seed_as = finder.mvie_affine_scaling(q_ellipse, p_mid, a_set, b_set)
    q_ellipse, p_seed = finder.mvie_mosek(a_set, b_set, p_mid)
    print(np.linalg.norm(q_ellipse - q_ellipse_as))
    print(np.linalg.norm(p_seed - p_seed_as))

    # seed = [np.array([0, 0, 0])]
    # obs = [[
    #     np.array([1, 0.9, 0]),
    #     np.array([-1, 0.9, 0]),
    #     np.array([1, 0.9, 1]),
    #     np.array([-1, 0.9, 1]),
    #     np.array([1, 1, 0]),
    #     np.array([-1, 1, 0]),
    #     np.array([1, 1, 1]),
    #     np.array([-1, 1, 1]),
    # ], [
    #     np.array([1, 0.2, 0]),
    #     np.array([-1, 0.2, 0]),
    #     np.array([1, 0.2, 1]),
    #     np.array([-1, 0.2, 1]),
    #     np.array([1, 1, 0]),
    #     np.array([-1, 1, 0]),
    #     np.array([1, 1, 1]),
    #     np.array([-1, 1, 1]),
    # ]]
    # a_ellip = np.eye(3)
    # d_ellip = 0.01 * np.eye(3)
    # d_ellip_inv = np.linalg.inv(d_ellip)
    # b_ellip = np.zeros(3)

    # # Transform obstacles and seed to simpler space
    # for p in seed:
    #     p = d_ellip_inv @ a_ellip.T @ (p - b_ellip)
    # for ob in obs:
    #     for p in ob:
    #         p = d_ellip_inv @ a_ellip.T @ (p - b_ellip)

    # # Compute halfspaces
    # start = time.perf_counter()
    # for ob in obs:
    #     e_constraint = np.concatenate((np.array(seed), -np.array(ob)))
    #     f_constraint = np.concatenate((np.ones(len(seed)), -np.ones(len(ob))))
    #     print(e_constraint)
    #     print(f_constraint)
    #     y = finder.recursive_halfspace_computation(e_constraint, f_constraint)
    #     a_i = y / (y.T @ y)
    #     h_i = [a_i, a_i.T @ a_i]
    #     print(h_i)
    #     print(y/np.linalg.norm(y), 1/np.linalg.norm(y))
    # stop = time.perf_counter()
    # print(stop-start)
