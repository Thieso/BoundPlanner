import casadi as ca
import numpy as np

solver_name = "ipopt"
ipopt_options = {
    "tol": 10e-4,
    "max_iter": 500,
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
    "fixed_mu_oracle": "quality-function",
    "line_search_method": "filter",
    "expect_infeasible_problem": "no",
    "print_level": 0,
}

solver_opts = {
    "verbose": False,
    "verbose_init": False,
    "print_time": False,
    "ipopt": ipopt_options,
}


def jac_SO3_inv_right(axis):
    if isinstance(axis, np.ndarray):
        angle = np.linalg.norm(axis) + 1e-6
        ident = np.eye(3)
    else:
        angle = ca.norm_2(axis) + 1e-6
        ident = ca.MX.eye(3)
    omega_mat = skew_matrix(axis)
    jac_inv = ident + 0.5 * omega_mat
    jac_inv += (
        (1 / angle**2 - (1 + np.cos(angle)) / (2 * angle * np.sin(angle)))
        * omega_mat
        @ omega_mat
    )
    return jac_inv


def jac_SO3_inv_left(axis):
    if isinstance(axis, np.ndarray):
        angle = np.linalg.norm(axis) + 1e-6
        ident = np.eye(3)
    else:
        angle = ca.norm_2(axis) + 1e-6
        ident = ca.MX.eye(3)
    omega_mat = skew_matrix(axis)
    jac_inv = ident - 0.5 * omega_mat
    jac_inv += (
        (1 / angle**2 - (1 + np.cos(angle)) / (2 * angle * np.sin(angle)))
        * omega_mat
        @ omega_mat
    )
    return jac_inv


def skew_matrix(omega):
    if isinstance(omega, ca.DM) or isinstance(omega, np.ndarray):
        mat = np.zeros((3, 3))
    else:
        mat = ca.SX.zeros((3, 3))
    mat[0, 1] = -omega[2]
    mat[1, 0] = omega[2]
    mat[0, 2] = omega[1]
    mat[2, 0] = -omega[1]
    mat[1, 2] = -omega[0]
    mat[2, 1] = omega[0]
    return mat


def rodrigues_matrix(omega, phi):
    """Compute rodrigues matrix given an axis of rotation and an angle.
    Parameters
    ----------
    omega : array 3x1
        unit axis of rotation
    phi : float
        angle
    Returns
    -------
    mat_rodrigues : matrix 3x3
        rotation matrix
    """
    if isinstance(omega, ca.DM) or isinstance(omega, np.ndarray):
        ident = np.eye(3)
    else:
        ident = ca.SX.eye(3)
    omega_mat = skew_matrix(omega)
    mat_rodrigues = (
        ident + ca.sin(phi) * omega_mat + (1 - ca.cos(phi)) * omega_mat @ omega_mat
    )
    return mat_rodrigues


def projection_opt_problem(max_set_size=20):
    a_set = ca.SX.sym("a set", max_set_size, 3)
    b_set = ca.SX.sym("b set", max_set_size)
    xd = ca.SX.sym("xd", 3)

    params = ca.vertcat(a_set.reshape((-1, 1)), b_set, xd)

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

    J = ca.sumsqr(x - xd)

    g += [a_set @ x - b_set]
    lbg += [-np.inf] * max_set_size
    ubg += [0] * max_set_size

    prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

    solver = ca.qpsol(
        "solver", "qpoases", prob, {"print_time": False, "printLevel": "none"}
    )
    return solver, lbu, ubu, lbg, ubg


def fit_opt_problem_sample(max_set_size):
    a_set = ca.SX.sym("a set", max_set_size, 3)
    b_set = ca.SX.sym("b set", max_set_size)
    l_ee = ca.SX.sym("l ee", 3)
    p_in_set = ca.SX.sym("p in set", 3)

    params = ca.vertcat(l_ee, a_set.reshape((-1, 1)), b_set)

    g = []
    lbg = []
    ubg = []
    u = []
    lbu = []
    ubu = []

    u += [p_in_set]
    lbu += [-np.inf] * 3
    ubu += [np.inf] * 3

    p_ee = p_in_set + l_ee

    J = 0

    g += [a_set @ p_in_set - b_set]
    lbg += [-np.inf] * max_set_size
    ubg += [0] * max_set_size

    g += [a_set @ p_ee - b_set]
    lbg += [-np.inf] * max_set_size
    ubg += [0] * max_set_size

    prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

    solver = ca.qpsol(
        "solver",
        "qpoases",
        prob,
        {
            "print_time": False,
            "printLevel": "none",
            "error_on_fail": False,
        },
    )
    return solver, lbu, ubu, lbg, ubg


def via_point_optimization_problem(nr_via=4, max_set_size=20):
    a_sets = ca.SX.sym("a set", max_set_size, 3, nr_via)
    b_sets = ca.SX.sym("b set", max_set_size, nr_via)
    p_start = ca.SX.sym("p start", 3)
    p_end = ca.SX.sym("p end", 3)

    params = ca.vertcat(p_start, p_end)
    for i in range(nr_via):
        params = ca.vertcat(params, a_sets[i].reshape((-1, 1)), b_sets[:, i])

    g = []
    lbg = []
    ubg = []
    u = []
    lbu = []
    ubu = []

    J = 0
    p_via_prev = p_start
    for i in range(nr_via):
        p_via = ca.SX.sym(f"p_via {i}", 3)
        u += [p_via]
        lbu += [-np.inf] * 3
        ubu += [np.inf] * 3

        J += ca.sumsqr(p_via - p_via_prev)
        p_via_prev = p_via

        g += [a_sets[i] @ p_via - b_sets[:, i]]
        lbg += [-np.inf] * max_set_size
        ubg += [0] * max_set_size
    J += ca.sumsqr(p_end - p_via)

    prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

    solver = ca.qpsol(
        "solver", "qpoases", prob, {"print_time": False, "printLevel": "none"}
    )
    return solver, lbu, ubu, lbg, ubg


def via_point_rot_optimization_problem(nr_via=4, max_set_size=20):
    a_inter = ca.SX.sym("a set inter", max_set_size, 3, nr_via)
    b_inter = ca.SX.sym("b set inter", max_set_size, nr_via)
    a_via = ca.SX.sym("a set via", max_set_size, 3, nr_via + 1)
    b_via = ca.SX.sym("b set via", max_set_size, nr_via + 1)
    w_size_via = ca.SX.sym("w size via", nr_via + 1)
    p_start = ca.SX.sym("p start", 3)
    p_end = ca.SX.sym("p end", 3)
    l_ee = ca.SX.sym("l ee", 3)
    omega = ca.SX.sym("omega", 3)
    omega_norm = ca.SX.sym("omega norm")

    params = ca.vertcat(p_start, p_end, l_ee, omega, omega_norm, w_size_via)
    for i in range(nr_via):
        params = ca.vertcat(params, a_inter[i].reshape((-1, 1)), b_inter[:, i])
    for i in range(nr_via + 1):
        params = ca.vertcat(params, a_via[i].reshape((-1, 1)), b_via[:, i])

    g = []
    lbg = []
    ubg = []
    u = []
    lbu = []
    ubu = []

    J = 0
    p_via_prev = p_start
    omega_via_prev = 0.0
    for i in range(nr_via):
        p_via = ca.SX.sym(f"p_via {i}", 3)
        omega_via = ca.SX.sym(f"omega_via {i}", 1)
        u += [p_via]
        lbu += [-np.inf] * 3
        ubu += [np.inf] * 3
        u += [omega_via]
        lbu += [0]
        ubu += [1]

        p_via_ee = p_via + rodrigues_matrix(omega, omega_norm * omega_via) @ l_ee

        J += w_size_via[i] * ca.sumsqr(p_via - p_via_prev)
        J += w_size_via[i] * ca.sumsqr(omega_via - omega_via_prev)

        g += [a_inter[i] @ p_via - b_inter[:, i]]
        lbg += [-np.inf] * max_set_size
        ubg += [0] * max_set_size

        for j in range(a_via[i].shape[0]):
            v = p_via - p_via_prev
            a = a_via[i][j, :].T
            b = b_via[j, i]

            phi_max = ca.SX.sym(f"phi_max {i} {j}")
            u += [phi_max]
            lbu += [0.0]
            ubu += [1.0]

            omega_mid = omega_via_prev + phi_max * (omega_via - omega_via_prev)
            l_max = rodrigues_matrix(omega, omega_norm * omega_mid) @ l_ee
            p_max = p_via_prev + phi_max * v
            p_max_ee = ca.dot(a, p_max + l_max)
            dp_max_ee = ca.jacobian(p_max_ee, phi_max)

            if i == 0:
                omega_prev = ca.SX.sym("omega_prev")
                f_max = ca.Function(
                    "f_max",
                    [
                        phi_max,
                        omega_via,
                        omega_prev,
                        a,
                        p_via,
                        p_via_prev,
                        l_ee,
                        omega,
                        omega_norm,
                    ],
                    [dp_max_ee],
                )
            else:
                f_max = ca.Function(
                    "f_max",
                    [
                        phi_max,
                        omega_via,
                        omega_via_prev,
                        a,
                        p_via,
                        p_via_prev,
                        l_ee,
                        omega,
                        omega_norm,
                    ],
                    [dp_max_ee],
                )
            constr = f_max(
                phi_max,
                omega_via,
                omega_via_prev,
                a,
                p_via,
                p_via_prev,
                l_ee,
                omega,
                omega_norm,
            )
            constr0 = f_max(
                0,
                omega_via,
                omega_via_prev,
                a,
                p_via,
                p_via_prev,
                l_ee,
                omega,
                omega_norm,
            )
            constr1 = f_max(
                1,
                omega_via,
                omega_via_prev,
                a,
                p_via,
                p_via_prev,
                l_ee,
                omega,
                omega_norm,
            )
            constr_with_bounds = ca.if_else(constr0 * constr1 < 0, constr, 0)

            g += [constr_with_bounds]
            lbg += [0]
            ubg += [0]
            g += [p_max_ee - b]
            lbg += [-np.inf]
            ubg += [0]

        g += [a_inter[i] @ p_via_ee - b_inter[:, i]]
        lbg += [-np.inf] * max_set_size
        ubg += [0] * max_set_size

        p_via_prev = p_via
        omega_via_prev = omega_via
    J += w_size_via[-1] * ca.sumsqr(p_end - p_via)
    J += w_size_via[-1] * ca.sumsqr(1 - omega_via)

    for pos in [0.25, 0.5]:
        p_mid = p_via + pos * (p_end - p_via)
        omega_mid = omega_via + pos * (1 - omega_via)
        p_mid_ee = p_mid + rodrigues_matrix(omega, omega_norm * omega_mid) @ l_ee

        g += [a_via[-1] @ p_mid_ee - b_via[:, -1]]
        lbg += [-np.inf] * max_set_size
        ubg += [0] * max_set_size

    prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

    solver = ca.nlpsol("solver", solver_name, prob, solver_opts)

    return solver, lbu, ubu, lbg, ubg
