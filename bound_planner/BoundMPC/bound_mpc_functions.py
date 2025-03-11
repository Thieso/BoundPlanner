import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..RobotModel import RobotModel
from ..utils.optimization_functions import rodrigues_matrix
from .jerk_trajectory_casadi import calcAcceleration, calcAngle, calcVelocity
from .mpc_utils_casadi import (
    compute_position_error,
    compute_rot_error_velocity,
    decompose_orthogonal_error,
    integrate_rot_error_diff,
)


def integrate_rotation_reference(pr_ref, omega, phi0, phi1):
    """Integrate the rotation reference by using the constant angular velocity
    omega over the interval phi1 - phi0.
    """
    r0 = R.from_rotvec(pr_ref).as_matrix()
    omega_norm = np.linalg.norm(omega)
    if omega_norm > 1e-4:
        dr = rodrigues_matrix(omega / omega_norm, (phi1 - phi0) * omega_norm)
        r1 = dr @ r0
    else:
        r1 = r0
    return R.from_matrix(r1).as_rotvec()


def compute_initial_rot_errors(pr, pr_ref, dp_normed_ref, br1, br2):
    tauc = R.from_rotvec(pr).as_matrix()
    taud = R.from_rotvec(pr_ref).as_matrix()
    dtau_init = R.from_matrix(tauc @ taud.T).as_rotvec()

    dp_normed = dp_normed_ref
    r01 = np.zeros((3, 3))
    r01[:, 0] = br2
    r01[:, 1] = dp_normed
    r01[:, 2] = br1
    dtau_01 = r01.T @ R.from_rotvec(dtau_init).as_matrix() @ r01
    eul = R.from_matrix(dtau_01).as_euler("zyx")
    dtau_init_orth2 = eul[2] * br2
    dtau_init_orth1 = eul[0] * br1
    dtau_init_par = eul[1] * dp_normed

    return [dtau_init, dtau_init_par, dtau_init_orth1, dtau_init_orth2]


def get_current_segments_split(phi, phi_switch, array):
    """Get the current entry of the array based on the value of phi if
    phi_switch gives the switching values between the entries.
    """
    if isinstance(array, np.ndarray):
        result = array[:2, :]
        for i in range(len(array) - 2):
            result = array[i + 1 : i + 3, :] if phi > phi_switch[i + 1] else result
    else:
        result = array[:2, :]
        for i in range(array.shape[0] - 2):
            result = ca.if_else(
                phi > phi_switch[i + 1], array[i + 1 : i + 3, :], result
            )
    return result[0, :], result[1, :]


def get_current_segments_1d(phi, phi_switch, array):
    """Get the current entry of the array based on the value of phi if
    phi_switch gives the switching values between the entries.
    """
    if isinstance(array, np.ndarray):
        result0 = array[0]
        result1 = array[1]
        for i in range(len(array) - 2):
            result0 = array[i + 1] if phi > phi_switch[i + 1] else result0
            result1 = array[i + 1] if phi > phi_switch[i + 2] else result1
    else:
        result0 = array[0]
        result1 = array[1]
        for i in range(len(array) - 2):
            result0 = ca.if_else(phi > phi_switch[i + 1], array[i + 1], result0)
            result1 = ca.if_else(phi > phi_switch[i + 1], array[i + 2], result1)
    return result0, result1


def reference_function(
    dp_ref,
    p_ref,
    p,
    v,
    phi_switch,
    bp1,
    bp2,
    br1,
    br2,
    v1,
    v2,
    v3,
    dp_normed_ref,
    e_r_bound,
    split_idx,
    idx,
    n_horizon,
    a_set,
    b_set,
):
    if isinstance(dp_ref, np.ndarray):
        p_d = np.zeros(6)
    else:
        p_d = ca.SX.zeros((6, 1))
    dp_d, dp_d_next = get_current_segments_split(idx, split_idx, dp_ref)
    phi_start, phi_end = get_current_segments_split(idx, split_idx, phi_switch)
    p_ref_current, p_ref_next = get_current_segments_split(idx, split_idx, p_ref)
    phi = (p[:3].T - p_ref_current[:3]) @ dp_d[:3].T
    phi_next = (p[:3].T - p_ref_next[:3]) @ dp_d_next[:3].T
    dphi = v[:3].T @ dp_d[:3].T
    if isinstance(dp_ref, np.ndarray):
        dphi = np.array(dphi).flatten()
        phi = np.array(phi).flatten()
        p_d[:3] = p_ref_current[:3].T + dp_d[:3].T * phi
        p_d[3:] = dp_d[3:].T * phi + p_ref_current[3:].T
        p_dr_next = dp_d_next[3:].T * phi_next + p_ref_next[3:].T
        dp_d[:3] = dp_d[:3].T
    else:
        p_d[:3] = p_ref_current[:3] + dp_d[:3] * phi
        p_d[3:] = dp_d[3:] * phi + p_ref_current[3:]
        p_dr_next = dp_d_next[3:] * phi_next + p_ref_next[3:]
    phi += phi_start

    e_r_boundc, e_r_boundn = get_current_segments_split(idx, split_idx, e_r_bound)
    r_e_upper = e_r_boundc[:3]
    r_e_lower = e_r_boundc[3:]
    r_e_upper_next = e_r_boundn[:3]
    r_e_lower_next = e_r_boundn[3:]

    bp10, _ = get_current_segments_split(idx, split_idx, bp1)
    bp20, _ = get_current_segments_split(idx, split_idx, bp2)
    bp10 = bp10.T
    bp20 = bp20.T
    if isinstance(dp_ref, np.ndarray):
        bp10 = bp10.T
        bp20 = bp20.T
    br10, br11 = get_current_segments_split(idx, split_idx, br1)
    br20, br21 = get_current_segments_split(idx, split_idx, br2)
    dp_normed_d, dp_normed_n = get_current_segments_split(idx, split_idx, dp_normed_ref)
    v10, v11 = get_current_segments_split(idx, split_idx, v1)
    v20, v21 = get_current_segments_split(idx, split_idx, v2)
    v30, v31 = get_current_segments_split(idx, split_idx, v3)
    if not isinstance(dp_ref, np.ndarray):
        br10 = br10.T
        br20 = br20.T
        br11 = br11.T
        br21 = br21.T
        v10 = v10.T
        v20 = v20.T
        v30 = v30.T
        v11 = v11.T
        v21 = v21.T
        v31 = v31.T

    ddp_d = 0 * dp_d.T
    a_current, _ = get_current_segments_1d(idx, split_idx, a_set)
    b_current, _ = get_current_segments_split(idx, split_idx, b_set)
    if isinstance(dp_ref, np.ndarray):
        if split_idx[1] == n_horizon:
            a_next = a_set[1]
            b_next = b_set[1]
            phi_end_seg = phi_switch[2]
        elif split_idx[2] == n_horizon:
            a_next = a_set[2]
            b_next = b_set[2]
            phi_end_seg = phi_switch[3]
        else:
            a_next = a_set[3]
            b_next = b_set[3]
            phi_end_seg = phi_switch[4]
    else:
        cond1 = ca.if_else(split_idx[2] == n_horizon, a_set[2], a_set[3])
        a_next = ca.if_else(split_idx[1] == n_horizon, a_set[1], cond1)
        cond1 = ca.if_else(split_idx[2] == n_horizon, b_set[2, :], b_set[3, :])
        b_next = ca.if_else(split_idx[1] == n_horizon, b_set[1, :], cond1)
        cond1 = ca.if_else(split_idx[2] == n_horizon, phi_switch[2], phi_switch[3])
        phi_end_seg = ca.if_else(split_idx[1] == n_horizon, phi_switch[1], cond1)

    outputs = [
        p_d,
        p_dr_next,
        p_ref_current[3:],
        dp_d,
        ddp_d,
        bp10,
        bp20,
        br10,
        br20,
        br11,
        br21,
        dp_normed_d,
        dp_normed_n,
        v10,
        v20,
        v30,
        v11,
        v21,
        v31,
        r_e_lower,
        r_e_upper,
        r_e_lower_next,
        r_e_upper_next,
        a_current,
        b_current,
        a_next,
        b_next,
        phi_end_seg,
        phi,
        dphi,
        phi_start,
    ]
    output_names = [
        "p_d",
        "p_dr_next",
        "p_r_omega0",
        "dp_d",
        "ddp_d",
        "bp1_current",
        "bp2_current",
        "br1_current",
        "br2_current",
        "br1_next",
        "br2_next",
        "dp_normed_d",
        "dp_normed_n",
        "v1_current",
        "v2_current",
        "v3_current",
        "v1_next",
        "v2_next",
        "v3_next",
        "r_bound_lower",
        "r_bound_upper",
        "r_bound_lower_next",
        "r_bound_upper_next",
        "a_current",
        "b_current",
        "a_next",
        "b_next",
        "phi_end_seg",
        "phi",
        "dphi",
        "phi_switchk",
    ]
    reference_data = {}
    for key, value in zip(output_names, outputs):
        reference_data[key] = value
    return reference_data


def error_function(
    p,
    pr_next,
    v,
    p_ref,
    dp_ref,
    ddp_ref,
    dp_normed_ref,
    dp_normed_refn,
    dphi,
    i_omega_0,
    i_omega_ref_0,
    i_omega_ref_seg,
    dtau_init,
    dtau_init_par,
    dtau_init_orth1,
    dtau_init_orth2,
    br1,
    br2,
    br1n,
    br2n,
    jac_dtau_l,
    jac_dtau_r,
    idx,
    split_idx,
    v1,
    v2,
    v3,
    v1n,
    v2n,
    v3n,
    n_horizon,
):
    # Compute position errors
    e_p_par, e_p_orth, de_p_par, de_p_orth, e_p, de_p = compute_position_error(
        p[:3], v[:3], p_ref[:3], dp_ref[:3], ddp_ref[:3].T, dphi
    )

    if isinstance(dtau_init, np.ndarray):
        if idx <= split_idx[1]:
            i_w_ref_0 = i_omega_ref_0
        else:
            i_w_ref_0 = i_omega_ref_seg
    else:
        i_w_ref_0 = ca.if_else(idx <= split_idx[1], i_omega_ref_0, i_omega_ref_seg)

    # Compute orientation error and its derivative
    e_init, _ = get_current_segments_split(idx, split_idx, dtau_init)
    if isinstance(dtau_init, np.ndarray):
        if split_idx[1] == n_horizon:
            e_initn = dtau_init[1, :]
        elif split_idx[2] == n_horizon:
            e_initn = dtau_init[2, :]
        else:
            e_initn = dtau_init[3, :]
        e_initn = e_initn.T
    else:
        cond1 = ca.if_else(split_idx[2] == n_horizon, dtau_init[2, :], dtau_init[3, :])
        e_initn = ca.if_else(split_idx[1] == n_horizon, dtau_init[1, :], cond1)
    e_r = integrate_rot_error_diff(
        e_init.T, p[3:], i_omega_0, p_ref[3:], i_w_ref_0, jac_dtau_l, jac_dtau_r
    )
    e_rn = integrate_rot_error_diff(
        e_initn.T, p[3:], i_omega_0, pr_next.T, i_w_ref_0, jac_dtau_l, jac_dtau_r
    )
    de_r = compute_rot_error_velocity(dp_ref[3:], v[3:], jac_dtau_l, jac_dtau_r, dphi)

    # Compute correct starting value based on the segment for the orientation
    # error
    e_par_init, e_par_initn = get_current_segments_split(idx, split_idx, dtau_init_par)
    e_orth1_init, e_orth1_initn = get_current_segments_split(
        idx, split_idx, dtau_init_orth1
    )
    e_orth2_init, e_orth2_initn = get_current_segments_split(
        idx, split_idx, dtau_init_orth2
    )
    e_par_init = e_par_init.T
    e_orth1_init = e_orth1_init.T
    e_orth2_init = e_orth2_init.T
    e_par_initn = e_par_initn.T
    e_orth1_initn = e_orth1_initn.T
    e_orth2_initn = e_orth2_initn.T

    # Project error onto path
    scal_orth1 = (e_r.T - e_init) @ v1
    scal_par = (e_r.T - e_init) @ v2
    scal_orth2 = (e_r.T - e_init) @ v3
    scal_orth1n = (e_rn.T - e_initn) @ v1n
    scal_parn = (e_rn.T - e_initn) @ v2n
    scal_orth2n = (e_rn.T - e_initn) @ v3n

    # Final decomposed orientation errors
    e_r_orth1 = e_orth1_init + scal_orth1 * br1
    e_r_par = e_par_init + scal_par * dp_normed_ref
    e_r_orth2 = e_orth2_init + scal_orth2 * br2
    e_r_orth1n = e_orth1_initn + scal_orth1n * br1n
    e_r_parn = e_par_initn + scal_parn * dp_normed_refn
    e_r_orth2n = e_orth2_initn + scal_orth2n * br2n

    outputs = [
        e_p_par,
        e_p_orth,
        de_p_par,
        de_p_orth,
        e_p,
        de_p,
        e_r,
        de_r,
        e_r_orth1,
        e_r_par,
        e_r_orth2,
        e_r_orth1n,
        e_r_parn,
        e_r_orth2n,
    ]
    output_names = [
        "e_p_par",
        "e_p_orth",
        "de_p_par",
        "de_p_orth",
        "e_p",
        "de_p",
        "e_r",
        "de_r",
        "e_r_orth1",
        "e_r_par",
        "e_r_orth2",
        "e_r_orth1n",
        "e_r_parn",
        "e_r_orth2n",
    ]
    error_data = {}
    for key, value in zip(output_names, outputs):
        error_data[key] = value
    return error_data


def objective_function(x_phi, e_r_par, x_phi_d, v_orth, u, dq, weights, k):
    """Create the objective function for the MPC."""
    # Extract weights
    w_p = weights[0]
    w_r = weights[1]
    w_v_p = weights[2]
    w_v_r = weights[3]
    w_phi = weights[4]
    w_dphi = weights[5]
    w_dq = weights[6]
    w_jerk = weights[7]

    # Create objective term
    objective_term = 0
    objective_term += w_r * ca.sumsqr(e_r_par)

    # Cartesian velocity and acceleration
    objective_term += w_v_p * ca.sumsqr(v_orth[:3])
    objective_term += w_v_r * ca.sumsqr(v_orth[3:])

    # Joint state
    objective_term += w_dq * ca.sumsqr(dq[2])
    objective_term += w_dq * ca.sumsqr(dq[3])
    objective_term += w_dq * ca.sumsqr(dq[4])
    objective_term += w_jerk * ca.sumsqr(u)

    # Path state
    # objective_term += w_phi * ca.sumsqr(x_phi_d[0] - x_phi[0])
    objective_term += w_phi * approx_one_norm(x_phi_d[0] - x_phi[0])
    objective_term += w_dphi * ca.sumsqr(x_phi_d[1] - x_phi[1])

    return objective_term


def approx_one_norm(x, alpha=0.1):
    return ca.sqrt(ca.sumsqr(x) + alpha**2) - alpha


def collision_function(q, obj_centers, obj_radii, alpha, beta):
    """Compute collision terms for the robot based on potential functions."""
    robot_model = RobotModel()
    p_list = [
        robot_model.fk_pos_j3(q),
        robot_model.fk_pos_j5(q),
        robot_model.fk_pos_j6(q),
        robot_model.fk_pos_j67(q),
        robot_model.fk_pos_elbow(q),
    ]
    radii = [0.09, 0.09, 0.09, 0.05, 0.11]

    f_col = 0
    for p, rad in zip(p_list, radii):
        for i in range(obj_centers.shape[0]):
            d = distance_sphere_sphere(p, rad, obj_centers[i, :].T, obj_radii[i])
            f_col += ca.log(1 + ca.exp(-alpha * (d + beta)))
    return f_col


def distance_sphere_sphere(p1, r1, p2, r2):
    d = ca.sumsqr(p1 - p2) - ca.sumsqr(r1 + r2)
    return d


def integration_function(
    nr_joints, nr_u, dt, q, dq, ddq, p_rot, phi, dphi, ddphi, u, u_prev
):
    """
    Create the function to integrate the state of dx = f(x, u).
    """
    jerk_matrix = ca.vertcat(u_prev.T, u.T).T
    u_prevn = u

    # Integrate the state using triangle functions
    # qn = calcAngle(jerk_matrix[:nr_joints, :], dt, q, dq, ddq, dt)
    # dqn = calcVelocity(jerk_matrix[:nr_joints, :], dt, dq, ddq, dt)
    # ddqn = calcAcceleration(jerk_matrix[:nr_joints, :], dt, ddq, dt)
    qn = (
        ddq * dt**2 / 2.0
        + dq * dt
        + q
        + u_prev[:7] * dt**3 / 8.0
        + u[:7] * dt**3 / 24.0
    )
    dqn = ddq * dt + dq + u_prev[:7] * dt**2 / 3.0 + u[:7] * dt**2 / 6.0
    ddqn = ddq + u_prev[:7] * dt / 2.0 + u[:7] * dt / 2.0

    robot_model = RobotModel()
    pn_pos = robot_model.fk_pos(qn)
    v_cart = robot_model.velocity_ee(qn, dqn)
    v_rot = robot_model.omega_ee(qn, dqn)
    vn = ca.vertcat(v_cart, v_rot)

    # RK4 integration of omega
    # pn_rot = p_rot + dt * omega_ee(q, dq)
    # q_mid = calcAngle(jerk_matrix[:, :nr_joints], dt/2, q, dq, ddq, dt)
    # dq_mid = calcVelocity(jerk_matrix[:, :nr_joints], dt/2, dq, ddq, dt)
    # k1 = omega_ee(q, dq)
    # k2 = omega_ee(q_mid, dq_mid)
    # k4 = omega_ee(qn, dqn)
    # pn_rot = p_rot + 1/6 * dt * (k1 + 4*k2 + k4)

    # Trapezoidal integration of omega
    k1 = robot_model.omega_ee(q, dq)
    k2 = v_rot
    pn_rot = p_rot + 1 / 2 * dt * (k1 + k2)

    pn = ca.vertcat(pn_pos, pn_rot)
    phin = calcAngle(jerk_matrix[-1, :], dt, phi, dphi, ddphi, dt)
    dphin = calcVelocity(jerk_matrix[-1, :], dt, dphi, ddphi, dt)
    ddphin = calcAcceleration(jerk_matrix[-1, :], dt, ddphi, dt)

    outputs = [qn, dqn, ddqn, pn, vn, phin, dphin, ddphin, u_prevn]
    output_names = [
        "q_new",
        "dq_new",
        "ddq_new",
        "p_new",
        "v_new",
        "phi_new",
        "dphi_new",
        "ddphi_new",
        "u_prev_new",
    ]
    int_data = {}
    for key, value in zip(output_names, outputs):
        int_data[key] = value
    return int_data


def decomp_function(e, e_off, b1, b2, p_lower, p_upper):
    e_plane = decompose_orthogonal_error(e, b1, b2)
    e_diff = e_plane - e_off
    bound = (p_upper - p_lower) / 2
    constraint0 = ca.sumsqr(e_diff[0]) - ca.sumsqr(bound[0])
    constraint1 = ca.sumsqr(e_diff[1]) - ca.sumsqr(bound[1])

    outputs = [constraint0, constraint1]
    output_names = ["constraint0", "constraint1"]
    decomp_data = {}
    for key, value in zip(output_names, outputs):
        decomp_data[key] = value
    return decomp_data
