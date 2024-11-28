import casadi as ca
import numpy as np

from ..RobotModel import RobotModel
from ..utils.lie_functions import rodrigues_matrix
from .jerk_trajectory_casadi import calcAcceleration, calcAngle, calcVelocity
from .mpc_utils_casadi import (
    compute_fourth_order_error_bound_general,
    compute_position_error,
    compute_rot_error_velocity,
    compute_sixth_order_error_bound_general,
    decompose_orthogonal_error,
    integrate_rot_error_diff,
)


def get_current_segment(phi, phi_switch, array):
    """Get the current entry of the array based on the value of phi if
    phi_switch gives the switching values between the entries.
    """
    result = array[-1, :]
    for i in reversed(range(array.shape[0] - 1)):
        result = ca.if_else(phi < phi_switch[i + 1], array[i, :], result)
    return result


def get_current_segment_split(phi, phi_switch, array):
    """Get the current entry of the array based on the value of phi if
    phi_switch gives the switching values between the entries.
    """
    result = array[2, :]
    for i in range(array.shape[0] - 1):
        result = ca.if_else(phi > phi_switch[i + 1], array[i, :], result)
    return result


def get_current_and_next_segment_1d(phi, phi_switch, array):
    """Get the current and the next entry of the array based on the value of
    phi if phi_switch gives the switching values between the entries."""
    result0 = array[-2]
    result1 = array[-1]
    for i in reversed(range(len(array) - 2)):
        result0 = ca.if_else(phi < phi_switch[i + 1], array[i], result0)
        result1 = ca.if_else(phi < phi_switch[i + 1], array[i + 1], result1)
    return result0, result1


def get_current_and_next_segment(phi, phi_switch, array):
    """Get the current and the next entry of the array based on the value of
    phi if phi_switch gives the switching values between the entries."""
    result = array[-2:, :]
    for i in reversed(range(array.shape[0] - 2)):
        result = ca.if_else(phi < phi_switch[i + 1], array[i : i + 2, :], result)
    return result[0, :], result[1, :]


def get_current_segments_split(phi, phi_switch, array):
    """Get the current entry of the array based on the value of phi if
    phi_switch gives the switching values between the entries.
    """
    result = array[:2, :]
    for i in range(array.shape[0] - 2):
        result = ca.if_else(phi > phi_switch[i + 1], array[i + 1 : i + 3, :], result)
    return result[0, :], result[1, :]


def get_current_segments_1d(phi, phi_switch, array):
    """Get the current entry of the array based on the value of phi if
    phi_switch gives the switching values between the entries.
    """
    result0 = array[0]
    result1 = array[1]
    for i in range(len(array) - 2):
        result0 = ca.if_else(phi > phi_switch[i + 1], array[i + 1], result0)
        result1 = ca.if_else(phi > phi_switch[i + 1], array[i + 2], result1)
    return result0, result1


def reference_function(
    dp_ref,
    p_ref,
    p_ref_p,
    dp_ref_p,
    phi_switch,
    phi,
    bp1,
    bp2,
    br1,
    br2,
    v1,
    v2,
    v3,
    dp_normed_ref,
    a6,
    a5,
    a4,
    a3,
    a2,
    a1,
    a0,
    spiral_a,
    spiral_l,
    spiral_theta,
    spiral_rot2d,
    spiral_rot_bp1,
    spiral_rot_bp1_norm,
    spiral_rot_bp2_norm,
    split_idx,
    idx,
    n_horizon,
    a_set,
    b_set,
):
    if isinstance(dp_ref, np.ndarray):
        p_e_bound = np.zeros((4, 1))
        r_e_upper = np.empty(3)
        r_e_lower = np.empty(3)
        p_d = np.zeros((6, 1))
        r0 = np.zeros((2, 2))
        r1 = np.zeros((2, 2))
        ddp_lin = np.zeros((3, 1))
    else:
        p_e_bound = ca.SX.zeros(4)
        r_e_upper = ca.SX.zeros(3)
        r_e_lower = ca.SX.zeros(3)
        p_d = ca.SX.zeros((6, 1))
        r0 = ca.SX.zeros((2, 2))
        r1 = ca.SX.zeros((2, 2))
        ddp_lin = ca.SX.zeros((3, 1))
    dp_d = get_current_segment(phi, phi_switch, dp_ref)
    phi_start, phi_end = get_current_and_next_segment(phi, phi_switch, phi_switch)
    p_ref_current = get_current_segment(phi, phi_switch, p_ref)
    p_d[:3] = p_ref_p
    p_d[3:] = dp_d[3:] * (phi - phi_start) + p_ref_current[3:]
    dp_d[:3] = dp_ref_p

    phi0 = phi_start
    a6c = get_current_segment(phi, phi_switch, a6)
    a5c = get_current_segment(phi, phi_switch, a5)
    a4c = get_current_segment(phi, phi_switch, a4)
    a3c = get_current_segment(phi, phi_switch, a3)
    a2c = get_current_segment(phi, phi_switch, a2)
    a1c = get_current_segment(phi, phi_switch, a1)
    a0c = get_current_segment(phi, phi_switch, a0)
    # for i in range(2):
    #     p_e_bound[i] = compute_fourth_order_error_bound_general(
    #         phi - phi0, a4c[i], a3c[i], a2c[i], a1c[i], a0c[i])
    #     p_e_bound[i + 2] = compute_fourth_order_error_bound_general(
    #         phi - phi0, a4c[i + 2], a3c[i + 2], a2c[i + 2], a1c[i + 2],
    #         a0c[i + 2])
    # for i in range(3):
    #     r_e_upper[i] = compute_fourth_order_error_bound_general(
    #         phi - phi0, a4c[i+4], a3c[i+4], a2c[i+4], a1c[i+4], a0c[i+4])
    #     r_e_lower[i] = compute_fourth_order_error_bound_general(
    #         phi - phi0, a4c[i+7], a3c[i+7], a2c[i+7], a1c[i+7], a0c[i+7])
    for i in range(2):
        p_e_bound[i] = compute_sixth_order_error_bound_general(
            phi - phi0, a6c[i], a5c[i], a4c[i], a3c[i], a2c[i], a1c[i], a0c[i]
        )
        p_e_bound[i + 2] = compute_sixth_order_error_bound_general(
            phi - phi0,
            a6c[i + 2],
            a5c[i + 2],
            a4c[i + 2],
            a3c[i + 2],
            a2c[i + 2],
            a1c[i + 2],
            a0c[i + 2],
        )
    for i in range(3):
        r_e_upper[i] = compute_sixth_order_error_bound_general(
            phi - phi0,
            a6c[i + 4],
            a5c[i + 4],
            a4c[i + 4],
            a3c[i + 4],
            a2c[i + 4],
            a1c[i + 4],
            a0c[i + 4],
        )
        r_e_lower[i] = compute_sixth_order_error_bound_general(
            phi - phi0,
            a6c[i + 7],
            a5c[i + 7],
            a4c[i + 7],
            a3c[i + 7],
            a2c[i + 7],
            a1c[i + 7],
            a0c[i + 7],
        )

    bound_lower = ca.vertcat(p_e_bound[2:], r_e_lower)
    bound_upper = ca.vertcat(p_e_bound[:2], r_e_upper)
    e_p_off = 0.5 * (p_e_bound[:2] + p_e_bound[2:])
    e_r_off = 0.5 * (r_e_upper + r_e_lower)
    bp10 = get_current_segment(phi, phi_switch, bp1).T
    bp20 = get_current_segment(phi, phi_switch, bp2).T
    if isinstance(dp_ref, np.ndarray):
        bp10 = bp10.T
        bp20 = bp20.T
    br10 = get_current_segment(phi, phi_switch, br1).T
    br20 = get_current_segment(phi, phi_switch, br2).T
    dp_normed_d = get_current_segment(phi, phi_switch, dp_normed_ref)
    v10 = get_current_segment(phi, phi_switch, v1).T
    v20 = get_current_segment(phi, phi_switch, v2).T
    v30 = get_current_segment(phi, phi_switch, v3).T

    # Basis vector rotation
    mat0 = ca.vertcat(bp10.T, bp20.T).T
    a, b = get_current_and_next_segment(phi, phi_switch, spiral_a)
    rot_2d0, rot_2d1 = get_current_and_next_segment_1d(phi, phi_switch, spiral_rot2d)
    if isinstance(dp_ref, np.ndarray):
        rot_2d0 = rot_2d0.T
        rot_2d1 = rot_2d1.T
    axis0, axis1 = get_current_and_next_segment(phi, phi_switch, spiral_rot_bp1)
    norm00, norm10 = get_current_and_next_segment(phi, phi_switch, spiral_rot_bp1_norm)
    norm01, norm11 = get_current_and_next_segment(phi, phi_switch, spiral_rot_bp2_norm)
    dp_d0, dp_d1 = get_current_and_next_segment(phi, phi_switch, dp_ref)

    l0, l1 = get_current_and_next_segment(phi, phi_switch, spiral_l)
    phi_spiral0 = l0 - (phi - phi_start)
    angle0 = (phi_spiral0**2) / (2 * l0**2)
    rot_mat = (
        rodrigues_matrix(axis0, angle0 * norm00).T
        @ rodrigues_matrix(dp_d0[:3], angle0 * norm01).T
    )
    mat_rotated0 = rot_mat @ mat0
    bp1_rotated0 = mat_rotated0[:, 0]
    bp2_rotated0 = mat_rotated0[:, 1]

    phi_spiral1 = phi - (phi_end - l1)
    angle1 = (phi_spiral1**2) / (2 * l1**2)
    rot_mat = rodrigues_matrix(axis1, angle1 * norm10) @ rodrigues_matrix(
        dp_d0[:3], angle1 * norm11
    )
    mat_rotated1 = rot_mat @ mat0
    bp1_rotated1 = mat_rotated1[:, 0]
    bp2_rotated1 = mat_rotated1[:, 1]

    # if isinstance(dp_ref, np.ndarray):
    #     if phi < phi_start + l_seg:
    #         print("Start: ", phi, angle0, norm01 * 180/np.pi, bp1_rotated0)
    #     elif phi > phi_end - l_seg:
    #         print("End: ", phi, angle1, norm11 * 180/np.pi, bp1_rotated1)
    #     else:
    #         print("Linear: ", phi, bp10)

    cond1 = ca.if_else(phi > phi_end - l1, bp1_rotated1, bp10)
    bp10 = ca.if_else(phi < phi_start + l0, bp1_rotated0, cond1)

    cond1 = ca.if_else(phi > phi_end - l1, bp2_rotated1, bp20)
    bp20 = ca.if_else(phi < phi_start + l0, bp2_rotated0, cond1)

    theta0 = get_current_segment(phi, phi_switch, spiral_theta)
    r0[0, 0] = ca.cos(theta0)
    r0[0, 1] = -ca.sin(theta0)
    r0[1, 0] = ca.sin(theta0)
    r0[1, 1] = ca.cos(theta0)
    ddspiral0 = r0 @ (
        2
        * -a
        * phi_spiral0
        * ca.vertcat(ca.sin(-a * phi_spiral0**2), -ca.cos(-a * phi_spiral0**2))
    )
    ddspiral0 = rot_2d0 @ ddspiral0
    ddspiral1 = (
        2
        * b
        * phi_spiral1
        * ca.vertcat(-ca.sin(b * phi_spiral1**2), ca.cos(b * phi_spiral1**2))
    )
    ddspiral1 = rot_2d1 @ ddspiral1
    cond1 = ca.if_else(phi > phi_end - l1, ddspiral1, ddp_lin)
    ddp_d = ca.if_else(phi < phi_start + l0, ddspiral0, cond1)
    # if isinstance(dp_ref, np.ndarray):
    #     if phi < phi_start + l_seg - offset0:
    #         print("Start: ", phi, ddspiral0)
    #     elif phi > phi_end + l_seg - offset1:
    #         print("Linear2: ", phi, ddp_lin)
    #     elif phi > phi_end - offset1:
    #         print("Start2: ", phi, ddspiral2)
    #     elif phi > phi_end - l_seg - offset1:
    #         print("End: ", phi, ddspiral1)
    #     else:
    #         print("Linear: ", phi, ddp_lin)
    #
    a_current, _ = get_current_segments_1d(idx, split_idx, a_set)
    b_current, _ = get_current_segments_split(idx, split_idx, b_set)
    cond1 = ca.if_else(split_idx[2] == n_horizon, a_set[2], a_set[3])
    a_next = ca.if_else(split_idx[1] == n_horizon, a_set[1], cond1)
    cond1 = ca.if_else(split_idx[2] == n_horizon, b_set[2, :], b_set[3, :])
    b_next = ca.if_else(split_idx[1] == n_horizon, b_set[1, :], cond1)
    _, phi_end_seg = get_current_segments_split(idx, split_idx, phi_switch)

    outputs = [
        p_d,
        dp_d,
        ddp_d,
        bp10,
        bp20,
        br10,
        br20,
        dp_normed_d,
        v10,
        v20,
        v30,
        bound_lower,
        bound_upper,
        e_p_off,
        e_r_off,
        a_current,
        b_current,
        a_next,
        b_next,
        phi_end_seg,
    ]
    output_names = [
        "p_d",
        "dp_d",
        "ddp_d",
        "bp1_current",
        "bp2_current",
        "br1_current",
        "br2_current",
        "dp_normed_d",
        "v1_current",
        "v2_current",
        "v3_current",
        "bound_lower",
        "bound_upper",
        "e_p_off",
        "e_r_off",
        "a_current",
        "b_current",
        "a_next",
        "b_next",
        "phi_end_seg",
    ]
    reference_data = {}
    for key, value in zip(output_names, outputs):
        reference_data[key] = value
    return reference_data


def compute_interpolated_rotation_bounds(reference, e_p_orth1, e_p_orth2, bound):
    err_p1 = e_p_orth1 - reference["e_p_off"][0]
    err_p2 = e_p_orth1 - reference["e_p_off"][1]
    pos1 = err_p1 / bound[0]
    pos2 = err_p2 / bound[1]
    i1 = 0.5 * (pos1 + 1)
    i2 = 0.5 * (pos2 + 1)
    # if isinstance(e_p, np.ndarray):
    #     print(bound_span)
    #     print(err_p1, err_p2)
    #     print(i1, i2)
    #     print("---")

    poly_upper1 = reference["bound_upper"][2:5]
    poly_upper2 = reference["bound_upper"][5:8]
    poly_upper3 = reference["bound_upper"][8:11]
    poly_upper4 = reference["bound_upper"][11:14]
    d_max_cross = -poly_upper4 + poly_upper1
    d_max_b1 = -poly_upper4 + poly_upper2
    d_max_b2 = -poly_upper4 + poly_upper3
    r_upper_bound = poly_upper4 + d_max_b1 * i1 * (1 - i2)
    r_upper_bound += d_max_b2 * i2 * (1 - i1)
    r_upper_bound += d_max_cross * i2 * i1

    poly_lower1 = reference["bound_lower"][2:5]
    poly_lower2 = reference["bound_lower"][5:8]
    poly_lower3 = reference["bound_lower"][8:11]
    poly_lower4 = reference["bound_lower"][11:14]
    d_min_cross = -poly_lower4 + poly_lower1
    d_min_b1 = -poly_lower4 + poly_lower2
    d_min_b2 = -poly_lower4 + poly_lower3
    r_lower_bound = poly_lower4 + d_min_b1 * i1 * (1 - i2)
    r_lower_bound += d_min_b2 * i2 * (1 - i1)
    r_lower_bound += d_min_cross * i2 * i1

    return r_upper_bound, r_lower_bound


def error_function(
    p,
    v,
    p_ref,
    dp_ref,
    ddp_ref,
    dp_normed_ref,
    dphi,
    i_omega_0,
    i_omega_ref_0,
    dtau_init,
    dtau_init_par,
    dtau_init_orth1,
    dtau_init_orth2,
    br1,
    br2,
    jac_dtau_l,
    jac_dtau_r,
    phi,
    phi_switch,
    v1,
    v2,
    v3,
):
    # Compute position errors
    e_p_par, e_p_orth, de_p_par, de_p_orth, e_p, de_p = compute_position_error(
        p[:3], v[:3], p_ref[:3], dp_ref[:3], ddp_ref[:3].T, dphi
    )

    # Compute orientation error and its derivative
    e_r = integrate_rot_error_diff(
        dtau_init, p[3:], i_omega_0, p_ref[3:], i_omega_ref_0, jac_dtau_l, jac_dtau_r
    )
    de_r = compute_rot_error_velocity(dp_ref[3:], v[3:], jac_dtau_l, jac_dtau_r, dphi)

    # Compute correct starting value based on the segment for the orientation
    # error
    e_par_init = get_current_segment(phi, phi_switch, dtau_init_par).T
    e_orth1_init = get_current_segment(phi, phi_switch, dtau_init_orth1).T
    e_orth2_init = get_current_segment(phi, phi_switch, dtau_init_orth2).T
    if isinstance(dtau_init_par, np.ndarray):
        e_par_init = e_par_init.T
        e_orth1_init = e_orth1_init.T
        e_orth2_init = e_orth2_init.T

    # Project error onto path
    scal_orth1 = ca.dot(e_r - dtau_init, v1)
    scal_par = ca.dot(e_r - dtau_init, v2)
    scal_orth2 = ca.dot(e_r - dtau_init, v3)

    # Final decomposed orientation errors
    e_r_orth1 = e_orth1_init + scal_orth1 * br1
    e_r_par = e_par_init + scal_par * dp_normed_ref
    e_r_orth2 = e_orth2_init + scal_orth2 * br2

    outputs = [
        e_p_par,
        e_p_orth,
        de_p_par,
        de_p_orth,
        e_p,
        de_p,
        e_r_par,
        e_r,
        de_r,
        e_r_orth1,
        e_r_orth2,
    ]
    output_names = [
        "e_p_par",
        "e_p_orth",
        "de_p_par",
        "de_p_orth",
        "e_p",
        "de_p",
        "e_r_par",
        "e_r",
        "de_r",
        "e_r_orth1",
        "e_r_orth2",
    ]
    error_data = {}
    for key, value in zip(output_names, outputs):
        error_data[key] = value
    return error_data


def objective_function(
    nr_joints,
    nr_u,
    x_phi,
    e_p_par,
    de_p,
    e_r_par,
    de_r,
    x_phi_d,
    v,
    v_ref,
    a,
    a_ref,
    u,
    q,
    dq,
    ddq,
    qd,
    weights,
    obj_centers,
    obj_radii,
):
    """Create the objective function for the MPC."""
    # Extract weights
    w_p = weights[0]
    w_r = weights[1]
    w_v_p = weights[2]
    w_v_r = weights[3]
    w_a_p = weights[5]
    w_a_r = weights[6]
    w_phi = weights[7]
    w_dphi = weights[8]
    w_ddphi = weights[9]
    w_dddphi = weights[10]
    w_q = weights[11]
    w_dq = weights[12]
    w_ddq = weights[13]
    w_jerk = weights[14]
    w_col = weights[16]
    w_col_alpha = weights[17]
    w_col_beta = weights[18]

    # Create objective term
    objective_term = 0
    objective_term += w_r * ca.sumsqr(e_r_par)
    objective_term += w_p * ca.sumsqr(e_p_par)

    # Cartesian velocity and acceleration
    objective_term += w_v_p * ca.sumsqr(v[:3] - v_ref[:3])
    objective_term += w_v_r * ca.sumsqr(v[3:] - v_ref[3:])
    # objective_term += w_a_p * ca.sumsqr(a[:3] - a_ref[:3])
    # objective_term += w_a_r * ca.sumsqr(a[3:] - a_ref[3:])

    # Joint state
    # objective_term += w_q * ca.sumsqr(q - qd)
    # objective_term += w_dq * ca.sumsqr(dq)
    # objective_term += w_ddq * ca.sumsqr(ddq)
    # objective_term += w_jerk * ca.sumsqr(u[:-1])
    objective_term += w_dq * ca.sumsqr(dq[2])
    # objective_term += w_ddq * ca.sumsqr(ddq[2])
    objective_term += w_dq * ca.sumsqr(dq[4])
    # objective_term += w_ddq * ca.sumsqr(ddq[4])
    objective_term += w_jerk * ca.sumsqr(u[:-1])

    # Path state
    # objective_term += w_phi * ca.sumsqr(x_phi_d[0] - x_phi[0])
    objective_term += w_phi * approx_one_norm(x_phi_d[0] - x_phi[0])
    objective_term += w_dphi * ca.sumsqr(x_phi_d[1] - x_phi[1])
    objective_term += w_ddphi * ca.sumsqr(x_phi_d[2] - x_phi[2])
    objective_term += w_dddphi * ca.sumsqr(u[-1])

    # Collision terms of the rest of the robot
    # objective_term += (
    #     w_col
    #     * 1
    #     / w_col_alpha
    #     * collision_function(q, obj_centers, obj_radii, w_col_alpha, w_col_beta)
    # )

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
