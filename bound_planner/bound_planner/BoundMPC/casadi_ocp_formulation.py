import casadi as ca
import numpy as np

from ..RobotModel import RobotModel
from .bound_mpc_functions import (
    error_function,
    get_current_and_next_segment,
    get_current_and_next_segment_1d,
    get_current_segments_split,
    objective_function,
    reference_function,
)


def setup_optimization_problem(
    N,
    nr_joints,
    nr_segs,
    dt,
    max_col_objs,
    solver_opts,
):
    """Build the optimization problem using symbolic variables such that it
    can be easily used later on by just inputting the new state.
    """
    robot_model = RobotModel()
    path_following = False
    nr_u = nr_joints + 1

    split_idx = ca.SX.sym("split_idx", nr_segs + 1)

    max_set_size = 15
    a_set = ca.SX.sym("a_set", max_set_size, 3, nr_segs)
    b_set = ca.SX.sym("b_set", nr_segs, max_set_size)
    a_set_joints = ca.SX.sym("a_set_joints", max_set_size, 3, 6)
    b_set_joints = ca.SX.sym("b_set_joints", 6, max_set_size)

    # Desired path parameter state
    x_phi_d = ca.SX.sym("x phi_desired", 3)
    phi_max = ca.SX.sym("max path parameter", 1)
    dphi_max = ca.SX.sym("max path parameter", 1)

    # Desired joint config
    qd = ca.SX.sym("q desired", nr_joints)

    # Reference trajectory parameters
    phi_switch = ca.SX.sym("path parameter switch", nr_segs + 1)
    p_ref = ca.SX.sym("linear ref position", nr_segs, 6)
    p_ref_prev0 = ca.SX.sym("ref velocity prev", 3)
    dp_ref = ca.SX.sym("linear ref velocity", nr_segs, 6)
    dp_ref_prev0 = ca.SX.sym("ref velocity prev", 3)
    dp_normed_ref = ca.SX.sym("norm of orientation reference", nr_segs, 3)
    bp1 = ca.SX.sym("orthogonal error basis 1", nr_segs, 3)
    bp2 = ca.SX.sym("orthogonal error basis 2", nr_segs, 3)
    br1 = ca.SX.sym("orthogonal error basis 1r", nr_segs, 3)
    br2 = ca.SX.sym("orthogonal error basis 2r", nr_segs, 3)
    v1 = ca.SX.sym("v1", nr_segs, 3)
    v2 = ca.SX.sym("v2", nr_segs, 3)
    v3 = ca.SX.sym("v3", nr_segs, 3)

    # Error computation variables
    dtau_init = ca.SX.sym("initial lie space error", 3)
    dtau_init_par = ca.SX.sym("initial lie space error par", 3, nr_segs)
    dtau_init_orth1 = ca.SX.sym("initial lie space error orth1", 3, nr_segs)
    dtau_init_orth2 = ca.SX.sym("initial lie space error orth2", 3, nr_segs)
    jac_dtau_r = ca.SX.sym("right jacobian at initial error", 3, 3)
    jac_dtau_l = ca.SX.sym("left jacobian at initial error", 3, 3)

    # Error bound parametrization
    a6 = ca.SX.sym("parameter 6 error function", nr_segs + 1, 10)
    a5 = ca.SX.sym("parameter 5 error function", nr_segs + 1, 10)
    a4 = ca.SX.sym("parameter 4 error function", nr_segs + 1, 10)
    a3 = ca.SX.sym("parameter 3 error function", nr_segs + 1, 10)
    a2 = ca.SX.sym("parameter 2 error function", nr_segs + 1, 10)
    a1 = ca.SX.sym("parameter 1 error function", nr_segs + 1, 10)
    a0 = ca.SX.sym("parameter 0 error function", nr_segs + 1, 10)

    # Euler spiral parameters
    spiral_a = ca.SX.sym("spiral a", nr_segs)
    spiral_l = ca.SX.sym("spiral l", nr_segs)
    spiral_theta = ca.SX.sym("spiral theta", nr_segs)
    spiral_rot2d = ca.SX.sym("spiral rot 2d", 3, 2, nr_segs)
    spiral_rot_bp1 = ca.SX.sym("spiral rot basis1", nr_segs, 3)
    spiral_rot_bp1_norm = ca.SX.sym("spiral rot basis1", nr_segs)
    spiral_rot_bp2_norm = ca.SX.sym("spiral rot basis2", nr_segs)

    # Objective function weights
    weights = ca.SX.sym("cost weights", 19)

    # Collision spheres
    obj_centers = ca.SX.sym("collision object centers", max_col_objs, 3)
    obj_radii = ca.SX.sym("collision object radii", max_col_objs)

    # Initialize variables
    J = 0
    g = []
    lbg = []
    ubg = []

    # states
    q = ca.SX.sym("q", N, nr_joints)
    dq = ca.SX.sym("dq", N, nr_joints)
    ddq = ca.SX.sym("ddq", N, nr_joints)
    p = ca.SX.sym("p", N, 6)
    v = ca.SX.sym("v", N, 6)
    phi = ca.SX.sym("phi", N)
    dphi = ca.SX.sym("dphi", N)
    ddphi = ca.SX.sym("ddphi", N)
    p_refx = ca.SX.sym("p_ref", N, 3)
    dp_refx = ca.SX.sym("dp_ref", N, 3)
    u = ca.SX.sym("u", N, nr_joints + 1)
    slacks = ca.SX.sym("s sets", 8)

    w = ca.vertcat(
        q[:],
        dq[:],
        ddq[:],
        u[:],
        phi[:],
        dphi[:],
        ddphi[:],
        p[:],
        v[:],
        p_refx[:],
        dp_refx[:],
        slacks,
    )

    i_omega_0 = p[0, 3:].T
    i_omega_ref_0 = ca.SX.sym("i_omega_ref_0", 3)

    for k in range(N - 1):
        q_new = (
            ddq[k, :] * dt**2 / 2.0
            + dq[k, :] * dt
            + q[k, :]
            + u[k, :7] * dt**3 / 8.0
            + u[k + 1, :7] * dt**3 / 24.0
        )
        dq_new = (
            ddq[k, :] * dt
            + dq[k, :]
            + u[k, :7] * dt**2 / 3.0
            + u[k + 1, :7] * dt**2 / 6.0
        )
        ddq_new = ddq[k, :] + u[k, :7] * dt / 2.0 + u[k + 1, :7] * dt / 2.0
        phi_new = (
            ddphi[k] * dt**2 / 2.0
            + dphi[k] * dt
            + phi[k]
            + u[k, 7] * dt**3 / 8.0
            + u[k + 1, 7] * dt**3 / 24.0
        )
        dphi_new = (
            ddphi[k] * dt + dphi[k] + u[k, 7] * dt**2 / 3.0 + u[k + 1, 7] * dt**2 / 6.0
        )
        ddphi_new = ddphi[k] + u[k, 7] * dt / 2.0 + u[k + 1, 7] * dt / 2.0

        v_cart = robot_model.velocity_ee(q_new, dq_new)
        v_rot = robot_model.omega_ee(q_new, dq_new)
        v_new = ca.vertcat(v_cart, v_rot).T

        robot_model = RobotModel()
        pn_pos = robot_model.fk_pos(q_new).T
        k1 = v[k, 3:]
        k2 = v_new[3:]
        pn_rot = p[k, 3:] + 1 / 2 * dt * (k1 + k2)
        p_new = ca.horzcat(pn_pos, pn_rot)

        # Euler spiral for p_ref
        dphi_euler = phi_new - phi[k]
        nr_steps = 3
        dphik = dphi_euler / nr_steps
        p_ref_prev_c = p_refx[k, :].T
        dp_ref_prev_c = dp_refx[k, :].T
        for i in range(nr_steps):
            phik = phi[k] + (i + 1) * dphik
            phi_startk, phi_endk = get_current_and_next_segment(
                phik, phi_switch, phi_switch
            )
            l0, l1 = get_current_and_next_segment(phik, phi_switch, spiral_l)
            dp_d_c, dp_d_n = get_current_and_next_segment(phik, phi_switch, dp_ref)
            dp_d_c = dp_d_c.T
            dp_d_n = dp_d_n.T
            a, b = get_current_and_next_segment(phik, phi_switch, spiral_a)
            theta0, theta1 = get_current_and_next_segment(
                phik, phi_switch, spiral_theta
            )
            r0 = ca.SX.zeros((2, 2))
            r1 = ca.SX.zeros((2, 2))
            r0[0, 0] = ca.cos(theta0)
            r0[0, 1] = -ca.sin(theta0)
            r0[1, 0] = ca.sin(theta0)
            r0[1, 1] = ca.cos(theta0)
            r1[0, 0] = ca.cos(theta1)
            r1[0, 1] = -ca.sin(theta1)
            r1[1, 0] = ca.sin(theta1)
            r1[1, 1] = ca.cos(theta1)
            rot_2d0, rot_2d1 = get_current_and_next_segment_1d(
                phik, phi_switch, spiral_rot2d
            )
            phi_spiral0 = l0 - (phik - phi_startk)
            dspiral0 = r0 @ ca.vertcat(
                ca.cos(-a * phi_spiral0**2), ca.sin(-a * phi_spiral0**2)
            )
            dspiral0 = rot_2d0 @ dspiral0

            phi_spiral1 = phik - (phi_endk - l1)
            dspiral1 = ca.vertcat(
                ca.cos(b * phi_spiral1**2), ca.sin(b * phi_spiral1**2)
            )
            dspiral1 = rot_2d1 @ dspiral1

            cond1 = ca.if_else(phik > phi_endk - l1, dspiral1, dp_d_c[:3])
            dp_d_c[:3] = ca.if_else(phik < phi_startk + l0, dspiral0, cond1)
            p_ref_new = p_ref_prev_c + 0.5 * dphik * (dp_d_c[:3] + dp_ref_prev_c)
            p_ref_prev_c = p_ref_new
            dp_ref_prev_c = dp_d_c[:3]
        dp_ref_new = dp_d_c[:3]

        # Dynamical system constraint
        g += [q_new - q[k + 1, :]]
        lbg += [0] * nr_joints
        ubg += [0] * nr_joints
        g += [dq_new - dq[k + 1, :]]
        lbg += [0] * nr_joints
        ubg += [0] * nr_joints
        g += [ddq_new - ddq[k + 1, :]]
        lbg += [0] * nr_joints
        ubg += [0] * nr_joints
        g += [phi_new - phi[k + 1]]
        lbg += [0]
        ubg += [0]
        g += [dphi_new - dphi[k + 1]]
        lbg += [0]
        ubg += [0]
        g += [ddphi_new - ddphi[k + 1]]
        lbg += [0]
        ubg += [0]
        g += [p_new - p[k + 1, :]]
        lbg += [0] * 6
        ubg += [0] * 6
        g += [v_new - v[k + 1, :]]
        lbg += [0] * 6
        ubg += [0] * 6
        g += [p_ref_new.T - p_refx[k + 1, :]]
        lbg += [0] * 3
        ubg += [0] * 3
        g += [dp_ref_new.T - dp_refx[k + 1, :]]
        lbg += [0] * 3
        ubg += [0] * 3

    # Formulate the NLP
    for k in range(1, N):
        pk = p[k, :].T
        vk = v[k, :].T
        vprev = v[k - 1, :].T

        # Compute reference trajectory
        reference = reference_function(
            dp_ref=dp_ref,
            p_ref=p_ref,
            dp_ref_p=dp_refx[k, :],
            p_ref_p=p_refx[k, :],
            phi_switch=phi_switch,
            phi=phi[k],
            bp1=bp1,
            bp2=bp2,
            br1=br1,
            br2=br2,
            v1=v1,
            v2=v2,
            v3=v3,
            dp_normed_ref=dp_normed_ref,
            a6=a6,
            a5=a5,
            a4=a4,
            a3=a3,
            a2=a2,
            a1=a1,
            a0=a0,
            spiral_a=spiral_a,
            spiral_l=spiral_l,
            spiral_theta=spiral_theta,
            spiral_rot2d=spiral_rot2d,
            spiral_rot_bp1=spiral_rot_bp1,
            spiral_rot_bp1_norm=spiral_rot_bp1_norm,
            spiral_rot_bp2_norm=spiral_rot_bp2_norm,
            split_idx=split_idx,
            idx=k,
            n_horizon=N,
            a_set=a_set,
            b_set=b_set,
        )
        p_d = reference["p_d"]
        dp_d = reference["dp_d"].T
        ddp_d = reference["ddp_d"].T
        bp1_current = reference["bp1_current"]
        bp2_current = reference["bp2_current"]
        br1_current = reference["br1_current"]
        br2_current = reference["br2_current"]
        v1_current = reference["v1_current"]
        v2_current = reference["v2_current"]
        v3_current = reference["v3_current"]
        dp_normed_d = reference["dp_normed_d"].T

        # Compute errors
        errors = error_function(
            p=pk,
            v=vk,
            p_ref=p_d,
            dp_ref=dp_d,
            ddp_ref=ddp_d,
            dp_normed_ref=dp_normed_d,
            dphi=dphi[k],
            i_omega_0=i_omega_0,
            i_omega_ref_0=i_omega_ref_0,
            dtau_init=dtau_init,
            dtau_init_par=dtau_init_par.T,
            dtau_init_orth1=dtau_init_orth1.T,
            dtau_init_orth2=dtau_init_orth2.T,
            br1=br1_current,
            br2=br2_current,
            jac_dtau_l=jac_dtau_l,
            jac_dtau_r=jac_dtau_r,
            phi=phi[k],
            phi_switch=phi_switch,
            v1=v1_current,
            v2=v2_current,
            v3=v3_current,
        )
        e_p_park = errors["e_p_par"]
        e_p = errors["e_p"]
        de_p = errors["de_p"]
        # de_p_park = errors["de_p_par"]
        e_r = errors["e_r"]
        de_r = errors["de_r"]
        e_r_park = errors["e_r_par"]
        e_r_orth1k = errors["e_r_orth1"]
        e_r_orth2k = errors["e_r_orth2"]

        # # Numerically differentiated velocity as acceleration
        ak = (vk - vprev) / dt
        v_ref = dphi[k] * dp_d
        a_ref = ddphi[k] * dp_d
        a_ref[:3] = a_ref[:3] + dphi[k] ** 2 * ddp_d.T

        # Increment objective value
        # A sigmoid function is used to be able to use the full error at the
        # end of the path which avoid oscillation and increases solver
        # speed.
        if path_following:
            e_p_obj = e_p
            e_r_obj = e_r
        else:
            sigm = 1 / (1 + ca.exp(-60 * (phi[k] - (phi_max - 0.05))))
            # e_p_obj = sigm * e_p + (1 - sigm) * e_p_park
            e_p_obj = e_p_park
            e_r_obj = 1 * sigm * e_r
            e_p_obj2 = 1 * sigm * e_p

            # e_r_proj1 = ca.dot(br1_current, e_r_orth1k)
            # e_r_proj2 = ca.dot(br2_current, e_r_orth2k)
            # e_r_proj_par = ca.dot(dp_normed_d, e_r_park)
            # e_r_obj = 10 * sigm * ca.vertcat(e_r_proj1, e_r_proj2)

            J = J + ca.sumsqr(e_r_obj)
            J = J + ca.sumsqr(e_p_obj2)
            # e_p_obj = e_p_park
            # e_r_obj = e_r_park
        x_phi = ca.vertcat(phi[k], dphi[k], ddphi[k])
        J = J + objective_function(
            nr_joints,
            nr_u,
            x_phi=x_phi,
            e_p_par=e_p_obj,
            de_p=de_p,
            e_r_par=e_r_obj,
            de_r=de_r,
            x_phi_d=x_phi_d,
            v=vk,
            v_ref=v_ref,
            a=ak,
            a_ref=a_ref,
            u=u[k, :].T,
            q=q[k, :].T,
            dq=dq[k, :].T,
            ddq=ddq[k, :].T,
            qd=qd,
            obj_centers=obj_centers,
            obj_radii=obj_radii,
            weights=weights,
        )

        # -----------------------------------------------------------------
        # CONSTRAINTS
        # -----------------------------------------------------------------
        # Orthogonal position error bounds
        # e_p_orth1k = ca.dot(bp1_current, e_p)
        # e_p_orth2k = ca.dot(bp2_current, e_p)
        # e_diff1 = e_p_orth1k - reference["e_p_off"][0]
        # e_diff2 = e_p_orth2k - reference["e_p_off"][1]
        # bound = (reference["bound_upper"][:2] - reference["bound_lower"][:2]) / 2
        # g += [ca.sumsqr(e_diff1) - ca.sumsqr(bound[0])]
        # g += [ca.sumsqr(e_diff2) - ca.sumsqr(bound[1])]
        # # g += [ca.sumsqr(e_p_orth1k) - ca.sumsqr(0.01)]
        # # g += [ca.sumsqr(e_p_orth2k) - ca.sumsqr(0.01)]
        # lbg += [-np.inf] * 2
        # ubg += [0] * 2

        # g += [(reference["a_current"] @ pk[:3]).T - reference["b_current"] - slacks[-3]]
        g += [(reference["a_current"] @ pk[:3]).T - reference["b_current"]]
        lbg += [-np.inf] * max_set_size
        ubg += [0] * max_set_size

        e_r_proj1 = ca.dot(br1_current, e_r_orth1k)
        e_r_proj2 = ca.dot(br2_current, e_r_orth2k)
        e_r_proj_par = ca.dot(dp_normed_d, e_r_park)

        # bound = (reference["bound_upper"][2:] - reference["bound_lower"][2:]) / 2
        # e_r_off = reference["e_r_off"]
        # g += [(e_r_proj1 - e_r_off[0]) ** 2 - bound[0] ** 2]
        # g += [(e_r_proj2 - e_r_off[2]) ** 2 - bound[2] ** 2]
        # g += [(e_r_proj_par - e_r_off[1]) ** 2 - bound[1] ** 2]
        # g += [e_r_proj1 - e_r_off[0] - bound[0]]
        # g += [-(e_r_proj1 - e_r_off[0]) - bound[0]]
        # g += [e_r_proj2 - e_r_off[2] - bound[2]]
        # g += [-(e_r_proj2 - e_r_off[2]) - bound[2]]
        # g += [e_r_proj_par - e_r_off[1] - bound[1]]
        # g += [-(e_r_proj_par - e_r_off[1]) - bound[1]]
        # lbg += [-np.inf] * 3
        # ubg += [0] * 3
        g += [e_r_proj1]
        g += [e_r_proj2]
        g += [e_r_proj_par]
        lbg += [-np.pi / 2] * 3
        ubg += [np.pi / 2] * 3

        p_list = [
            robot_model.fk_pos_j3(q[k, :]),
            robot_model.fk_pos_j5(q[k, :]),
            robot_model.fk_pos_j6(q[k, :]),
            robot_model.fk_pos_j67(q[k, :]),
            robot_model.fk_pos_j67(q[k, :], 2.5),
            robot_model.fk_pos_elbow(q[k, :]),
        ]
        for i, pl in enumerate(p_list):
            aj = a_set_joints[i]
            bj = b_set_joints[i, :]
            g += [(aj @ pl).T - bj - slacks[i]]
            # g += [(aj @ pl).T - bj]
            lbg += [-np.inf] * max_set_size
            ubg += [0] * max_set_size

        # Terminal cost
        J = J + 0.01 * ca.sumsqr(e_p)
        if k == N - 1:
            J = J + weights[16] * ca.sumsqr(slacks[:-2])
            J = J + weights[16] * ca.sumsqr(slacks[-1])
            J = J + weights[15] * ca.sumsqr(slacks[-2])

            # J = J + weights[15] / 10 * ca.sumsqr(dphi[k])
            # J = J + weights[15] / 10 * ca.sumsqr(ddphi[k])
            J = J + 0.01 * ca.sumsqr(e_r)
            J = J + 0.1 * ca.sumsqr(e_p)
            g += [
                (reference["a_next"] @ pk[:3]).T
                - (reference["b_next"] - 0.005)
                - slacks[-1]
            ]
            lbg += [-np.inf] * max_set_size
            ubg += [0] * max_set_size

            g += [phi[k] - (reference["phi_end_seg"] + 0.01 + slacks[-2])]
            lbg += [-np.inf]
            ubg += [0]

            an = reference["a_next"]
            bn = reference["b_next"]
            _, p_ref_next = get_current_segments_split(k, split_idx, p_ref)
            p_end = p_ref_next[:3]
            bnew = bn - (an @ p_end.T).T
            anew = an @ ca.horzcat(bp1_current, bp2_current)
            e_p_orth1k = ca.dot(bp1_current, e_p)
            e_p_orth2k = ca.dot(bp2_current, e_p)
            z = ca.vertcat(e_p_orth1k, e_p_orth2k)

            g += [(anew @ z).T - bnew - slacks[-1]]
            lbg += [-np.inf] * max_set_size
            ubg += [0] * max_set_size

    # Create an NLP solver
    params = ca.vertcat(
        split_idx,
        p_ref_prev0,
        dp_ref_prev0,
        i_omega_ref_0,
        dtau_init,
        dtau_init_par.reshape((-1, 1)),
        dtau_init_orth1.reshape((-1, 1)),
        dtau_init_orth2.reshape((-1, 1)),
        x_phi_d,
        phi_switch,
        jac_dtau_r.reshape((-1, 1)),
        jac_dtau_l.reshape((-1, 1)),
        p_ref.reshape((-1, 1)),
        dp_ref.reshape((-1, 1)),
        dp_normed_ref.reshape((-1, 1)),
        bp1.reshape((-1, 1)),
        bp2.reshape((-1, 1)),
        br1.reshape((-1, 1)),
        br2.reshape((-1, 1)),
        a6.reshape((-1, 1)),
        a5.reshape((-1, 1)),
        a4.reshape((-1, 1)),
        a3.reshape((-1, 1)),
        a2.reshape((-1, 1)),
        a1.reshape((-1, 1)),
        a0.reshape((-1, 1)),
        weights,
        phi_max,
        dphi_max,
        v1.reshape((-1, 1)),
        v2.reshape((-1, 1)),
        v3.reshape((-1, 1)),
        qd,
        spiral_rot_bp1.reshape((-1, 1)),
        spiral_rot_bp1_norm,
        spiral_rot_bp2_norm,
        spiral_a,
        spiral_l,
        spiral_theta,
        obj_centers.reshape((-1, 1)),
        obj_radii,
    )
    for i in range(len(spiral_rot2d)):
        params = ca.vertcat(params, spiral_rot2d[i].reshape((-1, 1)))
    for i in range(len(a_set)):
        params = ca.vertcat(params, a_set[i].reshape((-1, 1)))
    params = ca.vertcat(params, b_set.reshape((-1, 1)))
    for i in range(len(a_set_joints)):
        params = ca.vertcat(params, a_set_joints[i].reshape((-1, 1)))
    params = ca.vertcat(params, b_set_joints.reshape((-1, 1)))

    prob = {"f": J, "x": w, "g": ca.horzcat(*g), "p": params}
    solver = ca.nlpsol("solver", "ipopt", prob, solver_opts)
    # solver = ca.nlpsol('solver', 'fatrop', prob, solver_opts)

    return solver, lbg, ubg
