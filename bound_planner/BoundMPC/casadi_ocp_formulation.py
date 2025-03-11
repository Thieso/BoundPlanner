import casadi as ca
import numpy as np

from ..RobotModel import RobotModel
from .bound_mpc_functions import (
    error_function,
    get_current_segments_split,
    objective_function,
    reference_function,
)


def setup_optimization_problem(
    N,
    nr_joints,
    nr_segs,
    dt,
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

    # Desired joint config
    qd = ca.SX.sym("q desired", nr_joints)

    # Reference trajectory parameters
    phi_switch = ca.SX.sym("path parameter switch", nr_segs + 1)
    p_ref = ca.SX.sym("linear ref position", nr_segs, 6)
    dp_ref = ca.SX.sym("linear ref velocity", nr_segs, 6)
    dp_normed_ref = ca.SX.sym("norm of orientation reference", nr_segs, 3)
    bp1 = ca.SX.sym("orthogonal error basis 1", nr_segs, 3)
    bp2 = ca.SX.sym("orthogonal error basis 2", nr_segs, 3)
    br1 = ca.SX.sym("orthogonal error basis 1r", nr_segs, 3)
    br2 = ca.SX.sym("orthogonal error basis 2r", nr_segs, 3)
    v1 = ca.SX.sym("v1", nr_segs, 3)
    v2 = ca.SX.sym("v2", nr_segs, 3)
    v3 = ca.SX.sym("v3", nr_segs, 3)

    # Error computation variables
    dtau_init = ca.SX.sym("initial lie space error", 3, nr_segs)
    dtau_init_par = ca.SX.sym("initial lie space error par", 3, nr_segs)
    dtau_init_orth1 = ca.SX.sym("initial lie space error orth1", 3, nr_segs)
    dtau_init_orth2 = ca.SX.sym("initial lie space error orth2", 3, nr_segs)
    jac_dtau_r = ca.SX.sym("right jacobian at initial error", 3, 3)
    jac_dtau_l = ca.SX.sym("left jacobian at initial error", 3, 3)

    # Error bound parametrization
    e_r_bound = ca.SX.sym("error bounds orientation", nr_segs, 6)

    # Objective function weights
    weights = ca.SX.sym("cost weights", 11)

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
    u = ca.SX.sym("u", N, nr_joints)
    dslacks = ca.SX.sym("s sets", 6)
    slacks0 = ca.SX.sym("s sets", 6)
    rslacks = ca.SX.sym("s rot", N)
    drslacks = ca.SX.sym("s rot", N)
    pslacks = ca.SX.sym("s rot", N)
    dpslacks = ca.SX.sym("s rot", N)

    w = ca.vertcat(
        q[:],
        dq[:],
        ddq[:],
        u[:],
        p[:],
        v[:],
        dslacks,
        rslacks,
        drslacks,
        pslacks,
        dpslacks,
    )

    i_omega_0 = p[0, 3:].T
    i_omega_ref_0 = ca.SX.sym("i_omega_ref_0", 3)

    for k in range(N - 1):
        q_new = (
            ddq[k, :] * dt**2 / 2.0
            + dq[k, :] * dt
            + q[k, :]
            + u[k, :] * dt**3 / 8.0
            + u[k + 1, :] * dt**3 / 24.0
        )
        dq_new = (
            ddq[k, :] * dt
            + dq[k, :]
            + u[k, :] * dt**2 / 3.0
            + u[k + 1, :] * dt**2 / 6.0
        )
        ddq_new = ddq[k, :] + u[k, :] * dt / 2.0 + u[k + 1, :] * dt / 2.0
        v_cart = robot_model.velocity_ee(q[k + 1, :].T, dq[k + 1, :].T)
        v_rot = robot_model.omega_ee(q[k + 1, :].T, dq[k + 1, :].T)
        # v_cart = robot_model.velocity_ee(q_new.T, dq_new.T)
        # v_rot = robot_model.omega_ee(q_new.T, dq_new.T)
        v_new = ca.vertcat(v_cart, v_rot).T

        robot_model = RobotModel()
        pn_pos = robot_model.fk_pos(q[k + 1, :].T).T
        k1 = v[k, 3:]
        k2 = v[k + 1, 3:]
        pn_rot = p[k, 3:] + 1 / 2 * dt * (k1 + k2)
        # pn_rot = p[k, 3:] + dt * k1
        p_new = ca.horzcat(pn_pos, pn_rot)

        k1 = drslacks[k]
        k2 = drslacks[k + 1]
        rslacks_new = rslacks[k] + 1 / 2 * dt * (k1 + k2)

        k1 = dpslacks[k]
        k2 = dpslacks[k + 1]
        pslacks_new = pslacks[k] + 1 / 2 * dt * (k1 + k2)

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
        g += [p_new - p[k + 1, :]]
        lbg += [0] * 6
        ubg += [0] * 6
        g += [v_new - v[k + 1, :]]
        lbg += [0] * 6
        ubg += [0] * 6
        g += [rslacks_new - rslacks[k + 1, :]]
        lbg += [0]
        ubg += [0]
        g += [pslacks_new - pslacks[k + 1, :]]
        lbg += [0]
        ubg += [0]

    # Formulate the NLP
    slacks = slacks0 + dslacks
    for k in range(1, N):
        pk = p[k, :].T
        vk = v[k, :].T

        # Compute reference trajectory
        reference = reference_function(
            dp_ref=dp_ref,
            p_ref=p_ref,
            p=pk,
            v=vk,
            phi_switch=phi_switch,
            bp1=bp1,
            bp2=bp2,
            br1=br1,
            br2=br2,
            v1=v1,
            v2=v2,
            v3=v3,
            dp_normed_ref=dp_normed_ref,
            e_r_bound=e_r_bound,
            split_idx=split_idx,
            idx=k,
            n_horizon=N,
            a_set=a_set,
            b_set=b_set,
        )
        p_d = reference["p_d"]
        p_dr_next = reference["p_dr_next"]
        dp_d = reference["dp_d"].T
        ddp_d = reference["ddp_d"].T
        bp1_current = reference["bp1_current"]
        bp2_current = reference["bp2_current"]
        br1_current = reference["br1_current"]
        br2_current = reference["br2_current"]
        br1_next = reference["br1_next"]
        br2_next = reference["br2_next"]
        v1_current = reference["v1_current"]
        v2_current = reference["v2_current"]
        v3_current = reference["v3_current"]
        v1_next = reference["v1_next"]
        v2_next = reference["v2_next"]
        v3_next = reference["v3_next"]
        dp_normed_d = reference["dp_normed_d"].T
        dp_normed_n = reference["dp_normed_n"].T
        phi = reference["phi"]
        dphi = reference["dphi"]
        i_omega_ref_seg = reference["p_r_omega0"]

        # Compute errors
        errors = error_function(
            p=pk,
            pr_next=p_dr_next,
            v=vk,
            p_ref=p_d,
            dp_ref=dp_d,
            ddp_ref=ddp_d,
            dp_normed_ref=dp_normed_d,
            dp_normed_refn=dp_normed_n,
            dphi=dphi,
            i_omega_0=i_omega_0,
            i_omega_ref_0=i_omega_ref_0,
            i_omega_ref_seg=i_omega_ref_seg.T,
            dtau_init=dtau_init.T,
            dtau_init_par=dtau_init_par.T,
            dtau_init_orth1=dtau_init_orth1.T,
            dtau_init_orth2=dtau_init_orth2.T,
            br1=br1_current,
            br2=br2_current,
            br1n=br1_next,
            br2n=br2_next,
            jac_dtau_l=jac_dtau_l,
            jac_dtau_r=jac_dtau_r,
            idx=k,
            split_idx=split_idx,
            v1=v1_current,
            v2=v2_current,
            v3=v3_current,
            v1n=v1_next,
            v2n=v2_next,
            v3n=v3_next,
            n_horizon=N,
        )
        e_p_park = errors["e_p_par"]
        e_p = errors["e_p"]
        de_p = errors["de_p"]
        # de_p_park = errors["de_p_par"]
        e_r = errors["e_r"]
        e_r_par = errors["e_r_par"]
        de_r = errors["de_r"]
        e_r_park = errors["e_r_par"]
        e_r_orth1k = errors["e_r_orth1"]
        e_r_orth2k = errors["e_r_orth2"]

        # Velocity orthogonal to path
        v_ref = dphi * dp_d
        v_orth = vk - v_ref

        # -----------------------------------------------------------------
        # COST
        # -----------------------------------------------------------------
        if path_following:
            e_p_obj = e_p
            e_r_obj = e_r
        else:
            sigm = 1 / (1 + ca.exp(-60 * (phi - (phi_max - 0.05))))
            e_r_obj = 1 * sigm * e_r
            e_p_obj = 1 * sigm * e_p
            J = J + ca.sumsqr(e_r_obj)
            J = J + ca.sumsqr(e_p_obj)
        x_phi = ca.vertcat(phi, dphi)
        J = J + objective_function(
            x_phi=x_phi,
            e_r_par=e_r_par,
            x_phi_d=x_phi_d,
            v_orth=v_orth,
            u=u[k, :].T,
            dq=dq[k, :].T,
            weights=weights,
            k=k,
        )
        J = J + weights[0] * ca.sumsqr(e_p)
        J = J + weights[1] / 50 * ca.sumsqr(e_r_orth1k)
        J = J + weights[1] / 50 * ca.sumsqr(e_r_orth2k)
        # Terminal cost
        if k == N - 1:
            J = J + weights[8] * ca.sumsqr(slacks[:-2])
            J = J + weights[8] * ca.sumsqr(slacks[-1])
            J = J + weights[10] * ca.sumsqr(dslacks)
        J = J + weights[9] * ca.sumsqr(rslacks[k])
        J = J + weights[10] * ca.sumsqr(drslacks[k])
        J = J + weights[9] * ca.sumsqr(pslacks[k])
        J = J + weights[10] * ca.sumsqr(dpslacks[k])

        # -----------------------------------------------------------------
        # CONSTRAINTS
        # -----------------------------------------------------------------
        g += [(reference["a_current"] @ pk[:3]).T - reference["b_current"] - pslacks[k]]
        lbg += [-np.inf] * max_set_size
        ubg += [0] * max_set_size

        e_r_proj1 = ca.dot(br1_current, e_r_orth1k)
        e_r_proj2 = ca.dot(br2_current, e_r_orth2k)
        e_r_proj_par = ca.dot(dp_normed_d, e_r_park)

        g += [e_r_proj1 - reference["r_bound_upper"][0] - rslacks[k]]
        g += [e_r_proj_par - reference["r_bound_upper"][1] - rslacks[k]]
        g += [e_r_proj2 - reference["r_bound_upper"][2] - rslacks[k]]
        lbg += [-np.inf] * 3
        ubg += [0] * 3
        g += [e_r_proj1 - reference["r_bound_lower"][0] + rslacks[k]]
        g += [e_r_proj_par - reference["r_bound_lower"][1] + rslacks[k]]
        g += [e_r_proj2 - reference["r_bound_lower"][2] + rslacks[k]]
        lbg += [0] * 3
        ubg += [np.inf] * 3

        p_list = [robot_model.fk_pos_col(q[k, :].T, i) for i in range(6)]
        for i, pl in enumerate(p_list):
            aj = a_set_joints[i]
            bj = b_set_joints[i, :]
            g += [(aj @ pl).T - bj - slacks[i]]
            # g += [(aj @ pl).T - bj]
            lbg += [-np.inf] * max_set_size
            ubg += [0] * max_set_size

        g += [phi - (reference["phi_end_seg"] + 0.005)]
        lbg += [-np.inf]
        ubg += [0]

        # Terminal constraints
        if k == N - 1:
            # g += [
            #     (reference["a_next"] @ pk[:3]).T
            #     - (reference["b_next"] - 0.005)
            #     - slacks[-1]
            # ]
            # lbg += [-np.inf] * max_set_size
            # ubg += [0] * max_set_size

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

            J += 100 * ca.sumsqr(v[k, :])
            # g += [ca.sumsqr(dq[k, :])]
            # lbg += [0]
            # ubg += [0.01]

            e_r_parn = errors["e_r_par"]
            e_r_orth1n = errors["e_r_orth1"]
            e_r_orth2n = errors["e_r_orth2"]
            e_r_proj1n = ca.dot(br1_next, e_r_orth1n)
            e_r_proj2n = ca.dot(br2_next, e_r_orth2n)
            e_r_proj_parn = ca.dot(dp_normed_n, e_r_parn)
            g += [e_r_proj1n - reference["r_bound_upper_next"][0] - slacks[-1]]
            g += [e_r_proj_parn - reference["r_bound_upper_next"][1] - slacks[-1]]
            g += [e_r_proj2n - reference["r_bound_upper_next"][2] - slacks[-1]]
            lbg += [-np.inf] * 3
            ubg += [0] * 3
            g += [e_r_proj1n - reference["r_bound_lower_next"][0] + slacks[-1]]
            g += [e_r_proj_parn - reference["r_bound_lower_next"][1] + slacks[-1]]
            g += [e_r_proj2n - reference["r_bound_lower_next"][2] + slacks[-1]]
            lbg += [0] * 3
            ubg += [np.inf] * 3

    # Create an NLP solver
    params = ca.vertcat(
        split_idx,
        slacks0,
        i_omega_ref_0,
        dtau_init.reshape((-1, 1)),
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
        e_r_bound.reshape((-1, 1)),
        weights,
        phi_max,
        v1.reshape((-1, 1)),
        v2.reshape((-1, 1)),
        v3.reshape((-1, 1)),
        qd,
    )
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
