import casadi as ca
import numpy as np

from ..RobotModel import RobotModel


def setup_optimization_problem(
    N,
    nr_joints,
    dt,
    solver_opts,
):
    """Build the optimization problem using symbolic variables such that it
    can be easily used later on by just inputting the new state.
    """
    robot_model = RobotModel()

    max_set_size = 15
    a_set_joints = ca.SX.sym("a_set_joints", max_set_size, 3, 6)
    b_set_joints = ca.SX.sym("b_set_joints", 6, max_set_size)

    # Desired joint config
    qd = ca.SX.sym("q desired", nr_joints)
    w_cost = ca.SX.sym("w_cost", 1)
    desired_input = ca.SX.sym("u desired", 6)
    desired_input_dir = ca.SX.sym("u desired dir", 6)

    # Reference trajectory parameters
    dp_normed_ref = ca.SX.sym("norm of orientation reference", 3)
    br1 = ca.SX.sym("orthogonal error basis 1r", 3)
    br2 = ca.SX.sym("orthogonal error basis 2r", 3)
    v1 = ca.SX.sym("v1", 3)
    v2 = ca.SX.sym("v2", 3)
    v3 = ca.SX.sym("v3", 3)

    # Error computation variables
    dtau_init = ca.SX.sym("initial lie space error", 3)
    dtau_init_par = ca.SX.sym("initial lie space error par", 3)
    dtau_init_orth1 = ca.SX.sym("initial lie space error orth1", 3)
    dtau_init_orth2 = ca.SX.sym("initial lie space error orth2", 3)
    jac_dtau_r = ca.SX.sym("right jacobian at initial error", 3, 3)
    jac_dtau_l = ca.SX.sym("left jacobian at initial error", 3, 3)
    rot_upper = ca.SX.sym("Upper orientation bound", 3)
    rot_lower = ca.SX.sym("Lower orientation bound", 3)

    # Objective function weights
    weights = ca.SX.sym("cost weights", 19)

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
    slacks = ca.SX.sym("s sets", 6)

    w = ca.vertcat(
        q[:],
        dq[:],
        ddq[:],
        u[:],
        p[:],
        v[:],
        slacks,
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
        v_cart = robot_model.velocity_ee(q[k, :].T, dq[k, :].T)
        v_rot = robot_model.omega_ee(q[k, :].T, dq[k, :].T)
        v_new = ca.vertcat(v_cart, v_rot).T

        robot_model = RobotModel()
        pn_pos = robot_model.fk_pos(q[k, :].T).T
        k1 = v[k, 3:]
        k2 = v_new[3:]
        pn_rot = p[k, 3:] + 1 / 2 * dt * (k1 + k2)
        p_new = ca.horzcat(pn_pos, pn_rot)

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

    # Formulate the NLP
    for k in range(1, N):
        pk = p[k, :].T
        vk = v[k, :].T
        vprev = v[k - 1, :].T

        # ---------------------------------------------
        # Compute orientation error and its derivative
        e_r = dtau_init + jac_dtau_l @ (pk[3:] - i_omega_0)
        de_r = jac_dtau_l @ vk[3:]

        # Compute correct starting value based on the segment for the orientation
        # error
        e_par_init = dtau_init_par
        e_orth1_init = dtau_init_orth1
        e_orth2_init = dtau_init_orth2

        # Project error onto path
        scal_orth1 = ca.dot(e_r - dtau_init, v1)
        scal_par = ca.dot(e_r - dtau_init, v2)
        scal_orth2 = ca.dot(e_r - dtau_init, v3)

        # Final decomposed orientation errors
        e_r_orth1k = e_orth1_init + scal_orth1 * br1
        e_r_park = e_par_init + scal_par * dp_normed_ref
        e_r_orth2k = e_orth2_init + scal_orth2 * br2
        # ---------------------------------------------

        # # Numerically differentiated velocity as acceleration
        ak = (vk - vprev) / dt

        # Increment objective value
        # A sigmoid function is used to be able to use the full error at the
        # end of the path which avoid oscillation and increases solver
        # speed.
        if k <= 3:
            J_follow = 0
            J_follow += 100 * ca.sumsqr(vk[:3] - desired_input[:3])
            J_follow += 1 * ca.sumsqr(vk[3:] - desired_input[3:])

            projvp = (vk[:3].T @ desired_input_dir[:3]) * desired_input_dir[:3]
            projvr = (vk[3:].T @ desired_input_dir[3:]) * desired_input_dir[3:]
            projv = ca.vertcat(projvp, projvr)
            orthv = vk - projv
            J_set = 0
            J_set += 100 * ca.sumsqr(projv[:3] - desired_input[:3])
            J_set += 0.1 * ca.sumsqr(orthv[:3])
            J_set += 1 * ca.sumsqr(projv[3:] - desired_input[3:])
            J_set += 0.1 * ca.sumsqr(orthv[3:])

            J += w_cost * J_set + (1 - w_cost) * J_follow
            # J = J + 1 * ca.sumsqr(vk - desired_input)
        # else:
        #     J += 0.1 * ca.sumsqr(vk)

        # J += weights[11] * ca.sumsqr(q[k, :].T - qd)
        J += 0.1 * ca.sumsqr(ak)
        # J += 0.01 * ca.sumsqr(dq[k, :])
        # J += 0.001 * ca.sumsqr(ddq[k, :])
        J += weights[14] * ca.sumsqr(u)

        # -----------------------------------------------------------------
        # CONSTRAINTS
        # -----------------------------------------------------------------
        e_r_proj1 = ca.dot(br1, e_r_orth1k)
        e_r_proj2 = ca.dot(br2, e_r_orth2k)
        e_r_proj_par = ca.dot(dp_normed_ref, e_r_park)

        g += [e_r_proj1 - rot_lower[0]]
        g += [e_r_proj2 - rot_lower[1]]
        g += [e_r_proj_par - rot_lower[2]]
        lbg += [0] * 3
        ubg += [np.inf] * 3
        g += [e_r_proj1 - rot_upper[0]]
        g += [e_r_proj2 - rot_upper[1]]
        g += [e_r_proj_par - rot_upper[2]]
        lbg += [-np.inf] * 3
        ubg += [0] * 3

        p_list = [
            robot_model.fk_pos_j3(q[k, :].T),
            robot_model.fk_pos_j5(q[k, :].T),
            robot_model.fk_pos_j6(q[k, :].T),
            robot_model.fk_pos_j67(q[k, :].T),
            robot_model.fk_pos_elbow(q[k, :].T),
            pk[:3],
        ]
        for i, pl in enumerate(p_list):
            aj = a_set_joints[i]
            bj = b_set_joints[i, :]
            g += [(aj @ pl).T - bj - slacks[i]]
            # g += [(aj @ pl).T - bj]
            lbg += [-np.inf] * max_set_size
            ubg += [0] * max_set_size

        # Terminal cost
        J = J + weights[16] * ca.sumsqr(slacks)
        if k == N - 1:
            # g += [ca.sumsqr(dq[k, :])]
            # g += [ca.sumsqr(ddq[k, :])]
            # lbg += [0] * 2
            # ubg += [0.001] * 2
            J = J + 100 * ca.sumsqr(dq[k, :])
            J = J + 100 * ca.sumsqr(ddq[k, :])

    # Create an NLP solver
    params = ca.vertcat(
        i_omega_ref_0,
        dtau_init,
        dtau_init_par.reshape((-1, 1)),
        dtau_init_orth1.reshape((-1, 1)),
        dtau_init_orth2.reshape((-1, 1)),
        jac_dtau_r.reshape((-1, 1)),
        jac_dtau_l.reshape((-1, 1)),
        dp_normed_ref.reshape((-1, 1)),
        br1.reshape((-1, 1)),
        br2.reshape((-1, 1)),
        weights,
        v1.reshape((-1, 1)),
        v2.reshape((-1, 1)),
        v3.reshape((-1, 1)),
        rot_upper,
        rot_lower,
        qd,
        desired_input,
        desired_input_dir,
        w_cost,
    )
    for i in range(len(a_set_joints)):
        params = ca.vertcat(params, a_set_joints[i].reshape((-1, 1)))
    params = ca.vertcat(params, b_set_joints.reshape((-1, 1)))

    prob = {"f": J, "x": w, "g": ca.horzcat(*g), "p": params}
    solver = ca.nlpsol("solver", "ipopt", prob, solver_opts)
    # solver = ca.nlpsol('solver', 'fatrop', prob, solver_opts)

    return solver, lbg, ubg
