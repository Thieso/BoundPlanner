import numpy as np


def get_default_path(p0, r0, nr_segs=2):
    """
    Get necessary specifications for a default path.
    """
    # Position and orienation via points
    p_via = [p0] * nr_segs
    r_via = [r0] * nr_segs

    # Position bounds
    p_lower = [np.array([-1.0, -1.0])] * nr_segs
    p_upper = [np.array([1.0, 1.0])] * nr_segs
    p_limits = [p_lower, p_upper]

    # Orientation bounds
    r_lower = [np.array([-1.0, -1.0])] * nr_segs
    r_upper = [np.array([1.0, 1.0])] * nr_segs
    r_limits = [r_lower, r_upper]

    # Desired first basis vector
    bp1_list = [np.array([0.0, 0.0, 1.0])] * nr_segs
    br1_list = [np.array([0.0, 0.0, 1.0])] * nr_segs

    # Slope
    s = [0.0] * nr_segs

    # Position errors
    e_p_max = 0.5
    e_p_start = [np.array([e_p_max, e_p_max, -e_p_max, -e_p_max])] * (nr_segs - 1)
    e_p_mid = [np.array([e_p_max, e_p_max, -e_p_max, -e_p_max])] * (nr_segs - 1)
    e_p_end = [np.array([e_p_max, e_p_max, -e_p_max, -e_p_max])] * (nr_segs - 1)

    # e_p_mid[-1] = np.array(
    #     [
    #         e_p_max / 2,
    #         e_p_max / 2,
    #         -e_p_max / 2,
    #         -e_p_max / 2,
    #     ]
    # )
    # e_p_end[-1] = np.array([0.01, 0.01, -0.01, -0.01])

    # Orientation errors
    e_r_start = [np.array([90, 90, 90, -90, -90, -90]) * np.pi / 180] * (nr_segs - 1)
    e_r_end = [np.array([90, 90, 90, -90, -90, -90]) * np.pi / 180] * (nr_segs - 1)
    e_r_mid = [np.array([90, 90, 90, -90, -90, -90]) * np.pi / 180] * (nr_segs - 1)

    # e_r_start[0] = np.array([3, 3, 3, -3, -3, -3]) * np.pi / 180
    # e_r_mid[0] = np.array([45, 45, 45, -45, -45, -45]) * np.pi / 180
    # e_r_mid[-1] = np.array([45, 45, 45, -45, -45, -45]) * np.pi / 180
    # e_r_end[-1] = np.array([3, 3, 3, -3, -3, -3]) * np.pi / 180

    return (
        p_via,
        r_via,
        p_limits,
        r_limits,
        bp1_list,
        br1_list,
        s,
        e_p_start,
        e_p_end,
        e_p_mid,
        e_r_start,
        e_r_end,
        e_r_mid,
    )


def get_default_weights():
    w_p = 10
    w_r = 0.0
    w_v_p = 0.3
    w_v_r = 0.03
    w_a_p = 0.03
    w_a_r = 0.003
    w_speed = 0.5

    # w_phi = 8
    # w_dphi = 5
    # w_ddphi = 4
    # w_dddphi = 0.5
    w_phi = 10.5 * w_speed
    w_dphi = 4.06
    w_ddphi = 1.81
    w_dddphi = 0.14

    scal = 1.0 / w_phi
    w_phi *= scal
    w_dphi *= scal
    w_ddphi *= scal
    w_dddphi *= scal

    w_q = 0.01
    w_dq = 0.001
    w_ddq = 0.001
    w_jerk = 0.0001
    w_term = 1.0
    w_col = 100
    w_col_alpha = 1000
    w_col_beta = 0.002

    weights = np.array(
        [
            w_p,
            w_r,
            w_v_p,
            w_v_r,
            w_speed,
            w_a_p,
            w_a_r,
            w_phi,
            w_dphi,
            w_ddphi,
            w_dddphi,
            w_q,
            w_dq,
            w_ddq,
            w_jerk,
            w_term,
            w_col,
            w_col_alpha,
            w_col_beta,
        ]
    )
    return weights
