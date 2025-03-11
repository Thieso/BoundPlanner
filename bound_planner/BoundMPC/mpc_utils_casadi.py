import casadi as ca

""" Casadi utility functions for BoundMPC. """


def integrate_rot_error_diff(
    e_r0, i_omega_c, i_omega_0, i_omega_ref, i_omega_ref_0, jac_dtau_l, jac_dtau_r
):
    e_r = (
        e_r0
        + jac_dtau_l @ (i_omega_c - i_omega_0)
        - jac_dtau_r @ (i_omega_ref - i_omega_ref_0)
    )
    return e_r


def compute_rot_error_velocity(
    omega_desired, omega_current, jac_dtau_l, jac_dtau_r, dphi
):
    omega_desired_scaled = omega_desired * dphi
    de = jac_dtau_l @ omega_current - jac_dtau_r @ omega_desired_scaled
    return de


def compute_position_error(
    p_current, v_current, p_desired, dp_desired, ddp_desired, dphi
):
    """
    Compute the position error parallel and orthogonal to the path.
    Parameters
    ----------
    p_current : array
        current position
    v_current : array
        current velocity
    p_desired : array
        desired position
    dp_desired : array
        desired velocity
    dphi: float
        derivative of path parameter

    Returns
    -------
    e_par : array
        error parallel to path
    e_orth : array
        error orthogonal to path
    de_par : array
        derivative of error parallel to path
    de_orth : array
        derivative of error orthogonal to path
    """
    # Compute error vector
    e = p_current - p_desired

    # Project error onto the path
    e_par = (dp_desired.T @ e) * dp_desired

    # Compute orthogonal error
    e_orth = e - e_par

    # Compute time derivatives of the errors
    de = v_current - dp_desired * dphi
    de_par = (dp_desired.T @ de) * dp_desired
    de_par += ((ddp_desired * dphi).T @ e) * dp_desired
    de_par += (dp_desired.T @ e) * ddp_desired * dphi
    de_orth = de - de_par

    return e_par, e_orth, de_par, de_orth, e, de


def decompose_orthogonal_error(e_orth, v1, v2):
    """Decomposes the orthogonal error into two orthogonal directions which are
    based on the reference coordinate system.
    Parameters
    ----------
    e_orth : vector 3
        orthogonal position error
    v1 : vector 3
        basis vector 1 of orthogonal error plane
    v2 : vector 3
        basis vector 2 of orthogonal error plane

    Returns
    -------
    orth_coords : vector 2
        coordinates of orthogonal error in the error plane
    """
    orth_coords = ca.vertcat(ca.dot(e_orth, v1), ca.dot(e_orth, v2))

    return orth_coords


def compute_fourth_order_error_bound(phi, phi0, phi1, e0, e1, s0, s1, e_max):
    """Compute error bounds about the trajectory. This assumes a linear
    trajectory and constructs a fourth order polynomial as orthogonal error
    bounds.
    Parameters
    ----------
    phi : float
        current path parameter
    phi0 : float
        start of error curve
    phi1 : float
        end of error curve
    e0 : float
        allowed error at phi0
    e1 : float
        allowed error at phi1
    s0 : float
        slope at phi0
    s1 : float
        minus slope at phi1
    e_max : float
        maximum error at (phi1 - phi0)/2
    Returns
    -------
    e_bound : float
        maximum allowed error at phi
    """
    a0 = -(
        -(phi0**4) * phi1 * s1
        - phi0**3 * phi1**2 * s0
        + phi0**2 * phi1**3 * s1
        + phi0 * phi1**4 * s0
        + 5 * e0 * phi0**2 * phi1**2
        + 4 * e0 * phi0 * phi1**3
        - e0 * phi1**4
        - e1 * phi0**4
        + 4 * e1 * phi0**3 * phi1
        + 5 * e1 * phi0**2 * phi1**2
        - 16 * e_max * phi0**2 * phi1**2
    ) / (
        phi0**4
        - 4 * phi1 * phi0**3
        + 6 * phi1**2 * phi0**2
        - 4 * phi1**3 * phi0
        + phi1**4
    )
    a1 = (
        -(phi0**4) * s1
        - 2 * phi0**3 * phi1 * s0
        - 4 * phi0**3 * phi1 * s1
        - 3 * phi0**2 * phi1**2 * s0
        + 3 * phi0**2 * phi1**2 * s1
        + 4 * phi0 * phi1**3 * s0
        + 2 * phi0 * phi1**3 * s1
        + phi1**4 * s0
        + 10 * e0 * phi0**2 * phi1
        + 22 * e0 * phi0 * phi1**2
        + 22 * e1 * phi0**2 * phi1
        + 10 * e1 * phi0 * phi1**2
        - 32 * e_max * phi0**2 * phi1
        - 32 * e_max * phi0 * phi1**2
    ) / (
        phi0**4
        - 4 * phi1 * phi0**3
        + 6 * phi1**2 * phi0**2
        - 4 * phi1**3 * phi0
        + phi1**4
    )
    a2 = -(
        -(phi0**3) * s0
        - 4 * phi0**3 * s1
        - 6 * phi0**2 * phi1 * s0
        - 3 * phi0**2 * phi1 * s1
        + 3 * phi0 * phi1**2 * s0
        + 6 * phi0 * phi1**2 * s1
        + 4 * phi1**3 * s0
        + phi1**3 * s1
        + 5 * e0 * phi0**2
        + 32 * e0 * phi0 * phi1
        + 11 * e0 * phi1**2
        + 11 * e1 * phi0**2
        + 32 * e1 * phi0 * phi1
        + 5 * e1 * phi1**2
        - 16 * e_max * phi0**2
        - 64 * e_max * phi0 * phi1
        - 16 * e_max * phi1**2
    ) / (
        phi0**4
        - 4 * phi1 * phi0**3
        + 6 * phi1**2 * phi0**2
        - 4 * phi1**3 * phi0
        + phi1**4
    )
    a3 = (
        (
            -3 * phi0**2 * s0
            - 5 * phi0**2 * s1
            - 2 * phi0 * phi1 * s0
            + 2 * phi0 * phi1 * s1
            + 5 * phi1**2 * s0
            + 3 * phi1**2 * s1
            + 14 * e0 * phi0
            + 18 * e0 * phi1
            + 18 * e1 * phi0
            + 14 * e1 * phi1
            - 32 * e_max * phi0
            - 32 * e_max * phi1
        )
        / (phi0**3 - 3 * phi1 * phi0**2 + 3 * phi1**2 * phi0 - phi1**3)
        / (phi0 - phi1)
    )
    a4 = (
        -2
        * (-phi0 * s0 - phi0 * s1 + phi1 * s0 + phi1 * s1 + 4 * e0 + 4 * e1 - 8 * e_max)
        / (
            phi0**4
            - 4 * phi1 * phi0**3
            + 6 * phi1**2 * phi0**2
            - 4 * phi1**3 * phi0
            + phi1**4
        )
    )

    e_bound = a4 * phi**4 + a3 * phi**3 + a2 * phi**2 + a1 * phi + a0

    return e_bound


def compute_bound_params(phi0, phi1, e0, e1, s, e_max):
    a0 = -(
        -(phi0**4) * phi1 * s
        - phi0**3 * phi1**2 * s
        + phi0**2 * phi1**3 * s
        + phi0 * phi1**4 * s
        + 5 * e0 * phi0**2 * phi1**2
        + 4 * e0 * phi0 * phi1**3
        - e0 * phi1**4
        - e1 * phi0**4
        + 4 * e1 * phi0**3 * phi1
        + 5 * e1 * phi0**2 * phi1**2
        - 16 * e_max * phi0**2 * phi1**2
    ) / (
        phi0**4
        - 4 * phi1 * phi0**3
        + 6 * phi1**2 * phi0**2
        - 4 * phi1**3 * phi0
        + phi1**4
    )
    a1 = (
        -(phi0**4) * s
        - 2 * phi0**3 * phi1 * s
        - 4 * phi0**3 * phi1 * s
        - 3 * phi0**2 * phi1**2 * s
        + 3 * phi0**2 * phi1**2 * s
        + 4 * phi0 * phi1**3 * s
        + 2 * phi0 * phi1**3 * s
        + phi1**4 * s
        + 10 * e0 * phi0**2 * phi1
        + 22 * e0 * phi0 * phi1**2
        + 22 * e1 * phi0**2 * phi1
        + 10 * e1 * phi0 * phi1**2
        - 32 * e_max * phi0**2 * phi1
        - 32 * e_max * phi0 * phi1**2
    ) / (
        phi0**4
        - 4 * phi1 * phi0**3
        + 6 * phi1**2 * phi0**2
        - 4 * phi1**3 * phi0
        + phi1**4
    )
    a2 = -(
        -(phi0**3) * s
        - 4 * phi0**3 * s
        - 6 * phi0**2 * phi1 * s
        - 3 * phi0**2 * phi1 * s
        + 3 * phi0 * phi1**2 * s
        + 6 * phi0 * phi1**2 * s
        + 4 * phi1**3 * s
        + phi1**3 * s
        + 5 * e0 * phi0**2
        + 32 * e0 * phi0 * phi1
        + 11 * e0 * phi1**2
        + 11 * e1 * phi0**2
        + 32 * e1 * phi0 * phi1
        + 5 * e1 * phi1**2
        - 16 * e_max * phi0**2
        - 64 * e_max * phi0 * phi1
        - 16 * e_max * phi1**2
    ) / (
        phi0**4
        - 4 * phi1 * phi0**3
        + 6 * phi1**2 * phi0**2
        - 4 * phi1**3 * phi0
        + phi1**4
    )
    a3 = (
        (
            -3 * phi0**2 * s
            - 5 * phi0**2 * s
            - 2 * phi0 * phi1 * s
            + 2 * phi0 * phi1 * s
            + 5 * phi1**2 * s
            + 3 * phi1**2 * s
            + 14 * e0 * phi0
            + 18 * e0 * phi1
            + 18 * e1 * phi0
            + 14 * e1 * phi1
            - 32 * e_max * phi0
            - 32 * e_max * phi1
        )
        / (phi0**3 - 3 * phi1 * phi0**2 + 3 * phi1**2 * phi0 - phi1**3)
        / (phi0 - phi1)
    )
    a4 = (
        -2
        * (-phi0 * s - phi0 * s + phi1 * s + phi1 * s + 4 * e0 + 4 * e1 - 8 * e_max)
        / (
            phi0**4
            - 4 * phi1 * phi0**3
            + 6 * phi1**2 * phi0**2
            - 4 * phi1**3 * phi0
            + phi1**4
        )
    )

    return a4, a3, a2, a1, a0


def compute_bound_params_six(phi0, phi1, e0, e1, s, em):
    # a0 = (-phi0 ** 6 * phi1 * s1 + 6 * phi0 ** 5 * phi1 ** 2 * s1 - 6 * phi0 ** 4 * phi1 ** 3 * s0 + phi0 ** 4 * phi1 ** 3 * s1 + phi0 ** 3 * phi1 ** 4 * s0 - 6 * phi0 ** 3 * phi1 ** 4 * s1 + 6 * phi0 ** 2 * phi1 ** 5 * s0 - phi0 * phi1 ** 6 * s0 + 22 * e0 * phi0 ** 3 * phi1 ** 3 + 15 * e0 * phi0 ** 2 * phi1 ** 4 - 6 * e0 * phi0 * phi1 ** 5 + e0 * phi1 ** 6 + e1 * phi0 ** 6 - 6 * e1 * phi0 ** 5 * phi1 + 15 * e1 * phi0 ** 4 * phi1 ** 2 + 22 * e1 * phi0 ** 3 * phi1 ** 3 - 64 * em * phi0 ** 3 * phi1 ** 3) / (phi0 ** 6 - 6 * phi0 ** 5 * phi1 + 15 * phi0 ** 4 * phi1 ** 2 - 20 * phi0 ** 3 * phi1 ** 3 + 15 * phi0 ** 2 * phi1 ** 4 - 6 * phi0 * phi1 ** 5 + phi1 ** 6)
    # a1 = -(-phi0 ** 6 * s1 + 6 * phi0 ** 5 * phi1 * s1 - 18 * phi0 ** 4 * phi1 ** 2 * s0 + 33 * phi0 ** 4 * phi1 ** 2 * s1 - 20 * phi0 ** 3 * phi1 ** 3 * s0 - 20 * phi0 ** 3 * phi1 ** 3 * s1 + 33 * phi0 ** 2 * phi1 ** 4 * s0 - 18 * phi0 ** 2 * phi1 ** 4 * s1 + 6 * phi0 * phi1 ** 5 * s0 - phi1 ** 6 * s0 + 66 * e0 * phi0 ** 3 * phi1 ** 2 + 126 * e0 * phi0 ** 2 * phi1 ** 3 + 126 * e1 * phi0 ** 3 * phi1 ** 2 + 66 * e1 * phi0 ** 2 * phi1 ** 3 - 192 * em * phi0 ** 3 * phi1 ** 2 - 192 * em * phi0 ** 2 * phi1 ** 3) / (phi0 ** 6 - 6 * phi0 ** 5 * phi1 + 15 * phi0 ** 4 * phi1 ** 2 - 20 * phi0 ** 3 * phi1 ** 3 + 15 * phi0 ** 2 * phi1 ** 4 - 6 * phi0 * phi1 ** 5 + phi1 ** 6)
    # a2 = 6 * (-3 * phi0 ** 3 * s0 + 8 * phi0 ** 3 * s1 - 11 * phi0 ** 2 * phi1 * s0 + 6 * phi0 ** 2 * phi1 * s1 + 6 * phi0 * phi1 ** 2 * s0 - 11 * phi0 * phi1 ** 2 * s1 + 8 * phi1 ** 3 * s0 - 3 * phi1 ** 3 * s1 + 11 * e0 * phi0 ** 2 + 48 * e0 * phi0 * phi1 + 21 * e0 * phi1 ** 2 + 21 * e1 * phi0 ** 2 + 48 * e1 * phi0 * phi1 + 11 * e1 * phi1 ** 2 - 32 * em * phi0 ** 2 - 96 * em * phi0 * phi1 - 32 * em * phi1 ** 2) * phi0 * phi1 / (phi0 ** 6 - 6 * phi0 ** 5 * phi1 + 15 * phi0 ** 4 * phi1 ** 2 - 20 * phi0 ** 3 * phi1 ** 3 + 15 * phi0 ** 2 * phi1 ** 4 - 6 * phi0 * phi1 ** 5 + phi1 ** 6)
    # a3 = -2 * (-3 * phi0 ** 4 * s0 + 8 * phi0 ** 4 * s1 - 34 * phi0 ** 3 * phi1 * s0 + 44 * phi0 ** 3 * phi1 * s1 - 15 * phi0 ** 2 * phi1 ** 2 * s0 - 15 * phi0 ** 2 * phi1 ** 2 * s1 + 44 * phi0 * phi1 ** 3 * s0 - 34 * phi0 * phi1 ** 3 * s1 + 8 * phi1 ** 4 * s0 - 3 * phi1 ** 4 * s1 + 11 * e0 * phi0 ** 3 + 129 * e0 * phi0 ** 2 * phi1 + 159 * e0 * phi0 * phi1 ** 2 + 21 * e0 * phi1 ** 3 + 21 * e1 * phi0 ** 3 + 159 * e1 * phi0 ** 2 * phi1 + 129 * e1 * phi0 * phi1 ** 2 + 11 * e1 * phi1 ** 3 - 32 * em * phi0 ** 3 - 288 * em * phi0 ** 2 * phi1 - 288 * em * phi0 * phi1 ** 2 - 32 * em * phi1 ** 3) / (phi0 ** 6 - 6 * phi0 ** 5 * phi1 + 15 * phi0 ** 4 * phi1 ** 2 - 20 * phi0 ** 3 * phi1 ** 3 + 15 * phi0 ** 2 * phi1 ** 4 - 6 * phi0 * phi1 ** 5 + phi1 ** 6)
    # a4 = (-23 * phi0 ** 3 * s0 + 38 * phi0 ** 3 * s1 - 66 * phi0 ** 2 * phi1 * s0 + 51 * phi0 ** 2 * phi1 * s1 + 51 * phi0 * phi1 ** 2 * s0 - 66 * phi0 * phi1 ** 2 * s1 + 38 * phi1 ** 3 * s0 - 23 * phi1 ** 3 * s1 + 81 * e0 * phi0 ** 2 + 288 * e0 * phi0 * phi1 + 111 * e0 * phi1 ** 2 + 111 * e1 * phi0 ** 2 + 288 * e1 * phi0 * phi1 + 81 * e1 * phi1 ** 2 - 192 * em * phi0 ** 2 - 576 * em * phi0 * phi1 - 192 * em * phi1 ** 2) / (phi0 ** 3 - 3 * phi0 ** 2 * phi1 + 3 * phi0 * phi1 ** 2 - phi1 ** 3) ** 2
    # a5 = -3 * (-9 * phi0 ** 2 * s0 + 11 * phi0 ** 2 * s1 - 2 * phi0 * phi1 * s0 - 2 * phi0 * phi1 * s1 + 11 * phi1 ** 2 * s0 - 9 * phi1 ** 2 * s1 + 30 * e0 * phi0 + 34 * e0 * phi1 + 34 * e1 * phi0 + 30 * e1 * phi1 - 64 * em * phi0 - 64 * em * phi1) / (phi0 ** 5 - 5 * phi0 ** 4 * phi1 + 10 * phi0 ** 3 * phi1 ** 2 - 10 * phi0 ** 2 * phi1 ** 3 + 5 * phi0 * phi1 ** 4 - phi1 ** 5) / (phi0 - phi1)
    # a6 = 2 * (-5 * phi0 * s0 + 5 * phi0 * s1 + 5 * phi1 * s0 - 5 * phi1 * s1 + 16 * e0 + 16 * e1 - 32 * em) / (phi0 ** 6 - 6 * phi0 ** 5 * phi1 + 15 * phi0 ** 4 * phi1 ** 2 - 20 * phi0 ** 3 * phi1 ** 3 + 15 * phi0 ** 2 * phi1 ** 4 - 6 * phi0 * phi1 ** 5 + phi1 ** 6)

    a0 = (
        22 * e0 * phi0**3 * phi1**3
        + 15 * e0 * phi0**2 * phi1**4
        - 6 * e0 * phi0 * phi1**5
        + e0 * phi1**6
        + e1 * phi0**6
        - 6 * e1 * phi0**5 * phi1
        + 15 * e1 * phi0**4 * phi1**2
        + 22 * e1 * phi0**3 * phi1**3
        - 64 * em * phi0**3 * phi1**3
    ) / (
        phi0**6
        - 6 * phi0**5 * phi1
        + 15 * phi0**4 * phi1**2
        - 20 * phi0**3 * phi1**3
        + 15 * phi0**2 * phi1**4
        - 6 * phi0 * phi1**5
        + phi1**6
    )
    a1 = (
        -6
        * phi1**2
        * phi0**2
        * (
            11 * e0 * phi0
            + 21 * e0 * phi1
            + 21 * e1 * phi0
            + 11 * e1 * phi1
            - 32 * em * phi0
            - 32 * em * phi1
        )
        / (
            phi0**6
            - 6 * phi0**5 * phi1
            + 15 * phi0**4 * phi1**2
            - 20 * phi0**3 * phi1**3
            + 15 * phi0**2 * phi1**4
            - 6 * phi0 * phi1**5
            + phi1**6
        )
    )
    a2 = (
        6
        * phi1
        * phi0
        * (
            11 * phi0**2 * e0
            + 48 * e0 * phi0 * phi1
            + 21 * e0 * phi1**2
            + 21 * phi0**2 * e1
            + 48 * e1 * phi0 * phi1
            + 11 * e1 * phi1**2
            - 32 * phi0**2 * em
            - 96 * em * phi0 * phi1
            - 32 * em * phi1**2
        )
        / (
            phi0**6
            - 6 * phi0**5 * phi1
            + 15 * phi0**4 * phi1**2
            - 20 * phi0**3 * phi1**3
            + 15 * phi0**2 * phi1**4
            - 6 * phi0 * phi1**5
            + phi1**6
        )
    )
    a3 = (
        -2
        * (
            11 * phi0**3 * e0
            + 129 * e0 * phi0**2 * phi1
            + 159 * e0 * phi0 * phi1**2
            + 21 * e0 * phi1**3
            + 21 * phi0**3 * e1
            + 159 * e1 * phi0**2 * phi1
            + 129 * e1 * phi0 * phi1**2
            + 11 * e1 * phi1**3
            - 32 * phi0**3 * em
            - 288 * em * phi0**2 * phi1
            - 288 * em * phi0 * phi1**2
            - 32 * em * phi1**3
        )
        / (
            phi0**6
            - 6 * phi0**5 * phi1
            + 15 * phi0**4 * phi1**2
            - 20 * phi0**3 * phi1**3
            + 15 * phi0**2 * phi1**4
            - 6 * phi0 * phi1**5
            + phi1**6
        )
    )
    a4 = (
        3
        * (
            27 * phi0**2 * e0
            + 96 * e0 * phi0 * phi1
            + 37 * e0 * phi1**2
            + 37 * phi0**2 * e1
            + 96 * e1 * phi0 * phi1
            + 27 * e1 * phi1**2
            - 64 * phi0**2 * em
            - 192 * em * phi0 * phi1
            - 64 * em * phi1**2
        )
        / (
            phi0**6
            - 6 * phi0**5 * phi1
            + 15 * phi0**4 * phi1**2
            - 20 * phi0**3 * phi1**3
            + 15 * phi0**2 * phi1**4
            - 6 * phi0 * phi1**5
            + phi1**6
        )
    )
    a5 = (
        -6
        * (
            15 * e0 * phi0
            + 17 * e0 * phi1
            + 17 * e1 * phi0
            + 15 * e1 * phi1
            - 32 * em * phi0
            - 32 * em * phi1
        )
        / (
            phi0**5
            - 5 * phi0**4 * phi1
            + 10 * phi0**3 * phi1**2
            - 10 * phi0**2 * phi1**3
            + 5 * phi0 * phi1**4
            - phi1**5
        )
        / (phi0 - phi1)
    )
    a6 = (
        32
        * (e0 + e1 - 2 * em)
        / (
            phi0**6
            - 6 * phi0**5 * phi1
            + 15 * phi0**4 * phi1**2
            - 20 * phi0**3 * phi1**3
            + 15 * phi0**2 * phi1**4
            - 6 * phi0 * phi1**5
            + phi1**6
        )
    )

    return a6, a5, a4, a3, a2, a1, a0


def compute_bound_params_three(phi0, phi1, e0, e1, de0, dde0):
    a0 = (
        -(
            dde0 * phi0**4 * phi1
            - 2 * dde0 * phi0**3 * phi1**2
            + dde0 * phi0**2 * phi1**3
            - 4 * de0 * phi0**3 * phi1
            + 6 * de0 * phi0**2 * phi1**2
            - 2 * de0 * phi0 * phi1**3
            + 6 * e0 * phi0**2 * phi1
            - 6 * e0 * phi0 * phi1**2
            + 2 * e0 * phi1**3
            - 2 * e1 * phi0**3
        )
        / (phi0**3 - 3 * phi0**2 * phi1 + 3 * phi0 * phi1**2 - phi1**3)
        / 2
    )
    a1 = (
        (
            dde0 * phi0**4
            - 3 * dde0 * phi0**2 * phi1**2
            + 2 * dde0 * phi0 * phi1**3
            - 4 * de0 * phi0**3
            + 6 * de0 * phi0 * phi1**2
            - 2 * de0 * phi1**3
            + 6 * e0 * phi0**2
            - 6 * e1 * phi0**2
        )
        / (phi0**3 - 3 * phi0**2 * phi1 + 3 * phi0 * phi1**2 - phi1**3)
        / 2
    )
    a2 = (
        -(
            2 * dde0 * phi0**3
            - 3 * dde0 * phi0**2 * phi1
            + dde0 * phi1**3
            - 6 * de0 * phi0**2
            + 6 * de0 * phi0 * phi1
            + 6 * e0 * phi0
            - 6 * e1 * phi0
        )
        / (phi0**3 - 3 * phi0**2 * phi1 + 3 * phi0 * phi1**2 - phi1**3)
        / 2
    )
    a3 = (
        (
            dde0 * phi0**2
            - 2 * dde0 * phi0 * phi1
            + dde0 * phi1**2
            - 2 * de0 * phi0
            + 2 * de0 * phi1
            + 2 * e0
            - 2 * e1
        )
        / (phi0**3 - 3 * phi0**2 * phi1 + 3 * phi0 * phi1**2 - phi1**3)
        / 2
    )

    return a3, a2, a1, a0


def compute_fourth_order_error_bound_general(phi, a4, a3, a2, a1, a0):
    """Compute error bounds about the trajectory. This assumes a linear
    trajectory and constructs a fourth order polynomial as orthogonal error
    bounds.
    Parameters
    ----------
    phi : float
        current path parameter
    a4 : float
        parameter
    a3 : float
        parameter
    a2 : float
        parameter
    a1 : float
        parameter
    a0 : float
        parameter
    Returns
    -------
    e_bound : float
        maximum allowed error at phi
    """
    e_bound = a4 * phi**4 + a3 * phi**3 + a2 * phi**2 + a1 * phi + a0

    return e_bound


def compute_sixth_order_error_bound_general(phi, a6, a5, a4, a3, a2, a1, a0):
    e_bound = (
        a6 * phi**6
        + a5 * phi**5
        + a4 * phi**4
        + a3 * phi**3
        + a2 * phi**2
        + a1 * phi
        + a0
    )

    return e_bound
