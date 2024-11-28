import numpy as np
import casadi as ca


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
    """ Compute rodrigues matrix given an axis of rotation and an angle.
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
    mat_rodrigues = ident + ca.sin(phi) * omega_mat + (1 - ca.cos(phi)) * omega_mat @ omega_mat
    return mat_rodrigues


def jac_SO3_inv_right(axis):
    if isinstance(axis, np.ndarray):
        angle = np.linalg.norm(axis) + 1e-6
        ident = np.eye(3)
    else:
        angle = ca.norm_2(axis) + 1e-6
        ident = ca.MX.eye(3)
    omega_mat = skew_matrix(axis)
    jac_inv = ident + 0.5 * omega_mat
    jac_inv += (1 / angle**2 - (1 + np.cos(angle)) / (2 * angle * np.sin(angle))) * omega_mat @ omega_mat
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
    jac_inv += (1 / angle**2 - (1 + np.cos(angle)) / (2 * angle * np.sin(angle))) * omega_mat @ omega_mat
    return jac_inv
