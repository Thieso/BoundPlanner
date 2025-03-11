import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from bound_mpc.utils import gram_schmidt


def signed_angle(v1, v2):
    atan1 = np.arctan2(v1[1], v1[0])
    atan2 = np.arctan2(v2[1], v2[0])
    return atan1 - atan2


def create_euler_sprial(v1, v2, length=0.05):
    phi_a = length
    phi_b = 2 * phi_a

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    if np.linalg.norm(v1 - v2) < 1e-3:
        rot_2d = np.vstack((v1, v2))
        return 0.0, 0.0, 0.0, rot_2d, 0.0, [0.0, 0.0]

    # Build projection matrix for 2D
    v2_orth = gram_schmidt(v1, v2)
    v2_orth /= np.linalg.norm(v2_orth)
    rot_2d = np.vstack((v1, v2_orth))

    # Project to 2D
    dl1 = rot_2d @ v1
    dl2 = rot_2d @ v2

    theta = -signed_angle(dl1, dl2)
    a = (2 * theta) / (phi_b**2)
    # b = a - 4 * (theta / (phi_b**2))
    b = -a

    # Find the correct offset such that the spiral converges to the next linear
    # segment
    p_via = length * dl1
    p_mid = eval_euler_spiral(a, length)

    angle = theta + 2 * signed_angle(np.array([1, 0.0]), p_mid)
    rot_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    p_end = p_mid + rot_mat @ p_mid

    # Project the end point onto the linear segment
    v_via_end = p_end - p_via
    p_proj = p_via + np.dot(v_via_end, rot_2d @ v2) * (rot_2d @ v2)
    angle_proj = np.pi - signed_angle(np.array([1, 0.0]), p_proj - p_end)
    offset = np.linalg.norm(p_proj - p_end) / np.cos(angle_proj)

    # Compute the shortening of the path parameter due to the shortcut
    p_start = -dl1 * offset
    p_end += p_start
    lin_length1 = np.linalg.norm(p_start - p_via)
    lin_length2 = np.linalg.norm(p_end - p_via)
    shortenings = [lin_length1 - length, lin_length2 - length]

    return a, b, theta, rot_2d, offset, shortenings


def eval_euler_spiral(a, phi):
    int_cos = 0
    int_sin = 0
    for n in range(10):
        int_sin += (
            (-1) ** n
            * (a ** (2 * n + 1) * phi ** (4 * n + 3))
            / (math.factorial(2 * n + 1) * (4 * n + 3))
        )
        int_cos += (a ** (2 * n) * (-1 * (phi**4)) ** (n)) / (
            math.factorial(2 * n) * (1 + 4 * n)
        )
    int_cos *= phi
    p_euler = np.array([int_cos, int_sin])
    return p_euler


if __name__ == "__main__":
    v1 = np.array([1.0, 1.0, 0.0])
    v2 = np.array([0.0, -1.0, -0.3])
    length = 0.05
    a, b, theta, rot_2d, offset, shortenings = create_euler_sprial(v1, v2, length)
    print(f"Shortening 1 {shortenings[0]*1000:.1f}mm")
    print(f"Shortening 2 {shortenings[1]*1000:.1f}mm")
    print(f"Offset {offset*1000:.1f}mm")
    N = 500
    r1 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pi_a = np.zeros((N, 2))
    dpi_a = np.zeros((N, 2))
    ddpi_a = np.zeros((N, 2))
    dddpi_a = np.zeros((N, 2))
    phi = np.linspace(0, length, N)
    r0 = np.eye(3)
    for i in range(N - 1):
        dphi = phi[i + 1] - phi[i]
        dpi_a1 = np.array([np.cos(a * phi[i] ** 2), np.sin(a * phi[i] ** 2)])
        dpi_a2 = np.array([np.cos(a * phi[i + 1] ** 2), np.sin(a * phi[i + 1] ** 2)])
        pi_a[i + 1, :] = pi_a[i, :] + 0.5 * dphi * (dpi_a1 + dpi_a2)
        dpi_a[i + 1, :] = dpi_a2
        ddpi_a[i + 1, :] = (
            2
            * a
            * phi[i + 1]
            * np.array([-np.sin(a * phi[i + 1] ** 2), np.cos(a * phi[i + 1] ** 2)])
        )
        dddpi_a[i + 1, :] = (
            2
            * a
            * np.array([-np.sin(a * phi[i + 1] ** 2), np.cos(a * phi[i + 1] ** 2)])
        )
        dddpi_a[i + 1, :] += (
            4
            * a**2
            * phi[i + 1] ** 2
            * np.array([-np.cos(a * phi[i + 1] ** 2), -np.sin(a * phi[i + 1] ** 2)])
        )
        r0 = R.from_rotvec(rot_2d.T @ (dpi_a1 * dphi)).as_matrix() @ r0

    w = rot_2d.T @ pi_a[-1, :]
    r1_lin = R.from_rotvec(w).as_matrix()
    print(r0)
    print(r1_lin)
    print(np.linalg.norm(R.from_matrix(r0 @ r1_lin.T).as_rotvec()) * 180 / np.pi)

    pi_b = np.zeros((N, 2))
    dpi_b = np.zeros((N, 2))
    ddpi_b = np.zeros((N, 2))
    dddpi_b = np.zeros((N, 2))
    pi_b[0, :] = pi_a[-1, :]
    for i in range(N - 1):
        phi_current = 2 * length - (phi[i] + length)
        phi_next = 2 * length - (phi[i + 1] + length)
        dphi = phi_current - phi_next
        dpi_b1 = r1 @ np.array([np.cos(b * phi_current**2), np.sin(b * phi_current**2)])
        dpi_b2 = r1 @ np.array([np.cos(b * phi_next**2), np.sin(b * phi_next**2)])
        pi_b[i + 1, :] = pi_b[i, :] + 0.5 * dphi * (dpi_b1 + dpi_b2)
        dpi_b[i + 1, :] = dpi_b2
        ddpi_b[i + 1, :] = r1 @ (
            2
            * b
            * phi_next
            * np.array([np.sin(b * phi_next**2), -np.cos(b * phi_next**2)])
        )
        dddpi_b[i + 1, :] = r1 @ (
            -2 * b * np.array([np.sin(b * phi_next**2), -np.cos(b * phi_next**2)])
        )
        dddpi_b[i + 1, :] += r1 @ (
            4
            * b**2
            * phi_next**2
            * np.array([-np.cos(b * phi_next**2), -np.sin(b * phi_next**2)])
        )

    pi_a3 = np.zeros((N, 3))
    pi_b3 = np.zeros((N, 3))
    path_lin = np.zeros((N, 3))
    p_start = -v1 * offset
    for i in range(N):
        pi_a3[i, :] = rot_2d.T @ pi_a[i, :] + p_start
        pi_b3[i, :] = rot_2d.T @ pi_b[i, :] + p_start
        if phi[i] * 2 < length:
            path_lin[i, :] = phi[i] * 2 * v1
        else:
            path_lin[i, :] = (phi[i] * 2 - length) * v2 + length * v1

    plt.figure()
    plt.subplot(1, 1, 1, projection="3d")
    plt.plot(pi_a3[:, 0], pi_a3[:, 1], pi_a3[:, 2])
    plt.plot(pi_b3[:, 0], pi_b3[:, 1], pi_b3[:, 2])
    plt.plot(path_lin[:, 0], path_lin[:, 1], path_lin[:, 2])
    plt.plot(p_start[0], p_start[1], p_start[2], ".")
    # plt.plot(p_mid3[0], p_mid3[1], p_mid3[2], '.')
    # plt.plot(p_end3[0], p_end3[1], p_end3[2], '.')
    # plt.plot(p_via3[0], p_via3[1], p_via3[2], '.')
    # plt.plot(p_proj3[0], p_proj3[1], p_proj3[2], '.')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    plt.figure()
    plt.plot(pi_a[:, 0], pi_a[:, 1])
    plt.plot(pi_b[:, 0], pi_b[:, 1])
    # plt.plot(p_mid[0], p_mid[1], '.')
    # plt.plot(p_end[0], p_end[1], '.')
    # plt.plot(p_via[0], p_via[1], '.')
    # plt.plot(p_proj[0], p_proj[1], '.')
    plt.axis("equal")
    plt.figure()
    plt.subplot(411)
    for i in range(2):
        plt.plot(phi[1:], pi_a[1:, i], f"C{i}")
        plt.plot(phi[1:] + length, pi_b[1:, i], f"C{i}")
    plt.subplot(412)
    for i in range(2):
        plt.plot(phi[1:], dpi_a[1:, i], f"C{i}")
        plt.plot(phi[1:] + length, dpi_b[1:, i], f"C{i}")
    plt.subplot(413)
    for i in range(2):
        dphi = np.mean(np.diff(phi))
        plt.plot(phi[1:], ddpi_a[1:, i], f"C{i}")
        # plt.plot(phi[2:], np.diff(dpi_a[1:, i])/dphi, 'C0:')
        plt.plot(phi[1:] + length, ddpi_b[1:, i], f"C{i}")
        # plt.plot(phi[2:]+length, np.diff(dpi_b[1:, i])/dphi, 'C1:')
    plt.subplot(414)
    for i in range(2):
        dphi = np.mean(np.diff(phi))
        plt.plot(phi[1:], dddpi_a[1:, i], f"C{i}")
        plt.plot(phi[1:] + length, dddpi_b[1:, i], f"C{i}")
        plt.plot(phi[2:], np.diff(ddpi_a[1:, i]) / dphi, f"C{i}:")
        plt.plot(phi[2:] + length, np.diff(ddpi_b[1:, i]) / dphi, f"C{i}:")
    plt.figure()
    plt.subplot(211)
    for i in range(2):
        plt.plot(phi[1:], np.linalg.norm(ddpi_a[1:, :], axis=1), f"C{i}")
        plt.plot(phi[1:] + length, np.linalg.norm(ddpi_b[1:, :], axis=1), f"C{i}")
    plt.subplot(212)
    for i in range(2):
        plt.plot(phi[1:], np.linalg.norm(dddpi_a[1:, :], axis=1), f"C{i}")
        plt.plot(phi[1:] + length, np.linalg.norm(dddpi_b[1:, :], axis=1), f"C{i}")
    plt.show()
