import casadi
import numpy as np
import copy


def calcJ0(c1, p, h, t):
    if c1 <= t and t <= c1 + h:
        j0 = p * (c1 + h - t) / h
    else:
        j0 = 0
    return j0


def calcJn(c1, p, h, t):
    if c1 <= t and t <= c1 + h:
        jn = p * (t - c1) / h
    else:
        jn = 0
    return jn


def calcJk(c1, p, h, t):
    if c1 <= t and t <= c1 + h:
        jk = p * (t - c1) / h
    elif c1 + h < t and t <= c1 + 2 * h:
        jk = p * (c1 + 2 * h - t) / h
    else:
        jk = 0
    return jk


def calcJerk(traj, t, h):
    # j_full = casadi.DM.zeros((traj.shape[1], 1))
    # j_full = casadi.MX.zeros((traj.shape[1], 1))
    j_full = np.zeros((traj.shape[1],))
    for j in range(traj.shape[1]):
        if j == 0:
            j_full = j_full + calcJ0(j * h, traj[:, j], h, t)
        elif j == traj.shape[1] - 1:
            j_full = j_full + calcJn((j - 1) * h, traj[:, j], h, t)
        else:
            j_full = j_full + calcJk((j - 1) * h, traj[:, j], h, t)
    return j_full


def calcA0(c1, p, h, t):
    if c1 < t and t <= c1 + h:
        a0 = -p * (t - c1) * (t - 2 * h - c1) / h / 2
    elif t > c1 + h:
        a0 = p * h / 2
    else:
        a0 = 0
    return a0


def calcAn(c1, p, h, t):
    if c1 < t and t <= c1 + h:
        an = p * (t - c1)**2 / h / 2
    elif t > c1 + h:
        an = p * h / 2
    else:
        an = 0
    return an


def calcAk(c1, p, h, t):
    if c1 < t and t <= c1 + h:
        ak = p * (t - c1)**2 / h / 2
    elif c1 + h < t and t <= c1 + 2 * h:
        ak = -(h * h + (-2 * t + 2 * c1) * h + (t - c1)**2 / 2) * p / h
    elif t > c1 + 2 * h:
        ak = p * h
    else:
        ak = 0
    return ak


def calcAcceleration(traj, t, a_init, h):
    acc_full = copy.deepcopy(a_init)
    for j in range(traj.shape[1]):
        if j == 0:
            acc_full = acc_full + calcA0((j) * h, traj[:, j], h, t)
        elif j == traj.shape[1] - 1:
            acc_full = acc_full + calcAn((j - 1) * h, traj[:, j], h, t)
        else:
            acc_full = acc_full + calcAk((j - 1) * h, traj[:, j], h, t)
    return acc_full


def calcV0(c1, p, h, t):
    if c1 < t and t <= c1 + h:
        v0 = -p * (t - c1)**2 * (t-3 * h - c1) / h / 6
    elif t > c1 + h:
        v0 = p * h * (3 * t - h - 3 * c1) / 6
    else:
        v0 = 0
    return v0


def calcVn(c1, p, h, t):
    if c1 < t and t <= c1 + h:
        vn = -p * (-t + c1)**3 / h / 6
    elif t > c1 + h:
        vn = p * h * (3 * t - 2 * h - 3 * c1) / 6
    else:
        vn = 0
    return vn


def calcVk(c1, p, h, t):
    if c1 < t and t <= c1 + h:
        vk = -p * (-t + c1)**3 / h / 6
    elif c1 + h < t and t <= c1 + 2 * h:
        vk = p * (h**3 + (-3 * t + 3 * c1) * h * h + 3 * (t - c1)**2 * h - (t - c1)**3 / 2) / h / 3
    elif t > c1 + 2 * h:
        vk = -h * p * (c1 + h - t)
    else:
        vk = 0
    return vk


def calcVelocity(traj, t, v_init, a_init, h):
    vel_full = a_init * t + v_init
    for j in range(traj.shape[1]):
        if j == 0:
            vel_full = vel_full + calcV0((j) * h, traj[:, j], h, t)
        elif j == traj.shape[1] - 1:
            vel_full = vel_full + calcVn((j - 1) * h, traj[:, j], h, t)
        else:
            vel_full = vel_full + calcVk((j - 1) * h, traj[:, j], h, t)
    return vel_full


def calcQ0(c1, p, h, t):
    if c1 < t and t <= c1 + h:
        q0 = -p * (t - c1)**3 * (t - 4 * h - c1) / h / 24
    elif t > c1 + h:
        q0 = p * (h * h / 6 + (-2 / 3 * t + 2 / 3 * c1) * h + (t - c1)**2) * h / 4
    else:
        q0 = 0
    return q0


def calcQn(c1, p, h, t):
    if c1 < t and t <= c1 + h:
        qn = p * (-t + c1)**4 / h / 24
    elif t > c1 + h:
        qn = p * h * (h * h / 2 + (-4 / 3 * t + 4 / 3 * c1) * h + (t - c1)**2) / 4
    else:
        qn = 0
    return qn


def calcQk(c1, p, h, t):
    if c1 < t and t <= c1 + h:
        qk = p * (-t + c1)**4 / h / 24
    elif c1 + h < t and t <= c1 + 2 * h:
        qk = -(h**4 + (-4 * t + 4 * c1) * h**3 + 6 * (t - c1)**2 * h * h - 4 * (t - c1)**3 * h + (t - c1)**4 / 2) * p / h / 12
    elif t > c1 + 2 * h:
        qk = 7 / 12 * h * (h * h + (-12 / 7 * t + 12 / 7 * c1) * h + 6 / 7 * (t - c1)**2) * p
    else:
        qk = 0
    return qk


def calcAngle(traj, t, q_init, v_init, a_init, h):
    ang_full = a_init * t**2 / 2 + v_init * t + q_init
    for j in range(traj.shape[1]):
        if j == 0:
            ang_full = ang_full + calcQ0((j) * h, traj[:, j], h, t)
        elif j == traj.shape[1] - 1:
            ang_full = ang_full + calcQn((j - 1) * h, traj[:, j], h, t)
        else:
            ang_full = ang_full + calcQk((j - 1) * h, traj[:, j], h, t)
    return ang_full
