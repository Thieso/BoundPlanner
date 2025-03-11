from collections import namedtuple

import cdd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from ..BoundMPC.jerk_trajectory_casadi import calcAcceleration, calcAngle, calcVelocity

Params = namedtuple("Params", ["n", "dt", "build", "weights", "nr_segs"])


def get_default_params():
    w_p = 0.05
    w_r = 0.1
    w_v_p = 0.1
    w_v_r = 0.01
    w_speed = 0.5

    w_phi = 5.5 * w_speed
    w_dphi = 4.06

    scal = 0.5 / w_phi
    w_phi *= scal
    w_dphi *= scal

    w_dq = 0.001
    w_jerk = 0.0001
    w_term = 1.0
    w_slack = 10
    w_dslack = 500

    weights = np.array(
        [
            w_p,
            w_r,
            w_v_p,
            w_v_r,
            w_phi,
            w_dphi,
            w_dq,
            w_jerk,
            w_term,
            w_slack,
            w_dslack,
        ]
    )
    params = Params(n=10, dt=0.1, build=True, weights=weights, nr_segs=4)
    return params


def integrate_joint(model, jerk_matrix, q, dq, ddq, dt):
    qn = calcAngle(jerk_matrix, dt, q, dq, ddq, dt)
    dqn = calcVelocity(jerk_matrix, dt, dq, ddq, dt)
    ddqn = calcAcceleration(jerk_matrix, dt, ddq, dt)
    pn_lie, jac_fk, djac_fk = model.forward_kinematics(qn, dqn)
    ddjac_fk = 0 * djac_fk
    vn = jac_fk @ dqn
    vn = np.concatenate((model.velocity_ee(q, dq), model.omega_ee(q, dq)))
    an = djac_fk @ dqn + jac_fk @ ddqn
    jn = ddjac_fk @ dqn + 2 * djac_fk @ ddqn + jac_fk @ ddqn
    return (qn, dqn, ddqn, pn_lie, vn, an, jn)


def compute_polytope_vertices(a_set, b_set):
    b_set = b_set.reshape((b_set.shape[0], 1))
    array = np.hstack([b_set, -a_set])
    mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.INEQUALITY)
    poly = cdd.polyhedron_from_matrix(mat)
    g = cdd.copy_generators(poly)
    V = np.array(g.array)
    vertices = []
    for i in range(V.shape[0]):
        if V[i, 0] != 1:  # 1 = vertex, 0 = ray
            raise ValueError("Polyhedron is not a polytope")
        elif i not in g.lin_set:
            vertices.append(V[i, 1:])
    return vertices


def reduce_ineqs(a_set, b_set):
    b_set = b_set.reshape((b_set.shape[0], 1))
    array = np.hstack([b_set, -a_set])
    mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.INEQUALITY)
    cdd.matrix_redundancy_remove(mat)
    mat_np = np.array(mat.array)
    return [-mat_np[:, 1:], mat_np[:, 0]]


def plot_set(a_set, b_set, color=0):
    points = np.array(compute_polytope_vertices(a_set, b_set))
    hull = ConvexHull(points)
    faces = hull.simplices
    for face in faces:
        p1, p2, p3 = np.array(points)[face]
        dps = [[p1, p2], [p1, p3], [p2, p3]]
        for dp in dps:
            plt.plot(
                [dp[0][0], dp[1][0]],
                [dp[0][1], dp[1][1]],
                [dp[0][2], dp[1][2]],
                f"C{color}",
            )
    plt.axis("equal")


def gram_schmidt(v: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Do one Gram Schmidt orthogonolization step."""
    return b - gram_schmidt_w(v, b)


def gram_schmidt_w(v, b):
    """Project the vector b onto v. The jacobian matrix is used for
    orientation projections"""
    return (v.T @ b) * v


def normalize_set_size(sets, max_set_size=15):
    for set_iter in sets:
        a_norm = np.zeros((max_set_size, 3))
        b_norm = 10 * np.ones(max_set_size)
        set_size = set_iter[0].shape[0]
        if set_size <= max_set_size:
            a_norm[:set_size, :] = set_iter[0]
            b_norm[:set_size] = set_iter[1]
            set_iter[0] = a_norm
            set_iter[1] = b_norm
        else:
            print(
                f"(SetNormalizer) ERROR set size {set_size} exceeds max set size {max_set_size}"
            )
    return sets
