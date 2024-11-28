import matplotlib.pyplot as plt
import numpy as np
import rclpy
from bound_mpc_msg.msg import Vector
from bound_mpc_msg.srv._trajectory import Trajectory_Request
from pypoman import compute_polytope_vertices
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from ..BoundMPC.jerk_trajectory_casadi import calcAcceleration, calcAngle, calcVelocity
from .lie_functions import rodrigues_matrix


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


def create_obj_spheres_msg(t, centers, radii, ns="sphere", alpha=0.5):
    msg = MarkerArray()
    for i, (c, r) in enumerate(zip(centers, radii)):
        marker = Marker()
        marker.header.frame_id = "r1/world"
        marker.header.stamp = t
        d = rclpy.time.Duration(seconds=10000).to_msg()
        marker.lifetime = d
        marker.ns = f"{ns} {i}"
        marker.id = i
        marker.action = marker.ADD
        marker.type = marker.SPHERE
        marker.pose.position.x = c[0]
        marker.pose.position.y = c[1]
        marker.pose.position.z = c[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = r * 2
        marker.scale.y = r * 2
        marker.scale.z = r * 2
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = alpha
        msg.markers.append(marker)
    return msg


def get_ellipsoid_params(q_uncertainty):
    """Get the radii and the rotation of the ellipsoid based on the
    uncertainty matrix.

    Parameters
    ----------
    q_uncertainty : matrix 3x3
        Uncertainty matrix of ellipsoid

    Returns
    -------
    radii : vector 3x1
        Radii of the ellipsoid (based on the eigenvalues)
    rotation : matrix 3x3
        Rotation matrix of the ellipsoid
    """
    rotation, s, _ = np.linalg.svd(q_uncertainty)
    radii = np.sqrt(s)
    # Correct the rotation matrix if the determinant is -1 since then it is not
    # a valid rotation matrix but a reflection matrix
    if np.abs(np.linalg.det(rotation) + 1) < 1e-6:
        rotation *= -1
    return radii, rotation


def compute_initial_rot_errors(pr, pr_ref, dp_normed_ref, br1, br2):
    tauc = R.from_rotvec(pr).as_matrix()
    taud = R.from_rotvec(pr_ref).as_matrix()
    dtau_init = R.from_matrix(tauc @ taud.T).as_rotvec()

    dp_normed = dp_normed_ref
    r01 = np.zeros((3, 3))
    r01[:, 0] = br2
    r01[:, 1] = dp_normed
    r01[:, 2] = br1
    dtau_01 = r01.T @ R.from_rotvec(dtau_init).as_matrix() @ r01
    eul = R.from_matrix(dtau_01).as_euler("zyx")
    dtau_init_orth2 = eul[2] * br2
    dtau_init_orth1 = eul[0] * br1
    dtau_init_par = eul[1] * dp_normed

    return [dtau_init, dtau_init_par, dtau_init_orth1, dtau_init_orth2]


def create_traj_msg(
    p_via,
    r_via,
    e_p_start,
    e_p_end,
    e_p_mid,
    bp1_list,
    br1_list,
    s,
    e_r_start,
    e_r_end,
    e_r_mid,
    p0fk,
    q0,
    obj_centers,
    obj_radii,
    obstacles,
    sets_via,
    update=False,
):
    msg = Trajectory_Request()
    msg.update = update
    for i in range(len(p_via)):
        vec = Vector()
        vec.x = p_via[i].tolist()
        msg.p_via.append(vec)
        vec = Vector()
        vec.x = R.from_matrix(r_via[i]).as_rotvec().tolist()
        msg.r_via.append(vec)

    for i in range(len(p_via) - 1):
        vec = Vector()
        vec.x = bp1_list[i].tolist()
        msg.bp1.append(vec)
        vec = Vector()
        vec.x = br1_list[i].tolist()
        msg.br1.append(vec)

        vec = Vector()
        vec.x = e_p_start[i].tolist()
        msg.e_p_start.append(vec)
        vec = Vector()
        vec.x = e_p_end[i].tolist()
        msg.e_p_end.append(vec)
        vec = Vector()
        vec.x = e_p_mid[i].tolist()
        msg.e_p_mid.append(vec)

        vec = Vector()
        vec.x = e_r_start[i].tolist()
        msg.e_r_start.append(vec)
        vec = Vector()
        vec.x = e_r_end[i].tolist()
        msg.e_r_end.append(vec)
        vec = Vector()
        vec.x = e_r_mid[i].tolist()
        msg.e_r_mid.append(vec)

    for i in range(len(obj_centers)):
        vec = Vector()
        vec.x = obj_centers[i].tolist()
        msg.obj_centers.append(vec)
    for i in range(len(sets_via)):
        vec = Vector()
        vec.x = sets_via[i][0].flatten().tolist()
        msg.a_set.append(vec)
        vec = Vector()
        vec.x = sets_via[i][1].tolist()
        msg.b_set.append(vec)
    msg.obj_radii.x = obj_radii

    for i in range(len(obstacles)):
        vec = Vector()
        vec.x = obstacles[i]
        msg.obstacles.append(vec)

    msg.s.x = s
    msg.p0.x = p0fk.tolist()
    msg.q0.x = q0.tolist()
    return msg


def integrate_rotation_reference(pr_ref, omega, phi0, phi1):
    """Integrate the rotation reference by using the constant angular velocity
    omega over the interval phi1 - phi0.
    """
    r0 = R.from_rotvec(pr_ref).as_matrix()
    omega_norm = np.linalg.norm(omega)
    if omega_norm > 1e-4:
        dr = rodrigues_matrix(omega / omega_norm, (phi1 - phi0) * omega_norm)
        r1 = dr @ r0
    else:
        r1 = r0
    return R.from_matrix(r1).as_rotvec()


def project_position_bounds(corners_3d, p_ref, dp_ref, pidx, exact_points=True):
    """Project the positions bounds, given as corners of a rectangle, to the
    principle axis planes.
    """
    p = np.zeros(4)
    p_l = np.empty((corners_3d.shape[0], 2))
    p_u = np.empty((corners_3d.shape[0], 2))
    for i in range(corners_3d.shape[0]):
        # Get velocity orthogonal to the path in the plane
        vi = dp_ref[i, pidx] / np.linalg.norm(dp_ref[i, pidx])
        v_proj = np.array([vi[1], -vi[0]])
        # Project the corners onto the plane and onto the orthogonal velocity
        p[0] = np.dot(corners_3d[i, :3][pidx], v_proj)
        p[1] = np.dot(corners_3d[i, 3:6][pidx], v_proj)
        p[2] = np.dot(corners_3d[i, 6:9][pidx], v_proj)
        p[3] = np.dot(corners_3d[i, 9:12][pidx], v_proj)
        # Find the projected points by choosing the corner that is the most far
        # away or the highest projection on the orthogonal velocity vector
        if exact_points:
            idx = np.argmin(p)
            p_l[i, :] = corners_3d[i, 3 * idx : 3 * idx + 3][pidx]
            idx = np.argmax(p)
            p_u[i, :] = corners_3d[i, 3 * idx : 3 * idx + 3][pidx]
        else:
            p_l[i, :] = np.min(p) * v_proj
            p_u[i, :] = np.max(p) * v_proj
    # Add the reference
    p_l += p_ref[:, pidx]
    p_u += p_ref[:, pidx]
    return p_l, p_u


def move_robot_kinematic(robot_pub, t_ros, q_new):
    """Move the robot kinematically by just publishing the new joint state
    for Rviz (Only for visualization purposes).
    """
    robot_state_msg = JointState()
    robot_link_names = [
        "iiwa_joint_1",
        "iiwa_joint_2",
        "iiwa_joint_3",
        "iiwa_joint_4",
        "iiwa_joint_5",
        "iiwa_joint_6",
        "iiwa_joint_7",
    ]
    robot_state_msg.name = robot_link_names
    robot_state_msg.header.frame_id = "r1/world"
    robot_state_msg.header.stamp = t_ros
    robot_state_msg.position = q_new.tolist()
    robot_state_msg.velocity = np.zeros((7,)).tolist()
    robot_pub.publish(robot_state_msg)


def integrate_joint(model, jerk_matrix, q, dq, ddq, dt):
    qn = calcAngle(jerk_matrix, dt, q, dq, ddq, dt)
    dqn = calcVelocity(jerk_matrix, dt, dq, ddq, dt)
    ddqn = calcAcceleration(jerk_matrix, dt, ddq, dt)
    pn_lie, jac_fk, djac_fk = model.forward_kinematics(qn, dqn)
    ddjac_fk = model.ddjacobian_fk(q, dq, ddq)
    vn = jac_fk @ dqn
    an = djac_fk @ dqn + jac_fk @ ddqn
    jn = ddjac_fk @ dqn + 2 * djac_fk @ ddqn + jac_fk @ ddqn
    return (qn, dqn, ddqn, pn_lie, vn, an, jn)
