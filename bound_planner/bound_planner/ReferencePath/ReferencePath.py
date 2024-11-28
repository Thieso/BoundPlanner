import numpy as np
from scipy.spatial.transform import Rotation as R

from ..utils import gram_schmidt
from .euler_spiral import create_euler_sprial


class ReferencePath:
    """
    Reference Path class
    """

    def __init__(
        self,
        p,
        r,
        bp1,
        br1,
        e_p_start,
        e_p_end,
        e_p_mid,
        e_r_start,
        e_r_end,
        e_r_mid,
        a_sets,
        b_sets,
        nr_segs=2,
        phi_bias=0,
    ):
        # Position and orientation points
        self.p = p
        self.r = r
        l_traj = len(self.p)
        self.num_sectors = l_traj - 2

        # Length of euler spiral
        self.euler_spiral_length = 0.10

        # Number of linear segments
        self.nr_segs = nr_segs

        # Init a bias path parameter in case of replanning
        self.phi_bias = phi_bias

        # Switched flag indicates whether the next path element is used
        self.switched = True

        # Bound parameters
        self.e_p_start = e_p_start
        self.e_p_end = e_p_end
        self.e_p_mid = e_p_mid
        self.e_r_start = e_r_start
        self.e_r_end = e_r_end
        self.e_r_mid = e_r_mid
        self.a_sets = a_sets
        self.b_sets = b_sets
        for i in range(nr_segs - 1):
            self.e_p_start.append(self.e_p_end[-1])
            self.e_p_end.append(self.e_p_end[-1])
            self.e_p_mid.append(self.e_p_end[-1])
            self.e_r_start.append(self.e_r_end[-1])
            self.e_r_end.append(self.e_r_end[-1])
            self.e_r_mid.append(self.e_r_end[-1])
            self.a_sets.append(self.a_sets[-1])
            self.b_sets.append(self.b_sets[-1])

        # Which sector of the path is currently used
        self.sector = 0

        self.dr = []
        self.dr_normed = []
        self.iw = [np.zeros(3)]
        self.dr_limit = []
        omega_prev = np.array([0, 1.0, 0])
        for i in range(1, l_traj):
            drot = R.from_matrix(self.r[i] @ self.r[i - 1].T).as_rotvec()
            self.dr.append(drot)
            norm_dr = np.linalg.norm(drot)
            if norm_dr > 1e-4:
                self.dr_normed.append(drot / norm_dr)
                # Do not change the projection axis when just changing direction
                if np.linalg.norm(omega_prev + self.dr_normed[-1]) < 1e-4:
                    self.dr_normed[-1] *= -1
            else:
                self.dr_normed.append(omega_prev)
            omega_prev = np.copy(self.dr_normed[-1])
            self.iw.append(self.iw[i - 1] + self.dr[i - 1])
        for i in range(nr_segs - 1):
            self.dr.append(np.array(self.dr[-1]))
            self.dr_normed.append(self.dr_normed[-1])
            self.iw.append(self.iw[-1])
            self.r.append(self.r[-1])

        self.dp = []
        for i in range(1, l_traj):
            self.dp.append(self.p[i] - self.p[i - 1])
            if np.linalg.norm(self.dp[-1]) < 1e-3:
                if i > 1:
                    self.dp[-1] = self.dp[-2]
                else:
                    self.dp[-1] = np.array([0, 1.0, 0])
        for i in range(nr_segs - 1):
            self.p.append(self.p[-1])
            self.dp.append(self.dp[-1])

        # Compute the switching points and the arc length based on the length
        self.phi = [0]
        l = []
        l_total = 0
        for i in range(1, l_traj):
            li = np.linalg.norm(self.p[i] - self.p[i - 1])
            # If there is no change in position than there is most likely change
            # in orientation and then we need some path parameter there
            if np.linalg.norm(li) < 1e-3:
                li = np.linalg.norm(self.dr[i - 1]) / np.pi
            l.append(li)
            l_total += li
        for i in range(l_traj - 1):
            self.phi.append(l[i])
        for i in range(nr_segs - 1):
            self.phi.append(1)
        self.phi_max = l_total + self.phi_bias

        # Basis vectors for orthogonal plane
        self.bp1 = bp1
        self.br1 = br1
        self.bp2 = []
        self.br2 = []
        for i in range(len(self.bp1)):
            dp_normed = self.dp[i] / np.linalg.norm(self.dp[i])
            self.bp1[i] = gram_schmidt(dp_normed, self.bp1[i])

            orth_check = self.bp1[i] @ self.dp[i]
            if np.abs(orth_check) > 1e-6:
                print(f"[WARNING] Pos Basis vector {i} not orthogonal on path")
            if np.linalg.norm(self.bp1[i]) < 1e-3:
                print(f"[WARNING] Pos Basis vector {i} is too close to direction")

            self.bp1[i] = self.bp1[i] / np.linalg.norm(self.bp1[i])
            self.bp2.append(np.cross(dp_normed, self.bp1[i]))
            self.bp2[-1] /= np.linalg.norm(self.bp2[-1])

        for i in range(len(self.bp1)):
            self.br1[i] = gram_schmidt(self.dr_normed[i], self.br1[i])

            orth_check = self.br1[i] @ self.dr[i]
            if np.abs(orth_check) > 1e-6:
                print(f"[WARNING] Rot Basis vector {i} not orthogonal on path")
            if np.linalg.norm(self.br1[i]) < 1e-3:
                print(f"[WARNING] Rot Basis vector {i} is too close to direction")

            self.br1[i] = self.br1[i] / np.linalg.norm(self.br1[i])
            self.br2.append(np.cross(self.dr_normed[i], self.br1[i]))
            self.br2[-1] /= np.linalg.norm(self.br2[-1])

        for i in range(nr_segs - 1):
            self.bp1.append(self.bp1[-1])
            self.br1.append(self.br1[-1])
            self.bp2.append(self.bp2[-1])
            self.br2.append(self.br2[-1])

        # Compute rotation vectors of the basis vectors
        self.rot_bp1 = [np.zeros(3)]
        self.rot_bp1_norm = [0.0]
        self.rot_bp2_norm = [0.0]
        for i in range(len(self.dp) - 1):
            dp0_normed = self.dp[i] / np.linalg.norm(self.dp[i])
            dp1_normed = self.dp[i + 1] / np.linalg.norm(self.dp[i + 1])
            vec1 = np.cross(dp0_normed, dp1_normed)
            if np.linalg.norm(vec1) > 1e-3:
                vec1 /= np.linalg.norm(vec1)
            else:
                vec1 = np.array([0, 0, 1.0])
            norm1 = np.arccos(np.min((1, np.dot(dp0_normed, dp1_normed))))
            rot1 = R.from_rotvec(vec1 * norm1).as_matrix()
            norm2 = -np.arccos(np.min((1, np.dot(rot1 @ self.bp1[i], self.bp1[i + 1]))))

            # Check if you turn the correct way
            rbp1 = rot1 @ self.bp1[i]
            rbp2 = self.bp1[i + 1]
            rot2 = R.from_rotvec(dp1_normed * norm2).as_matrix()
            # rot2m = R.from_rotvec(dp1_normed * -norm2).as_matrix()
            if np.linalg.norm(rbp2 - rot2 @ rbp1) > 1e-6:
                norm2 *= -1
            # print(rbp2 - rot2 @ rbp1)
            # print(rbp2 - rot2m @ rbp1)
            if np.abs(norm1) > 1e-3:
                self.rot_bp1.append(vec1)
                self.rot_bp1_norm.append(norm1)
            else:
                self.rot_bp1.append(np.zeros(3))
                self.rot_bp1_norm.append(0.0)
            if np.abs(norm2) > 1e-3:
                self.rot_bp2_norm.append(norm2)
            else:
                self.rot_bp2_norm.append(0.0)

        # Euler spiral parameters
        self.spiral_a = np.zeros(len(self.dp))
        self.spiral_l = np.ones(len(self.dp)) * 0.001
        self.spiral_theta = np.zeros(len(self.dp))
        dp0_normed = self.dp[0] / np.linalg.norm(self.dp[0])
        dp1_normed = self.dp[1] / np.linalg.norm(self.dp[1])
        self.spiral_rot_2d = [np.vstack((dp0_normed, dp1_normed))] * len(self.dp)
        for i in range(len(self.dp) - 1):
            l_seg = self.phi[i + 1]
            l_seg2 = self.phi[i + 2]
            l_seg_min = np.min((l_seg, l_seg2))
            reduction = np.inf
            k = 0
            while reduction > l_seg_min / 2 or self.spiral_l[i + 1] > l_seg_min / 2:
                self.spiral_l[i + 1] = self.euler_spiral_length - k * 0.01
                if self.spiral_l[i + 1] < 1e-8:
                    print("Spiral length 0")
                    self.spiral_l[i + 1] = 0.05
                    a = 0.0
                    theta = 0.0
                    shortenings = [0.0, 0.0]
                    offset = 0.0
                    break
                a, b, theta, rot_2d, offset, shortenings = create_euler_sprial(
                    self.dp[i], self.dp[i + 1], self.spiral_l[i + 1]
                )
                reduction = shortenings[1] + shortenings[0] - offset
                k += 1
            # print(f"Offset {offset*1000:.1f}mm")
            # print(f"Shortening 1 {shortenings[0]*1000:.1f}mm")
            # print(f"Shortening 2 {shortenings[1]*1000:.1f}mm")
            print("Spiral l ", i, " ", self.spiral_l[i + 1])
            self.spiral_a[i + 1] = a
            self.spiral_theta[i + 1] = theta
            self.spiral_rot_2d[i + 1] = rot_2d

            # Correct the length of each segments based on the shorter spirals
            # and based on the offset
            self.phi[i + 1] -= offset
            self.phi[i + 2] += offset
            if np.max(shortenings) > l_seg / 2:
                pass
            if i < len(self.dp) - 1:
                self.phi[i + 2] -= shortenings[1] + shortenings[0]
            if self.phi[i + 2] < 0:
                print("ERROR euler angles too short")
                print(self.dp[i], self.dp[i + 1])
                raise RuntimeError
            self.phi_max -= shortenings[1] + shortenings[0]
        print(self.phi)

        # Scale the angular velocity to match it to the desired phi values
        for i in range(l_traj):
            self.dr[i] = self.dr[i] / self.phi[i + 1]

        # Initialize the reference parametrization
        self.pd = np.zeros((6, self.nr_segs))
        self.dpd = np.zeros((6, self.nr_segs))
        self.dpd_normed = np.zeros((3, self.nr_segs))
        self.ddpd = np.zeros((6, self.nr_segs))
        self.phi_switch = np.ones((self.nr_segs + 1,)) * self.phi_bias

        for i in range(self.nr_segs):
            self.set_point(i)

    def find_largest_proj(self, b_prev, b, p1, p2):
        sign = np.sign(np.dot(b_prev, b))
        max_proj = np.max(np.abs((np.dot(p1, b), np.dot(p2, b))))
        return sign * max_proj

    def set_point(self, idx):
        self.pd[:3, idx] = self.p[self.sector + idx]
        self.pd[3:, idx] = self.iw[self.sector + idx]
        # self.dpd[:3, idx] = self.dp[self.sector+idx] / self.phi[self.sector+idx+1]
        self.dpd[:3, idx] = self.dp[self.sector + idx] / np.linalg.norm(
            self.dp[self.sector + idx]
        )
        self.dpd[3:, idx] = self.dr[self.sector + idx]
        self.dpd_normed[:, idx] = self.dr_normed[self.sector + idx]
        self.phi_switch[idx + 1] = (
            np.array(self.phi).cumsum()[self.sector + idx + 1] + self.phi_bias
        )

    def update(self, switch, phi_current):
        # If the current phi is larger than the switching phi, update the
        # parameters
        if self.sector >= self.num_sectors or not switch:
            self.switched = False
            return
        else:
            self.switched = True
            self.sector += 1

            # Shift the segements to the front
            for i in range(self.nr_segs - 1):
                self.pd[:, i] = np.copy(self.pd[:, i + 1])
                self.dpd[:, i] = np.copy(self.dpd[:, i + 1])
                self.dpd_normed[:, i] = np.copy(self.dpd_normed[:, i + 1])
                self.phi_switch[i] = np.copy(self.phi_switch[i + 1])
            self.phi_switch[self.nr_segs - 1] = (
                np.copy(self.phi_switch[self.nr_segs]) + self.phi_bias
            )

            # Set new segment
            self.set_point(self.nr_segs - 1)

    def compute_phis(self):
        self.phi_switch = np.array(self.phi).cumsum()[self.sector]
        if self.sector + 2 < len(self.phi):
            self.phi_switch_next = np.array([sum(self.phi[: self.sector + 2])])
        else:
            self.phi_switch_next = np.array([self.phi_max])

    def get_parameters(self, switch, phi_current):
        self.update(switch, phi_current)
        return self.pd, self.dpd_normed, self.dpd, self.ddpd, self.phi_switch

    def get_basis_vectors(self):
        bp1 = np.array(self.bp1[self.sector : self.sector + self.nr_segs]).T
        bp2 = np.array(self.bp2[self.sector : self.sector + self.nr_segs]).T
        br1 = np.array(self.br1[self.sector : self.sector + self.nr_segs]).T
        br2 = np.array(self.br2[self.sector : self.sector + self.nr_segs]).T
        return bp1, bp2, br1, br2

    def get_bound_params(self):
        e_p_start = np.array(self.e_p_start[self.sector : self.sector + self.nr_segs])
        e_p_end = np.array(self.e_p_end[self.sector : self.sector + self.nr_segs])
        e_p_mid = np.array(self.e_p_mid[self.sector : self.sector + self.nr_segs])
        e_r_start = np.array(self.e_r_start[self.sector : self.sector + self.nr_segs])
        e_r_end = np.array(self.e_r_end[self.sector : self.sector + self.nr_segs])
        e_r_mid = np.array(self.e_r_mid[self.sector : self.sector + self.nr_segs])
        a_set = np.array(self.a_sets[self.sector : self.sector + self.nr_segs])
        b_set = np.array(self.b_sets[self.sector : self.sector + self.nr_segs])
        return e_p_start, e_p_end, e_p_mid, e_r_start, e_r_end, e_r_mid, a_set, b_set

    def get_spiral_params(self):
        a = np.array(self.spiral_a[self.sector : self.sector + self.nr_segs])
        l_spiral = np.array(self.spiral_l[self.sector : self.sector + self.nr_segs])
        theta = np.array(self.spiral_theta[self.sector : self.sector + self.nr_segs])
        rot_2d = np.array(self.spiral_rot_2d[self.sector : self.sector + self.nr_segs])
        rot_bp1 = np.array(self.rot_bp1[self.sector : self.sector + self.nr_segs])
        rot_bp1_norm = np.array(
            self.rot_bp1_norm[self.sector : self.sector + self.nr_segs]
        )
        rot_bp2_norm = np.array(
            self.rot_bp2_norm[self.sector : self.sector + self.nr_segs]
        )
        return a, l_spiral, theta, rot_2d, rot_bp1, rot_bp1_norm, rot_bp2_norm
