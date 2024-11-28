import numpy as np
import rclpy
from bound_mpc_msg.msg import MPCData
from rclpy.node import Node


class Logger(Node):
    def __init__(self, t0, N):
        super().__init__("logger")
        self.create_subscription(MPCData, "/mpc/mpc_data", self.callback_mpc_data, 0)
        self.i = 0
        self.t0 = t0
        self.phi = 0.0
        self.phi_max = 3.0
        self.log = False
        self.t_switch = []
        self.phi_switch = []
        self.fails = []
        self.sector = 0
        self.phi_switch_vector = []
        self.N = N

    def start(self):
        self.create_lists()
        self.log = True

    def stop(self):
        self.log = False

    def create_lists(self):
        # Create arrays for saving the values
        self.t_traj = []
        self.p_traj = []
        self.p_traj_horizon = [[] for _ in range(self.N - 1)]
        self.v_traj = []
        self.v_traj_real = []
        self.a_traj = []
        self.q_traj_real = []
        self.dq_traj_real = []
        self.q_traj = []
        self.q_traj_horizon = [[] for _ in range(self.N - 1)]
        self.dq_traj = []
        self.ddq_traj = []
        self.j_traj = []
        self.phi_traj = []
        self.phi_horizon = [[] for _ in range(self.N - 1)]
        self.dphi_traj = []
        self.dphi_horizon = [[] for _ in range(self.N - 1)]
        self.ddphi_traj = []
        self.dddphi_traj = []

        self.p_ref = []
        self.dp_ref = []
        self.dp_normed_ref = []
        self.dp_ref_horizon = [[] for _ in range(self.N - 1)]
        self.dp_normed_ref_horizon = [[] for _ in range(self.N - 1)]
        self.p_l = []
        self.p_u = []
        self.p_l_horizon = [[] for _ in range(self.N - 1)]
        self.p_u_horizon = [[] for _ in range(self.N - 1)]
        self.e_p_off = []
        self.e_r_off = []
        self.bp1 = []
        self.bp2 = []
        self.br1 = []
        self.br2 = []
        self.br1_horizon = [[] for _ in range(self.N - 1)]
        self.br2_horizon = [[] for _ in range(self.N - 1)]
        self.bp1_horizon = [[] for _ in range(self.N - 1)]
        self.bp2_horizon = [[] for _ in range(self.N - 1)]

        self.e = []
        self.e_par = []
        self.de_par = []
        self.de = []
        self.de_r = []
        self.e_orth = []
        self.e_orth_horizon = [[] for _ in range(self.N - 1)]
        self.e_r = []
        self.e_r_horizon = [[] for _ in range(self.N - 1)]
        self.e_r_par = []
        self.e_r_par_horizon = [[] for _ in range(self.N - 1)]
        self.e_r_orth1 = []
        self.e_r_orth2 = []
        self.e_r_orth1_horizon = [[] for _ in range(self.N - 1)]
        self.e_r_orth2_horizon = [[] for _ in range(self.N - 1)]
        self.de_orth = []
        self.iterations = []
        self.t_comp = []
        self.t_loops = []
        self.t_overhead = []
        self.fails = []

        self.a_set = []
        self.a_set_next = []
        self.a_set_j3 = []
        self.a_set_j5 = []
        self.a_set_j6 = []
        self.a_set_j67 = []
        self.a_set_elbow = []
        self.b_set = []
        self.b_set_next = []
        self.b_set_j3 = []
        self.b_set_j5 = []
        self.b_set_j6 = []
        self.b_set_j67 = []
        self.b_set_elbow = []

    def callback_mpc_data(self, msg):
        self.i += 1
        if self.log:
            self.phi = msg.phi.x[0]
            self.phi_max = msg.phi_max
            self.t_switch = np.array(msg.t_switch)
            self.phi_switch = np.array(msg.phi_switch)
            self.fails = np.array(msg.fails)
            t_traj = rclpy.time.Time.from_msg(msg.header.stamp)
            self.t_traj.append(t_traj.nanoseconds / 1e9 - self.t0)
            self.t_comp.append(msg.t_comp)
            self.t_loops.append(msg.t_loop)
            self.t_overhead.append(msg.t_overhead)
            self.iterations.append(msg.iterations)
            self.p_traj.append(np.array(msg.p[0].x))
            self.v_traj.append(np.array(msg.v[0].x))
            self.a_traj.append(np.array(msg.a[0].x))
            self.q_traj.append(np.array(msg.q[0].x))
            self.dq_traj.append(np.array(msg.dq[0].x))
            self.ddq_traj.append(np.array(msg.ddq[0].x))
            self.j_traj.append(np.array(msg.dddq[0].x))
            self.phi_traj.append(np.array(msg.phi.x[0]))
            self.dphi_traj.append(np.array(msg.dphi.x[0]))
            self.ddphi_traj.append(np.array(msg.ddphi.x[0]))
            self.dddphi_traj.append(np.array(msg.dddphi.x[0]))

            self.sector = msg.sector
            self.phi_switch_vector = np.array(msg.phi_switch_vector.x)

            if len(msg.e_p) > 0:
                self.a_set.append(np.array(msg.a_set.x))
                self.a_set_next.append(np.array(msg.a_set_next.x))
                self.a_set_j3.append(np.array(msg.a_set_j3.x))
                self.a_set_j5.append(np.array(msg.a_set_j5.x))
                self.a_set_j6.append(np.array(msg.a_set_j6.x))
                self.a_set_j67.append(np.array(msg.a_set_j67.x))
                self.a_set_elbow.append(np.array(msg.a_set_elbow.x))
                self.b_set.append(np.array(msg.b_set.x))
                self.b_set_next.append(np.array(msg.b_set_next.x))
                self.b_set_j3.append(np.array(msg.b_set_j3.x))
                self.b_set_j5.append(np.array(msg.b_set_j5.x))
                self.b_set_j6.append(np.array(msg.b_set_j6.x))
                self.b_set_j67.append(np.array(msg.b_set_j67.x))
                self.b_set_elbow.append(np.array(msg.b_set_elbow.x))

            err_data = False
            if len(msg.e_p) > 0:
                err_data = True
                self.e.append(np.array(msg.e_p[0].x))
                self.e_par.append(np.array(msg.e_p_par[0].x))
                self.de_par.append(np.array(msg.de_p_par[0].x))
                self.de.append(np.array(msg.de_p[0].x))
                self.de_r.append(np.array(msg.de_r[0].x))
                self.e_orth.append(np.array(msg.e_p_orth[0].x))
                self.e_r.append(np.array(msg.e_r[0].x))
                self.e_r_par.append(np.array(msg.e_r_par[0].x))
                self.e_r_orth1.append(np.array(msg.e_r_orth1[0].x))
                self.e_r_orth2.append(np.array(msg.e_r_orth2[0].x))
                self.de_orth.append(np.array(msg.de_p_orth[0].x))

            ref_data = False
            if len(msg.p_ref) > 0:
                ref_data = True
                self.p_ref.append(np.array(msg.p_ref[0].x))
                self.dp_ref.append(np.array(msg.dp_ref[0].x))
                self.dp_normed_ref.append(np.array(msg.dp_normed_ref[0].x))
                self.p_l.append(np.array(msg.p_lower[0].x))
                self.p_u.append(np.array(msg.p_upper[0].x))
                self.e_p_off.append(np.array(msg.e_p_off[0].x))
                self.e_r_off.append(np.array(msg.e_r_off[0].x))
                self.bp1.append(np.array(msg.bp1[0].x))
                self.bp2.append(np.array(msg.bp2[0].x))
                self.br1.append(np.array(msg.br1[0].x))
                self.br2.append(np.array(msg.br2[0].x))

            for i in range(self.N - 1):
                if i >= len(msg.phi.x):
                    self.phi_horizon[i].append(self.phi_horizon[i][-1])
                    self.dphi_horizon[i].append(self.dphi_horizon[i][-1])
                    self.p_traj_horizon[i].append(self.p_traj_horizon[i][-1])
                    self.q_traj_horizon[i].append(self.q_traj_horizon[i][-1])
                    if err_data:
                        self.e_orth_horizon[i].append(self.e_orth_horizon[i][-1])
                        self.e_r_horizon[i].append(self.e_r_horizon[i][-1])
                        self.e_r_par_horizon[i].append(self.e_r_par_horizon[i][-1])
                        self.e_r_orth1_horizon[i].append(self.e_r_orth1_horizon[i][-1])
                        self.e_r_orth2_horizon[i].append(self.e_r_orth2_horizon[i][-1])
                    if ref_data:
                        self.dp_ref_horizon[i].append(self.dp_ref_horizon[i][-1])
                        self.dp_normed_ref_horizon[i].append(
                            self.dp_normed_ref_horizon[i][-1]
                        )
                        self.p_l_horizon[i].append(self.p_l_horizon[i][-1])
                        self.p_u_horizon[i].append(self.p_u_horizon[i][-1])
                        self.bp1_horizon[i].append(self.bp1_horizon[i][-1])
                        self.bp2_horizon[i].append(self.bp2_horizon[i][-1])
                        self.br1_horizon[i].append(self.br1_horizon[i][-1])
                        self.br2_horizon[i].append(self.br2_horizon[i][-1])
                else:
                    self.phi_horizon[i].append(np.array(msg.phi.x[i]))
                    self.dphi_horizon[i].append(np.array(msg.dphi.x[i]))
                    self.p_traj_horizon[i].append(np.array(msg.p[i].x))
                    self.q_traj_horizon[i].append(np.array(msg.q[i].x))
                    if err_data:
                        self.e_orth_horizon[i].append(np.array(msg.e_p_orth[i].x))
                        self.e_r_horizon[i].append(np.array(msg.e_r[i].x))
                        self.e_r_par_horizon[i].append(np.array(msg.e_r_par[i].x))
                        self.e_r_orth1_horizon[i].append(np.array(msg.e_r_orth1[i].x))
                        self.e_r_orth2_horizon[i].append(np.array(msg.e_r_orth2[i].x))
                    if ref_data:
                        self.dp_ref_horizon[i].append(np.array(msg.dp_ref[i].x))
                        self.dp_normed_ref_horizon[i].append(
                            np.array(msg.dp_normed_ref[i].x)
                        )
                        self.p_l_horizon[i].append(np.array(msg.p_lower[i].x))
                        self.p_u_horizon[i].append(np.array(msg.p_upper[i].x))
                        self.bp1_horizon[i].append(np.array(msg.bp1[i].x))
                        self.bp2_horizon[i].append(np.array(msg.bp2[i].x))
                        self.br1_horizon[i].append(np.array(msg.br1[i].x))
                        self.br2_horizon[i].append(np.array(msg.br2[i].x))


def main():
    rclpy.init()
    node = Logger()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
