import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation as R


class RobotModel:
    def __init__(self):
        # Geometric length of the links
        self.d1 = 0.1575
        self.d2 = 0.2025
        self.d3 = 0.2375
        self.d4 = 0.1825
        self.d5 = 0.2175
        self.d6 = 0.1825
        self.d7 = 0.081
        self.d8 = 0.071 + 0.145
        # d8 = 0.071

        # Joint limits
        self.q_lim_lower = [
            -165 * np.pi / 180,
            -115 * np.pi / 180,
            -165 * np.pi / 180,
            -115 * np.pi / 180,
            -165 * np.pi / 180,
            -115 * np.pi / 180,
            -170 * np.pi / 180,
        ]
        self.q_lim_upper = [
            165 * np.pi / 180,
            115 * np.pi / 180,
            165 * np.pi / 180,
            115 * np.pi / 180,
            165 * np.pi / 180,
            115 * np.pi / 180,
            170 * np.pi / 180,
        ]
        self.dq_lim_lower = [
            -85 * np.pi / 180,
            -85 * np.pi / 180,
            -100 * np.pi / 180,
            -75 * np.pi / 180,
            -130 * np.pi / 180,
            -135 * np.pi / 180,
            -135 * np.pi / 180,
        ]
        self.dq_lim_upper = [
            85 * np.pi / 180,
            85 * np.pi / 180,
            100 * np.pi / 180,
            75 * np.pi / 180,
            130 * np.pi / 180,
            135 * np.pi / 180,
            135 * np.pi / 180,
        ]
        self.tau_lim_lower = [-320, -320, -176, -176, -110, -40, -40]
        self.tau_lim_upper = [320, 320, 176, 176, 110, 40, 40]
        self.u_max = 35
        self.u_min = -35

        self.setup_ik_problem()

    def get_robot_limits(self):
        return (
            self.q_lim_upper,
            self.q_lim_lower,
            self.dq_lim_upper,
            self.dq_lim_lower,
            self.tau_lim_upper,
            self.tau_lim_lower,
            self.u_max,
            self.u_min,
        )

    def forward_kinematics(self, q: np.ndarray, dq: np.ndarray):
        """Compute forward kinematics to get the position of the end effector in
        cartesian space. Also computes the jacobian and its derivative.
        """
        jac_ee = self.jacobian_fk(q)
        djac_ee = self.djacobian_fk(q, dq)
        p_robot = self.fk(q)
        return p_robot, jac_ee, djac_ee

    def get_robot_params(self):
        return self.d1, self.d2, self.d3, self.d4, self.d5, self.d6, self.d7, self.d8

    def setup_ik_problem(self):
        q = ca.SX.sym("q", 7)
        pd = ca.SX.sym("p desired", 3)
        rd = ca.SX.sym("r desired", 3, 3)
        r_ee = self.hom_transform_endeffector(q)[:3, :3]
        J = ca.sumsqr(self.fk_pos(q) - pd[:3])
        J += ca.sumsqr(r_ee @ rd.T - ca.SX.eye(3))
        w = [q]
        lbw = self.q_lim_lower
        ubw = self.q_lim_upper

        params = ca.vertcat(pd, rd.reshape((-1, 1)))

        prob = {
            "f": J,
            "x": ca.vertcat(*w),
            # 'g': ca.vertcat(*g),
            "p": params,
        }
        self.lbu = lbw
        self.ubu = ubw
        # lbg = lbg
        # ubg = ubg
        ipopt_options = {
            "tol": 10e-4,
            "max_iter": 500,
            "limited_memory_max_history": 6,
            "limited_memory_initialization": "scalar1",
            "limited_memory_max_skipping": 2,
            "linear_solver": "ma57",
            "linear_system_scaling": "mc19",
            "ma57_automatic_scaling": "no",
            "ma57_pre_alloc": 100,
            "mu_strategy": "adaptive",
            "adaptive_mu_globalization": "kkt-error",
            "print_info_string": "no",
            "fast_step_computation": "yes",
            "warm_start_init_point": "yes",
            "mu_oracle": "loqo",
            "fixed_mu_oracle": "quality-function",
            "line_search_method": "filter",
            "expect_infeasible_problem": "no",
            "print_level": 0,
        }

        solver_opts = {
            "verbose": False,
            "verbose_init": False,
            "print_time": False,
            "ipopt": ipopt_options,
        }
        self.ik_solver = ca.nlpsol("solver", "ipopt", prob, solver_opts)

    def inverse_kinematics(self, pd, rd, q0):
        """Inverse kinematics based on optimization."""
        params = np.concatenate((pd, rd.T.flatten()))
        sol = self.ik_solver(x0=q0, lbx=self.lbu, ubx=self.ubu, p=params)
        q_ik = np.array(sol["x"]).flatten()
        if not self.ik_solver.stats()["success"]:
            print("(IK) ERROR No convergence in IK optimization")
        h_ik = self.hom_transform_endeffector(q_ik)
        pos_error = np.linalg.norm(pd - h_ik[:3, 3])
        rot_error = np.linalg.norm(R.from_matrix(h_ik[:3, :3] @ rd.T).as_rotvec())
        print(f"(IK) Position error {pos_error}m")
        print(f"(IK) Rotation error {rot_error * 180 / np.pi} deg")
        return q_ik

    def fk_pos(self, q):
        """Compute the end effector position of the robot in cartesian space
        given the joint configuration.
        """
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()
        if isinstance(q, np.ndarray):
            m = np.zeros(3)
        else:
            m = ca.SX.zeros(3)
        t776 = d7 + d8
        t777 = q[5]
        t788 = ca.sin(t777) * t776
        t764 = t776 * ca.cos(t777) + d5 + d6
        t779 = q[3]
        t767 = ca.sin(t779)
        t787 = t767 * t764
        t778 = q[4]
        t766 = ca.sin(t778)
        t780 = q[2]
        t768 = ca.sin(t780)
        t786 = t768 * t766
        t771 = ca.cos(t778)
        t772 = ca.cos(t779)
        t785 = t771 * t772
        t784 = t764 * t772 + d3 + d4
        t783 = t771 * t788
        t782 = q[0]
        t781 = q[1]
        t775 = ca.cos(t782)
        t774 = ca.cos(t781)
        t773 = ca.cos(t780)
        t770 = ca.sin(t782)
        t769 = ca.sin(t781)
        t762 = -t768 * t787 + (t766 * t773 + t768 * t785) * t788
        t761 = (
            (-t764 * t773 * t774 + t769 * t783) * t767
            + t774 * (t773 * t785 - t786) * t788
            + t784 * t769
        )
        m[0] = t761 * t775 - t762 * t770
        m[1] = t761 * t770 + t762 * t775
        m[2] = (
            (t767 * t783 + t784) * t774
            + ((-t772 * t783 + t787) * t773 + t786 * t788) * t769
            + d1
            + d2
        )
        return m

    def fk_pos_g1(self, q, d9=0.07):
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()
        if isinstance(q, np.ndarray):
            m = np.zeros(3)
        else:
            m = ca.SX.zeros(3)

        t35 = q[4]
        t34 = q[5]
        t33 = q[6]
        t19 = q[2]
        t10 = ca.sin(t19)
        t25 = ca.sin(t33)
        t28 = ca.cos(t34)
        t22 = t28 * t25
        t26 = ca.sin(t35)
        t27 = ca.cos(t33)
        t29 = ca.cos(t35)
        t17 = d7 + d8
        t8 = ca.sin(t34)
        t31 = t17 * t8
        t3 = t26 * t31 + d9 * (-t22 * t26 + t27 * t29)
        t32 = t10 * t3
        t18 = q[3]
        t13 = ca.cos(t18)
        t4 = t29 * t31 - d9 * (t22 * t29 + t26 * t27)
        t30 = t4 * t13
        t7 = d9 * t25 * t8 + t17 * t28 + d5 + d6
        t24 = t13 * t7 + d3 + d4
        t9 = ca.sin(t18)
        t23 = -t7 * t9 + t30
        t21 = q[0]
        t20 = q[1]
        t16 = ca.cos(t21)
        t15 = ca.cos(t20)
        t14 = ca.cos(t19)
        t12 = ca.sin(t21)
        t11 = ca.sin(t20)
        t2 = t10 * t23 + t14 * t3
        t1 = (-t14 * t15 * t7 + t11 * t4) * t9 + (t14 * t30 - t32) * t15 + t24 * t11
        m[0] = t1 * t16 - t12 * t2
        m[2] = (-t14 * t23 + t32) * t11 + (t4 * t9 + t24) * t15 + d1 + d2
        m[1] = t1 * t12 + t16 * t2
        return m

    def fk_pos_elbow(self, q):
        """Compute the elbow position of the robot in cartesian space
        given the joint configuration.
        """
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()
        if isinstance(q, np.ndarray):
            m = np.zeros(3)
        else:
            m = ca.SX.zeros(3)
        t280 = d3 + d4
        t283 = ca.sin(q[1]) * t280
        m[0] = ca.cos(q[0]) * t283
        m[1] = ca.sin(q[0]) * t283
        m[2] = t280 * ca.cos(q[1]) + d1 + d2
        return m

    def fk_pos_j3(self, q):
        """Compute the elbow position of the robot in cartesian space
        given the joint configuration.
        """
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()
        if isinstance(q, np.ndarray):
            m = np.zeros(3)
        else:
            m = ca.SX.zeros(3)
        t481 = q[1]
        t483 = d3 * ca.sin(t481)
        t482 = q[0]
        m[0] = ca.cos(t482) * t483
        m[1] = ca.sin(t482) * t483
        m[2] = ca.cos(t481) * d3 + d2 + d1
        return m

    def fk_pos_j5(self, q):
        """Compute the elbow position of the robot in cartesian space
        given the joint configuration.
        """
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()
        if isinstance(q, np.ndarray):
            m = np.zeros(3)
        else:
            m = ca.SX.zeros(3)
        t493 = q[3]
        t499 = d5 * ca.sin(t493)
        t494 = q[2]
        t498 = ca.sin(t494) * t499
        t497 = ca.cos(t494) * t499
        t496 = q[0]
        t495 = q[1]
        t492 = ca.cos(t496)
        t491 = ca.cos(t495)
        t489 = ca.sin(t496)
        t488 = ca.sin(t495)
        t485 = ca.cos(t493) * d5 + d3 + d4
        t484 = t488 * t485 - t491 * t497
        m[2] = t491 * t485 + t488 * t497 + d1 + d2
        m[1] = t484 * t489 - t492 * t498
        m[0] = t492 * t484 + t489 * t498
        return m

    def fk_pos_j6(self, q):
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()
        if isinstance(q, np.ndarray):
            m = np.zeros(3)
        else:
            m = ca.SX.zeros(3)
        t10 = d5 + d6
        t11 = q[3]
        t17 = t10 * ca.sin(t11)
        t12 = q[2]
        t16 = ca.sin(t12) * t17
        t15 = ca.cos(t12) * t17
        t14 = q[0]
        t13 = q[1]
        t9 = ca.cos(t14)
        t8 = ca.cos(t13)
        t6 = ca.sin(t14)
        t5 = ca.sin(t13)
        t2 = t10 * ca.cos(t11) + d3 + d4
        t1 = -t15 * t8 + t2 * t5
        m[0] = t1 * t9 + t16 * t6
        m[1] = t6 * t1 - t16 * t9
        m[2] = t15 * t5 + t2 * t8 + d1 + d2
        return m

    def fk_pos_j67(self, q, scal=1.3):
        """Compute the end effector position of the robot in cartesian space
        given the joint configuration.
        """
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()
        if isinstance(q, np.ndarray):
            m = np.zeros(3)
        else:
            m = ca.SX.zeros(3)
        t776 = scal * d7
        t777 = q[5]
        t788 = ca.sin(t777) * t776
        t764 = t776 * ca.cos(t777) + d5 + d6
        t779 = q[3]
        t767 = ca.sin(t779)
        t787 = t767 * t764
        t778 = q[4]
        t766 = ca.sin(t778)
        t780 = q[2]
        t768 = ca.sin(t780)
        t786 = t768 * t766
        t771 = ca.cos(t778)
        t772 = ca.cos(t779)
        t785 = t771 * t772
        t784 = t764 * t772 + d3 + d4
        t783 = t771 * t788
        t782 = q[0]
        t781 = q[1]
        t775 = ca.cos(t782)
        t774 = ca.cos(t781)
        t773 = ca.cos(t780)
        t770 = ca.sin(t782)
        t769 = ca.sin(t781)
        t762 = -t768 * t787 + (t766 * t773 + t768 * t785) * t788
        t761 = (
            (-t764 * t773 * t774 + t769 * t783) * t767
            + t774 * (t773 * t785 - t786) * t788
            + t784 * t769
        )
        m[0] = t761 * t775 - t762 * t770
        m[1] = t761 * t770 + t762 * t775
        m[2] = (
            (t767 * t783 + t784) * t774
            + ((-t772 * t783 + t787) * t773 + t786 * t788) * t769
            + d1
            + d2
        )
        return m

    def fk(self, q):
        """Compute the end effector position of the robot in cartesian space
        given the joint configuration.
        """
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()

        if isinstance(q, np.ndarray):
            m = np.zeros(6)
        else:
            m = ca.SX.zeros(6)
        h = self.hom_transform_endeffector(q)
        m[:3] = h[:3, 3]
        m[3:] = R.from_matrix(h[:3, :3]).as_rotvec()

        return m

    def hom_transform_endeffector(self, q):
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()
        if isinstance(q, np.ndarray):
            m = np.zeros((4, 4))
        else:
            m = ca.SX.zeros((4, 4))
        t98 = q[2]
        t83 = np.sin(t98)
        t95 = q[5]
        t87 = np.cos(t95)
        t109 = t83 * t87
        t96 = q[4]
        t88 = np.cos(t96)
        t97 = q[3]
        t89 = np.cos(t97)
        t106 = t88 * t89
        t80 = np.sin(t95)
        t82 = np.sin(t97)
        t74 = t106 * t87 + t80 * t82
        t81 = np.sin(t96)
        t90 = np.cos(t98)
        t69 = t109 * t81 - t74 * t90
        t111 = t82 * t88
        t71 = t111 * t87 - t80 * t89
        t99 = q[1]
        t84 = np.sin(t99)
        t91 = np.cos(t99)
        t116 = t69 * t91 - t71 * t84
        t105 = t89 * t90
        t110 = t83 * t81
        t75 = t105 * t88 - t110
        t115 = t75 * t91
        t93 = d7 + d8
        t77 = t87 * t93 + d5 + d6
        t114 = t77 * t82
        t113 = t80 * t93
        t112 = t81 * t82
        t107 = t87 * t90
        t104 = t90 * t91
        t103 = t77 * t89 + d3 + d4
        t102 = t113 * t88
        t101 = t80 * t111
        t100 = q[0]
        t94 = q[6]
        t92 = np.cos(t100)
        t86 = np.cos(t94)
        t85 = np.sin(t100)
        t79 = np.sin(t94)
        t73 = t106 * t83 + t81 * t90
        t72 = -t110 * t89 + t88 * t90
        t70 = t105 * t81 + t83 * t88
        t67 = -t109 * t82 + t73 * t80
        t66 = t112 * t84 + t70 * t91
        t65 = t107 * t81 + t74 * t83
        t64 = t113 * t73 - t114 * t83
        t63 = (t111 * t84 + t115) * t80 + t87 * (-t104 * t82 + t84 * t89)
        t61 = -t65 * t79 + t72 * t86
        t60 = (t102 * t84 - t104 * t77) * t82 + t115 * t113 + t103 * t84
        t59 = t116 * t79 - t66 * t86
        m[3, 0] = 0
        m[0, 2] = t63 * t92 - t67 * t85
        m[2, 2] = (t107 * t82 - t75 * t80) * t84 + t91 * (t87 * t89 + t101)
        m[1, 0] = (-t116 * t85 + t65 * t92) * t86 - (t66 * t85 - t72 * t92) * t79
        m[0, 3] = t60 * t92 - t64 * t85
        m[0, 1] = t59 * t92 - t61 * t85
        m[1, 3] = t60 * t85 + t64 * t92
        m[3, 1] = 0
        m[2, 3] = (
            (t101 * t93 + t103) * t91
            + ((-t102 * t89 + t114) * t90 + t110 * t113) * t84
            + d1
            + d2
        )
        m[1, 1] = t59 * t85 + t61 * t92
        m[0, 0] = (-t116 * t86 - t66 * t79) * t92 - t85 * (t65 * t86 + t72 * t79)
        m[3, 3] = 1
        m[3, 2] = 0
        m[2, 0] = (t69 * t84 + t71 * t91) * t86 + (-t112 * t91 + t70 * t84) * t79
        m[1, 2] = t63 * t85 + t67 * t92
        m[2, 1] = (-t69 * t79 + t70 * t86) * t84 - t91 * (t112 * t86 + t71 * t79)
        return m

    def rotvec_endeffector(self, q):
        if isinstance(q, np.ndarray):
            m = np.zeros(3)
        else:
            m = ca.SX.zeros(3)
        t1 = q[5]
        t2 = ca.cos(t1)
        t3 = q[3]
        t4 = ca.cos(t3)
        t5 = t4 * t2
        t6 = q[4]
        t7 = ca.cos(t6)
        t9 = ca.sin(t3)
        t10 = ca.sin(t1)
        t12 = t10 * t9 + t5 * t7
        t13 = q[2]
        t14 = ca.cos(t13)
        t16 = ca.sin(t13)
        t17 = t16 * t2
        t18 = ca.sin(t6)
        t19 = t18 * t17
        t20 = t12 * t14 - t19
        t21 = q[1]
        t22 = ca.cos(t21)
        t24 = ca.sin(t21)
        t28 = -t2 * t7 * t9 + t10 * t4
        t29 = t28 * t24
        t30 = t20 * t22 - t29
        t31 = q[6]
        t32 = ca.cos(t31)
        t34 = ca.sin(t31)
        t37 = t16 * t7
        t38 = t14 * t18 * t4 + t37
        t42 = t18 * t24 * t9 + t22 * t38
        t45 = q[0]
        t46 = ca.cos(t45)
        t51 = t14 * t18 * t2 + t12 * t16
        t56 = -t16 * t18 * t4 + t14 * t7
        t59 = ca.sin(t45)
        t63 = -t12 * t14 + t19
        t68 = t34 * (t22 * t63 + t29) - t42 * t32
        t73 = t32 * t56 - t34 * t51
        t78 = -t14 * t4 * t7 + t16 * t18

        t89 = (
            t46 * (t30 * t32 - t34 * t42) / 0.2e1
            - t59 * (t32 * t51 + t34 * t56) / 0.2e1
            + t59 * t68 / 0.2e1
            + t46 * t73 / 0.2e1
            + t24 * (t14 * t2 * t9 + t10 * t78) / 0.2e1
            + (t10 * t7 * t9 + t5) * t22 / 0.2e1
            - 0.1e1 / 0.2e1
        )

        t90 = ca.acos(t89)
        t91 = t89**2
        t93 = ca.sqrt(0.1e1 - t91)
        t95 = 0.1e1 / t93 * t90
        t100 = -t28
        t117 = t10 * (t24 * t7 * t9 - t22 * t78) + (-t14 * t22 * t9 + t24 * t4) * t2
        t124 = t10 * (t14 * t18 + t37 * t4) - t9 * t17
        m[0] = (
            (
                t24 * (t20 * t34 + t32 * t38)
                - (t18 * t32 * t9 + t100 * t34) * t22
                - t59 * t117
                - t124 * t46
            )
            * t95
            / 0.2e1
        )
        m[1] = (
            (
                t46 * t117
                - t59 * t124
                - t32 * (t100 * t22 + t24 * t63)
                - (-t18 * t22 * t9 + t24 * t38) * t34
            )
            * t95
            / 0.2e1
        )
        m[2] = (
            (
                t32 * (t30 * t59 + t46 * t51)
                - (t42 * t59 - t46 * t56) * t34
                - t46 * t68
                + t59 * t73
            )
            * t95
            / 0.2e1
        )
        return m

    def jacobian_fk(self, q):
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()
        if isinstance(q, np.ndarray):
            m = np.zeros((6, 7))
        else:
            m = ca.SX.zeros((6, 7))
        t1 = q[4]
        t2 = ca.cos(t1)
        t3 = q[1]
        t4 = ca.sin(t3)
        t5 = t2 * t4
        t6 = d7 + d8
        t7 = q[5]
        t8 = ca.sin(t7)
        t9 = t8 * t6
        t11 = ca.cos(t3)
        t12 = q[2]
        t13 = ca.cos(t12)
        t14 = t13 * t11
        t15 = ca.cos(t7)
        t17 = t15 * t6 + d5 + d6
        t19 = t14 * t17 - t5 * t9
        t20 = q[3]
        t21 = ca.sin(t20)
        t24 = ca.cos(t20)
        t27 = ca.sin(t12)
        t28 = ca.sin(t1)
        t29 = t28 * t27
        t30 = t13 * t2 * t24 - t29
        t32 = t8 * t30 * t6 * t11
        t33 = t24 * t17
        t35 = t4 * (t33 + d3 + d4)
        t37 = q[0]
        t38 = ca.sin(t37)
        t40 = ca.cos(t37)
        t43 = t27 * t2
        t45 = t13 * t28
        t46 = t24 * t43 + t45
        t48 = -t17 * t21 * t27 + t46 * t9
        m[0, 0] = t38 * (t19 * t21 - t32 - t35) - t48 * t40
        t50 = t8 * t2
        t52 = t21 * t6 * t50
        t53 = t33 + t52 + d3 + d4
        t57 = t24 * t6 * t50
        t58 = -t17 * t21 + t57
        t60 = t9 * t29
        t63 = t11 * t53 - (t13 * t58 - t60) * t4
        m[0, 1] = t63 * t40
        t64 = -t58
        t67 = t27 * t64 - t45 * t9
        t68 = t67 * t11
        t71 = t13 * t64 + t60
        m[0, 2] = t38 * t71 + t40 * t68
        t73 = -t19
        t75 = t13 * t2
        t81 = t24 * t73 - t21 * (t11 * t75 * t9 + t17 * t4)
        t84 = t52 + t33
        m[0, 3] = t27 * t38 * t84 + t40 * t81
        t87 = t21 * t4
        t88 = t14 * t24 + t87
        t90 = t27 * t24
        t97 = t11 * t27 * t40 + t13 * t38
        m[0, 4] = -t8 * (t28 * (-t38 * t90 + t40 * t88) + t2 * t97) * t6
        t104 = t11 * t30 + t21 * t5
        t109 = t11 * t13 * t21 - t24 * t4
        t114 = t21 * t27
        m[0, 5] = (t40 * (t104 * t15 + t109 * t8) - (t114 * t8 + t15 * t46) * t38) * t6
        m[0, 6] = 0.0e0
        m[1, 0] = t40 * (t21 * t73 + t32 + t35) - t48 * t38
        m[1, 1] = t38 * t63
        m[1, 2] = t40 * (t13 * (t21 * (-t15 * t6 - d5 - d6) + t57) - t60) + t38 * t68
        m[1, 3] = -t27 * t40 * t84 + t38 * t81
        t140 = -t28 * t90 + t75
        m[1, 4] = -t8 * t6 * (t38 * (t11 * t43 + t28 * t88) - t140 * t40)
        t150 = t40 * t114
        m[1, 5] = t6 * (t15 * (t104 * t38 + t40 * t46) + t8 * (t38 * t109 + t150))
        m[1, 6] = 0.0e0
        m[2, 0] = 0.0e0
        m[2, 1] = t11 * t71 - t4 * t53
        m[2, 2] = -t67 * t4
        m[2, 3] = t13 * t4 * t84 + t11 * t58
        t160 = t13 * t4
        t165 = t27 * t4
        m[2, 4] = (t28 * (-t11 * t21 + t160 * t24) + t165 * t2) * t9
        t169 = t2 * t21
        t175 = t11 * t24 + t160 * t21
        m[2, 5] = -(t15 * (-t11 * t169 + t30 * t4) + t8 * t175) * t6
        m[2, 6] = 0.0e0
        m[3, 0] = 0.0e0
        m[3, 1] = -t38
        m[3, 2] = t4 * t40
        m[3, 3] = t97
        t179 = -t109
        m[3, 4] = t114 * t38 + t179 * t40
        t184 = -t13 * t24 * t28 - t43
        t187 = t11 * t184 - t28 * t87
        t189 = -t140
        m[3, 5] = t187 * t40 + t189 * t38
        t194 = t2 * t24 * t8 - t15 * t21
        t197 = t28 * t27 * t8
        t202 = t15 * t24 + t169 * t8
        t204 = t11 * (t13 * t194 - t197) + t202 * t4
        t208 = -t194
        t210 = -t13 * t28 * t8 + t208 * t27
        m[3, 6] = t204 * t40 + t210 * t38
        m[4, 0] = 0.0e0
        m[4, 1] = t40
        m[4, 2] = t4 * t38
        m[4, 3] = t11 * t27 * t38 - t13 * m[4, 1]
        m[4, 4] = t179 * t38 - t150
        m[4, 5] = t187 * t38 - t189 * m[4, 1]
        m[4, 6] = t204 * t38 - t210 * m[4, 1]
        m[5, 0] = 0.1e1
        m[5, 1] = 0.0e0
        m[5, 2] = t11
        m[5, 3] = -t165
        m[5, 4] = t175
        m[5, 5] = -t21 * t28 * m[5, 2] - t184 * t4
        m[5, 6] = t4 * (t13 * t208 + t197) + t202 * m[5, 2]
        return m

    def djacobian_fk(self, q, dq):
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()

        if isinstance(q, np.ndarray):
            m = np.zeros((6, 7))
        else:
            m = ca.SX.zeros((6, 7))
        t1219 = q[4]
        t1205 = np.cos(t1219)
        t1212 = dq[5]
        t1214 = dq[3]
        t1182 = t1205 * t1214 - t1212
        t1220 = q[3]
        t1200 = np.sin(t1220)
        t1206 = np.cos(t1220)
        t1213 = dq[4]
        t1190 = t1206 * t1213
        t1215 = dq[2]
        t1180 = t1190 + t1215
        t1199 = np.sin(t1219)
        t1308 = t1199 * t1180
        t1256 = t1182 * t1200 + t1308
        t1193 = t1215 * t1206
        t1184 = t1193 + t1213
        t1221 = q[2]
        t1201 = np.sin(t1221)
        t1216 = dq[1]
        t1150 = t1184 * t1201 - t1200 * t1216
        t1207 = np.cos(t1221)
        t1268 = t1207 * t1214
        t1143 = t1200 * t1268 + t1150
        t1217 = dq[0]
        t1196 = t1217 * t1201
        t1222 = q[1]
        t1202 = np.sin(t1222)
        t1168 = t1202 * t1214 - t1196
        t1263 = t1216 * t1202
        t1177 = t1207 * t1263
        t1149 = -t1168 + t1177
        t1208 = np.cos(t1222)
        t1165 = t1200 * t1213 - t1201 * t1216
        t1279 = t1202 * t1165
        t1126 = ((t1180 * t1208 + t1217) * t1207 + t1279) * t1205 - (
            t1143 * t1208 + t1149 * t1206
        ) * t1199
        t1223 = q[0]
        t1203 = np.sin(t1223)
        t1209 = np.cos(t1223)
        t1269 = t1207 * t1208
        t1244 = t1206 * t1269
        t1280 = t1201 * t1205
        t1189 = t1202 * t1217
        t1166 = t1201 * t1214 - t1189
        t1283 = t1200 * t1166
        t1197 = t1217 * t1208
        t1300 = t1197 + t1215
        t1225 = (t1184 * t1207 + t1217 * t1244 - t1283) * t1199 + (
            t1190 + t1300
        ) * t1280
        t1310 = t1126 * t1203 + t1209 * t1225
        t1309 = t1126 * t1209 - t1203 * t1225
        t1195 = t1216 * t1206
        t1243 = t1205 * t1195
        t1153 = t1199 * t1217 + t1202 * t1243
        t1147 = t1153 * t1207
        t1297 = -t1150 * t1205 - t1207 * t1256
        t1301 = t1168 * t1205 - t1202 * t1212
        t1307 = -t1199 * t1279 + t1206 * t1301 + t1208 * t1297 - t1147
        t1274 = t1205 * t1216
        t1233 = t1182 * t1207 - t1274
        t1306 = t1184 * t1280 + t1200 * t1233
        t1275 = t1205 * t1212
        t1181 = -t1214 + t1275
        t1264 = t1215 * t1200
        t1167 = t1199 * t1212 - t1264
        t1272 = t1206 * t1207
        t1226 = -t1167 * t1201 + t1181 * t1272 + t1195
        t1130 = t1226 * t1208 + t1200 * (t1181 * t1202 + t1177 + t1196)
        t1299 = -t1165 * t1199 + t1182 * t1206 - t1207 * t1243
        t1218 = q[5]
        t1204 = np.cos(t1218)
        t1210 = d7 + d8
        t1265 = t1210 * t1204
        t1179 = d5 + d6 + t1265
        t1176 = t1214 * t1179
        t1246 = t1212 * t1265
        t1145 = t1205 * t1246 - t1176
        t1211 = d3 + d4
        t1238 = t1199 * t1246
        t1251 = t1201 * t1264
        t1296 = (t1251 + t1195) * t1179 + t1145 * t1272 - t1201 * t1238 + t1216 * t1211
        t1288 = t1199 * t1201
        t1155 = t1205 * t1272 - t1288
        t1276 = t1205 * t1207
        t1258 = t1155 * t1197 + t1184 * t1276
        t1295 = (-t1200 * t1212 + t1308) * t1201 + t1205 * t1283 - t1258
        t1267 = t1207 * t1216
        t1185 = -t1214 + t1267
        t1287 = t1199 * t1206
        t1293 = (-t1143 * t1199 + t1180 * t1276) * t1202 + (
            -t1165 * t1205 + t1185 * t1287
        ) * t1208
        t1292 = t1300 * t1201
        t1198 = np.sin(t1218)
        t1291 = t1198 * t1205
        t1290 = t1198 * t1207
        t1289 = t1198 * t1210
        t1286 = t1199 * t1207
        t1285 = t1199 * t1213
        t1282 = t1200 * t1179
        t1281 = t1200 * t1208
        t1278 = t1204 * t1212
        t1273 = t1206 * t1145
        t1271 = t1206 * t1208
        t1270 = t1207 * t1200
        t1266 = t1208 * t1215
        t1262 = t1216 * t1208
        t1261 = t1217 * t1179
        t1260 = t1217 * t1203
        t1259 = t1217 * t1209
        t1254 = t1198 * t1285
        t1253 = t1205 * t1289
        t1252 = t1199 * t1278
        t1250 = t1215 * t1280
        t1249 = t1201 * t1266
        t1247 = t1204 * t1275
        t1163 = t1179 * t1267
        t1242 = t1200 * t1261
        t1230 = (t1200 * t1202 + t1244) * t1199 + t1208 * t1280
        t1227 = t1247 - t1254
        t1157 = t1202 * t1163
        t1224 = (
            (
                t1306 * t1208
                + t1147
                + (-t1182 * t1202 + t1196 * t1205) * t1206
                + (t1180 * t1269 + t1279) * t1199
            )
            * t1289
            - t1296 * t1208
            - t1200 * (t1145 * t1202 + t1179 * t1196 + t1157)
        )
        t1183 = -t1216 + t1268
        t1175 = t1179 * t1216
        t1162 = t1179 * t1264
        t1160 = t1185 + t1275
        t1154 = -t1201 * t1287 + t1276
        t1148 = -t1201 * t1263 + t1207 * (t1217 + t1266)
        t1140 = t1166 * t1206 + t1270 * t1300
        t1137 = (t1149 + t1249) * t1200 - t1183 * t1271
        t1136 = t1206 * (-t1181 * t1201 - t1189) + (t1197 * t1200 - t1167) * t1207
        t1131 = (-t1155 * t1289 + t1179 * t1270) * t1202 + (
            t1179 * t1206 + t1200 * t1253 + t1211
        ) * t1208
        t1127 = -t1299 * t1289 - t1200 * (t1145 + t1163)
        t1123 = (
            -(-t1166 * t1205 + t1201 * t1212) * t1289 + t1207 * t1300 * t1179
        ) * t1206 + t1200 * (
            -t1179 * t1166
            + (-t1201 * t1254 + (t1201 * t1278 + t1290 * t1300) * t1205) * t1210
        )
        t1121 = -(-t1180 * t1286 - t1306) * t1289 - t1296
        t1120 = (
            t1208 * t1242 + t1162 + (-(t1197 * t1206 + t1184) * t1291 - t1252) * t1210
        ) * t1201 - t1207 * ((t1197 * t1199 + t1256) * t1289 - t1273)
        t1119 = t1136 * t1204 + t1198 * t1295
        t1118 = (
            ((-t1182 * t1201 + t1189 * t1205) * t1200 - t1180 * t1288 + t1258) * t1289
            + (t1145 * t1201 + t1202 * t1261) * t1206
            + t1211 * t1189
            + (-t1282 * t1300 + t1238) * t1207
        )
        t1117 = (
            (t1208 * t1256 + t1153) * t1289 - t1145 * t1271 - t1263 * t1282
        ) * t1201 + t1207 * (
            -(-t1199 * t1263 + (t1184 * t1208 + t1206 * t1217) * t1205) * t1289
            + (t1162 - t1238) * t1208
            + t1242
        )
        t1116 = t1130 * t1204 + t1198 * t1307
        t1115 = (
            t1157
            + ((-t1202 * t1285 - t1208 * t1233) * t1198 + t1202 * t1247) * t1210
            + (-t1168 + t1249) * t1179
        ) * t1206 - t1200 * (
            -((t1207 * t1285 + t1250) * t1208 + t1205 * t1177 - t1301) * t1289
            + (t1145 * t1207 + t1175) * t1208
        )
        t1114 = t1121 * t1202 - t1127 * t1208
        m[2, 2] = (
            (-t1256 * t1289 + t1273) * t1201
            - t1207 * (t1162 + (-t1184 * t1291 - t1252) * t1210)
        ) * t1202 - ((-t1206 * t1253 + t1282) * t1201 - t1286 * t1289) * t1262
        m[4, 2] = t1202 * t1259 + t1203 * t1262
        m[3, 5] = -t1309
        m[5, 6] = (-t1198 * t1297 - t1204 * t1226) * t1202 - t1208 * (
            -t1160 * t1200 * t1204 - t1198 * t1299
        )
        m[2, 3] = (
            (-t1179 * t1201 * t1193 + (-t1176 * t1207 + t1175) * t1200) * t1202
            + (t1163 - t1176) * t1271
            + (
                (
                    (t1182 * t1290 - t1198 * t1274) * t1206
                    + (-t1198 * t1250 + t1207 * t1227) * t1200
                )
                * t1202
                + (t1227 * t1206 + t1200 * t1198 * (t1185 * t1205 + t1212)) * t1208
            )
            * t1210
        )
        m[1, 2] = t1117 * t1203 + t1120 * t1209
        m[4, 4] = t1137 * t1203 - t1140 * t1209
        m[4, 3] = t1148 * t1203 + t1209 * t1292
        m[1, 6] = 0
        m[5, 0] = 0
        m[0, 3] = t1115 * t1209 + t1123 * t1203
        m[0, 4] = -(t1309 * t1198 + (t1154 * t1203 + t1209 * t1230) * t1278) * t1210
        m[3, 0] = 0
        m[1, 3] = t1115 * t1203 - t1123 * t1209
        m[5, 5] = t1293
        m[3, 6] = t1116 * t1209 + t1119 * t1203
        m[3, 4] = t1137 * t1209 + t1140 * t1203
        m[0, 0] = -t1118 * t1209 + t1203 * t1224
        m[1, 4] = -(t1310 * t1198 - (t1154 * t1209 - t1203 * t1230) * t1278) * t1210
        m[3, 2] = -t1202 * t1260 + t1209 * t1262
        m[3, 1] = -t1259
        m[4, 0] = 0
        m[0, 1] = t1114 * t1209 - t1131 * t1260
        m[1, 0] = -t1118 * t1203 - t1209 * t1224
        m[5, 3] = -t1202 * t1207 * t1215 - t1201 * t1262
        m[1, 1] = t1114 * t1203 + t1131 * t1259
        m[2, 6] = 0
        m[0, 5] = (
            -(
                (t1130 * t1198 - t1204 * t1307) * t1209
                + t1203 * (t1136 * t1198 - t1204 * t1295)
            )
            * t1210
        )
        m[2, 0] = 0
        m[5, 2] = -t1263
        m[0, 6] = 0
        m[4, 6] = t1116 * t1203 - t1119 * t1209
        m[4, 1] = -t1260
        m[2, 1] = t1121 * t1208 + t1127 * t1202
        m[3, 3] = t1148 * t1209 - t1203 * t1292
        m[4, 5] = -t1310
        m[1, 5] = (
            -(
                (-t1203 * t1307 + t1209 * t1295) * t1204
                + t1198 * (t1130 * t1203 - t1136 * t1209)
            )
            * t1210
        )
        m[2, 4] = (
            -(
                -t1293 * t1198
                + ((-t1199 * t1272 - t1280) * t1202 + t1199 * t1281) * t1278
            )
            * t1210
        )
        m[5, 1] = 0
        m[2, 5] = t1210 * (
            (-t1202 * t1297 + t1208 * t1299) * t1204
            + t1198 * (-t1160 * t1281 + t1202 * t1226)
        )
        m[0, 2] = t1117 * t1209 - t1120 * t1203
        m[5, 4] = (t1183 * t1206 - t1251) * t1202 + t1185 * t1281
        return m

    def ddjacobian_fk(self, q, q_p, q_pp):
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()
        if isinstance(q, np.ndarray):
            m = np.zeros((6, 7))
        else:
            m = ca.SX.zeros((6, 7))
        t304 = q[2]
        t277 = np.cos(t304)
        t288 = d3 + d4
        t293 = q_pp[1]
        t303 = q[3]
        t276 = np.cos(t303)
        t298 = q_p[2]
        t302 = q[4]
        t269 = np.sin(t302)
        t295 = q_p[5]
        t301 = q[5]
        t274 = np.cos(t301)
        t286 = d7 + d8
        t497 = t286 * t274
        t413 = t295 * t497
        t360 = t269 * t413
        t335 = t298 * t360
        t291 = q_pp[3]
        t265 = t291 / 2
        t296 = q_p[4]
        t478 = t295 * t296
        t405 = t269 * t478
        t186 = t265 + t405
        t287 = d5 + d6
        t253 = t291 * t287
        t275 = np.cos(t302)
        t289 = q_pp[5]
        t510 = t275 * t289
        t53 = -t253 + (-2 * t186 + t510) * t497
        t543 = t276 * t53 - 2 * t335
        t603 = -t277 * t543 - t288 * t293
        t280 = t295**2
        t281 = t296**2
        t299 = q_p[1]
        t284 = t299**2
        t300 = q_p[0]
        t285 = t300**2
        t241 = t284 + t285
        t283 = t298**2
        t381 = t283 + t241
        t346 = t281 + t381
        t153 = t280 + t346
        t107 = t269 * t153
        t271 = np.sin(t304)
        t270 = np.sin(t303)
        t247 = t299 * t270
        t421 = t296 * t247
        t370 = 2 * t421
        t477 = t295 * t298
        t402 = t270 * t477
        t290 = q_pp[4]
        t297 = q_p[3]
        t471 = t297 * t298
        t422 = t270 * t471
        t194 = -t290 + 2 * t422
        t245 = t293 * t270
        t462 = t194 * t271 + t245
        t466 = t299 * t297
        t292 = q_pp[2]
        t488 = t292 * t271
        t189 = -2 * t466 + t488
        t469 = t298 * t296
        t426 = t269 * t469
        t372 = 2 * t426
        t467 = t299 * t295
        t581 = -t189 * t275 + t271 * t372 - 2 * t467
        t51 = t581 * t276
        t602 = -t51 - t462 * t275 + t269 * t370 + (-t107 + 2 * t402) * t271
        t475 = t296 * t297
        t425 = t269 * t475
        t181 = t425 + t289 / 2
        t509 = t275 * t291
        t105 = -t509 / 2 + t181
        t208 = t275 * t297 - t295
        t561 = -2 * t299
        t451 = t208 * t561
        t139 = t277 * t451
        t305 = q[1]
        t272 = np.sin(t305)
        t106 = t272 * t139
        t282 = t297**2
        t204 = t282 + t241
        t347 = t281 + t204
        t152 = t280 + t347
        t108 = t152 * t275
        t142 = -t275 * t292 + t372
        t465 = t300 * t298
        t486 = t293 * t272
        t167 = t465 + t486 / 2
        t244 = t290 * t269
        t472 = t297 * t295
        t248 = -2 * t472
        t191 = t244 + t248
        t205 = t497 + t287
        t278 = np.cos(t305)
        t474 = t296 * t299
        t236 = t275 * t474
        t352 = t272 * t236
        t397 = t275 * t469
        t227 = 2 * t397
        t242 = t269 * t292
        t459 = t227 + t242
        t175 = t282 + t381
        t333 = t281 + t175
        t110 = t280 + t333
        t367 = t110 * t275 + t244
        t56 = t248 + t367
        t50 = t56 * t276
        t435 = t50 + t459
        t491 = t290 * t275
        t375 = -(t107 - t491) * t271 + t435 * t277
        t473 = t296 * t300
        t391 = t269 * t473
        t357 = t275 * t413
        t342 = -2 * t357
        t155 = t297 * t342
        t174 = t280 + t204
        t42 = t174 * t497 + t204 * t287 + t155
        t234 = t275 * t478
        t496 = t289 * t269
        t129 = t234 + t496 / 2
        t432 = t129 * t497
        t443 = 2 * t473
        t230 = t275 * t443
        t294 = q_pp[0]
        t468 = t298 * t299
        t394 = t272 * t468
        t187 = t394 - t294 / 2
        t577 = -2 * t269
        t190 = 2 * t465 + t486
        t572 = 2 * t272
        t588 = t299 * t572
        t373 = t269 * t588
        t314 = -t190 * t275 + t296 * t373
        t594 = t314 * t276
        t548 = t187 * t577 - t594
        t433 = t230 + t548
        t514 = t271 * t300
        t447 = -2 * t514
        t516 = t271 * t298
        t448 = -2 * t516
        t485 = t293 * t275
        t268 = np.sin(t301)
        t527 = t268 * t286
        t198 = -t294 + 2 * t394
        t532 = t198 * t275
        t193 = t289 + 2 * t425
        t113 = -t193 + t509
        t231 = t269 * t474
        t545 = t113 * t277 + 2 * t231
        t154 = t280 + t175
        t412 = t275 * t472
        t368 = 2 * t412
        t35 = -t175 * t287 + (-t154 + t368) * t497
        t551 = t35 * t277
        t573 = 2 * t271
        t574 = -2 * t271
        t337 = t271 * t360
        t326 = -2 * t337
        t71 = t241 * t288 + t299 * t326
        t78 = -t205 * t297 + t357
        t601 = (
            (
                (-t205 * t488 + t561 * t78 + t551) * t270
                - (t205 * t293 + t448 * t78) * t276
                + t432 * t573
                + t603
            )
            * t278
            + (
                (
                    (t208 * t448 - t485 + t545) * t270
                    + (-t142 * t271 + t451) * t276
                    + t375
                )
                * t278
                + (t106 + t208 * t447 + t272 * (t108 + t191)) * t270
                + t433 * t277
                + ((-2 * t391 - t532) * t271 + t105 * t572) * t276
                + (t167 * t269 + t352) * t574
            )
            * t527
            - (-t272 * t42 + t447 * t78) * t276
            + t272 * t71
        )
        t522 = t270 * t272
        t450 = -2 * t522
        t229 = 2 * t391
        t481 = t294 * t275
        t143 = t229 - t481
        t104 = t143 * t276
        t583 = t208 * t270 + t269 * t298
        t553 = t300 * t583
        t599 = t104 + 2 * t553
        t388 = t189 * t276 - t462
        t455 = t281 + t283
        t384 = t280 + t455
        t124 = t269 * (t285 + t384)
        t511 = t272 * t300
        t414 = t299 * t511
        t558 = -t292 / 2
        t185 = t414 + t558
        t131 = t269 * t185
        t182 = t422 - t290 / 2
        t354 = t473 * t522
        t336 = t269 * t354
        t411 = t275 * t465
        t350 = t276 * t411
        t524 = t269 * t294
        t379 = t524 / 2
        t517 = t271 * t296
        t159 = -t247 + t517
        t507 = t275 * t300
        t393 = t159 * t507
        t476 = t295 * t300
        t403 = t272 * t476
        t164 = t472 - t244 / 2
        t538 = t164 * t271
        t255 = t280 / 2
        t259 = t283 / 2
        t456 = t259 + t285 / 2
        t341 = t282 / 2 + t255 + t456
        t470 = t297 * t300
        t400 = t272 * t470
        t60 = -(t281 / 2 + t341) * t271 + t400
        t130 = t391 - t481 / 2
        t96 = t130 * t276
        t597 = (
            ((t96 + t553) * t277 + (t350 + t379) * t271 + t393) * t278
            + (-t275 * t60 + t403 - t538) * t276
            + ((t185 * t275 + t426) * t276 + t182 * t275 - t402 + t124 / 2) * t277
            - (t181 * t270 + t131) * t271
            + t336
        )
        t575 = 2 * t270
        t508 = t275 * t295
        t207 = -t297 + t508
        t595 = (t207 * t448 + t293) * t276 - 2 * t159 * t508
        t592 = t164 * t450 + (-t190 * t269 + t476 * t575) * t271
        t327 = t198 * t271 + t272 * t291
        t591 = t181 * t572 - t275 * t327
        t567 = 2 * t276
        t503 = t276 * t299
        t521 = t270 * t298
        t586 = t300 * (t271 * (t269 * t295 - t521) - t503)
        t490 = t291 * t270
        t188 = 2 * t469 + t490
        t146 = t193 * t270
        t390 = t188 * t275 - t146 + t242
        t163 = t272 * t297 - t514
        t399 = t271 * t470
        t225 = 2 * t399
        t495 = t289 * t270
        t407 = t275 * t495
        t415 = t272 * t503
        t420 = t270 * t466
        t427 = t269 * t476
        t404 = t269 * t477
        t219 = -2 * t404
        t339 = -2 * t405 - t291 + t510
        t369 = -2 * t412
        t436 = t270 * t369 + t276 * t339 + t219
        t513 = t272 * t174
        t14 = (
            (
                t271 * (t270 * t292 - t496)
                - 2 * t420
                + (t154 * t270 + t436) * t277
                + t595
            )
            * t278
            + t272 * t407
            + (2 * t163 * t508 + t225 - t513) * t276
            + (t190 * t270 - 2 * t207 * t415 - 2 * t427) * t277
            + (-t198 * t270 + t295 * t373) * t271
            + t186 * t450
        )
        t576 = -2 * t270
        t306 = q[0]
        t273 = np.sin(t306)
        t571 = -2 * t273
        t570 = 2 * t273
        t569 = -2 * t274
        t568 = -2 * t275
        t566 = 2 * t277
        t565 = 2 * t278
        t279 = np.cos(t306)
        t564 = -2 * t279
        t563 = 2 * t279
        t562 = -2 * t287
        t243 = t272 * t294
        t560 = t243 / 2
        t559 = t268 / 2
        t232 = t271 * t468
        t184 = t232 + t265
        t484 = t293 * t277
        t321 = t184 * t562 + t205 * t484
        t500 = t277 * t299
        t445 = 2 * t500
        t374 = t78 * t445
        t94 = t186 + t232
        t557 = ((2 * t94 - t510) * t497 - t321) * t270 + t276 * t374
        t419 = t271 * t471
        t195 = -t293 - 2 * t419
        t233 = t271 * t477
        t34 = t195 * t275 + 2 * t233 + t545
        t556 = t270 * t34 - t51
        t417 = t271 * t465
        t479 = t294 * t277
        t127 = t417 - t479 / 2
        t254 = t299 * t288
        t331 = t205 * t299 + t277 * t78
        t530 = t205 * t270
        t555 = t127 * t530 + (t276 * t331 + t254 - t337) * t300
        t504 = t276 * t298
        t363 = t296 + t504
        t554 = t271 * (t363 * t507 + t379)
        t85 = t271 * t363 - t247
        t552 = t300 * t85
        t197 = -t292 + 2 * t414
        t79 = t197 * t275 + t372
        t547 = t276 * t79 + t124
        t537 = (t466 - t488 / 2) * t205
        t546 = t299 * t342 + 2 * t537
        t203 = t282 + t283 + t285
        t348 = t281 + t203
        t151 = t280 + t348
        t223 = -2 * t400
        t544 = 2 * t403 + (t151 * t271 + t223) * t275
        t380 = t530 / 2
        t64 = t78 * t504
        t541 = t292 * t380 - t64
        t235 = t270 * t475
        t506 = t276 * t290
        t115 = 2 * t235 - t292 - t506
        t92 = t115 * t277
        t401 = t270 * t470
        t540 = (-t272 * t474 - t401) * t271
        t539 = t129 * t274
        t122 = (t469 + t490 / 2) * t271
        t183 = t419 + t293 / 2
        t535 = t183 * t205
        t531 = t203 * t287
        t529 = t205 * t277
        t123 = t269 * (t284 + t384)
        t525 = t269 * t276
        t520 = t270 * t300
        t519 = t271 * t203
        t518 = t271 * t293
        t515 = t271 * t299
        t505 = t276 * t296
        t502 = t276 * t300
        t501 = t277 * t298
        t499 = t277 * t300
        t498 = t278 * t300
        t494 = t289 * t271
        t493 = t289 * t276
        t492 = t290 * t270
        t489 = t291 * t271
        t487 = t293 * t269
        t482 = t294 * t270
        t480 = t294 * t276
        t464 = t270 * t560 - t122
        t454 = t282 + t284
        t382 = t281 + t454
        t169 = t280 + t382
        t418 = t271 * t474
        t463 = t169 * t270 - 2 * t418
        t196 = 2 * t232 + t291
        t246 = t294 * t271
        t460 = -t196 * t272 + t246
        t458 = t233 + t231
        t457 = t236 + t487 / 2
        t453 = t283 + t284
        t452 = -2 * t530
        t449 = 2 * t520
        t446 = 2 * t503
        t444 = 2 * t498
        t359 = t272 * t420
        t82 = 2 * t540
        t442 = (t190 * t276 - 2 * t359 + t443) * t277 - t327 * t276 + t82
        t102 = (-2 * t231 + t485) * t276
        t65 = (-t196 * t275 + t193) * t276
        t441 = (t468 * t577 + t102) * t277 + t65 + t457 * t574
        t202 = t282 + t453
        t171 = t280 + t202
        t41 = -t202 * t287 + (-t171 + t368) * t497
        t440 = t277 * t41 + t546
        t439 = t78 * t502
        t98 = -t208 * t277 + t275 * t299
        t438 = t98 * t520
        t410 = t276 * t469
        t437 = t410 * t573 - 2 * t421 + t92
        t349 = t281 + t202
        t150 = t280 + t349
        t62 = t150 * t275 + t191
        t54 = t62 * t276
        t434 = t54 + t459
        t431 = t185 * t529
        t206 = t298 + t505
        t430 = t206 * t499
        t429 = t207 * t504
        t428 = t269 * t515
        t424 = t269 * t465
        t409 = t276 * t475
        t408 = t269 * t494
        t160 = t271 * t297 - t511
        t406 = t160 * t508
        t398 = t208 * t501
        t396 = t206 * t500
        t395 = t277 * t466
        t392 = t207 * t502
        t387 = t139 + t191
        t383 = t280 + t454
        t385 = t287 * t454 + t383 * t497 + t155
        t377 = t495 / 2
        t376 = -t482 / 2
        t371 = t299 * t450
        t365 = t243 - t489
        t361 = t78 * t415
        t356 = t270 * t406
        t355 = t271 * t391
        t353 = t270 * t400
        t351 = t274 * t427
        t345 = t510 / 2 - t186
        t343 = -t491 + t547
        t138 = 2 * t208 * t521
        t340 = t142 * t276 + t138 - t491
        t49 = -t253 / 2 + t345 * t497
        t338 = (t272 * t49 + (t167 * t277 - t187 * t271) * t205) * t270 + (
            -t286 * t351 - t361
        ) * t277
        t334 = t274 * t275 * t233
        t332 = (t270 * t472 - t493 / 2) * t277 + t295 * t85
        t330 = -t281 / 2 - t410
        t329 = (-t276 * t293 + 2 * t420) * t277 + t196 * t276
        t325 = -2 * t410 - t455
        t324 = t235 - t506 / 2 + t558
        t319 = t188 * t271 - t272 * t482
        t318 = 2 * t357 * t517 + t408 * t497 + (t286 * t334 - t535) * t567 + t603
        t317 = (
            -t272 * t492 - 2 * t163 * t505 + t190 * t271 + (t206 * t588 - t294) * t277
        )
        t315 = 2 * (t359 - t473) * t275 + t295 * t371
        t313 = 2 * t350 + t524
        t308 = (
            (t561 * t583 + t102) * t277 + t65 + t463 * t275 + t191 * t270 - t271 * t487
        )
        t307 = (
            (t207 * t446 - t245) * t277
            + (t369 + t383) * t276
            - t407
            + (2 * t405 + t196) * t270
            - 2 * t295 * t428
        )
        t240 = -t246 / 2
        t226 = -2 * t399
        t222 = 2 * t400
        t180 = -2 * t355
        t173 = t280 + t203
        t161 = -t270 * t296 + t515
        t144 = t299 * t444 + t243
        t126 = t231 - t485 / 2
        t116 = -t241 * t272 + t278 * t293
        t114 = -t277 * t291 - t195
        t89 = t113 * t270
        t87 = t152 * t522
        t83 = t295 * t206 * t277 + t494 / 2
        t72 = t205 * t521 - t360
        t63 = t151 * t275 + t191
        t45 = t169 * t275 + t387
        t44 = (t294 * t278 / 2 - t185) * t271 + t277 * (t278 * t465 + t456)
        t36 = (-t278 * t381 - t190) * t271 + t277 * (t278 * t292 - t198)
        t23 = ((-2 * t417 + t479) * t278 - t197 * t277 + t222 - t519) * t270 + t276 * (
            (t277 * t297 - t299) * t444 + t471 * t566 - t365
        )
        t22 = (t271 * t525 / 2 - t275 * t277 / 2) * t289 + (
            (-t270 * t160 + (t296 + (t298 + t498) * t276) * t277) * t269
            + t275 * (t206 + t498) * t271
        ) * t295
        t21 = (t171 * t270 + t436) * t277 + t189 * t270 - t408 + t595
        t20 = ((t175 * t277 + t189) * t278 + t190 * t277 + t460) * t270 + t276 * (
            t114 * t278 + t225 + (-t204 + 2 * t395) * t272
        )
        t19 = (
            -t332 * t278 + t295 * t163 * t276 + (-t276 * t277 * t467 + t377) * t272
        ) * t269 - (-t83 * t278 + (t161 * t272 - t499) * t295) * t275
        t18 = (
            (t54 + t390) * t277
            - t51
            + (t195 * t270 + t271 * t290) * t275
            + t458 * t575
            - t271 * t123
        )
        t17 = (
            ((t376 + t392) * t277 - t586) * t278
            + (t185 * t270 + t129 + t429) * t277
            + (t271 * t345 + t560) * t276
            - t356
            + (t270 * t341 - t404) * t271
            - t353
        )
        t16 = (
            ((-2 * t401 + t480) * t277 - 2 * t552) * t278
            + (-t197 * t276 - t194) * t277
            + (-t271 * t348 + t222) * t276
            - t319
        ) * t269 + (
            (t246 + 2 * t430) * t278
            + (t285 - t325) * t277
            + 2 * t354
            + (-t115 - 2 * t414) * t271
        ) * t275
        t15 = (
            ((t480 / 2 - t401) * t277 - t552) * t278
            + (-t185 * t276 - t182) * t277
            + t60 * t276
            + t464
        ) * t269 - t275 * (
            (t240 - t430) * t278
            + (-t283 / 2 - t285 / 2 - t280 / 2 + t330) * t277
            - t354
            + (t324 + t414) * t271
        )
        t13 = (
            ((t276 * t333 + t188) * t277 + t388) * t278 + t347 * t522 + t442
        ) * t269 + ((t271 * t346 + t437) * t278 + t317) * t275
        t12 = (((t110 * t276 + t188) * t277 + t388) * t278 + t87 + t442) * t269 + (
            (t153 * t271 + t437) * t278 + t317
        ) * t275
        t11 = ((t438 + (-t96 - t424) * t277 - t554) * t527 + t555) * t272 - t278 * (
            -2 * (t130 * t270 + t300 * (-t428 + (t275 * t500 - t208) * t276)) * t527
            + (t205 * t500 + t78) * t449
            + (t205 * t276 + t288) * t294
        ) / 2
        t9 = (
            ((t50 + t390) * t277 + t602) * t278
            + (-t315 + t548) * t277
            + (t180 + t591) * t276
            + (t82 + t87) * t275
            + t592
        )
        t8 = (
            (t205 * t127 * t565 + 2 * t431 + (t223 + t519) * t287) * t276
            + (t331 * t498 + t78 * t501 + t365 * t287 / 2) * t576
            + (
                (
                    (
                        -t398
                        + (t560 - t489 / 2) * t275
                        + t271 * t181
                        + (-t269 * t272 * t296 + t278 * t98) * t300
                    )
                    * t567
                    + (
                        (t143 * t277 + t411 * t573) * t278
                        + t79 * t277
                        - 2 * t538
                        + t544
                    )
                    * t270
                )
                * t268
                + (
                    (t223 - 2 * t406) * t276
                    - t243 * t270
                    + (t173 * t276 + t345 * t576) * t271
                )
                * t274
            )
            * t286
        )
        t7 = (
            (t205 * t482 - 2 * t439) * t278
            - 2 * t64
            + t185 * t452
            + ((t278 * t599 + t138 + t343) * t268 - 2 * t539) * t286
        ) * t271 + t277 * (
            -((t230 + t313) * t278 + t63 * t276 + t89 + t227 - 2 * t131) * t527
            + t72 * t444
            + (t173 * t497 + t155 + t531) * t270
            + t543
        )
        t6 = (
            (t271 * t313 + t277 * t599 + 2 * t393) * t278
            + (t194 * t275 - 2 * t402 + t547) * t277
            + (t191 * t271 + t544) * t276
            + t319 * t275
            + (-t197 * t269 - t146) * t271
            + 2 * t336
        ) * t268 - (
            ((2 * t392 - t482) * t277 - 2 * t586) * t278
            + (t197 * t270 + 2 * t234 + 2 * t429 + t496) * t277
            + (t271 * t339 + t243) * t276
            - 2 * t356
            + (t173 * t270 + t219) * t271
            - 2 * t353
        ) * t274
        t5 = (
            (t270 * t35 - t543) * t278
            + 2 * t361
            + t167 * t452
            + (((t89 + t435) * t278 + t208 * t371 + t433) * t268 + 2 * t351) * t286
        ) * t271 + (
            t541 * t278
            - t439
            - t187 * t530
            + (
                ((t107 + t340) * t278 + (t229 + t532) * t276 + t208 * t449 + 2 * t352)
                * t559
                - t278 * t539
                + (t272 * t274 * t467 + t167 * t268) * t269
            )
            * t286
        ) * t566
        t4 = (
            -(
                (-2 * t438 + (t104 + 2 * t424) * t277 + 2 * t554) * t278
                + (t113 * t271 + t130 * t572 + 2 * t398) * t270
                + t343 * t277
                + (-2 * t208 * t511 + t271 * t63) * t276
                + (t131 - t397) * t574
            )
            * t527
            / 2
            + t555 * t278
            + (t431 + (t531 / 2 + (-t412 + t173 / 2) * t497) * t271 + t78 * t511) * t270
            + (t64 + t432) * t277
            + (t205 * t560 + t271 * t49) * t276
            - t271 * t335
            + t288 * t560
        )
        t3 = (
            ((-t551 - 2 * t537) * t278 + 2 * t167 * t529 + (t184 * t272 + t240) * t562)
            * t276
            + t270
            * (
                (-t277 * t53 - 2 * t535) * t278
                + t272 * t374
                + (t204 * t272 + t226) * t287
            )
            + (
                (
                    -(
                        t34 * t278
                        + t106
                        + (t152 * t272 + t226) * t275
                        + t191 * t272
                        + t476 * t573
                    )
                    * t276
                    + t270
                    * (
                        (t277 * t56 - t581) * t278
                        - t314 * t277
                        + t460 * t275
                        + t193 * t272
                        + t180
                    )
                )
                * t268
                + (
                    (-t272 * t94 - t240) * t567
                    + t270 * (t226 + t513)
                    + (t272 * t493 + (t278 * t446 + (t278 * t516 - t163) * t575) * t295)
                    * t275
                )
                * t274
            )
            * t286
        )
        t2 = (
            (((-t367 + 2 * t472) * t276 - t390) * t277 - t602) * t278
            + (t198 * t269 + t315 + t594) * t277
            + (2 * t355 - t591) * t276
            + (-t87 - 2 * t540) * t275
            - t592
        ) * t268 + t274 * t14
        t1 = ((t375 + t556) * t527 + (t551 + t546) * t270 + t318) * t272 - (
            ((t108 + t387) * t270 + t441) * t527 + t42 * t276 + t71 + t557
        ) * t278
        m[3, 5] = t13 * t279 + t16 * t273
        m[4, 4] = t20 * t273 - t23 * t279
        m[5, 2] = -t278 * t284 - t486
        m[3, 2] = t116 * t279 - t144 * t273
        m[5, 0] = 0
        m[5, 6] = (t18 * t268 - t21 * t274) * t272 - t278 * (t268 * t308 + t274 * t307)
        m[1, 1] = t1 * t273 + t11 * t564
        m[1, 4] = (
            (t12 * t273 + t15 * t564) * t268 + (t19 * t273 + t22 * t279) * t569
        ) * t286
        m[0, 3] = -t273 * t8 + t279 * t3
        m[5, 3] = (t271 * t453 - t277 * t292) * t272 - t278 * (t298 * t445 + t518)
        m[4, 6] = t2 * t273 - t279 * t6
        m[5, 4] = ((-t202 * t277 - t189) * t270 - t276 * t114) * t272 + (
            (-t232 + t484 / 2 - t291 / 2) * t270 + t276 * (t395 - t284 / 2 - t282 / 2)
        ) * t565
        m[5, 1] = 0
        m[1, 2] = t273 * t5 + t279 * t7
        m[2, 0] = 0
        m[0, 1] = t1 * t279 + t11 * t570
        m[0, 2] = -t273 * t7 + t279 * t5
        m[1, 6] = 0
        m[4, 0] = 0
        m[3, 4] = t20 * t279 + t23 * t273
        m[0, 0] = t273 * t601 + t338 * t571 + t4 * t564
        m[2, 4] = (
            (
                (
                    ((-t150 * t276 - t188) * t277 - t388) * t269
                    + (t324 * t277 - t421 + (t255 + t284 / 2 + t259 - t330) * t271)
                    * t568
                )
                * t272
                + t278
                * ((-t329 + t463) * t269 + (-t396 + t409 - t518 / 2 + t492 / 2) * t568)
            )
            * t268
            + (
                (t269 * t332 - t275 * t83) * t272
                + t278 * (t269 * t377 + (-t275 * t161 + (t297 - t500) * t525) * t295)
            )
            * t569
        ) * t286
        m[3, 0] = 0
        m[1, 0] = -t279 * t601 + t338 * t563 + t4 * t571
        m[4, 5] = t13 * t273 - t16 * t279
        m[5, 5] = (
            ((-t276 * t349 - t188) * t277 - t388) * t269
            + t275 * (t370 - t92 + (-t284 + t325) * t271)
        ) * t272 - (
            (-t270 * t382 + t329 + 2 * t418) * t269
            + t275 * (-2 * t396 + 2 * t409 + t492 - t518)
        ) * t278
        m[3, 3] = t279 * t36 + t44 * t571
        m[4, 2] = t116 * t273 + t144 * t279
        m[0, 6] = 0
        m[1, 3] = t273 * t3 + t279 * t8
        m[2, 5] = (
            (t18 * t272 - t278 * t308) * t274 + t268 * (t21 * t272 + t278 * t307)
        ) * t286
        m[2, 2] = (
            (-(t89 + t434) * t527 - t41 * t270 + t543) * t271
            - 2 * t277 * (((t123 + t340) * t559 - t539) * t286 + t541)
        ) * t272 - 2 * (
            ((t126 * t276 + t299 * t583) * t527 - t78 * t503 + t293 * t380) * t271
            + (-(t275 * t276 * t468 + t457) * t527 + t299 * t72) * t277
        ) * t278
        m[2, 3] = (
            (2 * (-t105 * t277 - t183 * t275 + t458) * t527 + t440) * t276
            + (t535 + t49 * t277 + (-(t277 * t62 - t581) * t268 / 2 - t334) * t286)
            * t575
        ) * t272 + (
            t321 * t276
            + t270 * (t374 + t385)
            + (
                (-t268 * t45 + t274 * t510 + t569 * t94) * t276
                + (t126 * t277 + t184 * t275 - t181) * t268 * t576
            )
            * t286
        ) * t278
        m[0, 4] = (
            (t12 * t279 + t15 * t570) * t268 + (t19 * t279 - t22 * t273) * t569
        ) * t286
        m[4, 1] = -t273 * t294 - t279 * t285
        m[2, 1] = (
            (t434 * t277 - (t123 - t491) * t271 + t556) * t527 + t440 * t270 + t318
        ) * t278 + (
            (t270 * t45 + t441) * t527 + t385 * t276 + t299 * (t254 + t326) + t557
        ) * t272
        m[3, 1] = t273 * t285 - t279 * t294
        m[1, 5] = (
            -(
                (t9 * t273 + (t275 * t464 - t597) * t564) * t274
                + t268 * (t14 * t273 + t17 * t563)
            )
            * t286
        )
        m[0, 5] = -t286 * (
            (t14 * t268 + t274 * t9) * t279
            + (((t272 * t376 + t122) * t275 + t597) * t274 + t268 * t17) * t571
        )
        m[4, 3] = t273 * t36 + t44 * t563
        m[3, 6] = t2 * t279 + t273 * t6
        m[2, 6] = 0
        return m

    def velocity_ee(self, q, dq):
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()
        if isinstance(q, np.ndarray):
            m = np.zeros(3)
        else:
            m = ca.SX.zeros(3)
        t574 = q[4]
        t566 = ca.cos(t574)
        t557 = dq[3] * t566 - dq[5]
        t575 = q[3]
        t567 = ca.cos(t575)
        t558 = dq[2] * t567 + dq[4]
        t561 = ca.sin(t575)
        t576 = q[2]
        t568 = ca.cos(t576)
        t562 = ca.sin(t576)
        t590 = t562 * t566
        t596 = -(dq[1] * t566 - t557 * t568) * t561 + t558 * t590
        t577 = q[1]
        t563 = ca.sin(t577)
        t595 = dq[0] * t563
        t594 = dq[1] * t568
        t593 = dq[4] * t561
        t573 = q[5]
        t559 = ca.sin(t573)
        t571 = d7 + d8
        t592 = t559 * t571
        t560 = ca.sin(t574)
        t591 = t562 * t560
        t589 = t566 * t568
        t588 = t567 * t568
        t587 = t568 * t561
        t565 = ca.cos(t573)
        t586 = t571 * t565
        t585 = dq[5] * t565 * t560
        t583 = t562 * t585
        t555 = d5 + d6 + t586
        t553 = dq[5] * t566 * t586 - dq[3] * t555
        t581 = dq[2] * t561 * t562 + dq[1] * t567
        t572 = d3 + d4
        t580 = -dq[1] * t572 - t553 * t588
        t579 = t566 * t588 - t591
        t578 = q[0]
        t570 = ca.cos(t578)
        t569 = ca.cos(t577)
        t564 = ca.sin(t578)
        t556 = dq[4] * t567 + dq[2]
        t552 = (
            t572 * t595
            + t553 * t562 * t567
            + (
                (
                    dq[0] * t579 * t569
                    + (-t557 * t562 + t566 * t595) * t561
                    + t558 * t589
                    - t556 * t591
                )
                * t559
                + t568 * t585
            )
            * t571
            + (-dq[2] * t587 + (t563 * t567 - t569 * t587) * dq[0]) * t555
        )
        t551 = (
            -(
                t596 * t569
                + (dq[0] * t590 + (dq[1] * t589 - t557) * t563) * t567
                + ((t556 * t569 + dq[0]) * t568 + (-dq[1] * t562 + t593) * t563) * t560
            )
            * t592
            + (-t571 * t583 - t580) * t569
            + t553 * t563 * t561
            + (t581 * t569 + (dq[0] * t562 + t563 * t594) * t561) * t555
        )
        m[0] = t551 * t570 - t552 * t564
        m[1] = t551 * t564 + t552 * t570
        m[2] = (
            (-(-t556 * t560 * t568 - t596) * t559 + t583) * t571 - t581 * t555 + t580
        ) * t563 - (
            (dq[1] * t579 - t557 * t567 + t560 * t593) * t592
            - (t555 * t594 + t553) * t561
        ) * t569
        return m

    def acceleration_ee(self, q, dq, ddq):
        d1, d2, d3, d4, d5, d6, d7, d8 = self.get_robot_params()
        if isinstance(q, np.ndarray):
            m = np.zeros(6)
        else:
            m = ca.SX.zeros(6)
        t1 = q[4]
        t2 = ca.sin(t1)
        t3 = dq[3]
        t5 = dq[4]
        t6 = t5 * t3 * t2
        t7 = 0.2e1 * t6
        t8 = ca.cos(t1)
        t9 = ddq[3]
        t10 = t9 * t8
        t11 = ddq[5]
        t12 = -t7 + t10 - t11
        t13 = q[2]
        t14 = ca.cos(t13)
        t15 = t14 * t12
        t16 = dq[2]
        t18 = dq[5]
        t19 = t3 * t8 - t18
        t20 = t19 * t16
        t21 = ca.sin(t13)
        t24 = dq[1]
        t25 = t24 * t2
        t27 = 0.2e1 * t5 * t25
        t28 = ddq[1]
        t29 = t8 * t28
        t31 = q[3]
        t32 = ca.sin(t31)
        t34 = dq[0]
        t35 = t34**2
        t36 = t24**2
        t37 = t16**2
        t38 = t3**2
        t39 = t5**2
        t40 = t18**2
        t43 = t18 * t3
        t44 = 0.2e1 * t43
        t45 = ddq[4]
        t46 = t2 * t45
        t48 = ca.cos(t31)
        t50 = t5 * t16
        t51 = t8 * t50
        t52 = 0.2e1 * t51
        t53 = ddq[2]
        t54 = t53 * t2
        t59 = 0.2e1 * t16 * t2 * t5
        t63 = t19 * t24
        t67 = t8 * t45
        t73 = q[1]
        t74 = ca.cos(t73)
        t76 = ca.sin(t73)
        t77 = t76 * t24
        t90 = t76 * t28
        t91 = t16 * t34
        t100 = t5 * t34
        t101 = t8 * t100
        t103 = t16 * t24
        t104 = t76 * t103
        t105 = ddq[0]
        t107 = t104 - t105 / 0.2e1
        t115 = t2 * t100
        t116 = 0.2e1 * t115
        t126 = t5 * t24
        t130 = t91 + t90 / 0.2e1
        t136 = d7 + d8
        t138 = q[5]
        t139 = ca.sin(t138)
        t141 = ca.cos(t138)
        t142 = t136 * t141
        t143 = t8 * t142
        t144 = t143 * t43
        t145 = 0.2e1 * t144
        t150 = d5 + d6
        t154 = t142 + d5 + d6
        t158 = t8 * t136
        t161 = -t141 * t158 * t18 + t154 * t3
        t162 = t161 * t24
        t167 = t158 * t141 * t11
        t168 = t2 * t18
        t169 = t5 * t168
        t170 = t9 / 0.2e1
        t173 = t141 * t136 * (t169 + t170)
        t175 = t150 * t9
        t176 = t167 - 0.2e1 * t173 - t175
        t178 = t18 * t16
        t180 = t136 * t2 * t141
        t182 = 0.2e1 * t180 * t178
        t185 = t16 * t161
        t188 = t154 * t28
        t192 = t8 * t18 * t5
        t193 = t2 * t11
        t196 = (t192 + t193 / 0.2e1) * t141
        t197 = t21 * t136
        t200 = d3 + d4
        t201 = t28 * t200
        t210 = -t167 / 0.2e1 + t173 + t175 / 0.2e1
        t215 = t48 * t161
        t234 = t18 * t24
        t242 = (
            -t139
            * t136
            * (
                t74
                * (
                    t32 * (-0.2e1 * t20 * t21 + t15 + t27 - t29)
                    + t14
                    * (
                        t48 * (t8 * (t35 + t36 + t37 + t38 + t39 + t40) - t44 + t46)
                        + t52
                        + t54
                    )
                    + t48 * (t21 * (t53 * t8 - t59) - 0.2e1 * t63)
                    - (-t67 + (t35 + t36 + t37 + t39 + t40) * t2) * t21
                )
                + t32
                * (
                    -0.2e1 * t14 * t19 * t77
                    - 0.2e1 * t21 * t19 * t34
                    + t76 * (t8 * (t35 + t36 + t38 + t39 + t40) - t44 + t46)
                )
                + t14
                * (
                    t48 * (t8 * (t90 + 0.2e1 * t91) - 0.2e1 * t25 * t76 * t5)
                    + 0.2e1 * t101
                    - 0.2e1 * t107 * t2
                )
                + t48
                * (
                    t21 * (t8 * (-0.2e1 * t104 + t105) - t116)
                    + 0.2e1 * t76 * (t6 - t10 / 0.2e1 + t11 / 0.2e1)
                )
                - 0.2e1 * t21 * (t126 * t76 * t8 + t130 * t2)
            )
            + t74
            * (
                t32
                * (
                    t14
                    * (
                        -t145
                        + t141 * t136 * (t35 + t36 + t37 + t38 + t40)
                        + t150 * (t35 + t36 + t37 + t38)
                    )
                    + t21 * t53 * t154
                    - 0.2e1 * t162
                )
                + t14 * (t176 * t48 - t182)
                + t48 * (0.2e1 * t185 * t21 + t188)
                - 0.2e1 * t197 * t196
                + t201
            )
            + t32
            * (
                -0.2e1 * t107 * t154 * t21
                + 0.2e1 * t130 * t14 * t154
                - 0.2e1 * t210 * t76
            )
            + t14 * (-0.2e1 * t18 * t180 * t34 + 0.2e1 * t215 * t77)
            + t48
            * (
                0.2e1 * t21 * t34 * t161
                - (
                    -t145
                    + t141 * t136 * (t35 + t36 + t38 + t40)
                    + t150 * (t35 + t36 + t38)
                )
                * t76
            )
            - t76 * (-0.2e1 * t180 * t21 * t234 + t200 * (t35 + t36))
        )
        t243 = q[0]
        t244 = ca.cos(t243)
        t246 = ca.sin(t243)
        t254 = t8 * t105
        t279 = t34 * t76
        t280 = t24 * t279
        t299 = t280 - t53 / 0.2e1
        t318 = t21 * t18
        t319 = t180 * t318
        t320 = t24 * t200
        t351 = t141 * t178
        t352 = t2 * t21
        t358 = (
            -t139
            * t136
            * (
                t74
                * (
                    -0.2e1 * t32 * t34 * (-t14 * t19 + t24 * t8)
                    + t14 * (t48 * (t116 - t254) + 0.2e1 * t2 * t91)
                    + 0.2e1 * (t8 * t48 * t91 + t101 + t105 * t2 / 0.2e1) * t21
                )
                + t32
                * (0.2e1 * t14 * t20 + t21 * t12 + 0.2e1 * t76 * (-t254 / 0.2e1 + t115))
                + t14
                * (
                    t48 * (t8 * (0.2e1 * t280 - t53) + t59)
                    - t67
                    + (t35 + t37 + t39 + t40) * t2
                )
                + t48
                * (
                    t21 * (t8 * (t35 + t37 + t38 + t39 + t40) - t44 + t46)
                    - 0.2e1 * t19 * t279
                )
                - 0.2e1 * (t2 * t299 - t51) * t21
            )
            / 0.2e1
            + t74
            * (
                t32 * t154 * (t16 * t34 * t21 - t14 * t105 / 0.2e1)
                + (t154 * t24 * t48 - t14 * t215 - t319 + t320) * t34
            )
            + t32
            * (
                t14 * t154 * t299
                + t21
                * (
                    -t144
                    + t141 * t136 * (t35 + t37 + t38 + t40) / 0.2e1
                    + t150 * (t35 + t37 + t38) / 0.2e1
                )
                - t34 * t161 * t76
            )
            + t14 * (t136 * t196 - t185 * t48)
            + t48 * (-t21 * t210 + t105 * t154 * t76 / 0.2e1)
            - t136 * t352 * t351
            + t200 * t76 * t105 / 0.2e1
        )
        m[0] = t242 * t244 - 0.2e1 * t246 * t358
        m[1] = t242 * t246 + 0.2e1 * t244 * t358
        t364 = t16 * t21
        t365 = t3 * t364
        t379 = t21 * t53
        t380 = t3 * t24
        t443 = t24 * t21
        t444 = t16 * t443
        m[2] = t76 * (
            t139
            * (
                t32 * (t15 + t8 * (-0.2e1 * t365 - t28) + t27 + 0.2e1 * t16 * t318)
                + t14
                * (t48 * (t8 * (t36 + t37 + t38 + t39 + t40) - t44 + t46) + t52 + t54)
                + t48 * (t8 * (t379 - 0.2e1 * t380) - 0.2e1 * t50 * t352 + 0.2e1 * t234)
                - (-t67 + (t36 + t37 + t39 + t40) * t2) * t21
            )
            * t136
            + t32
            * (
                t14
                * (
                    t145
                    - t141 * t136 * (t36 + t37 + t38 + t40)
                    - t150 * (t36 + t37 + t38)
                )
                - 0.2e1 * t143 * t234
                + 0.2e1 * (t380 - t379 / 0.2e1) * t154
            )
            + t14 * (-t176 * t48 + t182)
            + t48 * (0.2e1 * t8 * t197 * t351 - 0.2e1 * t154 * (t365 + t28 / 0.2e1))
            + 0.2e1 * t136 * t21 * t141 * t192
            + t180 * t21 * t11
            - t201
        ) - t74 * (
            t139
            * t136
            * (
                t32 * (-0.2e1 * t14 * t63 + t8 * (t36 + t38 + t39 + t40) - t44 + t46)
                + t14 * (t48 * (-t27 + t29) - 0.2e1 * t2 * t103)
                + t48 * (t8 * (-0.2e1 * t444 - t9) + t7 + t11)
                - 0.2e1 * (t8 * t126 + t2 * t28 / 0.2e1) * t21
            )
            + t32
            * (
                -t14 * t188
                - t167
                + 0.2e1 * t141 * (t444 + t169 + t170) * t136
                + 0.2e1 * t150 * (t444 + t170)
            )
            - 0.2e1 * t14 * t48 * t162
            + t48 * (-t145 + t141 * t136 * (t36 + t38 + t40) + t150 * (t36 + t38))
            + (-0.2e1 * t319 + t320) * t24
        )
        t484 = ddq[6]
        t485 = t139 * t484
        t486 = dq[6]
        t487 = t141 * t486
        t488 = -t487 + t5
        t489 = t488 * t18
        t490 = t485 - t489
        t492 = t486 * t5
        t493 = t139 * t2
        t494 = t493 * t492
        t495 = t487 + t5
        t496 = t495 * t3
        t506 = -t139 * t3 * t486 * t8 + t139 * t18 * t486 - t141 * t484 + t2 * t43 - t45
        t510 = t8 * t18
        t511 = -t139 * t2 * t486 + t3 - t510
        t512 = t511 * t16
        t515 = t8 * t486
        t517 = -t139 * t515 + t168
        t523 = -t139 * t492 - t11
        t528 = (
            -t139 * t2 * t484
            + t16 * t32 * t495
            + t16 * t48 * t517
            + t2 * t489
            + t523 * t8
            + t9
        )
        t530 = t48 * t495
        t531 = -t517
        t537 = t48 * t517
        t539 = t495 * t32
        t555 = -t490 * t8 + t193 + t494 + t496
        t560 = (
            t74
            * (
                t14 * (t48 * (t490 * t8 - t193 - t494 - t496) + t32 * t506 + t512)
                + t21 * t528
                + (t32 * t531 + t16 + t530) * t24
            )
            + t14 * (t34 * t511 + t537 * t77 + t539 * t77)
            + t21 * (t32 * t34 * t495 + t34 * t48 * t517 - t511 * t77)
            - t48 * t506 * t76
            - t32 * t555 * t76
            + t76 * t53
            - t24 * t34
        )
        t571 = -t506
        t572 = t32 * t571
        t579 = (
            -t74 * (t14 * (t48 * t531 - t539) + t511 * t21) * t34
            + t14 * t528
            + t21 * (t48 * t555 - t512 + t572)
            - t530 * t279
            + t32 * t517 * t279
            - t16 * t279
            - t28
        )
        m[3] = t244 * t560 + t246 * t579
        m[4] = -t244 * t579 + t246 * t560
        t583 = -t488
        t585 = -t18 * t583 - t485
        m[5] = (
            t76
            * (
                t14 * (t48 * (t585 * t8 + t193 + t494 + t496) + t572 - t511 * t16)
                + t48 * (t139 * t16 * t21 * t515 - t16 * t2 * t318 - t24 * t495)
                + t32 * (-t139 * t24 * t515 + t168 * t24 - t364 * t495)
                - t8 * t523 * t21
                + t493 * t21 * t484
                + t2 * t583 * t318
                - t21 * t9
                - t103
            )
            + t74
            * (
                t14 * (t537 + t539) * t24
                + t48 * t571
                + t32 * (-t585 * t8 - t193 - t494 - t496)
                + t24 * t493 * t21 * t486
                + t443 * t510
                - t3 * t443
                + t53
            )
            + t105
        )
        return m

    def omega_ee(self, q, dq):
        if isinstance(q, np.ndarray):
            m = np.zeros(3)
        else:
            m = ca.SX.zeros(3)
        t708 = q[4]
        t697 = ca.sin(t708)
        t702 = ca.cos(t708)
        t707 = q[5]
        t715 = dq[6] * ca.sin(t707)
        t693 = dq[5] * t702 + t697 * t715 - dq[3]
        t710 = q[2]
        t699 = ca.sin(t710)
        t704 = ca.cos(t710)
        t694 = -dq[5] * t697 + t702 * t715
        t695 = dq[6] * ca.cos(t707) + dq[4]
        t709 = q[3]
        t698 = ca.sin(t709)
        t703 = ca.cos(t709)
        t713 = t694 * t703 - t695 * t698
        t716 = -t693 * t699 + t704 * t713
        t712 = q[0]
        t711 = q[1]
        t706 = ca.cos(t712)
        t705 = ca.cos(t711)
        t701 = ca.sin(t712)
        t700 = ca.sin(t711)
        t691 = t694 * t698 + t695 * t703 + dq[2]
        t690 = -t693 * t704 - t699 * t713 - dq[1]
        t689 = t691 * t700 + t705 * t716
        m[0] = t689 * t706 + t690 * t701
        m[1] = t689 * t701 - t690 * t706
        m[2] = t691 * t705 - t700 * t716 + dq[0]
        return m

    def manipulability_measure(self, q):
        t366 = ca.cos(q[4])
        t356 = 0.17640000e0 * t366**2
        t368 = ca.cos(q[2])
        t358 = 0.16000000e0 * t368**2
        t371 = q[3]
        t364 = ca.sin(t371)
        t370 = q[5]
        t365 = ca.cos(t370)
        t367 = ca.cos(t371)
        t373 = (
            -0.4200e0 * t364 * ca.sin(t370) * t366 * (0.4200e0 * t367 + 0.4000e0) * t365
            + 0.33640000e0
        )
        t372 = q[1]
        t369 = ca.cos(t372)
        t361 = t367**2
        t359 = t365**2
        t357 = 0.33600000e0 * t367
        m = (
            0.5644800000e-1
            * t364**2
            * (
                (-t356 * t361 + t356 - 0.1e1 * t357 - 0.33640000e0) * t359
                + t357
                + (
                    0.4000e0
                    * ca.sin(t372)
                    * t368
                    * (t365 - 0.1e1)
                    * t364
                    * (t365 + 0.1e1)
                    * (0.4000e0 * t367 + 0.4200e0)
                    + (
                        (
                            (t356 + t358) * t361
                            + t357
                            - 0.1e1 * t358
                            - 0.1e1 * t356
                            + 0.33640000e0
                        )
                        * t359
                        - t361 * t358
                        - 0.1e1 * t357
                        + t358
                        - t373
                    )
                    * t369
                )
                * t369
                + t373
            )
        )
        return m


# if __name__ == '__main__':
#     model = RobotModel()
