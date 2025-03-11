from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from scipy.spatial.transform import Rotation as R

USE_IIWA = True


class RobotModel:
    def __init__(self):
        pinocchio_model_dir = Path(__file__).parent
        model_path = pinocchio_model_dir
        mesh_dir = pinocchio_model_dir
        if USE_IIWA:
            urdf_filename = "iiwa.urdf"
        else:
            urdf_filename = "gen3_arm.urdf"
        urdf_model_path = model_path / urdf_filename
        self.model, collision_model, visual_model = pin.buildModelsFromUrdf(
            urdf_model_path, package_dirs=mesh_dir
        )
        self.ee_id = self.model.getFrameId("end_effector_link")
        self.col_ids = [
            self.model.getJointId("joint_3"),
            self.model.getJointId("joint_4"),
            self.model.getJointId("joint_5"),
            self.model.getJointId("joint_6"),
            self.model.getJointId("joint_7"),
            self.model.getFrameId("link4_col_link"),
            self.model.getFrameId("end_effector_col_link"),
        ]
        if USE_IIWA:
            self.col_joint_sizes = [0.09, 0.12, 0.09, 0.10, 0.07, 0.09, 0.075]
        else:
            self.col_joint_sizes = [0.09, 0.09, 0.06, 0.06, 0.06, 0.06, 0.075]
        self.cmodel = cpin.Model(self.model)
        # [self.cmodel.names[i] for i in range(7)]
        # Joint limits
        self.q_lim_lower = self.model.lowerPositionLimit
        self.q_lim_upper = self.model.upperPositionLimit
        if not USE_IIWA:
            self.q_lim_lower[[0, 2, 4, 6]] = -np.inf
            self.q_lim_upper[[0, 2, 4, 6]] = np.inf
        self.dq_lim_lower = -self.model.velocityLimit
        self.dq_lim_upper = self.model.velocityLimit
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
        if isinstance(q, np.ndarray):
            data = self.model.createData()
            pin.framesForwardKinematics(self.model, data, q)
        else:
            data = self.cmodel.createData()
            cpin.framesForwardKinematics(self.cmodel, data, q)
        p = data.oMf[self.ee_id].translation
        return p

    def fk_pos_col(self, q, i):
        if isinstance(q, np.ndarray):
            data = self.model.createData()
            pin.forwardKinematics(self.model, data, q)
            pin.framesForwardKinematics(self.model, data, q)
        else:
            data = self.cmodel.createData()
            cpin.forwardKinematics(self.cmodel, data, q)
            cpin.framesForwardKinematics(self.cmodel, data, q)
        if i < 5:
            p = data.oMi[self.col_ids[i]].translation
        else:
            p = data.oMf[self.col_ids[i]].translation
        return p

    def fk(self, q):
        """Compute the end effector position of the robot in cartesian space
        given the joint configuration.
        """
        if isinstance(q, np.ndarray):
            m = np.zeros(6)
        else:
            m = ca.SX.zeros(6)
        h = self.hom_transform_endeffector(q)
        m[:3] = h[:3, 3]
        m[3:] = R.from_matrix(h[:3, :3]).as_rotvec()

        return m

    def hom_transform_endeffector(self, q):
        if isinstance(q, np.ndarray):
            data = self.model.createData()
            pin.framesForwardKinematics(self.model, data, q)
            p = data.oMf[self.ee_id].homogeneous
        else:
            data = self.cmodel.createData()
            cpin.framesForwardKinematics(self.cmodel, data, q)
            p = data.oMf[self.ee_id].homogeneous
        return p

    def jacobian_fk(self, q):
        if isinstance(q, np.ndarray):
            data = self.model.createData()
            pin.computeForwardKinematicsDerivatives(self.model, data, q, q, q)
            jac = pin.getFrameJacobian(
                self.model, data, self.ee_id, pin.LOCAL_WORLD_ALIGNED
            )
        else:
            data = self.cmodel.createData()
            cpin.computeForwardKinematicsDerivatives(self.cmodel, data, q, q, q)
            jac = cpin.getFrameJacobian(
                self.cmodel, data, self.ee_id, pin.LOCAL_WORLD_ALIGNED
            )
        return jac

    def djacobian_fk(self, q, dq):
        if isinstance(q, np.ndarray):
            data = self.model.createData()
            pin.computeForwardKinematicsDerivatives(self.model, data, q, dq, dq)
            djac = pin.getFrameJacobianTimeVariation(
                self.model, data, self.ee_id, pin.LOCAL_WORLD_ALIGNED
            )
        else:
            data = self.cmodel.createData()
            cpin.computeForwardKinematicsDerivatives(self.cmodel, data, q, dq, dq)
            djac = cpin.getFrameJacobianTimeVariation(
                self.cmodel, data, self.ee_id, pin.LOCAL_WORLD_ALIGNED
            )
        return djac

    def ddjacobian_fk(self, q, dq, ddq):
        if isinstance(q, np.ndarray):
            data = self.model.createData()
            pin.computeForwardKinematicsDerivatives(self.model, data, q, dq, ddq)
            ddjac = data.ddJ
        else:
            data = self.cmodel.createData()
            cpin.computeForwardKinematicsDerivatives(self.cmodel, data, q, dq, ddq)
            cpin.forwardKinematics(self.cmodel, data, q)
            ddjac = data.ddJ
        return ddjac

    def velocity_ee(self, q, dq):
        jac = self.jacobian_fk(q)
        v = jac @ dq
        return v[:3]

    def acceleration_ee(self, q, dq, ddq):
        jac = self.jacobian_fk(q)
        djac = self.djacobian_fk(q, dq)
        a = djac @ dq + jac @ ddq
        return a

    def omega_ee(self, q, dq):
        jac = self.jacobian_fk(q)
        w = jac @ dq
        return w[3:]


if __name__ == "__main__":
    model = RobotModel()
    p = model.fk(np.zeros(7))
    model.fk(np.zeros(7))
    model.velocity_ee(np.zeros(7), np.zeros(7))
