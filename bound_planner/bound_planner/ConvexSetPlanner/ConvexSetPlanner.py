import time
from concurrent.futures import ThreadPoolExecutor

import casadi as ca

# import graphviz
# import pathlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from geometry_msgs.msg import Point
from pydrake.common import RandomGenerator
from pydrake.geometry.all import HPolyhedron
from pydrake.geometry.optimization import GraphOfConvexSets, GraphOfConvexSetsOptions
from pypoman import compute_polytope_vertices
from rclpy.node import Node
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray

from bound_planner.BoundMPC.mpc_utils_casadi import (
    compute_bound_params_six,
    compute_bound_params_three,
    compute_sixth_order_error_bound_general,
)
from bound_planner.ConvexSetPlanner.ConvexSetFinder import ConvexSetFinder
from bound_planner.RobotModel import RobotModel
from bound_planner.utils import (
    get_ellipsoid_params,
    gram_schmidt,
    normalize_set_size,
    plot_set,
)
from bound_planner.utils.lie_functions import rodrigues_matrix


def transform_poly(poly, r, trans, obs_size_increase=0.05):
    b_trans = poly.b() + obs_size_increase + poly.A() @ r @ trans
    trans_poly = HPolyhedron(poly.A() @ r, b_trans)
    return trans_poly


def adapt_to_ee_size(poly, obs_size_increase=0.05):
    b_trans = poly.b() + obs_size_increase
    adapted_poly = HPolyhedron(poly.A(), b_trans)
    return adapted_poly


def squared_error(x, xd, xdn, A, b):
    # return np.sum((x - xd)**2) + np.sum((x - xdn)**2) + 10 * (np.max(A @ x - b) + 0.03)**2
    return (
        np.sum((x - xd) ** 2)
        + np.sum((x - xdn) ** 2)
        + 3 * np.min(((A @ x - b) + 0.05) ** 2)
    )


def max_rect_volume_loss(x):
    return -(x[0] - x[2]) * (x[1] - x[3])


def projection_opt_problem(max_set_size=20):
    a_set = ca.SX.sym("a set", max_set_size, 3)
    b_set = ca.SX.sym("b set", max_set_size)
    xd = ca.SX.sym("xd", 3)

    params = ca.vertcat(a_set.reshape((-1, 1)), b_set, xd)

    g = []
    lbg = []
    ubg = []
    u = []
    lbu = []
    ubu = []

    x = ca.SX.sym("x", 3)
    u += [x]
    lbu += [-np.inf] * 3
    ubu += [np.inf] * 3

    J = ca.sumsqr(x - xd)

    g += [a_set @ x - b_set]
    lbg += [-np.inf] * max_set_size
    ubg += [0] * max_set_size

    prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

    solver = ca.qpsol(
        "solver", "qpoases", prob, {"print_time": False, "printLevel": "none"}
    )
    return solver, lbu, ubu, lbg, ubg


class ConvexSetPlanner(Node):
    def __init__(
        self,
        obstacles=[],
        e_p_max=0.5,
        obs_size_increase=0.05,
        workspace_max=[1.0, 1.0, 1.2],
        workspace_min=[-1.0, -1.0, 0.0],
    ):
        super().__init__("convex_planner")
        self.rviz_pub = self.create_publisher(MarkerArray, "/r1/obs_marker", 1)
        self.plan_msgs = []
        self.iris_executor = ThreadPoolExecutor(max_workers=15)
        self.robot_model = RobotModel()
        self.replanning = False

        self.visualize = True
        self.visualize_graph = False
        self.visualize_plan = False
        self.visualize_via_path = False
        self.visualize_obstacles = True
        self.visualize_sets = False
        self.obs_size_increase = obs_size_increase
        self.rviz_marker_msg = None
        self.comp_time_set = 0
        self.comp_time_edge = 0
        self.comp_time_fit = 0
        self.comp_time_total = 0
        self.comp_time_graph = 0
        self.comp_time_path = 0
        self.comp_time_via = 0
        self.ginfo = {}
        self.vinfo = {}
        self.w_size = 0.1
        self.w_bias = 0.01
        self.generator = RandomGenerator()
        self.rng = np.random.default_rng()
        self.max_set_size = 20
        self.workspace_max = workspace_max
        self.workspace_min = workspace_min
        self.max_iters = 20
        self.max_samples = 500

        self.solver_name = "ipopt"
        self.ipopt_options = {
            "tol": 10e-4,
            "max_iter": 500,
            "limited_memory_max_history": 6,
            "limited_memory_initialization": "scalar1",
            "limited_memory_max_skipping": 2,
            # "linear_solver": "ma57",
            # "linear_system_scaling": "mc19",
            # "ma57_automatic_scaling": "no",
            # "ma57_pre_alloc": 100,
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

        self.solver_opts = {
            "verbose": False,
            "verbose_init": False,
            "print_time": False,
            "ipopt": self.ipopt_options,
        }

        # Maximum position error
        self.e_p_max = e_p_max

        # Projection solver
        (
            self.proj_solver,
            self.proj_lbu,
            self.proj_ubu,
            self.proj_lbg,
            self.proj_ubg,
        ) = projection_opt_problem(self.max_set_size)

        # Fit solver
        # self.solver_fit, self.lbu_fit, self.ubu_fit, self.lbg_fit, self.ubg_fit = self.fit_opt_problem(self.max_set_size)
        (
            self.solver_fit,
            self.lbu_fit,
            self.ubu_fit,
            self.lbg_fit,
            self.ubg_fit,
        ) = self.fit_opt_problem_sample(self.max_set_size)

        # Via path optimizers and bound function optimizer
        self.solver_via = []
        self.lbu_via = []
        self.ubu_via = []
        self.lbg_via = []
        self.ubg_via = []
        self.solver_via_rot = []
        self.lbu_via_rot = []
        self.ubu_via_rot = []
        self.lbg_via_rot = []
        self.ubg_via_rot = []
        for i in range(1, 5):
            solver, lbu, ubu, lbg, ubg = self.via_point_optimization_problem(
                i, self.max_set_size
            )
            self.solver_via.append(solver)
            self.lbu_via.append(lbu)
            self.ubu_via.append(ubu)
            self.lbg_via.append(lbg)
            self.ubg_via.append(ubg)
            solver, lbu, ubu, lbg, ubg = self.via_point_rot_optimization_problem(
                i, self.max_set_size
            )
            self.solver_via_rot.append(solver)
            self.lbu_via_rot.append(lbu)
            self.ubu_via_rot.append(ubu)
            self.lbg_via_rot.append(lbg)
            self.ubg_via_rot.append(ubg)

        # Obstacles
        self.obs = []
        for ob in obstacles:
            self.obs.append(HPolyhedron.MakeBox(ob[:3], ob[3:]))
        for ob in obstacles:
            ob[0] -= self.obs_size_increase
            ob[1] -= self.obs_size_increase
            ob[2] -= self.obs_size_increase
            ob[3] += self.obs_size_increase
            ob[4] += self.obs_size_increase
            ob[5] += self.obs_size_increase
        self.adapted_obs = []
        for i in range(len(self.obs)):
            adapted_ob = adapt_to_ee_size(self.obs[i], self.obs_size_increase)
            self.adapted_obs.append(adapted_ob)

        # Compute obstacle points
        self.obs_points = np.empty((0, 3))
        self.obs_sets = []
        self.obs_points_sets = []
        for i in range(len(self.obs)):
            points = np.array(
                compute_polytope_vertices(
                    self.adapted_obs[i].A(), self.adapted_obs[i].b()
                )
            )
            self.obs_points = np.concatenate((self.obs_points, points))
            self.obs_points_sets.append(points)
            self.obs_sets.append([self.adapted_obs[i].A(), self.adapted_obs[i].b()])

        self.obs_sets = normalize_set_size(self.obs_sets)

        # Create set finder object
        self.set_finder = ConvexSetFinder(
            self.obs_sets,
            self.obs_points,
            self.obs_points_sets,
            self.workspace_max,
            self.workspace_min,
        )

    def print_computation_time(self):
        print(
            f"(PosPath) Computed {self.nr_sets} sets with {self.nr_inter_set} intersections leading to {self.nr_edges} edges"
        )
        print(f"(PosPath) Building graph of convex sets: {self.comp_time_graph:.4f}s")
        print(
            f"(PosPath) -> Shortest path of sets computation: {self.comp_time_path:.4f}s"
        )
        print(f"(PosPath) -> Via point optimization: {self.comp_time_via:.4f}s")
        print(f"(PosPath) -> Set computation: {self.comp_time_set:.4f}s")
        print(f"(PosPath) --> MVIE solver total {self.set_finder.ell_time:.4f}s")
        print(
            f"(PosPath) --> Projection finder total time: {self.set_finder.proj_time:.4f}s"
        )
        print(f"(PosPath) -> Edge computation: {self.comp_time_edge:.4f}s")
        print(f"(PosPath) --> Fit computation: {self.comp_time_fit:.4f}s")
        print(f"(PosPath) Total time: {self.comp_time_total:.4f}s")
        print("---------------------------")
        print()

    def plan_convex_set_path(self, start, end, r0, r1, replanning=False, p_horizon=[]):
        start_time_total = time.perf_counter()
        self.ginfo = {}
        self.vinfo = {}
        self.replanning = replanning
        self.p_horizon = p_horizon
        self.set_finder.proj_time = 0.0
        self.set_finder.ell_time = 0.0
        self.comp_time_set = 0.0
        self.comp_time_edge = 0.0
        self.comp_time_fit = 0.0
        self.comp_time_path = 0.0
        self.comp_time_via = 0.0
        if self.visualize:
            self.rviz_marker_msg = MarkerArray()
            msg = Marker()
            msg.header.frame_id = "r1/world"
            msg.action = Marker.DELETEALL
            self.rviz_marker_msg.markers.append(msg)
            self.rviz_pub.publish(self.rviz_marker_msg)

        if self.visualize and self.visualize_obstacles:
            for i, ob in enumerate(self.obs):
                self.rviz_marker_msg.markers.append(
                    self.create_marker_msg(ob, f"Obs {i}", 1, 0, 0, 0.2)
                )

        # Max workspace
        bounding_box = HPolyhedron.MakeBox([-2, -2, 0], [2, 2, 1.5])

        # Project end point to collision free space
        for ob in self.adapted_obs:
            constraint_viol = ob.A() @ end - ob.b()
            if not np.any(constraint_viol > 0):
                print("(PosPath) Projecting end point to collision free space")
                idx = np.argmax(constraint_viol)
                end -= constraint_viol[idx] * ob.A()[idx, :]

        # Compute direction of end effector at start and end point
        self.omega = R.from_matrix(r1 @ r0.T).as_rotvec()
        self.omega_norm = np.linalg.norm(self.omega)
        self.omega_normed = self.omega / self.omega_norm
        self.l_ee = r0 @ np.array([0, 0, -self.robot_model.d8])
        self.l_ee_end = r1 @ np.array([0, 0, -self.robot_model.d8])
        b1rd = np.array([0, 0, 1.0])
        self.br1 = gram_schmidt(self.omega_normed, b1rd)
        self.br1 /= np.linalg.norm(self.br1)
        self.br2 = np.cross(self.omega_normed, self.br1)
        self.br2[-1] /= np.linalg.norm(self.br2[-1])

        # Create graphs
        graph = GraphOfConvexSets()
        inter_graph = nx.Graph()
        self.nr_sets = 0
        self.nr_edges = 0
        self.nr_inter_set = 0

        # Grow convex sets
        start_time = time.perf_counter()
        self.known_sets = []
        t_start_set = time.perf_counter()
        if self.replanning:
            max_horizon_idx = 0
            for k, s in enumerate(self.sets_via_prev):
                dist_start = s[0] @ start - s[1]
                dist_horizon = s[0] @ np.array(p_horizon).T - np.expand_dims(s[1], 1)
                start_in = np.max(dist_start) < 1e-8
                horizon_in = np.max(dist_horizon, axis=0) < 1e-8
                horizon_idx = np.where(np.logical_not(horizon_in))[0]
                start_in_set = False
                if start_in:
                    print(f"(Replanning) Start in set {k+1} {np.max(dist_start)}")
                    start_in_set = True
                else:
                    print(f"(Replanning) Start NOT in set {k+1} {np.max(dist_start)}")
                if horizon_idx.shape[0] > 0:
                    if horizon_idx[0] == 0:
                        print(f"(Replanning) Horizon NOT in set {k+1} {horizon_in}")
                    else:
                        print(
                            f"(Replanning) Horizon in set {k+1} up to idx {horizon_idx[0]}"
                        )
                        if start_in_set:
                            max_horizon_idx = np.max(
                                (max_horizon_idx, horizon_idx[0] - 1)
                            )
                            print(f"(Replanning) New horizon idx {max_horizon_idx}")
                elif horizon_idx.shape[0] == 0:
                    print(f"(Replanning) Full Horizon in set {k+1}")
                    if start_in_set:
                        max_horizon_idx = len(p_horizon) - 1
                        print(f"(Replanning) New horizon idx {max_horizon_idx}")
                        break
            self.p_horizon_max = p_horizon[max_horizon_idx]
            print(f"(Replanning) p horizon: {self.p_horizon_max}")
            # (
            #     a_set,
            #     b_set,
            #     q_ellipse_start,
            #     p_mid_start,
            # ) = self.set_finder.find_set_around_line(
            #     start, self.p_horizon_max - start, optimize=False
            # )
            (
                a_set,
                b_set,
                q_ellipse_start,
                p_mid_start,
            ) = self.set_finder.find_set_collision_avoidance(
                start, self.p_horizon_max, True
            )
        else:
            (
                a_set,
                b_set,
                q_ellipse_start,
                p_mid_start,
            ) = self.set_finder.find_set_around_point(start, fixed_mid=True)
            if np.max(a_set @ (start + self.l_ee) - b_set) > 1e-8:
                (
                    a_set,
                    b_set,
                    q_ellipse_start,
                    p_mid_start,
                ) = self.set_finder.find_set_collision_avoidance(
                    start, start + self.l_ee, True
                )
                # (
                #     a_set,
                #     b_set,
                #     q_ellipse_start,
                #     p_mid_start,
                # ) = self.set_finder.find_set_around_line(
                #     start, self.l_ee, optimize=False
                # )
        self.comp_time_set += time.perf_counter() - t_start_set
        self.known_sets.append((a_set, b_set))

        region = HPolyhedron(a_set, b_set)
        region_start = region.ReduceInequalities(1e-4)
        vertex_start = graph.AddVertex(region_start, name="Vertex start")
        self.id = 0
        inter_graph.add_node(region_start, id=self.id, name="Vertex start")
        self.ginfo[self.id] = {
            "vertex0": vertex_start,
            "vertex1": vertex_start,
            "conn_to_start": True,
            "conn_to_end": False,
            "p_proj": start,
            "edge": None,
            "p_via": np.concatenate((start, [0.0])),
        }
        self.vinfo[vertex_start.id()] = {
            # "size": vertex_start.set().CalcVolumeViaSampling(self.generator, 0.05, 1000).volume,
            "size": 1 / np.linalg.det(q_ellipse_start),
            "q_ellipse": q_ellipse_start,
            "p_mid": p_mid_start,
            "a_set": np.array(a_set),
            "b_set": np.array(b_set),
        }
        self.nr_sets += 1
        # if self.replanning and not in_same_set:
        #     if vertex_start.set().IntersectsWith(vertex_horizon.set()):
        #         inter_set = vertex_start.set().Intersection(vertex_horizon.set())
        #         vertex_inter_horizon = inter_graph.AddVertex(
        #             inter_set, name="Vertex horizon"
        #         )
        #         edge_inter = inter_graph.AddEdge(vertex_start, vertex_inter_horizon)
        #         edge_inter.AddCost(0.0)
        #         edge_inter = inter_graph.AddEdge(vertex_inter_horizon, vertex_start)
        #         edge_inter.AddCost(0.0)
        #         self.ginfo[vertex_inter_horizon.id()] = {
        #             "vertex0": vertex_start,
        #             "vertex1": vertex_horizon,
        #             "conn_to_start": True,
        #             "conn_to_end": False,
        #             "p_proj": p_horizon[-1],
        #             "edge": None,
        #             "p_via": np.concatenate((p_horizon[-1], [0.1])),
        #         }

        t_start_edge = time.perf_counter()
        connected = self.add_edges(
            vertex_start, region_start, graph, inter_graph, end, start
        )
        self.comp_time_edge += time.perf_counter() - t_start_edge

        t_start_set = time.perf_counter()
        # a_set, b_set, q_ellipse_end, p_mid_end = self.set_finder.find_set_around_line(
        #     end, self.l_ee_end, optimize=False
        # )
        (
            a_set,
            b_set,
            q_ellipse_end,
            p_mid_end,
        ) = self.set_finder.find_set_collision_avoidance(end, end + self.l_ee_end, True)
        self.comp_time_set += time.perf_counter() - t_start_set
        self.known_sets.append((a_set, b_set))
        region = HPolyhedron(a_set, b_set)
        region_end = region.ReduceInequalities(1e-4)
        vertex_end = graph.AddVertex(region_end, name="Vertex end")
        self.id += 1
        inter_graph.add_node(region_end, id=self.id, name="Vertex end")
        self.ginfo[self.id] = {
            "vertex0": vertex_end,
            "vertex1": vertex_end,
            "conn_to_start": False,
            "conn_to_end": True,
            "p_proj": end,
            "edge": None,
            "p_via": np.concatenate((end, [1.0])),
        }
        self.vinfo[vertex_end.id()] = {
            # "size": vertex_end.set().CalcVolumeViaSampling(self.generator, 0.05, 1000).volume,
            "size": 1 / np.linalg.det(q_ellipse_end),
            "q_ellipse": q_ellipse_end,
            "p_mid": p_mid_end,
            "a_set": np.array(a_set),
            "b_set": np.array(b_set),
        }
        t_start_edge = time.perf_counter()
        self.nr_sets += 1
        conn = self.add_edges(vertex_end, region_end, graph, inter_graph, end, start)
        connected = conn or connected
        self.comp_time_edge += time.perf_counter() - t_start_edge

        if self.visualize and self.visualize_sets:
            self.rviz_marker_msg.markers.append(
                self.create_marker_msg(vertex_start.set(), "set start", 0, 0, 1)
            )
            self.rviz_marker_msg.markers.append(
                self.create_marker_msg(vertex_end.set(), "set end", 0, 0, 1)
            )
            self.rviz_pub.publish(self.rviz_marker_msg)

        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = 10
        # self.obs_pub.publish(self.rviz_obs_msg)
        # Sample points until solution is found
        j = 0
        nr_samples = 0
        success = False
        p_via_old = None
        while not success:
            via_sample = False
            if connected:
                t_start_path = time.perf_counter()
                path = nx.shortest_path(
                    inter_graph, region_start, region_end, weight="weight"
                )
                id_dict = nx.get_node_attributes(inter_graph, "id")
                self.comp_time_path += time.perf_counter() - t_start_path
                t_start_path = time.perf_counter()
                p_via, p_via_list, omega_via, sets_via, _, _ = self.compute_via_points(
                    path, start, end, id_dict
                )
                self.comp_time_via += time.perf_counter() - t_start_path

                if p_via_old is not None and p_via_old.shape == p_via.shape:
                    if np.linalg.norm(p_via_old - p_via) < 1e-4:
                        print("(PosPath) Found path solution")
                        success = True
                        break
                    else:
                        print("(PosPath) Sampling via points to refine graph")
                        samples = p_via_list[1:-1]
                    # success = True
                else:
                    samples = p_via_list[1:-1]
                    print("(PosPath) Sampling via points to improve graph")
                    via_sample = True
                p_via_old = np.copy(p_via)
            else:
                in_collision = True
                in_safe = True
                nr_sampled = 0
                while (in_collision or in_safe) and nr_sampled <= self.max_samples:
                    in_collision = False
                    in_safe = False
                    sample = self.rng.uniform(self.workspace_min, self.workspace_max, 3)
                    nr_sampled += 1
                    for ob in self.adapted_obs:
                        constraint_viol = ob.A() @ sample - ob.b()
                        if np.max(constraint_viol) < 1e-3:
                            in_collision = True
                            break
                    for a, b in self.known_sets:
                        set_viol = a @ sample - b
                        if np.max(set_viol) < 1e-3:
                            in_safe = True
                            break
                if nr_sampled >= self.max_samples:
                    raise RuntimeError("Could not find solution")
                samples = [sample]
                if (
                    sample[0] < -0.425
                    and sample[0] > -0.875
                    and sample[2] > 0.5
                    and sample[2] < 0.8
                    and np.abs(sample[1]) < 0.25
                ):
                    pass
                print(f"(PosPath) Adding random point {sample} to graph")
                nr_samples += 1
                if nr_samples > self.max_iters:
                    print("(PosPath) Exceeded max iterations")
                    raise RuntimeError
            for sample in samples:
                print(
                    f"(PosPath) Evaluating sample [{sample[0]:.3f}, {sample[1]:.3f}, {sample[2]:.3f}]"
                )
                j += 1
                t_start_set = time.perf_counter()
                a_set, b_set, q_ellipse, p_mid = self.set_finder.find_set_around_point(
                    sample, fixed_mid=not via_sample
                )
                # a_set, b_set, q_ellipse, p_mid = self.set_finder.find_set_around_point(
                #     sample, fixed_mid=True
                # )
                self.comp_time_set += time.perf_counter() - t_start_set
                dvertex = np.inf
                for key in self.vinfo:
                    dq = np.linalg.norm(q_ellipse - self.vinfo[key]["q_ellipse"])
                    dp_mid = np.linalg.norm(p_mid - self.vinfo[key]["p_mid"])
                    dvertex_current = dq + dp_mid
                    if dvertex_current < dvertex:
                        dvertex = dvertex_current
                if dvertex > 0.01:
                    self.known_sets.append((a_set, b_set))
                    region = HPolyhedron(a_set, b_set)
                    region = region.ReduceInequalities(1e-4)
                    vertex_new = graph.AddVertex(region, name=f"Vertex {j}")
                    self.vinfo[vertex_new.id()] = {
                        # "size": vertex_new.set().CalcVolumeViaSampling(self.generator, 0.05, 1000).volume,
                        "size": 1 / np.linalg.det(q_ellipse),
                        "q_ellipse": q_ellipse,
                        "p_mid": p_mid,
                        "a_set": np.array(a_set),
                        "b_set": np.array(b_set),
                    }
                    t_start_edge = time.perf_counter()
                    self.nr_sets += 1
                    conn = self.add_edges(
                        vertex_new, region, graph, inter_graph, end, start
                    )
                    connected = conn or connected
                    self.comp_time_edge += time.perf_counter() - t_start_edge
                    if self.visualize and self.visualize_sets:
                        self.rviz_marker_msg.markers.append(
                            self.create_marker_msg(vertex_new.set(), f"{j}", 0, 0, 1)
                        )
                        self.rviz_pub.publish(self.rviz_marker_msg)
                else:
                    print(
                        f"(PosPath) Set already known, minimum distance: {dvertex:.3f}"
                    )
            print("---------------------------")

            # if result.is_success():
            #     if p_via_old is not None and p_via_old.shape == p_via.shape:
            #         if np.linalg.norm(p_via_old - p_via) < 1e-4:
            #             print("(PosPath) Found path solution")
            #             success = True
            #         # success = True
            #     p_via_old = np.copy(p_via)

        self.comp_time_graph = time.perf_counter() - start_time

        if self.visualize and self.visualize_graph:
            plt.figure(3)
            # nx.draw(inter_graph)
            plt.subplot(111, projection="3d")
            for edge0 in inter_graph.nodes.items():
                for edge1 in inter_graph.nodes.items():
                    if inter_graph.has_edge(edge0[0], edge1[0]):
                        p0 = self.ginfo[edge0[1]["id"]]["p_proj"]
                        p1 = self.ginfo[edge1[1]["id"]]["p_proj"]
                        plt.plot(p0[0], p0[1], p0[2], "C0.")
                        plt.plot(p1[0], p1[1], p1[2], "C0.")
                        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], "C0")
            plt.plot(start[0], start[1], start[2], "C2.")
            plt.plot(end[0], end[1], end[2], "C2.")
            for vertex in graph.Vertices():
                plot_set(vertex.set().A(), vertex.set().b(), 1)
            for obs_set in self.obs_sets:
                plot_set(obs_set[0], obs_set[1], 2)
            plt.show()

        start_time = time.perf_counter()
        # # Compute intersection of the connected path sets
        # path = graph.GetSolutionPath(vertex_start, vertex_end, result)
        # p_via, p_via_list, sets_via = self.compute_via_points(path, start, end)
        print(p_via)
        (
            p_via,
            p_via_list,
            omega_via,
            sets_via,
            q_via,
            p_mid_via,
        ) = self.compute_via_points(
            path,
            start,
            end,
            id_dict,
            with_rot=True,
            p_via_guess=p_via_list,
        )
        self.sets_via_prev = sets_via.copy()
        self.q_ellipse_prev = q_via.copy()
        self.p_mid_prev = p_mid_via.copy()
        self.comp_time_via += time.perf_counter() - start_time

        b1d = np.array([0, 0, 1.0])
        bp1_list = []
        bp2_list = []
        for i in range(len(p_via) - 1):
            dp = p_via[i + 1] - p_via[i]
            dp /= np.linalg.norm(dp)
            b1 = gram_schmidt(dp, b1d)
            b1 /= np.linalg.norm(b1)
            bp1_list.append(b1)
            b2 = np.cross(dp, b1)
            b2 /= np.linalg.norm(b2)
            bp2_list.append(b2)

        if self.visualize and self.visualize_via_path:
            plt.figure(2)
            plt.subplot(111, projection="3d")
            # for k, set_current in enumerate(sets_via):
            #     plot_set(set_current[0], set_current[1], k+1)
            plt.axis("equal")
            plt.plot(p_via[:, 0], p_via[:, 1], p_via[:, 2], "C0")
            plt.plot(p_via[:, 0], p_via[:, 1], p_via[:, 2], "C0.")
            if self.replanning:
                p_horizon_np = np.array(p_horizon)
                plt.plot(
                    p_horizon_np[:, 0], p_horizon_np[:, 1], p_horizon_np[:, 2], "C1."
                )
            for set_via in sets_via:
                plot_set(set_via[0], set_via[1], 1)
            for obs_set in self.obs_sets:
                plot_set(obs_set[0], obs_set[1], 2)
            plt.show()

        if self.visualize and self.visualize_plan:
            self.rviz_marker_msg.markers += self.plan_msgs
        if self.visualize:
            self.rviz_pub.publish(self.rviz_marker_msg)
        self.comp_time_total = time.perf_counter() - start_time_total
        self.print_computation_time()

        # Compute orientation via points
        # phi_switch = np.array(phi_switch)
        # omega_via = phi_switch / np.max(phi_switch) * self.omega_norm
        r_via = [R.from_rotvec(x * self.omega).as_matrix() @ r0 for x in omega_via]

        sets_via_normed = normalize_set_size(sets_via, 15)

        return p_via_list, r_via, bp1_list, sets_via_normed

    def compute_via_points(
        self, path, start, end, id_dict, with_rot=False, p_via_guess=None
    ):
        # Compute intersection sets
        x0 = np.empty(0)
        self.plan_msgs = []
        sets_inter = []
        for i, edge in enumerate(path[1:-1]):
            sets_inter.append([edge.A(), edge.b()])
            # Add a small constant to move the via point away from the border,
            # important for the next set optimization
            idx = np.linalg.norm(edge.A(), axis=1) > 1e-4
            # pv = self.ginfo[path[i].v().id()]["p_via"]
            # np.max(sets_inter[-1][0] @ pv[:3] - sets_inter[-1][1])
            if p_via_guess is not None:
                x0 = np.concatenate(
                    (
                        x0,
                        p_via_guess[i + 1],
                        [self.ginfo[id_dict[edge]]["p_via"][3]],
                    )
                )
            else:
                x0 = np.concatenate((x0, self.ginfo[id_dict[edge]]["p_via"]))
            # TODO compute automatically
            sets_inter[-1][1][idx] -= 0.001

            # if self.visualize and self.visualize_plan and with_rot:
            #     self.plan_msgs.append(
            #         self.create_marker_msg(v_inter, f"Plan Inter {i+1}", 0, 1, 0)
            #     )
        # if self.visualize and self.visualize_plan and with_rot:
        #     self.plan_msgs.append(
        #         self.create_marker_msg(path[0].v().set(), "Plan Inter 0", 0, 1, 0)
        #     )

        sets = []
        sets_via = []
        q_ellipse = []
        p_mid = []
        w_size_via = []
        for i in range(len(path)):
            if i == 0:
                a_set = path[i].A()
                b_set = path[i].b()
                last_id = self.ginfo[id_dict[path[i]]]["vertex0"].id()
                det_via = self.vinfo[last_id]["size"]
                q_via = self.vinfo[last_id]["q_ellipse"]
                p_mid_via = self.vinfo[last_id]["p_mid"]
                w_size_via.append(det_via)
            else:
                v0 = self.ginfo[id_dict[path[i]]]["vertex0"]
                v1 = self.ginfo[id_dict[path[i]]]["vertex1"]
                if v0.id() != last_id:
                    a_set = v0.set().A()
                    b_set = v0.set().b()
                    det_via = self.vinfo[v0.id()]["size"]
                    q_via = self.vinfo[v0.id()]["q_ellipse"]
                    p_mid_via = self.vinfo[v0.id()]["p_mid"]
                    w_size_via.append(det_via)
                    last_id = v0.id()
                elif v1.id() != last_id:
                    a_set = v1.set().A()
                    b_set = v1.set().b()
                    det_via = self.vinfo[v1.id()]["size"]
                    q_via = self.vinfo[v1.id()]["q_ellipse"]
                    p_mid_via = self.vinfo[v1.id()]["p_mid"]
                    w_size_via.append(det_via)
                    last_id = v1.id()
            sets.append([a_set, b_set])
            sets_via.append([a_set, b_set])
            q_ellipse.append(q_via)
            p_mid.append(p_mid_via)
            if self.visualize and self.visualize_plan and with_rot:
                v1 = HPolyhedron(a_set, b_set)
                self.plan_msgs.append(self.create_marker_msg(v1, f"Plan {i}", 0, 1, 0))

        # Adapted sizes such that large sizes are penalized less
        # w_size_via /= np.max(w_size_via)
        # w_size_via = 1.1 - w_size_via
        w_size_via = 1 - np.cbrt(w_size_via)

        # Optimize via points
        sets_inter = normalize_set_size(sets_inter, self.max_set_size)
        sets_via = normalize_set_size(sets_via, self.max_set_size)
        nr_via = len(sets_inter)
        if not with_rot:
            pass
            # params = np.concatenate((start, end))
            # x0 = np.zeros((3 * nr_via,))
            # for i in range(nr_via):
            #     params = np.concatenate(
            #         (params, sets_inter[i][0].T.flatten(), sets_inter[i][1]))
            # sol = self.solver_via[nr_via - 1](x0=x0,
            #                                   lbx=self.lbu_via[nr_via - 1],
            #                                   ubx=self.ubu_via[nr_via - 1],
            #                                   lbg=self.lbg_via[nr_via - 1],
            #                                   ubg=self.ubg_via[nr_via - 1],
            #                                   p=params)
            # if not self.solver_via[nr_via - 1].stats()["success"]:
            #     g = sol['g']
            #     g_viol = -np.sum(g[np.where(g < np.array(self.lbg_via[nr_via - 1]) - 1e-6)[0]])
            #     g_viol += np.sum(g[np.where(g > np.array(self.ubg_via[nr_via - 1]) + 1e-6)[0]])
            #     print("(PosOpt) ERROR No convergence in via point optimization")
            #     print(f"(PosOpt) -> Constraint violations: {g_viol}")
            # else:
            #     print("(PosOpt) Found via point path through graph")
        else:
            params = np.concatenate(
                (
                    start,
                    end,
                    self.l_ee,
                    self.omega_normed,
                    [self.omega_norm],
                    w_size_via,
                )
            )
            for i in range(nr_via):
                params = np.concatenate(
                    (params, sets_inter[i][0].T.flatten(), sets_inter[i][1])
                )
            for i in range(nr_via + 1):
                params = np.concatenate(
                    (params, sets_via[i][0].T.flatten(), sets_via[i][1])
                )
            x0 = np.concatenate((x0, 0.5 * np.ones(self.max_set_size * nr_via)))
            sol = self.solver_via_rot[nr_via - 1](
                x0=x0,
                lbx=self.lbu_via_rot[nr_via - 1],
                ubx=self.ubu_via_rot[nr_via - 1],
                lbg=self.lbg_via_rot[nr_via - 1],
                ubg=self.ubg_via_rot[nr_via - 1],
                p=params,
            )

            if not self.solver_via_rot[nr_via - 1].stats()["success"]:
                g = sol["g"]
                g_viol = -np.sum(
                    g[np.where(g < np.array(self.lbg_via_rot[nr_via - 1]) - 1e-6)[0]]
                )
                g_viol += np.sum(
                    g[np.where(g > np.array(self.ubg_via_rot[nr_via - 1]) + 1e-6)[0]]
                )
                print("(PosOpt) ERROR No convergence in via point rot optimization")
                print(f"(PosOpt) -> Constraint violations: {g_viol}")
            else:
                print("(PosOpt) Found via point path with rot through graph")

        # Only add the via points that do not coincide with each other
        sets_via = []
        p_via = [start]
        omega_via = [0.0]
        for i in range(len(sets_inter)):
            if with_rot:
                step = 4 + self.max_set_size
                via_opt = np.array(sol["x"][step * i : step * (i + 1)]).flatten()
                p_via_opt = via_opt[:3]
                omega_opt = via_opt[3]
                if np.linalg.norm(p_via_opt - p_via[-1]) > 1e-4:
                    p_via.append(p_via_opt)
                    omega_via.append(omega_opt)
                    sets_via.append(sets[i])
                # Extend the first segment backwards until it hits the bounds
                if self.replanning and i == 0:
                    a_set0 = sets_via[0][0]
                    b_set0 = sets_via[0][1]
                    b_trans0 = b_set0 - a_set0 @ p_via[0]
                    dp0 = p_via[1] - p_via[0]
                    dp0 /= np.linalg.norm(dp0)
                    dp_horizon = self.p_horizon - p_via[0]
                    sol_lin = linprog(
                        np.ones(1),
                        A_ub=np.expand_dims(a_set0 @ dp0, 1),
                        b_ub=b_trans0,
                        bounds=(None, None),
                    )
                    phi_horizon = np.min(dp0 @ dp_horizon.T)
                    phi_horizon = np.min((phi_horizon, -0.5))
                    # replanning_phi = -np.max((sol_lin.x, [phi_horizon]))
                    replanning_phi = -phi_horizon
                    replanning_phi = np.max((replanning_phi, 0.0))
                    print(f"(Replanning) Horizon phi: {phi_horizon:.3f}")
                    print(f"(Replanning) Linprog phi: {sol_lin.x[0]:.3f}")
                    print(f"(Replanning) Replan phi: {replanning_phi:.3f}")
                    if phi_horizon < sol_lin.x[0]:
                        print(f"(Replanning) Horizon needs deviations")
                    p_via[0] = p_via[0] - replanning_phi * dp0
            else:
                # via_opt = np.array(sol['x'][3*i:3*(i+1)]).flatten()
                via_opt = x0[4 * i : 4 * (i + 1)]
                p_via_opt = via_opt[:3]
                omega_opt = via_opt[3]
                if np.linalg.norm(p_via_opt - p_via[-1]) > 1e-4:
                    p_via.append(p_via_opt)
                    omega_via.append(omega_opt)
                    sets_via.append(sets[i])

        p_via.append(end)
        omega_via.append(1.0)
        sets_via.append(sets[-1])
        return np.array(p_via), p_via, omega_via, sets_via, q_ellipse, p_mid

    def via_point_optimization_problem(self, nr_via=4, max_set_size=20):
        a_sets = ca.SX.sym("a set", max_set_size, 3, nr_via)
        b_sets = ca.SX.sym("b set", max_set_size, nr_via)
        p_start = ca.SX.sym("p start", 3)
        p_end = ca.SX.sym("p end", 3)

        params = ca.vertcat(p_start, p_end)
        for i in range(nr_via):
            params = ca.vertcat(params, a_sets[i].reshape((-1, 1)), b_sets[:, i])

        g = []
        lbg = []
        ubg = []
        u = []
        lbu = []
        ubu = []

        J = 0
        p_via_prev = p_start
        for i in range(nr_via):
            p_via = ca.SX.sym(f"p_via {i}", 3)
            u += [p_via]
            lbu += [-np.inf] * 3
            ubu += [np.inf] * 3

            J += ca.sumsqr(p_via - p_via_prev)
            p_via_prev = p_via

            g += [a_sets[i] @ p_via - b_sets[:, i]]
            lbg += [-np.inf] * max_set_size
            ubg += [0] * max_set_size
        J += ca.sumsqr(p_end - p_via)

        prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

        solver = ca.qpsol(
            "solver", "qpoases", prob, {"print_time": False, "printLevel": "none"}
        )
        # solver = ca.qpsol('solver', 'osqp', prob, {
        #     "print_time": False,
        #     "osqp": {"verbose": False}
        # })

        return solver, lbu, ubu, lbg, ubg

    def via_point_rot_optimization_problem(self, nr_via=4, max_set_size=20):
        a_inter = ca.SX.sym("a set inter", max_set_size, 3, nr_via)
        b_inter = ca.SX.sym("b set inter", max_set_size, nr_via)
        a_via = ca.SX.sym("a set via", max_set_size, 3, nr_via + 1)
        b_via = ca.SX.sym("b set via", max_set_size, nr_via + 1)
        w_size_via = ca.SX.sym("w size via", nr_via + 1)
        p_start = ca.SX.sym("p start", 3)
        p_end = ca.SX.sym("p end", 3)
        l_ee = ca.SX.sym("l ee", 3)
        omega = ca.SX.sym("omega", 3)
        omega_norm = ca.SX.sym("omega norm")

        params = ca.vertcat(p_start, p_end, l_ee, omega, omega_norm, w_size_via)
        for i in range(nr_via):
            params = ca.vertcat(params, a_inter[i].reshape((-1, 1)), b_inter[:, i])
        for i in range(nr_via + 1):
            params = ca.vertcat(params, a_via[i].reshape((-1, 1)), b_via[:, i])

        g = []
        lbg = []
        ubg = []
        u = []
        lbu = []
        ubu = []

        J = 0
        p_via_prev = p_start
        omega_via_prev = 0.0
        for i in range(nr_via):
            p_via = ca.SX.sym(f"p_via {i}", 3)
            omega_via = ca.SX.sym(f"omega_via {i}", 1)
            u += [p_via]
            lbu += [-np.inf] * 3
            ubu += [np.inf] * 3
            u += [omega_via]
            lbu += [0]
            ubu += [1]

            p_via_ee = p_via + rodrigues_matrix(omega, omega_norm * omega_via) @ l_ee

            J += w_size_via[i] * ca.sumsqr(p_via - p_via_prev)
            J += w_size_via[i] * ca.sumsqr(omega_via - omega_via_prev)

            g += [a_inter[i] @ p_via - b_inter[:, i]]
            lbg += [-np.inf] * max_set_size
            ubg += [0] * max_set_size

            for j in range(a_via[i].shape[0]):
                v = p_via - p_via_prev
                a = a_via[i][j, :].T
                b = b_via[j, i]

                phi_max = ca.SX.sym(f"phi_max {i} {j}")
                u += [phi_max]
                lbu += [0.0]
                ubu += [1.0]

                omega_mid = omega_via_prev + phi_max * (omega_via - omega_via_prev)
                l_max = rodrigues_matrix(omega, omega_norm * omega_mid) @ l_ee
                p_max = p_via_prev + phi_max * v
                p_max_ee = ca.dot(a, p_max + l_max)
                dp_max_ee = ca.jacobian(p_max_ee, phi_max)

                if i == 0:
                    omega_prev = ca.SX.sym("omega_prev")
                    f_max = ca.Function(
                        "f_max",
                        [
                            phi_max,
                            omega_via,
                            omega_prev,
                            a,
                            p_via,
                            p_via_prev,
                            l_ee,
                            omega,
                            omega_norm,
                        ],
                        [dp_max_ee],
                    )
                else:
                    f_max = ca.Function(
                        "f_max",
                        [
                            phi_max,
                            omega_via,
                            omega_via_prev,
                            a,
                            p_via,
                            p_via_prev,
                            l_ee,
                            omega,
                            omega_norm,
                        ],
                        [dp_max_ee],
                    )
                constr = f_max(
                    phi_max,
                    omega_via,
                    omega_via_prev,
                    a,
                    p_via,
                    p_via_prev,
                    l_ee,
                    omega,
                    omega_norm,
                )
                constr0 = f_max(
                    0,
                    omega_via,
                    omega_via_prev,
                    a,
                    p_via,
                    p_via_prev,
                    l_ee,
                    omega,
                    omega_norm,
                )
                constr1 = f_max(
                    1,
                    omega_via,
                    omega_via_prev,
                    a,
                    p_via,
                    p_via_prev,
                    l_ee,
                    omega,
                    omega_norm,
                )
                constr_with_bounds = ca.if_else(constr0 * constr1 < 0, constr, 0)

                g += [constr_with_bounds]
                lbg += [0]
                ubg += [0]
                g += [p_max_ee - b]
                lbg += [-np.inf]
                ubg += [0]

            g += [a_inter[i] @ p_via_ee - b_inter[:, i]]
            lbg += [-np.inf] * max_set_size
            ubg += [0] * max_set_size

            p_via_prev = p_via
            omega_via_prev = omega_via
        J += w_size_via[-1] * ca.sumsqr(p_end - p_via)
        J += w_size_via[-1] * ca.sumsqr(1 - omega_via)

        for pos in [0.25, 0.5]:
            p_mid = p_via + pos * (p_end - p_via)
            omega_mid = omega_via + pos * (1 - omega_via)
            p_mid_ee = p_mid + rodrigues_matrix(omega, omega_norm * omega_mid) @ l_ee

            g += [a_via[-1] @ p_mid_ee - b_via[:, -1]]
            lbg += [-np.inf] * max_set_size
            ubg += [0] * max_set_size

        prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

        solver = ca.nlpsol("solver", self.solver_name, prob, self.solver_opts)
        # codegenopt = {'cpp': True}
        # solver.generate_dependencies(f'via_point_opt_{nr_via}.cpp', codegenopt)
        # solver_file = f'/home/toeleric/research/bound_mpc_rewrite/ros2_ws/src/bound_mpc/via_point_opt_{nr_via}.so'
        # solver = ca.nlpsol('solver', self.solver_name,
        #                    solver_file, self.solver_opts)

        return solver, lbu, ubu, lbg, ubg

    def check_intersection(self, a_set, b_set, l_ee, sample):
        sets = normalize_set_size([[a_set, b_set - 0.001]], self.max_set_size)
        a_setc = sets[0][0]
        b_setc = sets[0][1]
        success = False
        p_inside = np.concatenate((sample, [0]))
        for i in range(20):
            omega_sample = i / 19
            l_eec = (
                rodrigues_matrix(self.omega_normed, self.omega_norm * omega_sample)
                @ l_ee
            )
            params = np.concatenate((l_eec, a_setc.T.flatten(), b_setc))
            sol = self.solver_fit(
                x0=sample,
                lbx=self.lbu_fit,
                ubx=self.ubu_fit,
                lbg=self.lbg_fit,
                ubg=self.ubg_fit,
                p=params,
            )
            if self.solver_fit.stats()["success"]:
                success = True
                p_inside = np.concatenate((sample, [omega_sample]))
                break

        # return self.solver_fit.stats()['success'], sol['x'].full().flatten()
        return success, p_inside

    def fit_opt_problem(self, max_set_size):
        a_set = ca.SX.sym("a set", max_set_size, 3)
        b_set = ca.SX.sym("b set", max_set_size)
        rot_axis = ca.SX.sym("rot axis", 3)
        l_ee = ca.SX.sym("l ee", 3)
        p_in_set = ca.SX.sym("p in set", 3)
        omega_in_set = ca.SX.sym("omega in set")
        angle_norm = ca.SX.sym("angle norm")

        params = ca.vertcat(l_ee, rot_axis, angle_norm, a_set.reshape((-1, 1)), b_set)

        g = []
        lbg = []
        ubg = []
        u = []
        lbu = []
        ubu = []

        u += [p_in_set]
        lbu += [-np.inf] * 3
        ubu += [np.inf] * 3
        u += [omega_in_set]
        lbu += [0.0]
        ubu += [1.0]

        p_ee = p_in_set + rodrigues_matrix(rot_axis, angle_norm * omega_in_set) @ l_ee

        # J = ca.sumsqr(omega_in_set - 0.5)
        J = 0

        g += [a_set @ p_in_set - b_set]
        lbg += [-np.inf] * max_set_size
        ubg += [0] * max_set_size

        g += [a_set @ p_ee - b_set]
        lbg += [-np.inf] * max_set_size
        ubg += [0] * max_set_size

        prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

        solver = ca.nlpsol("solver", self.solver_name, prob, self.solver_opts)
        return solver, lbu, ubu, lbg, ubg

    def fit_opt_problem_sample(self, max_set_size):
        a_set = ca.SX.sym("a set", max_set_size, 3)
        b_set = ca.SX.sym("b set", max_set_size)
        l_ee = ca.SX.sym("l ee", 3)
        p_in_set = ca.SX.sym("p in set", 3)

        params = ca.vertcat(l_ee, a_set.reshape((-1, 1)), b_set)

        g = []
        lbg = []
        ubg = []
        u = []
        lbu = []
        ubu = []

        u += [p_in_set]
        lbu += [-np.inf] * 3
        ubu += [np.inf] * 3

        p_ee = p_in_set + l_ee

        J = 0

        g += [a_set @ p_in_set - b_set]
        lbg += [-np.inf] * max_set_size
        ubg += [0] * max_set_size

        g += [a_set @ p_ee - b_set]
        lbg += [-np.inf] * max_set_size
        ubg += [0] * max_set_size

        prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

        solver = ca.qpsol(
            "solver",
            "qpoases",
            prob,
            {
                "print_time": False,
                "printLevel": "none",
                "error_on_fail": False,
            },
        )
        return solver, lbu, ubu, lbg, ubg

    def add_edges(self, vertex_new, largest_region, graph, inter_graph, end, start):
        connected = False
        for vertex in graph.Vertices():
            intersects = largest_region.IntersectsWith(vertex.set())
            if intersects and vertex != vertex_new:
                set_inter = vertex.set().Intersection(vertex_new.set())
                # Check if end-effector fits into the intersection set
                sampling_success = True
                try:
                    sample = set_inter.UniformSample(self.generator)
                except ValueError:
                    sampling_success = False
                if sampling_success:
                    t_start_fit = time.perf_counter()
                    a_set = set_inter.A()
                    b_set = set_inter.b()
                    fits = False
                    # for i in range(30):
                    #     sample = set_inter.UniformSample(self.generator, sample)
                    #     omega, success = self.find_viable_angle(sample, a_set, b_set)
                    #     if success:
                    #         fits = True
                    #         via = np.concatenate((sample, omega))
                    #         break
                    # for j in range(100):
                    #     omega = self.rng.uniform(0, 1, 1)
                    #     p_ee = sample + R.from_rotvec(self.omega * omega).as_matrix() @ self.l_ee
                    #     if np.max(a_set @ p_ee - b_set) < 1e-3:
                    #         fits = True
                    #         via = np.concatenate((sample, omega))
                    #         break
                    # if fits:
                    #     break
                    fits, via = self.check_intersection(a_set, b_set, self.l_ee, sample)
                    self.comp_time_fit += time.perf_counter() - t_start_fit
                else:
                    fits = False
                if not fits:
                    print("(Edges) EE does not fit in intersection")
                if fits:
                    self.id += 1
                    inter_graph.add_node(
                        set_inter, id=self.id, name=f"Interset {self.id}"
                    )
                    self.nr_inter_set += 2
                    self.ginfo[self.id] = {
                        "vertex0": vertex,
                        "vertex1": vertex_new,
                        "conn_to_start": False,
                        "conn_to_end": False,
                        "p_proj": None,
                        "p_via": via,
                    }
                    for edge in inter_graph.nodes.items():
                        v0 = self.ginfo[edge[1]["id"]]["vertex0"]
                        v1 = self.ginfo[edge[1]["id"]]["vertex1"]
                        cond1 = v0.id() == vertex.id() or v1.id() == vertex.id()
                        cond2 = v0.id() == vertex_new.id() or v1.id() == vertex_new.id()
                        if cond1:
                            size = self.vinfo[vertex.id()]["size"]
                        elif cond2:
                            size = self.vinfo[vertex_new.id()]["size"]
                        cond = self.id != edge[1]["id"]
                        if cond and (cond1 or cond2):
                            self.nr_edges += 2
                            p_proj = self.ginfo[edge[1]["id"]]["p_proj"]
                            if p_proj is None:
                                p_proj = end
                            if self.ginfo[self.id]["p_proj"] is None:
                                normed_sets = normalize_set_size(
                                    [[set_inter.A(), set_inter.b()]],
                                    max_set_size=self.max_set_size,
                                )
                                params = np.concatenate(
                                    (
                                        normed_sets[0][0].T.flatten(),
                                        normed_sets[0][1],
                                        p_proj,
                                    )
                                )
                                sol = self.proj_solver(
                                    x0=sample,
                                    lbx=self.proj_lbu,
                                    ubx=self.proj_ubu,
                                    lbg=self.proj_lbg,
                                    ubg=self.proj_ubg,
                                    p=params,
                                )
                                self.ginfo[self.id]["p_proj"] = (
                                    sol["x"].full().flatten()
                                )
                            dist = np.linalg.norm(
                                self.ginfo[self.id]["p_proj"] - p_proj
                            )

                            conn_to_start = (
                                self.ginfo[self.id]["conn_to_start"]
                                or self.ginfo[edge[1]["id"]]["conn_to_start"]
                            )
                            conn_to_end = (
                                self.ginfo[self.id]["conn_to_end"]
                                or self.ginfo[edge[1]["id"]]["conn_to_end"]
                            )
                            self.ginfo[self.id]["conn_to_start"] = conn_to_start
                            self.ginfo[self.id]["conn_to_end"] = conn_to_end
                            self.ginfo[edge[1]["id"]]["conn_to_start"] = conn_to_start
                            self.ginfo[edge[1]["id"]]["conn_to_end"] = conn_to_end
                            if conn_to_start and conn_to_end:
                                connected = True
                            else:
                                connected = False

                            c_size = np.tanh(0.25 - np.cbrt(size))
                            cost = dist * (1 + self.w_size * c_size) + self.w_bias
                            inter_graph.add_edge(set_inter, edge[0], weight=cost)
                            print(
                                f"(Costs) cost {cost:.3f}, c_size {c_size:.3f} {np.cbrt(size):.3f}"
                            )
        return connected

    def create_marker_msg(self, poly, i, r, g, b, a=0.1):
        marker = Marker()
        marker.header.frame_id = "r1/world"
        marker.action = Marker.ADD
        marker.type = Marker.TRIANGLE_LIST
        marker.ns = f"Set {i}"
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)

        try:
            points = compute_polytope_vertices(poly.A(), poly.b())
            hull = ConvexHull(points)
            faces = hull.simplices
            for face in faces:
                p1, p2, p3 = np.array(points)[face]
                marker.points.append(self.create_point(p1))
                marker.points.append(self.create_point(p2))
                marker.points.append(self.create_point(p3))
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        except RuntimeError:
            print("(Visualization) Could not compute polytope for RViz")
        return marker

    def create_point(self, point):
        p = Point()
        p.x, p.y, p.z = point
        return p
