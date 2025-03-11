import copy
import time

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation as R

from bound_planner.utils import (
    compute_polytope_vertices,
    gram_schmidt,
    normalize_set_size,
    reduce_ineqs,
)
from bound_planner.utils.optimization_functions import (
    fit_opt_problem_sample,
    projection_opt_problem,
    rodrigues_matrix,
    via_point_optimization_problem,
    via_point_rot_optimization_problem,
)

from .ConvexSetFinder import ConvexSetFinder


class BoundPlanner:
    def __init__(
        self,
        obstacles=[],
        e_p_max=0.5,
        obs_size_increase=0.08,
        workspace_max=[1.0, 1.0, 1.2],
        workspace_min=[-1.0, -1.0, 0.0],
    ):
        self.plan_msgs = []
        self.replanning = False
        self.sets_via_prev = []

        self.obs_size_increase = obs_size_increase
        self.comp_time_set = 0
        self.comp_time_edge = 0
        self.comp_time_fit = 0
        self.comp_time_total = 0
        self.comp_time_graph = 0
        self.comp_time_path = 0
        self.comp_time_via = 0
        self.w_size = 0.1
        self.c_fit = 1.0
        self.w_bias = 0.01
        self.rng = np.random.default_rng()
        self.max_set_size = 20
        self.workspace_max = workspace_max
        self.workspace_min = workspace_min
        self.length_ee = 0.05
        self.max_iters = 20
        self.nr_optimized = 10
        self.nr_free_mid = 5
        self.max_samples = 500

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
        (
            self.solver_fit,
            self.lbu_fit,
            self.ubu_fit,
            self.lbg_fit,
            self.ubg_fit,
        ) = fit_opt_problem_sample(self.max_set_size)

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
            solver, lbu, ubu, lbg, ubg = via_point_optimization_problem(
                i, self.max_set_size
            )
            self.solver_via.append(solver)
            self.lbu_via.append(lbu)
            self.ubu_via.append(ubu)
            self.lbg_via.append(lbg)
            self.ubg_via.append(ubg)
            solver, lbu, ubu, lbg, ubg = via_point_rot_optimization_problem(
                i, self.max_set_size
            )
            self.solver_via_rot.append(solver)
            self.lbu_via_rot.append(lbu)
            self.ubu_via_rot.append(ubu)
            self.lbg_via_rot.append(lbg)
            self.ubg_via_rot.append(ubg)

        # Obstacles
        self.obs = []
        self.obs_points = np.empty((0, 3))
        self.obs_sets = []
        self.obs_sets_orig = []
        self.obs_points_sets = []
        self.add_obstacle_reps(obstacles)

        # Create set finder object
        self.set_finder = ConvexSetFinder(
            self.obs_sets,
            self.obs_points_sets,
            self.workspace_max,
            self.workspace_min,
        )

    def make_box(self, lb, ub):
        a_set = np.concatenate((np.eye(3), -np.eye(3)))
        b_set = np.concatenate((ub, -np.array(lb)))
        return [a_set, b_set]

    def add_obstacle_reps(self, obstacles, update=False, reset=False):
        if reset:
            self.obs = []
            self.obs_points = np.empty((0, 3))
            self.obs_sets = []
            self.obs_sets_orig = []
            self.obs_points_sets = []
        for ob in obstacles:
            set_ob = self.make_box(ob[:3], ob[3:])
            adapted_ob = copy.deepcopy(set_ob)
            adapted_ob[1] += self.obs_size_increase
            points = np.array(compute_polytope_vertices(adapted_ob[0], adapted_ob[1]))

            self.obs_sets_orig.append(set_ob)
            self.obs_points = np.concatenate((self.obs_points, points))
            self.obs_points_sets.append(points)
            self.obs_sets.append(adapted_ob)
        self.obs_sets = normalize_set_size(self.obs_sets)

        if update:
            self.set_finder.obs_sets = self.obs_sets.copy()
            self.set_finder.obs_points_sets = self.obs_points_sets.copy()

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

    def plan_convex_set_path(
        self,
        start,
        end,
        r0,
        r1,
        replanning=False,
        p_horizon=[],
        first_sample=None,
        new_obs=False,
    ):
        sampled_first = False
        start_time_total = time.perf_counter()
        self.replanning = replanning
        self.replanning_phi = 0.0
        self.p_horizon = p_horizon
        self.set_finder.proj_time = 0.0
        self.set_finder.ell_time = 0.0
        self.comp_time_set = 0.0
        self.comp_time_edge = 0.0
        self.comp_time_fit = 0.0
        self.comp_time_path = 0.0
        self.comp_time_via = 0.0

        # Project end point to collision free space
        for ob in self.obs_sets:
            constraint_viol = ob[0] @ end - ob[1]
            if not np.any(constraint_viol > 0):
                print("(PosPath) Projecting end point to collision free space")
                idx = np.argmax(constraint_viol)
                end -= (constraint_viol[idx] - self.obs_size_increase) * ob[0][idx, :]

        # Compute direction of end effector at start and end point
        self.omega = R.from_matrix(r1 @ r0.T).as_rotvec()
        self.omega_norm = np.linalg.norm(self.omega)
        if self.omega_norm > 1e-6:
            self.omega_normed = self.omega / self.omega_norm
        else:
            self.omega_normed = np.array([0, 0, 1.0])
        self.l_ee = r0 @ np.array([-self.length_ee, 0, 0])
        self.l_ee_end = r1 @ np.array([-self.length_ee, 0, 0])
        b1rd = np.array([0, 0, 1.0])
        self.br1 = gram_schmidt(self.omega_normed, b1rd)
        self.br1 /= np.linalg.norm(self.br1)
        self.br2 = np.cross(self.omega_normed, self.br1)
        self.br2[-1] /= np.linalg.norm(self.br2[-1])

        # Create graphs
        graph = nx.Graph()
        inter_graph = nx.Graph()
        self.nr_sets = 0
        self.nr_edges = 0
        self.nr_inter_set = 0

        # Grow convex sets
        start_time = time.perf_counter()
        t_start_set = time.perf_counter()
        if self.replanning:
            max_horizon_idx = 1
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
            if new_obs:
                print("New obstacle requires setting the horizon idx to 1")
                max_horizon_idx = 1
            self.p_horizon_max = p_horizon[max_horizon_idx]
            print(f"(Replanning) p horizon: {self.p_horizon_max}")
            (
                a_set,
                b_set,
                q_ellipse_start,
                p_mid_start,
                collision,
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
            collision = False
            if np.max(a_set @ (start + self.l_ee) - b_set) > 1e-8:
                (
                    a_set,
                    b_set,
                    q_ellipse_start,
                    p_mid_start,
                    collision,
                ) = self.set_finder.find_set_collision_avoidance(
                    start, start + self.l_ee, True
                )
        # Use last set if we cannot find a new set
        if collision:
            if new_obs:
                print(
                    "[WARNING] Start point in collision, projecting to collision free space"
                )
                # Project start point to collision free space
                for ob in self.obs_sets:
                    constraint_viol = ob[0] @ start - ob[1]
                    if not np.any(constraint_viol > 0):
                        print(
                            "(PosPath) Projecting start point to collision free space"
                        )
                        idx = np.argmax(constraint_viol)
                        start -= (
                            constraint_viol[idx] - self.obs_size_increase
                        ) * ob.A()[idx, :]
                    print(f"(PosPath) New start point {start}")
                (
                    a_set,
                    b_set,
                    q_ellipse_start,
                    p_mid_start,
                ) = self.set_finder.find_set_around_point(start, fixed_mid=True)
            else:
                print("[WARNING] Could not find start set, reusing old end set")
                a_set = copy.deepcopy(self.sets_via_prev[-1][0])
                b_set = copy.deepcopy(self.sets_via_prev[-1][1])
                p_mid_start = start
                q_ellipse_start = np.eye(3)
        self.comp_time_set += time.perf_counter() - t_start_set

        a_set, b_set = reduce_ineqs(a_set, b_set)
        set_start = [a_set, b_set]
        self.id_inter = 0
        self.id_graph = 0
        graph.add_node(
            self.id_graph,
            cset=set_start,
            name="Vertex start",
            size=1 / np.linalg.det(q_ellipse_start),
            q_ellipse=q_ellipse_start,
            p_mid=p_mid_start,
            a_set=np.array(a_set),
            b_set=np.array(b_set),
        )
        inter_graph.add_node(
            self.id_inter,
            cset=set_start,
            name="Vertex start",
            id0=self.id_graph,
            id1=self.id_graph,
            set0=set_start,
            set1=set_start,
            conn_to_start=True,
            conn_to_end=False,
            p_proj=start,
            edge=None,
            p_via=np.concatenate((start, [0.0])),
        )
        self.nr_sets += 1

        t_start_edge = time.perf_counter()
        connected = self.add_edges(self.id_graph, graph, inter_graph, end, start)
        self.comp_time_edge += time.perf_counter() - t_start_edge

        if (
            np.max(a_set @ end - b_set) < 1e-8
            and np.max(a_set @ (end + self.l_ee_end) - b_set) < 1e-8
        ):
            print("(PosPath) End point in start set, finishing ...")
            p_via_list = [start, end]
            omega_via = [0.0, 1.0]
            r_via = [R.from_rotvec(x * self.omega).as_matrix() @ r0 for x in omega_via]
            bp1_list = [np.array([0, 0, 1.0])]
            sets_via = [[a_set, b_set]]
            sets_via_normed = normalize_set_size(sets_via, 15)
            self.sets_via_prev = sets_via_normed.copy()
            self.graph = graph
            self.inter_graph = inter_graph
            return p_via_list, r_via, bp1_list, sets_via_normed
        else:
            t_start_set = time.perf_counter()
            # a_set, b_set, q_ellipse_end, p_mid_end = self.set_finder.find_set_around_line(
            #     end, self.l_ee_end, optimize=False
            # )
            (
                a_set,
                b_set,
                q_ellipse_end,
                p_mid_end,
                collision,
            ) = self.set_finder.find_set_collision_avoidance(
                end, end + self.l_ee_end, True
            )
            a_set, b_set = reduce_ineqs(a_set, b_set)
            self.comp_time_set += time.perf_counter() - t_start_set
            set_end = [a_set, b_set]
            self.id_graph += 1
            self.id_inter += 1
            graph.add_node(
                self.id_graph,
                cset=set_end,
                name="Vertex end",
                size=1 / np.linalg.det(q_ellipse_end),
                q_ellipse=q_ellipse_end,
                p_mid=p_mid_end,
                a_set=np.array(a_set),
                b_set=np.array(b_set),
            )
            inter_graph.add_node(
                self.id_inter,
                cset=set_end,
                name="Vertex end",
                id0=self.id_graph,
                id1=self.id_graph,
                set0=set_end,
                set1=set_end,
                conn_to_start=False,
                conn_to_end=True,
                p_proj=end,
                edge=None,
                p_via=np.concatenate((end, [1.0])),
            )
            t_start_edge = time.perf_counter()
            self.nr_sets += 1
            conn = self.add_edges(self.id_graph, graph, inter_graph, end, start)
            connected = conn or connected
            self.comp_time_edge += time.perf_counter() - t_start_edge

        # Sample points until solution is found
        j = 0
        nr_samples = 0
        success = False
        p_via_old = None
        while not success:
            via_sample = False
            if connected:
                t_start_path = time.perf_counter()
                path = nx.shortest_path(inter_graph, 0, 1, weight="weight")
                self.comp_time_path += time.perf_counter() - t_start_path
                t_start_path = time.perf_counter()
                p_via, p_via_list, omega_via, sets_via, _, _ = self.compute_via_points(
                    path, start, end, graph, inter_graph
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
                        via_sample = True
                else:
                    samples = p_via_list[1:-1]
                    print("(PosPath) Sampling via points to improve graph")
                    via_sample = True
                p_via_old = np.copy(p_via)
            elif not sampled_first and first_sample is not None:
                samples = [first_sample]
            else:
                in_collision = True
                in_safe = True
                nr_sampled = 0
                while (in_collision or in_safe) and nr_sampled <= self.max_samples:
                    in_collision = False
                    in_safe = False
                    sample = self.rng.uniform(self.workspace_min, self.workspace_max, 3)
                    nr_sampled += 1
                    for ob in self.obs_sets:
                        constraint_viol = ob[0] @ sample - ob[1]
                        if np.max(constraint_viol) < 1e-3:
                            in_collision = True
                            break
                    for setc in graph.nodes.items():
                        set_viol = setc[1]["a_set"] @ sample - setc[1]["b_set"]
                        if np.max(set_viol) < 1e-3:
                            in_safe = True
                            break
                if nr_sampled >= self.max_samples:
                    raise RuntimeError("(PosPath) Could not find collision-free sample")
                samples = [sample]
                print(f"(PosPath) Adding random point {sample} to graph")
                nr_samples += 1
                if nr_samples > self.max_iters:
                    raise RuntimeError("(PosPath) Exceeded max iterations")
            for sample in samples:
                print(
                    f"(PosPath) Evaluating point [{sample[0]:.3f}, {sample[1]:.3f}, {sample[2]:.3f}]"
                )
                j += 1
                t_start_set = time.perf_counter()
                if nr_samples >= self.nr_optimized:
                    optimize = False
                else:
                    optimize = True
                fixed_mid = (via_sample or (not sampled_first),)
                if nr_samples >= self.nr_free_mid:
                    fixed_mid = True
                a_set, b_set, q_ellipse, p_mid = self.set_finder.find_set_around_point(
                    sample,
                    fixed_mid=fixed_mid,
                    optimize=optimize,
                )
                a_set, b_set = reduce_ineqs(a_set, b_set)
                sampled_first = True
                self.comp_time_set += time.perf_counter() - t_start_set
                dvertex = np.inf
                for vertex in graph.nodes.items():
                    dq = np.linalg.norm(q_ellipse - vertex[1]["q_ellipse"])
                    dp_mid = np.linalg.norm(p_mid - vertex[1]["p_mid"])
                    dvertex_current = dq + dp_mid
                    if dvertex_current < dvertex:
                        dvertex = dvertex_current
                if dvertex > 0.01:
                    set_new = [a_set, b_set]
                    self.id_graph += 1
                    graph.add_node(
                        self.id_graph,
                        cset=set_new,
                        name=f"Vertex {j}",
                        size=1 / np.linalg.det(q_ellipse),
                        q_ellipse=q_ellipse,
                        p_mid=p_mid,
                        a_set=np.array(a_set),
                        b_set=np.array(b_set),
                    )
                    t_start_edge = time.perf_counter()
                    self.nr_sets += 1
                    conn = self.add_edges(self.id_graph, graph, inter_graph, end, start)
                    connected = conn or connected
                    self.comp_time_edge += time.perf_counter() - t_start_edge
                else:
                    print(
                        f"(PosPath) Set already known, minimum distance: {dvertex:.3f}"
                    )
            print("---------------------------")

        self.comp_time_graph = time.perf_counter() - start_time

        start_time = time.perf_counter()
        # Compute via path
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
            graph,
            inter_graph,
            with_rot=True,
            p_via_guess=p_via_list,
        )
        self.sets_via_prev = sets_via.copy()
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

        self.comp_time_total = time.perf_counter() - start_time_total
        self.print_computation_time()

        # Compute orientation via points
        r_via = [R.from_rotvec(x * self.omega).as_matrix() @ r0 for x in omega_via]
        r0_adapted = R.from_rotvec(-self.replanning_phi * self.omega).as_matrix() @ r0
        r_via[0] = r0_adapted

        sets_via_normed = normalize_set_size(sets_via, 15)

        self.graph = graph
        self.inter_graph = inter_graph
        return p_via_list, r_via, bp1_list, sets_via_normed

    def compute_via_points(
        self, path, start, end, graph, inter_graph, with_rot=False, p_via_guess=None
    ):
        # Compute intersection sets
        x0 = np.empty(0)
        self.plan_msgs = []
        sets_inter = []
        for i, edge in enumerate(path[1:-1]):
            sets_inter.append(inter_graph.nodes[edge]["cset"])
            x0 = np.concatenate((x0, inter_graph.nodes[edge]["p_proj"], [0.5]))
            # Add a small constant to move the via point away from the border,
            # important for the next set optimization
            # TODO compute automatically
            idx = np.linalg.norm(sets_inter[-1][0], axis=1) > 1e-4
            sets_inter[-1][1][idx] -= 0.001

        sets = []
        sets_via = []
        q_ellipse = []
        p_mid = []
        w_size_via = []
        for i in range(len(path)):
            if i == 0:
                a_set = inter_graph.nodes[path[i]]["cset"][0]
                b_set = inter_graph.nodes[path[i]]["cset"][1]
                last_id = inter_graph.nodes[path[i]]["id0"]
                det_via = graph.nodes[last_id]["size"]
                q_via = graph.nodes[last_id]["q_ellipse"]
                p_mid_via = graph.nodes[last_id]["p_mid"]
                w_size_via.append(det_via)
            else:
                id0 = inter_graph.nodes[path[i]]["id0"]
                id1 = inter_graph.nodes[path[i]]["id1"]
                if id0 != last_id:
                    a_set = graph.nodes[id0]["cset"][0]
                    b_set = graph.nodes[id0]["cset"][1]
                    det_via = graph.nodes[id0]["size"]
                    q_via = graph.nodes[id0]["q_ellipse"]
                    p_mid_via = graph.nodes[id0]["p_mid"]
                    w_size_via.append(det_via)
                    last_id = id0
                elif id1 != last_id:
                    a_set = graph.nodes[id1]["cset"][0]
                    b_set = graph.nodes[id1]["cset"][1]
                    det_via = graph.nodes[id1]["size"]
                    q_via = graph.nodes[id1]["q_ellipse"]
                    p_mid_via = graph.nodes[id1]["p_mid"]
                    w_size_via.append(det_via)
                    last_id = id1
            sets.append([a_set, b_set])
            sets_via.append([a_set, b_set])
            q_ellipse.append(q_via)
            p_mid.append(p_mid_via)

        # Adapted sizes such that large sizes are penalized less
        # w_size_via /= np.max(w_size_via)
        # w_size_via = 1.1 - w_size_via
        w_size_via = 1 - np.cbrt(w_size_via)

        # Optimize via points
        sets_inter = normalize_set_size(sets_inter, self.max_set_size)
        sets_via = normalize_set_size(sets_via, self.max_set_size)
        nr_via = len(sets_inter)
        if with_rot:
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
                    self.replanning_phi = -phi_horizon
                    self.replanning_phi = np.max((self.replanning_phi, 0.0))
                    print(f"(Replanning) Horizon phi: {phi_horizon:.3f}")
                    print(f"(Replanning) Linprog phi: {sol_lin.x[0]:.3f}")
                    print(f"(Replanning) Replan phi: {self.replanning_phi:.3f}")
                    if phi_horizon < sol_lin.x[0]:
                        print("(Replanning) Horizon needs deviations")
                    p_via[0] = p_via[0] - self.replanning_phi * dp0
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

    def set_intersection(self, set1, set2, tol=0.0):
        set_inter = [
            np.concatenate((set1[0], set2[0])),
            np.concatenate((set1[1], set2[1])),
        ]
        sol_lin = linprog(
            np.zeros(3),
            A_ub=set_inter[0],
            b_ub=set_inter[1] - tol,
            bounds=(None, None),
        )
        point_inside_intersection = sol_lin.x
        success = sol_lin.success
        return point_inside_intersection, set_inter, success

    def add_edges(self, id_new, graph, inter_graph, end, start):
        connected = False
        set_new = graph.nodes[id_new]["cset"]
        for vertex in graph.nodes.items():
            if vertex[0] != id_new:
                setc = vertex[1]["cset"]
                idc = vertex[0]
                p_intersect, set_inter, intersects = self.set_intersection(
                    setc, set_new, tol=0.01
                )
            else:
                intersects = False
            if intersects:
                # Check if end-effector fits into the intersection set
                t_start_fit = time.perf_counter()
                a_set = set_inter[0]
                b_set = set_inter[1]
                fits = False
                fits, via = self.check_intersection(
                    a_set, b_set, self.l_ee, p_intersect
                )
                self.comp_time_fit += time.perf_counter() - t_start_fit

                self.id_inter += 1
                inter_graph.add_node(
                    self.id_inter,
                    cset=set_inter,
                    name=f"Interset {self.id_inter}",
                    id0=idc,
                    id1=id_new,
                    set0=setc,
                    set1=set_new,
                    conn_to_start=False,
                    conn_to_end=False,
                    p_proj=None,
                    p_via=via,
                )
                self.nr_inter_set += 2
                for edge in inter_graph.nodes.items():
                    v0 = edge[1]["id0"]
                    v1 = edge[1]["id1"]
                    cond1 = v0 == vertex[0] or v1 == vertex[0]
                    cond2 = v0 == id_new or v1 == id_new
                    if cond1:
                        size = vertex[1]["size"]
                    elif cond2:
                        size = graph.nodes[id_new]["size"]
                    cond = self.id_inter != edge[0]
                    if cond and (cond1 or cond2):
                        self.nr_edges += 2
                        p_proj = edge[1]["p_proj"]
                        if p_proj is None:
                            p_proj = end
                        if inter_graph.nodes[self.id_inter]["p_proj"] is None:
                            normed_sets = normalize_set_size(
                                [[set_inter[0], set_inter[1]]],
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
                                x0=p_intersect,
                                lbx=self.proj_lbu,
                                ubx=self.proj_ubu,
                                lbg=self.proj_lbg,
                                ubg=self.proj_ubg,
                                p=params,
                            )
                            inter_graph.nodes[self.id_inter]["p_proj"] = (
                                sol["x"].full().flatten()
                            )
                        dist = np.linalg.norm(
                            inter_graph.nodes[self.id_inter]["p_proj"] - p_proj
                        )

                        conn_to_start = (
                            inter_graph.nodes[self.id_inter]["conn_to_start"]
                            or edge[1]["conn_to_start"]
                        )
                        conn_to_end = (
                            inter_graph.nodes[self.id_inter]["conn_to_end"]
                            or edge[1]["conn_to_end"]
                        )
                        inter_graph.nodes[self.id_inter]["conn_to_start"] = (
                            conn_to_start
                        )
                        inter_graph.nodes[self.id_inter]["conn_to_end"] = conn_to_end
                        edge[1]["conn_to_start"] = conn_to_start
                        edge[1]["conn_to_end"] = conn_to_end
                        if conn_to_start and conn_to_end:
                            connected = True
                        else:
                            connected = False

                        c_size = np.tanh(0.25 - np.cbrt(size))
                        cost = dist * (1 + self.w_size * c_size) + self.w_bias
                        if not fits:
                            cost += self.c_fit
                        inter_graph.add_edge(self.id_inter, edge[0], weight=cost)
                        # print(
                        #     f"(Costs) cost {cost:.3f}, c_size {c_size:.3f} {np.cbrt(size):.3f}"
                        # )
        return connected
