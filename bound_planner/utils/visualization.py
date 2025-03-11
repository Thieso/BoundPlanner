import matplotlib.pyplot as plt
import numpy as np

from .util_functions import plot_set


def plot_via_path(p_via, r_via, sets_via, obs_sets, p_horizon=[]):
    plt.figure()
    plt.subplot(111, projection="3d")
    plt.axis("equal")
    p_via = np.array(p_via)
    plt.plot(p_via[:, 0], p_via[:, 1], p_via[:, 2], "C0")
    plt.plot(p_via[:, 0], p_via[:, 1], p_via[:, 2], "C0.")
    if p_horizon != []:
        p_horizon_np = np.array(p_horizon)
        plt.plot(p_horizon_np[:, 0], p_horizon_np[:, 1], p_horizon_np[:, 2], "C1.")
    for set_via in sets_via:
        plot_set(set_via[0], set_via[1], 1)
    for obs_set in obs_sets:
        plot_set(obs_set[0], obs_set[1], 2)


def plot_graph(p0, p1, graph, intersection_graph, obs_sets):
    plt.figure()
    plt.subplot(111, projection="3d")
    for edge0 in intersection_graph.nodes.items():
        for edge1 in intersection_graph.nodes.items():
            if intersection_graph.has_edge(edge0[0], edge1[0]):
                p0 = edge0[1]["p_proj"]
                p1 = edge1[1]["p_proj"]
                plt.plot(p0[0], p0[1], p0[2], "C0.")
                plt.plot(p1[0], p1[1], p1[2], "C0.")
                plt.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], "C0")
    plt.plot(p0[0], p0[1], p0[2], "C2.")
    plt.plot(p1[0], p1[1], p1[2], "C2.")
    for vertex in graph.nodes.items():
        plot_set(vertex[1]["cset"][0], vertex[1]["cset"][1], 1)
    for obs_set in obs_sets:
        plot_set(obs_set[0], obs_set[1], 2)
