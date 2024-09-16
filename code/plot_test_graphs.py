# This script creates a visualization of the graph algorithms performed on 100 random points.

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

points = np.random.rand(100, 2) * 10

distance_matrix = pd.DataFrame(
    np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
)
def create_rng(distance_matrix):
    n = len(distance_matrix)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = distance_matrix[i][j]
            if not any(
                distance_matrix[i][k] < dist_ij and distance_matrix[j][k] < dist_ij
                for k in range(n)
                if k != i and k != j
            ):
                G.add_edge(i, j)
    return G

def create_gg(distance_matrix, points):
    n = len(points)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = distance_matrix[i][j]
            mid_point = (points[i] + points[j]) / 2
            radius = dist_ij / 2
            if not any(
                np.linalg.norm(mid_point - points[k]) < radius for k in range(n)if k != i and k != j):
                G.add_edge(i, j)
    return G

def create_connected_knng(distance_matrix):
    n = len(distance_matrix)
    dists = distance_matrix.values
    indices = np.argsort(dists, axis=1)
    knng = nx.Graph()
    knng.add_nodes_from(range(n))
    k = 1
    while not nx.is_connected(knng):
        for i in range(n):
            if not knng.has_edge(i, indices[i, k]):
                knng.add_edge(i, indices[i, k])
        k += 1
    return knng

def create_nng(distance_matrix):
    n = len(distance_matrix)
    dists = distance_matrix.values
    indices = np.argsort(dists, axis=1)
    nng = nx.Graph()
    nng.add_nodes_from(range(n))
    for i in range(n):
        if not nng.has_edge(i, indices[i, 1]):
            nng.add_edge(i, indices[i, 1])
    return nng

knng = create_connected_knng(distance_matrix)
nng = create_nng(distance_matrix)
rng = create_rng(distance_matrix)
gg = create_gg(distance_matrix, points)

fig, axs = plt.subplots(1, 4, figsize=(22, 6))

nx.draw(nng, points, ax=axs[0], node_size=20, with_labels=False, edge_color="orange")
axs[0].set_title("NNG (Nearest Neighbor Graph)")

nx.draw(rng, points, ax=axs[1], node_size=20, with_labels=False, edge_color="blue")
axs[1].set_title("RNG (Relative Neighbor Graph)")

nx.draw(gg, points, ax=axs[2], node_size=20, with_labels=False, edge_color="red")
axs[2].set_title("GG (Gabriel Graph)")

nx.draw(knng, points, ax=axs[3], node_size=20, with_labels=False, edge_color="orange")
axs[3].set_title("Connected k-NNG")

plt.tight_layout()

plt.savefig("../results/other/graph_visualization.png")
