# This script creates a visualization of the clustering algorithm on random blobs of points that was used in my final presentation.

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def create_nng(points, distance_matrix):
    n = len(points)
    dists = distance_matrix.values
    indices = np.argsort(dists, axis=1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    k = 1
    while not nx.is_connected(G):
        for i in range(n):
            if not G.has_edge(i, indices[i, k]):
                G.add_edge(i, indices[i, k])
        k += 1
    return G

def create_rng(points, distance_matrix):
    n = len(points)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    dist_matrix = distance_matrix.values

    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = dist_matrix[i][j]
            if not any(dist_matrix[i][k] < dist_ij and dist_matrix[j][k] < dist_ij for k in range(n) if k != i and k != j):
                G.add_edge(i, j)
    return G

def create_gg(points, distance_matrix):
    n = len(points)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    dist_matrix = distance_matrix.values

    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = dist_matrix[i][j]
            mid_point = (points[i] + points[j]) / 2
            radius = dist_ij / 2
            if not any(np.linalg.norm(mid_point - points[k]) < radius for k in range(n) if k != i and k != j):
                G.add_edge(i, j)
    return G

def perform_clustering(graph, k):
    L = nx.laplacian_matrix(graph).astype(float)
    eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
    k_eigenvectors = eigenvectors[:, 1:k+1]
    norm_k_eigenvectors = k_eigenvectors / np.linalg.norm(k_eigenvectors, axis=1, keepdims=True)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(norm_k_eigenvectors)
    return kmeans.labels_

def plot_results(points, clusters, graphs, titles):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axes[0]):
        ax.scatter(points[:, 0], points[:, 1], c=clusters[i], cmap='viridis', marker='o', edgecolor='k')
        ax.set_title(titles[i] + ' Clusters')
    for i, ax in enumerate(axes[1]):
        ax.set_title(titles[i])
        pos = {i: points[i] for i in range(len(points))}
        nx.draw(graphs[i], pos, ax=ax, node_size=50, node_color='blue', edge_color='gray', with_labels=False)
        ax.set_xlim(min(points[:,0])-1, max(points[:,0])+1)
        ax.set_ylim(min(points[:,1])-1, max(points[:,1])+1)
    return fig 


def save_plots_to_file(fig, filename):
    fig.savefig(filename, format='png', bbox_inches='tight')
    print(f"Plot saved to {filename}")

# Generate test data
n_samples = 100
n_features = 2
centers = 3
points, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)
distance_matrix = pd.DataFrame(squareform(pdist(points, 'euclidean')))

# Create graphs
nng = create_nng(points, distance_matrix)
rng = create_rng(points, distance_matrix)
gg = create_gg(points, distance_matrix)

# Perform clustering
nng_clusters = perform_clustering(nng, 3)
rng_clusters = perform_clustering(rng, 3)
gg_clusters = perform_clustering(gg, 3)

# Plot results
graphs = [rng, gg, nng]
clusters = [nng_clusters, rng_clusters, gg_clusters]
titles = ['RNG', 'GG', 'connected kNNG']
fig = plot_results(points, clusters, graphs, titles)
save_plots_to_file(fig, "../results/other/clustering_visualization.png")