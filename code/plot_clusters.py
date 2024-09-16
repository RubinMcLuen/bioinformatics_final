# This script creates a visualization of the clusters from a given similarity measure, graph type, and number of clusters.

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

def load_graph(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_clusters(file_path):
    return np.load(file_path)

def plot_graph_clusters(graph, clusters, title, file_path):
    plt.figure(figsize=(10, 8))
    ax = plt.gca() 
    pos = nx.spring_layout(graph)

    num_clusters = len(set(clusters))
    cmap = plt.get_cmap('viridis', num_clusters)
    colors = [cmap(i / num_clusters) for i in range(num_clusters)]

    node_colors = [colors[cluster] for cluster in clusters]
    nx.draw_networkx_nodes(graph, pos, node_size=70, node_color=node_colors, alpha=0.8, ax=ax)

    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.3, ax=ax)

    plt.title(title)

    for i in range(num_clusters):
        plt.scatter([], [], color=colors[i], label=f'Cluster {i+1}')

    plt.legend(title="Cluster Number", loc='best', bbox_to_anchor=(1, 1)) 
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

base_dir = '../results'
similarity_measure = 'mahalanobis'
graph_type = 'rng' 
k = 8 

results_dir = os.path.join(base_dir, similarity_measure)
graph_file = os.path.join(results_dir, f'{graph_type}_graph.pickle')
clusters_file = os.path.join(results_dir, f'clusters_k{k}_{graph_type}.npy')
title = f'{graph_type.upper()} Clusters'

if os.path.exists(graph_file) and os.path.exists(clusters_file):
    graph = load_graph(graph_file)
    clusters = load_clusters(clusters_file)
    plot_path = os.path.join(results_dir, f'visualization/{graph_type}_clusters_k{k}.png')
    plot_graph_clusters(graph, clusters, title + f' (k={k})', plot_path)
    print(f'Plot saved at {plot_path}')
else:
    print(f'Graph or cluster data not found for {similarity_measure} with {graph_type} and k={k}')
