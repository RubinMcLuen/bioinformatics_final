import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

def load_normalized_data(results_folder):
    normalized_data_path = os.path.join(results_folder, 'normalized_data.csv')
    return pd.read_csv(normalized_data_path, index_col=0)

def load_graphs(results_folder):
    graphs_paths = {
        'c_knng': os.path.join(results_folder, 'c_knng_graph.pickle'),
        'rng': os.path.join(results_folder, 'rng_graph.pickle'),
        'gg': os.path.join(results_folder, 'gg_graph.pickle')
    }
    return {key: pickle.load(open(path, 'rb')) for key, path in graphs_paths.items()}

def plot_clusters(clusters, graph_type, normalized_data, results_folder, num_clusters):
    cluster_labels = set(clusters)
    visualization_folder = os.path.join(results_folder, f'visualization/{num_clusters}_clusters_expression_profiles')
    os.makedirs(visualization_folder, exist_ok=True)

    for cluster_label in cluster_labels:
        cluster_genes = normalized_data[clusters == cluster_label]
        plt.figure()
        for gene_name in cluster_genes.index:
            plt.plot(cluster_genes.columns, cluster_genes.loc[gene_name], label=gene_name)


        title = f'{graph_type.upper()} - Cluster {cluster_label}'

        plt.title(title)
        plt.xlabel('Conditions')
        plt.ylabel('Expression Level')
        plt.legend(loc='upper left', fontsize='small')
        plt.grid(True)
        
        plot_path = os.path.join(visualization_folder, f'plot_{graph_type}_cluster_{cluster_label}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved at {plot_path}")

def main(similarity_measure, graph_type, num_clusters):
    results_folder = f'../results/{similarity_measure}'
    normalized_data = load_normalized_data(results_folder)
    graphs = load_graphs(results_folder)

    clusters_path = os.path.join(results_folder, f'clusters_k{num_clusters}_{graph_type}.npy')
    clusters = np.load(clusters_path)
    
    plot_clusters(clusters, graph_type, normalized_data, results_folder, num_clusters)

if __name__ == "__main__":
    similarity_measure = 'mahalanobis'
    graph_type = 'c_knng' 
    num_clusters = 19  

    main(similarity_measure, graph_type, num_clusters)
