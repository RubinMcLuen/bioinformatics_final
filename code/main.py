import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
import pickle
from similarity_measures import *

# This class contains all steps of the main experiment for this project.
# It loads and normalizes the data, calculates the distance matrix, and constructs graphs.
# Then, it runs the spectral community detection algorithm on the graphs.
# Finally it calculates the figure of merit of the resulting clusters.
# The class is set up to do the entire process for one given similarity measure, and it will
# calculate the figure of merit for 1 to k clusters.

# At most important points throughout, it saves the state of the process, so that it can
# continue from where it left off. The normalized data and distance matrix are saved as .csv.
# The graphs are saved as .pickle, and the clusters are saved as .npy. 
class DataAnalysisProcessor:
    def __init__(self, expression_file, gene_names_file, similarity_measure):
        self.expression_file = expression_file
        self.gene_names_file = gene_names_file
        self.similarity_measure = similarity_measure
        self.normalized_data = None
        self.points = None
        self.distance_matrix = None
        self.graphs = {}
        self.clusters = None
        self.removed_condition_data = None

        self.distance_functions = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'mahalanobis': mahalanobis_distance,
            'pearson': pearson_correlation,
            'uncentered': uncentered_correlation,
            'spearman': spearman_rank_correlation,
            'absolute': absolute_correlation,
            'squared': squared_correlation,
        }

        self.results_folder = os.path.join('../results', similarity_measure)
        os.makedirs(self.results_folder, exist_ok=True)
        print(f"Results folder: {self.results_folder}")

    # This function merges the original data files into a pandas dataframe where the
    # rows are genes and columns are conditions. Then, it normalizes the data by
    # z-score normalization to make the mean 0, and variance 1. 
    def load_and_normalize_data(self):
        print("Loading and normalizing data...")
        normalized_path = os.path.join(self.results_folder, 'normalized_data.csv')
        if os.path.exists(normalized_path):
            print("Loading existing normalized data from CSV")
            self.normalized_data = pd.read_csv(normalized_path, index_col=0)
            self.points = self.normalized_data.values
        else:
            print("Loading and normalizing data...")
            expression_data = pd.read_csv(self.expression_file, sep='\s+', header=None)
            with open(self.gene_names_file) as f:
                gene_names = [line.strip() for line in f]
            expression_data.index = gene_names
            self.normalized_data = (expression_data - expression_data.mean()) / expression_data.std()
            self.points = self.normalized_data.values
            self.save_state(normalized_path, self.normalized_data)
            print("Data normalized and saved.")
    
    # This function removes the selected condition from the original dataset, for the purpose
    # of calculating figure of merit. I set the experiment up to remove condition 5.
    def remove_condition(self, condition_to_remove):
        print(f"Removing condition {condition_to_remove}")
        self.removed_condition_data = self.normalized_data.iloc[:, condition_to_remove]
        self.normalized_data = self.normalized_data.drop(self.normalized_data.columns[condition_to_remove], axis=1)
        self.points = self.normalized_data.values
        print("Condition removed.")

    # This funciton calculates the distance matrix for the given similarity measure.
    # It calls the functions for the similarity measures from similarity_measures.py.
    def calculate_distance_matrix(self):
        distance_matrix_path = os.path.join(self.results_folder, 'distance_matrix.csv')
        if os.path.exists(distance_matrix_path):
            print("Loading existing distance matrix from CSV")
            self.distance_matrix = pd.read_csv(distance_matrix_path, index_col=0)
        else:
            print("Calculating new distance matrix")
            distance_function = self.distance_functions[self.similarity_measure]
            n = len(self.normalized_data)
            distance_matrix = np.zeros((n, n))
            
            if self.similarity_measure == 'mahalanobis':
                # Compute the covariance matrix for Mahalanobis distance
                cov_matrix = np.cov(self.points, rowvar=False)
                inv_cov_matrix = np.linalg.inv(cov_matrix)

                for i in range(n):
                    for j in range(i + 1, n):
                        distance = distance_function(self.points[i], self.points[j], inv_cov_matrix)
                        distance_matrix[i][j] = distance
                        distance_matrix[j][i] = distance
            else:
                for i in range(n):
                    for j in range(i + 1, n):
                        distance = distance_function(self.points[i], self.points[j])
                        distance_matrix[i][j] = distance
                        distance_matrix[j][i] = distance

            self.distance_matrix = pd.DataFrame(distance_matrix, index=self.normalized_data.index, columns=self.normalized_data.index)
            self.save_state(distance_matrix_path, self.distance_matrix)
            print("Distance matrix calculated and saved")

    # This function constructs the connected k-nearest neighbor graph, based on the
    # distance matrix. Starting at 1, it iteratively finds the k nearest neighbors,
    # and then checks if the graph is connected or not.
    def create_c_knng(self):
        file_path = os.path.join(self.results_folder, 'c_knng_graph.pickle')
        self.graphs['c_knng'] = self.load_graph(file_path)
        if self.graphs['c_knng'] is not None:
            print("c_knng loaded from file.")
            return

        print("Creating Connected k-Nearest Neighbor Graph")
        n = len(self.points)
        dists = self.distance_matrix.values
        indices = np.argsort(dists, axis=1)
        self.graphs['c_knng'] = nx.Graph()
        self.graphs['c_knng'].add_nodes_from(range(n))
        k = 1
        while not nx.is_connected(self.graphs['c_knng']):
            for i in range(n):
                if not self.graphs['c_knng'].has_edge(i, indices[i, k]):
                    self.graphs['c_knng'].add_edge(i, indices[i, k])
            k += 1
        self.save_graph(file_path, self.graphs['c_knng'])
        print("c_knng created and saved.")

    # This function constructs the relative neighbor graph, based on the distance
    # matrix. It goes through every gene and checks if every other gene has a third
    # that's closer to both of them, than they are to each other.
    def create_rng(self):
        file_path = self.results_folder + '/rng_graph.pickle'
        self.graphs['rng'] = self.load_graph(file_path)
        if self.graphs['rng'] is not None:
            print("RNG loaded from file.")
            return

        print("Creating Relative Neighbor Graph")
        n = len(self.points)
        dist_matrix = self.distance_matrix.values
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                dist_ij = dist_matrix[i][j]
                no_close_neighbors = True
                for k in range(n):
                    if k != i and k != j and dist_matrix[i][k] < dist_ij and dist_matrix[j][k] < dist_ij:
                        no_close_neighbors = False
                        break
                if no_close_neighbors:
                    G.add_edge(i, j)
        self.graphs['rng'] = G
        self.save_graph(file_path, G)
        print("RNG created and saved.")

    # This funcitons constructs the Gabriel Graph based on the distance matrix
    # It goes through every gene, and checks if there's any point within a circle
    # with the diameter of the circle being the distance between the two genes,
    # calculated by the same similarity measure as the distance matrix.
    def create_gg(self):
        file_path = self.results_folder + '/gg_graph.pickle'
        self.graphs['gg'] = self.load_graph(file_path)
        if self.graphs['gg'] is not None:
            print("GG loaded from file.")
            return

        print("Creating Gabriel Graph")
        n = len(self.points)
        dist_matrix = self.distance_matrix.values
        distance_function = self.distance_functions[self.similarity_measure]
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            print(i)
            for j in range(i + 1, n):
                dist_ij = dist_matrix[i, j]
                mid_point = (self.points[i] + self.points[j]) / 2
                radius = dist_ij / 2
                no_interference = True
                for k in range(n):
                    if k != i and k != j:
                        if self.similarity_measure == 'mahalanobis':
                            cov_matrix = np.cov(self.points, rowvar=False)
                            inv_cov_matrix = np.linalg.inv(cov_matrix)
                            dist_mk = distance_function(mid_point, self.points[k], inv_cov_matrix)
                        else:
                            dist_mk = distance_function(mid_point, self.points[k])
                        if dist_mk < radius:
                            no_interference = False
                            break
                if no_interference:
                    G.add_edge(i, j, weight=dist_ij)

        self.graphs['gg'] = G
        self.save_graph(file_path, G)
        print("GG created and saved.")

    # This function creates k clusters, where genes are randomly assigned to clusters
    # for the sake of figure of merit comparison.
    def perform_random_clustering(self, k):
        print(f"Performing random clustering with {k} clusters...")
        n = len(self.normalized_data)
        random_labels = np.random.randint(0, k, n)
        self.clusters = random_labels
        print("Random clustering completed.")

    # This function performs the spectral community detection on the given graph. It calculates
    # the laplacian matrix, calculates the eigenvectors of that matrix, selects the first k
    # eigenvectors, and then performs k-means clustering on those eigenvectors. It outputs
    # k clusters in the form of a list of genes and their associated cluster number.
    def perform_clustering(self, k, graph_type):
        print(f"Performing clustering with {k} clusters and graph type '{graph_type}'...")
        if graph_type not in self.graphs:
            raise ValueError(f"No graph set for {graph_type}.")

        graph = self.graphs[graph_type]
        L = nx.laplacian_matrix(graph).astype(float)
        eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
        k_eigenvectors = eigenvectors[:, 1:k + 1]

        if np.any(np.isnan(k_eigenvectors)) or np.any(np.isinf(k_eigenvectors)):
            print("Data contains NaN or infinite values")
        else:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(k_eigenvectors)
            self.clusters = kmeans.labels_

            clustering_path = os.path.join(self.results_folder, f'clusters_k{k}_{graph_type}.npy')
            self.save_state(clustering_path, self.clusters)
            print(f"Clustering completed with {k} clusters, results saved at {clustering_path}.")

    # This function saves the graph as a .pickle file so that it's memory efficient
    # and can be reloaded quickly.
    def save_graph(self, file_path, graph):
        with open(file_path, 'wb') as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
        print(f"Graph saved at {file_path}.")

    # This function loads the saved .pickle graph file.
    def load_graph(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
        
    # This function saves the clusters as a .npy file so that they're memory efficient.
    def save_state(self, file_path, data):
        if isinstance(data, np.ndarray):
            np.save(file_path, data)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(file_path)
        else:
            pd.Series(data).to_csv(file_path)
        print(f"State saved at {file_path}.")

    # This function calculates the figure of merit, using the process given in the report.
    def calculate_fom(self, normalized_data, clusters, condition_to_remove):
        print("Calculating Figure of Merit (FOM)")
        n = len(normalized_data)  # Total number of genes
        k = len(set(clusters))  # Number of clusters

        cluster_means = {i: [] for i in range(k)}
        for i, label in enumerate(clusters):
            cluster_means[label].append(i)

        deviations = []  # Store squared deviations for the specified condition
        for cluster_index in cluster_means:
            # Get data for the genes in the cluster
            cluster_data = normalized_data.iloc[cluster_means[cluster_index]]

            # Mean expression level for the removed condition within this cluster
            cluster_mean = cluster_data.iloc[:, condition_to_remove].mean()

            # Calculate the squared deviation from the cluster mean for the removed condition
            deviations.extend((cluster_data.iloc[:, condition_to_remove] - cluster_mean) ** 2)

        # Root mean square deviation
        root_mean_square_deviation = np.sqrt(sum(deviations) / n)

        # Normalizing factor accounting for the number of clusters
        normalization_factor = np.sqrt((n - k) / n)

        # Calculate the FOM
        fom = root_mean_square_deviation / normalization_factor

        print("FOM calculation completed.")
        return fom

    # This functions runs most of the entire process. With a given similarity measure,
    # it runs the functions to create the different graphs, perform the community detection,
    # and calculate figure of merit. It runs this process 1 to k times, where k is the
    # desired number of output clusters. This function outputs
    # a file with the figure of merit for each number of clusters.
    def calculate_fom_series(self, max_k, condition_to_remove):
        """Calculate FOM series for multiple cluster sizes and graph types."""
        print("Calculating FOM series")
        # graph_types = ['c_knng', 'rng', 'gg', 'random'] 
        graph_types = ['gg']
        fom_results = {graph_type: [] for graph_type in graph_types}

        for graph_type in graph_types:
            for k in range(1, max_k + 1):
                print(f"Calculating FOM for cluster size {k} with graph type '{graph_type}'...")
                if graph_type == 'random':
                    self.perform_random_clustering(k)
                else:
                    self.set_current_graph(graph_type)
                    self.perform_clustering(k, graph_type)
                fom = self.calculate_fom(self.normalized_data, self.clusters, condition_to_remove) 
                fom_results[graph_type].append((k, fom)) 

            fom_series_path = os.path.join(self.results_folder, f'fom_series_{graph_type}.csv')
            df_fom_series = pd.DataFrame(fom_results[graph_type], columns=['Cluster Size', 'FOM'])
            df_fom_series.to_csv(fom_series_path, index=False)
            print(f"FOM series saved at {fom_series_path}")

        return fom_results

    # This function calls for the creation of the different graph types.
    def set_current_graph(self, graph_type):
        self.calculate_distance_matrix()
        
        if graph_type == 'c_knng':
            print("Setting current graph to c_kNNG...")
            self.create_c_knng()
        elif graph_type == 'rng':
            print("Setting current graph to RNG...")
            self.create_rng()
        elif graph_type == 'gg':
            print("Setting current graph to GG...")
            self.create_gg()
        else:
            raise ValueError("Unsupported graph type.")
        
# Example usage of the class
# The class is set up to run one similarity measure at a time,
# you can set the input files, similarity measure, condition to remove,
# and range of cluster numbers.
expression_file = '../data/yeastEx.txt'
gene_names_file = '../data/yeastNames.txt'
similarity_measure = 'mahalanobis'
max_k = 20
condition_to_remove = 5

processor = DataAnalysisProcessor(expression_file, gene_names_file, similarity_measure)
processor.load_and_normalize_data()
processor.remove_condition(condition_to_remove)
processor.calculate_fom_series(max_k, condition_to_remove)

# The results are saved in the results directory, within the directory of the appropriate similarity measure.