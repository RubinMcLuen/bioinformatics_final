# This script creates a plot of the figure of merit for a range of cluster numbers for a given similarity measure.

import matplotlib.pyplot as plt
import pandas as pd
import os

class PlotFOMSeries:
    def __init__(self, results_folder, similarity_measure, output_filename='FOM_plot.png'):
        self.results_folder = results_folder
        self.similarity_measure = similarity_measure.capitalize()
        self.output_filename = output_filename

    def plot_fom_series(self):
        graph_types = ['random', 'c_knng', 'rng', 'gg']
        plt.figure(figsize=(10, 8))

        labels = {
            'random': 'Random',
            'c_knng': 'Connected kNNG',
            'rng': 'RNG',
            'gg': 'GG'
        }

        max_cluster_size = 0

        for graph_type in graph_types:
            fom_series_path = os.path.join(self.results_folder, f'fom_series_{graph_type}.csv')
            if os.path.exists(fom_series_path):
                df = pd.read_csv(fom_series_path)
                df['Cluster Size'] = df['Cluster Size'].astype(int)
                plt.plot(df['Cluster Size'], df['FOM'], label=labels[graph_type])
                max_cluster_size = max(max_cluster_size, max(df['Cluster Size']))
            else:
                print(f"No FOM series data found for {graph_type}. Ensure the data has been generated.")

        plt.xlabel('Number of Clusters')
        plt.ylabel('Figure of Merit (FOM)')
        plt.title(f'{self.similarity_measure} - FOM Across Different Graph Types')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0) 
        plt.xticks(range(1, max_cluster_size + 1)) 

        plt.savefig(os.path.join(self.results_folder, self.output_filename))

similarity_measure = 'euclidean'
results_folder = f'../results/{similarity_measure}'
fom_plotter = PlotFOMSeries(results_folder, similarity_measure)
fom_plotter.plot_fom_series()
