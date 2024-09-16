# This script creates a table that displays the figure of merit of every combination of similarity measures and graph types

import pandas as pd
import os

class FOMDataChart:
    def __init__(self, base_results_folder):
        self.base_results_folder = base_results_folder

    def generate_fom_chart(self, similarity_measures, graph_types):
        data = [] 

        for sim_measure in similarity_measures:
            results_folder = os.path.join(self.base_results_folder, sim_measure)
            for graph_type in graph_types:
                fom_series_path = os.path.join(results_folder, f'fom_series_{graph_type}.csv')
                if os.path.exists(fom_series_path):
                    df = pd.read_csv(fom_series_path)
                    df['Cluster Size'] = df['Cluster Size'].astype(int)
                    df['Similarity Measure'] = sim_measure
                    df['Graph Type'] = graph_type
                    data.append(df)
                else:
                    print(f"No FOM series data found for {graph_type} with {sim_measure}.")

        full_data = pd.concat(data, ignore_index=True)

        pivot_table = full_data.pivot_table(index='Cluster Size', columns=['Similarity Measure', 'Graph Type'], values='FOM')
        pivot_table = pivot_table.reindex(columns=graph_types, level='Graph Type')

        pivot_table = pivot_table.applymap(lambda x: round(x, 3) if not pd.isna(x) else "")

        min_fom = full_data.groupby(['Similarity Measure', 'Graph Type'])['FOM'].min().reset_index()
        min_fom = min_fom.loc[min_fom.groupby('Similarity Measure')['FOM'].idxmin()]
        min_fom['Metric'] = 'Min FOM'
        min_fom_pivot = min_fom.pivot(index='Metric', columns=['Similarity Measure', 'Graph Type'], values='FOM')
        min_fom_pivot = min_fom_pivot.reindex(columns=graph_types, level='Graph Type')
        min_fom_pivot = min_fom_pivot.applymap(lambda x: round(x, 3) if pd.notnull(x) else "") 

        final_table = pd.concat([pivot_table, min_fom_pivot], axis=0)

        return final_table

similarity_measures = ['euclidean', 'manhattan', 'mahalanobis', 'pearson', 'uncentered', 'spearman', 'absolute', 'squared']
graph_types = ['c_knng', 'rng', 'gg']
results_folder = '../results'
fom_data_chart = FOMDataChart(results_folder)
fom_chart = fom_data_chart.generate_fom_chart(similarity_measures, graph_types)

print(fom_chart)
# fom_chart.to_excel('../results/other/fom_chart.xlsx', engine='openpyxl')

