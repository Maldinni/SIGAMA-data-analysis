import os
import leidenalg
import numpy as np
import pandas as pd
import igraph as ig
from pathlib import Path
from collections import deque

from utils.parsing import parse_args, load_config
from utils.load_data import ensure_dirs, save_organized_clusters
from utils.organizing import ordenate_clusters, split_clusters

def main():
    os.chdir('..')  # Change to project root directory
    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    csv_directory = f'{cfg["paths"]["processed"]}'
    graph_directory = f'{cfg["paths"]["processed"]}/graphs'

    cluster_input_directory = f'{cfg["paths"]["processed"]}'
    cluster_output_directory = f'{cfg["paths"]["processed"]}/separated clusters'

    csv_file = os.path.join(csv_directory, 'chamado_abrir_202603041306_limpo.csv')
    graph_file = os.path.join(graph_directory, 'article_similarity.graphml')

    print('loading graph')
    G = ig.Graph.Read(graph_file, format='graphml')
    texts = G.vs['texts']

    print(
        'running Leiden community detection for different resolution parameters'
    )
    num_resolution_parameter = cfg['community_detection'][
        'num_resolution_parameter']
    max_resolution_parameter = cfg['community_detection'][
        'max_resolution_parameter']
    min_resolution_parameter = max_resolution_parameter / num_resolution_parameter

    resolution_parameters = np.linspace(min_resolution_parameter,
                                        max_resolution_parameter,
                                        num_resolution_parameter)

    modularity_values = np.zeros(num_resolution_parameter)
    num_unique_clusters = np.zeros(num_resolution_parameter)

    decreasing = deque(np.zeros(5, dtype=bool))

    for i, resolution_parameter in enumerate(resolution_parameters):
        partition = leidenalg.find_partition(
            G,
            leidenalg.CPMVertexPartition,
            weights='weight',
            resolution_parameter=resolution_parameter)

        modularity_values[i] = G.modularity(partition.membership,
                                            weights='weight')
        num_unique_clusters[i] = len(np.unique(partition.membership))
        print(f'number of unique clusters: {num_unique_clusters[i]}')
        print(f'modularity values: {modularity_values[i]}')

        if i > 0:
            decreasing.popleft()
            decreasing.append(modularity_values[i] < modularity_values[i - 1])

        if all(decreasing):
            print('modularity is decreasing, breaking')
            break

    best_resolution_parameter = resolution_parameters[np.argmax(
        modularity_values)]
    partition = leidenalg.find_partition(
        G,
        leidenalg.CPMVertexPartition,
        weights='weight',
        resolution_parameter=best_resolution_parameter)

    text_cluster = dict(zip(texts, partition.membership))

    print('saving cluster')
    df = pd.read_csv(csv_file)
    df['Cluster ID'] = df['ds_problema'].map(text_cluster)

    new_csv_file = csv_file.replace('.csv', '_clustered.csv')
    df.to_csv(new_csv_file, index=False)

    cluster_input_directory = Path(cluster_input_directory)
    cluster_output_directory = Path(cluster_output_directory)

    cluster_separated_file = os.path.join(cluster_input_directory, 'chamado_abrir_202603041306_limpo_clustered.csv')

    df_organized = pd.read_csv(cluster_separated_file)

    df_organized = ordenate_clusters(df_organized)

    save_organized_clusters(df_organized, cluster_input_directory)

    split_clusters(df_organized, cluster_output_directory)

    print("Organização finalizada.")

if __name__ == '__main__':
    main()