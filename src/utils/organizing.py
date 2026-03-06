import os
import pandas as pd

def ordenate_clusters(df):
    df_ordered = df.sort_values(by="Cluster ID", ascending=True)
    return df_ordered

def split_clusters(df, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for cluster_id, group in df.groupby("Cluster ID"):

        file_path = os.path.join(output_folder, f"cluster_{cluster_id}.csv")

        group.to_csv(file_path, index=False)

    print(f"{df['Cluster ID'].nunique()} clusters salvos em {output_folder}")