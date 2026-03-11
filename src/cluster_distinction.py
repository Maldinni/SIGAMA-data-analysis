import os
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from utils.parsing import parse_args, load_config, parse_distinction_file, extract_cluster_id
from utils.hypersphere import get_centroids
from utils.load_data import load_embedding_shards, align_to_df
from utils.semantic import get_article_strings

def main():

    DEFINITION_PROMPT = PromptTemplate(
        input_variables=["cluster_a", "cluster_b"],
        template="""
        Compare os dois grupos de textos.

        Cluster A:
        {cluster_a}

        Cluster B:
        {cluster_b}

        Identifique no máximo 3 diferenças principais entre os clusters.

        Saída:

        Diferenças:
        -
        -
        -
        """
    )

    args = parse_args()
    cfg = load_config(args)

    llm = ChatOllama(
        model=cfg["llm"]["model_name"],
        temperature=cfg["llm"]["temperature"]
    )

    distinction_chain = DEFINITION_PROMPT | llm

    processed_path = Path(cfg["paths"]["processed"])
    output_path = Path(cfg["paths"]["output"])

    news_file = processed_path / "articles_merged_cleaned_clustered.csv"
    clusters_file = processed_path / "clusters_solicitacoes_defined.csv"

    output_dir = output_path / "cluster_distinctions"

    os.makedirs(output_dir, exist_ok=True)

    news_df = pd.read_csv(news_file)
    cluster_df = pd.read_csv(clusters_file)

    # carregar embeddings
    shard_directory = Path(cfg["paths"]["raw"]) / "shards_h5"
    files = glob(os.path.join(shard_directory, "*.h5"))

    embeddings, texts = load_embedding_shards(files)

    aligned_embeddings, aligned_texts = align_to_df(
        embeddings,
        texts,
        news_df
    )

    centroids = get_centroids(
        aligned_embeddings,
        news_df["Cluster ID"].values
    )

    labels = news_df["Cluster ID"].values
    texts = news_df["text"].values

    print("Iniciando distinção entre clusters...")

    for label, centroid in enumerate(centroids):

        similar_cluster_label = int(
            cluster_df.loc[label, "Most Similar Cluster"]
        )

        cluster_texts = texts[labels == label]
        cluster_embeddings = aligned_embeddings[labels == label]

        similar_cluster_texts = texts[labels == similar_cluster_label]
        similar_cluster_embeddings = aligned_embeddings[
            labels == similar_cluster_label
        ]

        number_cluster = min(5, len(cluster_texts))
        number_similar = min(5, len(similar_cluster_texts))

        cluster_a = get_article_strings(
            centroids[similar_cluster_label],
            cluster_embeddings,
            cluster_texts,
            number_cluster
        )

        cluster_b = get_article_strings(
            centroid,
            similar_cluster_embeddings,
            similar_cluster_texts,
            number_similar
        )

        chain_input = {
            "cluster_a": "\n\n".join(cluster_a),
            "cluster_b": "\n\n".join(cluster_b)
        }

        raw_output = distinction_chain.invoke(chain_input)

        output_file = output_dir / f"cluster_{label}_distinction.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(raw_output.content)

        print(f"Cluster {label} salvo.")


def json_converter():

    args = parse_args()
    cfg = load_config(args)

    processed_path = Path(cfg["paths"]["processed"])
    output_path = Path(cfg["paths"]["output"])

    clusters_file = processed_path / "clusters_defined.csv"
    cluster_df = pd.read_csv(clusters_file)

    distinction_dir = output_path / "cluster_distinctions"

    files = glob(os.path.join(distinction_dir, "*.txt"))

    distinctions = []

    for file in files:

        cluster_id = extract_cluster_id(file)

        features = parse_distinction_file(file)

        distinctions.append({
            "Cluster ID": cluster_id,
            "Distinguishing Features": features
        })

    dist_df = pd.DataFrame(distinctions)

    final_df = cluster_df.merge(
        dist_df,
        on="Cluster ID",
        how="left"
    )

    output_file = processed_path / "clusters_defined_distinguished.csv"

    final_df.to_csv(output_file, index=False)

    print("Dataset final criado.")

if __name__ == '__main__':
    main()
    json_converter()