import os
import numpy as np
import pandas as pd
import sys
from glob import glob
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.json import SimpleJsonOutputParser

from utils.parsing import parse_args, load_config
from utils.hypersphere import get_representative_articles, get_centroids
from utils.load_data import ensure_dirs, load_embedding_shards, align_to_df

def truncate(text, max_chars=1300):
    return text[:max_chars]

def main():
    os.chdir('..')
# Define prompt templates
    DEFINITION_PROMPT = PromptTemplate(
        input_variables=["articles"],
        template="""
                Leia os textos abaixo e identifique o tema principal.

                Use apenas informações dos textos.

                Saída:

                Palavras-chave:
                - 
                - 
                - 
                - 
                - 

                Título:

                Descrição:
                1-2 frases.

                Textos:
                {articles}
                """,
    )

    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    required_fields = cfg['definition']['required_fields']

    llm = ChatOllama(
        model="mistral:7b",
        temperature=cfg['llm']['temperature']
    )

    cluster_directory = Path(cfg["paths"]["processed"]) / "separated clusters"
    cluster_files = glob(os.path.join(cluster_directory, "cluster_*.csv"))
    output_directory = f'{cfg["paths"]["output"]}'

    solicitacoes_file = Path(cfg["paths"]["processed"]) / "primeiro_acesso_historico_202603041232_limpo_llm_clustered.csv"

    solicitacoes_df = pd.read_csv(solicitacoes_file)

    cluster_csv_file = 'clusters_defined.csv'
    output_path = os.path.join(output_directory, cluster_csv_file)

    if os.path.exists(output_path):
        cluster_definitions_df = pd.read_csv(output_path)
        cluster_definitions = cluster_definitions_df.to_dict('records')
        processed_clusters = set(cluster_definitions_df['Cluster ID'].values.tolist())
    else:
        cluster_definitions = []
        processed_clusters = set()
    
    shard_directory = Path(cfg["paths"]["raw"]) / "shards_h5"
    files = glob(os.path.join(shard_directory, '*.h5'))

    embeddings, texts = load_embedding_shards(files)

    aligned_embeddings, aligned_texts = align_to_df(
        embeddings,
        texts,
        solicitacoes_df
    )

    centroids = get_centroids(
    aligned_embeddings,
    solicitacoes_df['Cluster ID'].values
    )

    #sources = solicitacoes_df['source'].values
    definition_chain = DEFINITION_PROMPT | llm

    print("Iniciando processamento de clusters...")

    for cluster_file in cluster_files:

        label = int(Path(cluster_file).stem.split("_")[1])

        if label in processed_clusters:
            continue

        print(f"Processando cluster {label}...")

        cluster_df = pd.read_csv(cluster_file)

        cluster_articles = cluster_df["ds_observacao"].dropna().values

        number_of_articles = min(
            cfg['definition']['max_article_number'],
            len(cluster_articles)
        )

        cluster_mask = solicitacoes_df['Cluster ID'] == label

        cluster_embeddings = aligned_embeddings[cluster_mask]

        cluster_texts = solicitacoes_df.loc[cluster_mask, "ds_observacao"].values

        cluster_articles = get_representative_articles(
            centroids[label],
            cluster_embeddings,
            cluster_texts,
            number_of_articles
        )

        cluster_articles = [truncate(a) for a in cluster_articles]

        chain_input = {"articles": "\n\n".join(cluster_articles)}

        raw_output = definition_chain.invoke(chain_input)

        with open(f"{output_directory}/cluster_{label}.txt", "w", encoding="utf-8") as f:
            f.write(raw_output.content)

        print(f"Cluster {label} salvo.")

if __name__ == '__main__':
    main()