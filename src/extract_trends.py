import os
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from utils.parsing import parse_args, load_config, extract_cluster_id, clean_llm_text
from utils.load_data import ensure_dirs


def truncate(text, max_chars=2000):
    return text[:max_chars]

def main():

    TRENDS_PROMPT = PromptTemplate(
        input_variables=["title", "date", "old_news", "recent_news"],
        template="""
            Compare dois conjuntos de textos sobre o tema "{title}".

            Conjunto 1: textos antigos (antes de {date})
            Conjunto 2: textos recentes (depois de {date})

            Identifique diferenças entre os dois conjuntos.

            Saída:

            Narrativas em alta:
            -
            -
            -

            Atores em alta:
            -
            -
            -

            Narrativas em baixa:
            -
            -
            -

            Atores em baixa:
            -
            -
            -

            Textos antigos:
            {old_news}

            Textos recentes:
            {recent_news}
            """
    )

    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    num_news_per_set = cfg['trends']['max_article_number'] // 2
    recent_cutoff = pd.to_datetime(str(cfg['trends']['recent_cutoff']) + "-01-01")
    older_cutoff = pd.to_datetime(str(cfg['trends']['older_cutoff']) + "-01-01")

    llm = ChatOllama(
        model=cfg["llm"]["model_name"],
        temperature=cfg["llm"]["temperature"]
    )

    trends_chain = TRENDS_PROMPT | llm

    processed_path = Path(cfg["paths"]["processed"])

    news_file = processed_path / "primeiro_acesso_historico_202603041232_limpo_llm_clustered.csv"
    clusters_file = processed_path / "clusters_solicitacoes_defined.csv"

    news_df = pd.read_csv(news_file)
    cluster_df = pd.read_csv(clusters_file)

    news_df["dt_cadastro"] = pd.to_datetime(
        news_df["dt_cadastro"],
        format="ISO8601",
        utc=True
    ).dt.tz_localize(None)

    labels = news_df["Cluster ID"].unique()

    output_dir = Path(cfg["paths"]["output"]) / "extract_trends"
    os.makedirs(output_dir, exist_ok=True)

    print("Iniciando processamento de clusters...")

    for label in tqdm(labels):

        older_df = news_df[
            (news_df["Cluster ID"] == label) &
            (news_df["dt_cadastro"] >= older_cutoff) &
            (news_df["dt_cadastro"] < recent_cutoff)
        ].sort_values("dt_cadastro", ascending=True)


        older_news_list = older_df["ds_observacao"].dropna().values

        n_old = min(num_news_per_set, len(older_news_list))

        if n_old == 0:
            continue

        older_news = "\n\n".join(
            [truncate(t) for t in older_news_list[:n_old]]
        )

        recent_df = news_df[
            (news_df["Cluster ID"] == label) &
            (news_df["dt_cadastro"] >= recent_cutoff)
        ].sort_values("dt_cadastro", ascending=False)

        recent_news_list = recent_df["ds_observacao"].dropna().values[:num_news_per_set]

        if len(recent_news_list) == 0:
            continue

        recent_news = "\n\n".join(
            [truncate(t) for t in recent_news_list]
        )

        title_row = cluster_df[cluster_df["Cluster ID"] == label]

        if title_row.empty:
            cluster_title = f"Cluster {label}"
        else:
            cluster_title = title_row.iloc[0]["Title"]

        chain_input = {
            "title": cluster_title,
            "date": str(recent_cutoff),
            "old_news": older_news,
            "recent_news": recent_news
        }

        raw_output = trends_chain.invoke(chain_input)

        output_file = output_dir / f"cluster_{label}_trends.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(raw_output.content)

    print("Processamento concluído.")

def json_converter():

    args = parse_args()
    cfg = load_config(args)

    processed_path = Path(cfg["paths"]["processed"])
    output_path = Path(cfg["paths"]["output"])

    clusters_file = processed_path / "clusters_solicitacoes_defined_distinguished.csv"
    cluster_df = pd.read_csv(clusters_file)

    trends_dir = output_path / "extract_trends"

    files = glob(os.path.join(trends_dir, "*.txt"))

    records = []

    for file in files:

        cluster_id = extract_cluster_id(file)

        with open(file, "r", encoding="utf-8") as f:
            content = clean_llm_text(f.read())

        records.append({
            "Cluster ID": cluster_id,
            "Trends": content
        })

    trends_df = pd.DataFrame(records)

    final_df = cluster_df.merge(
        trends_df,
        on="Cluster ID",
        how="left"
    )

    output_file = processed_path / "clusters_solicitacoes_defined_distinguished_trends.csv"

    final_df.to_csv(output_file, index=False)

    print("Dataset final criado.")

if __name__ == '__main__':
    os.chdir("..")
    main()
    json_converter()