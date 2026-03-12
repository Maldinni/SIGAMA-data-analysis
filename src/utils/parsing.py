import argparse
import tomllib
from pathlib import Path
from typing import Any, Dict


def load_toml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    raw = p.read_bytes()
    with open(p, "rb") as f:
        return tomllib.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="News scraping prototype")

    p.add_argument("--directories", default="parameters/directories.toml")
    p.add_argument("--clustering", default="parameters/clustering.toml")
    p.add_argument("--initial_embedding", default="parameters/ingestion/initial_embedding.toml")
    p.add_argument("--semantic", default="parameters/analysis/semantic.toml")

    p.add_argument("--source", default=None, help="Rodar apenas uma fonte por nome.")
    p.add_argument("--max_articles_per_source", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    dirs_file = load_toml(args.directories)
    initial_embedding_file = load_toml(args.initial_embedding)
    graph_construction_file = load_toml(args.clustering)
    cluster_definition_file = load_toml(args.semantic)
    trends_file = load_toml(args.semantic)

    cfg = {
        "paths": dirs_file["paths"],
        "initial_embedding": initial_embedding_file,
        "graph_construction": graph_construction_file["graph_construction"],
        "community_detection": graph_construction_file["community_detection"],
        "definition": cluster_definition_file["definition"],
        "llm": cluster_definition_file["llm"],
        "trends": trends_file["trends"],
    }

    return cfg

def extract_cluster_id(filename):

    match = re.search(r"(\d+)", filename)
    return int(match.group(1))

def load_and_parse_cluster_file(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # keywords
    keywords_section = re.search(
        r"Palavras-chave:(.*?)Título:",
        text,
        re.S
    )

    keywords = []
    if keywords_section:
        keywords = re.findall(r"- (.*)", keywords_section.group(1))

    # title
    title_match = re.search(
        r"Título:\s*(.*?)\n",
        text
    )

    title = title_match.group(1).strip() if title_match else ""

    # description
    desc_match = re.search(
        r"Descrição:\s*(.*?)\n\s*Foco:",
        text,
        re.S
    )

    description = desc_match.group(1).strip() if desc_match else ""

    focus_match = re.search(
        r"Foco:\s*(.*)",
        text
    )

    focus = focus_match.group(1).strip() if focus_match else ""

    return keywords, title, description, focus

def parse_distinction_file(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    matches = re.findall(r"- (.*)", text)

    return "; ".join(matches)

import re

def clean_llm_text(text):

    text = re.sub(r"#+", "", text)      
    text = re.sub(r"\*\*", "", text)    
    text = re.sub(r"- ", "", text)       
    text = text.replace("\n", " ")       
    text = re.sub(r"\s+", " ", text)     

    return text.strip()