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

    p.add_argument("--source", default=None, help="Rodar apenas uma fonte por nome.")
    p.add_argument("--max_articles_per_source", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    dirs_file = load_toml(args.directories)
    initial_embedding_file = load_toml(args.initial_embedding)
    graph_construction_file = load_toml(args.clustering)

    cfg = {
        "paths": dirs_file["paths"],
        "initial_embedding": initial_embedding_file,
    }

    return cfg