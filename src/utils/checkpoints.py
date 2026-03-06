import json
from pathlib import Path
from typing import Set


def load_processed_documents(path: str | Path) -> Set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    with p.open("r", encoding="utf-8") as f:
        return set(json.load(f))


def save_processed_documents(path: str | Path, documents: Set[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(sorted(documents), f, ensure_ascii=False, indent=2)