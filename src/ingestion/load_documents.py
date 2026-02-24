from pathlib import Path
from config.settings import DOCUMENTOS_DIR

def list_documents() -> list[Path]:
    if not DOCUMENTOS_DIR.exists():
        raise FileNotFoundError(f"{DOCUMENTOS_DIR} não encontrado")

    return list(DOCUMENTOS_DIR.glob("*"))