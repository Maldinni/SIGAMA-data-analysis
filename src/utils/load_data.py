import os
import csv
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

def load_csv(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega um arquivo CSV e retorna um DataFrame.
    
    :param caminho_arquivo: Caminho do arquivo CSV
    :return: DataFrame com os dados carregados
    """
    try:
        df = pd.read_csv(caminho_arquivo)
        print("CSV carregado com sucesso!")
        return df
    except FileNotFoundError:
        print("Arquivo não encontrado.")
    except pd.errors.EmptyDataError:
        print("O arquivo está vazio.")
    except Exception as e:
        print(f"Erro ao carregar o CSV: {e}")

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def determine_output_filename(output_dir: str, ext: str = "csv") -> Tuple[str, int]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing = sorted(out.glob(f"shard_*.{ext}"))
    if not existing:
        return str(out / f"shard_{0:04d}.{ext}"), 0

    last = existing[-1].stem
    shard_id = int(last.split("_")[1]) + 1
    return str(out / f"shard_{shard_id:04d}.{ext}"), shard_id