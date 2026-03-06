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

def save_to_hdf5(news_list, filename, disable_tqdm=False):
    """
    Save dataset to HDF5.

    Parameters:
    - news_list: list of dicts or objects with attributes
    - filename: str
    """

    with h5py.File(filename, 'w') as file:

        n = len(news_list)

        # String datasets
        texts = file.create_dataset('text', (n,),
                                      dtype=h5py.string_dtype())

        # Optional embedding group (if you generate later)
        embedding_group = file.create_group('embeddings')

        for i, article in tqdm(enumerate(news_list),
                               total=n,
                               disable=disable_tqdm):

            texts[i] = str(article["text"]) if article["text"] else ""

            # Se tiver embedding
            if "embedding" in article and article["embedding"] is not None:
                embedding_group.create_dataset(
                    str(i),
                    data=np.array(article["embedding"], dtype=np.float32),
                    dtype='f'
                )
            else:
                embedding_group.create_dataset(str(i), shape=(0,), dtype='f')

def load_embeddings_from_hdf5(filename):
    """
    Load embeddings from a single HDF5 shard.

    Returns:
    - embeddings: np.ndarray (N, D)
    - urls: list[str]
    """

    with h5py.File(filename, "r") as f:

        # Carrega URLs (identificador)
        texts = f["text"][:]
        texts = [
            u.decode("utf-8") if isinstance(u, bytes) else u
            for u in texts
        ]

        total_articles = len(texts)

        # Carrega embeddings
        embedding_group = f["embeddings"]

        # Pega dimensão do primeiro embedding
        first_embedding = np.array(embedding_group["0"])
        dim = first_embedding.shape[0]

        embeddings = np.zeros((total_articles, dim), dtype=np.float32)

        for i in range(total_articles):
            embeddings[i] = np.array(embedding_group[str(i)])

    return embeddings, texts

def load_embedding_shards(embeddings_files, disable_tqdm=False):
    """
    Load embeddings from multiple shards and concatenate them into a single array. Also return the associated PMIDs.

    Parameters:
    - embeddings_files: List of strings, the paths to the embeddings files.
    - disable_tqdm: bool

    Returns:
    - embeddings: np.array, the concatenated embeddings.
    - urls: List of strings, the URLs of the articles.
    """

    embeddings = []
    texts = []

    for file in tqdm(embeddings_files, disable=disable_tqdm):
        embeddings_shard, texts_shard = load_embeddings_from_hdf5(file)
        embeddings.append(embeddings_shard)
        texts.extend(texts_shard)

    return np.concatenate(embeddings, axis=0), texts