import glob
import os
import time
import pickle
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from utils.parsing import parse_args, load_config
from utils.load_data import ensure_dirs, determine_output_filename, save_to_hdf5
from utils.checkpoints import load_processed_documents, save_processed_documents
from utils.embedding import unpack_embedding_parameters, save_embeddings, SentenceTransformerEmbeddings

from classes.data_types import Embeddings

def main():
    os.chdir('..')  # Change to project root directory
    args = parse_args()
    cfg = load_config(args)

    embedding_parameters = cfg["initial_embedding"]

    model_name, sleep_time, batch_size, items_per_shard = unpack_embedding_parameters(
        embedding_parameters
    )

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    cleaned_directory = f'{cfg["paths"]["processed"]}'
    embedded_directory = f'{cfg["paths"]["raw"]}/embedded'
    shard_directory = f'{cfg["paths"]["raw"]}/shards_h5'

    if not os.path.exists(cleaned_directory):
        print(f"Erro: O diretório {cleaned_directory} não existe.")
    else:
        cleaned_files = glob.glob(os.path.join(cleaned_directory, 'primeiro_acesso_historico_202603041232_limpo_llm.csv'))
        print(f"Arquivos encontrados no diretório de processados: {cleaned_files}")

    # Se não houver arquivos CSV, aparece um erro
    if not cleaned_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado no diretório {cleaned_directory}")
    dfs = []
    for file in cleaned_files:
        dfs.append(pd.read_csv(file))

    df = pd.concat(dfs, ignore_index=True)

    # Initialize the embedding model
    embedding_model = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=batch_size
    )

    processed_path = f'{paths["checkpoints"]}/processed_embedded_documents.json'
    processed = load_processed_documents(processed_path)

    # Remove already embedded articles from the dataframe
    df = df[~df['id_primeiro_acesso_historico'].isin(processed)]

    # Ensure embedding output directory exists
    os.makedirs(embedded_directory, exist_ok=True)

    # Ensure converted embed output directory exists
    os.makedirs(shard_directory, exist_ok=True)

    output_file, shard_id = determine_output_filename(
        embedded_directory, 'pkl')

    for start in tqdm(range(0, len(df), items_per_shard)):
        end = start + items_per_shard

        text_embeddings = Embeddings()

        selection = df.iloc[start:end]
        texts = selection['ds_observacao'].apply(lambda x: str(x) if isinstance(x, (str, int, float)) else '').tolist()
        
        # Generate embeddings for texts
        embedded_texts = embedding_model.embed_documents(texts)

        # Store PMIDs and embeddings
        text_embeddings.ids = selection['id_primeiro_acesso_historico'].tolist()
        text_embeddings.texts = texts
    
        text_embeddings.embeddings = embedded_texts

        # Save the current shard of embeddings to disk
        save_embeddings(text_embeddings, output_file)

        # Update the list of processed articles
        processed.update(text_embeddings.texts)
        save_processed_documents(processed_path, processed)

        # Prepare next shard filename
        shard_id += 1
        output_file = os.path.join(embedded_directory, f'shard_{shard_id:04d}.pkl')

        # Wait before next batch to avoid overloading resources
        time.sleep(sleep_time)

    embedded_directory = Path(embedded_directory)
    shard_directory = Path(shard_directory)

    for pkl_file in embedded_directory.glob("*.pkl"):

        print(f"Convertendo {pkl_file.name}...")

        # Carrega pickle
        with open(pkl_file, "rb") as f:
            obj = pickle.load(f)

        print(type(obj))
        print(obj.__dict__.keys())
        # Define nome do .h5
        h5_filename = shard_directory / (pkl_file.stem + ".h5")

        # Salva em HDF5
        news_list = []

        for i in range(len(obj.texts)):
            article = {
                "text": obj.texts[i],
                "embedding": obj.embeddings[i] if i < len(obj.embeddings) else None
            }
            news_list.append(article)

        save_to_hdf5(news_list, str(h5_filename))

    print("Conversão finalizada.")

if __name__ == "__main__":
    main()