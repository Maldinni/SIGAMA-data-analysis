import glob
import os
import time
from tqdm import tqdm
import pandas as pd

from src.utils.parsing import parse_args, load_config
from src.utils.io import ensure_dirs, determine_output_filename
from src.utils.checkpoints import load_processed_urls, save_processed_urls
from src.utils.embedding import unpack_embedding_parameters, save_embeddings, SentenceTransformerEmbeddings

from src.classes.data_types import Embeddings

def main():
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

    cleaned_files = glob.glob(os.path.join(cleaned_directory, '*.csv'))

    dfs = []
    for file in cleaned_files:
        dfs.append(pd.read_csv(file))

    df = pd.concat(dfs, ignore_index=True)

    # Initialize the embedding model
    embedding_model = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=batch_size
    )

    processed_path = f'{paths["checkpoints"]}/processed_embedded_urls.json'
    processed = load_processed_urls(processed_path)

    # Remove already embedded articles from the dataframe
    df = df[~df['canonical_url'].isin(processed)]

    # Ensure embedding output directory exists
    os.makedirs(embedded_directory, exist_ok=True)

    output_file, shard_id = determine_output_filename(
        embedded_directory, 'pkl')

    for start in tqdm(range(0, len(df), items_per_shard)):
        end = start + items_per_shard

        text_embeddings = Embeddings()

        selection = df.iloc[start:end]
        texts = selection['ds_observacao'].tolist()
        
        # Generate embeddings for texts
        embedded_texts = embedding_model.embed_documents(texts)

        # Store PMIDs and embeddings
        text_embeddings.ids = selection['id_primeiro_acesso_historico'].tolist()
        text_embeddings.texts = texts
    
        text_embeddings.embeddings = embedded_texts

        # Save the current shard of embeddings to disk
        save_embeddings(text_embeddings, output_file)

        # Update the list of processed articles
        processed.update(text_embeddings.urls)
        save_processed_urls(processed_path, processed)

        # Prepare next shard filename
        shard_id += 1
        output_file = os.path.join(embedded_directory, f'shard_{shard_id:04d}.pkl')

        # Wait before next batch to avoid overloading resources
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()