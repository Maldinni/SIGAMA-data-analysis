import pickle
import tomllib
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
        # Initializes the embedding model using SentenceTransformers
        # model_name: pre-trained model for semantic embedding generation
        # batch_size: number of documents to process per iteration
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed_documents(self, texts):
        # Converts a list of texts into numerical embeddings
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

def unpack_embedding_parameters(config):
    """
    Unpack the embedding parameters from the embedding dict.
    
    Parameters:
    - config: dict
    
    Returns:
    - items_per_shard: int
    """

    model = config['model']
    sleep_time = config['api_calls']['sleep_time']
    batch_size = config['api_calls']['batch_size']
    items_per_shard = config['items_per_shard']

    return model, sleep_time, batch_size, items_per_shard


def save_embeddings(embeddings, output_file):
    """
    Save the embeddings to the output file.

    Parameters:
    - embeddings: Embeddings
    - output_file: str
    """
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)