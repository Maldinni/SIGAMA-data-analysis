from ingestion.load_multiple_controles import load_all_controles
from processing.normalization import normalize_columns

def build_dataset():
    df = load_all_controles()

    df = normalize_columns(df)

    return df