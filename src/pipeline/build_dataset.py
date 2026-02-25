from ingestion.load_multiple_controles import load_all_controles
from processing.normalization import normalize_columns
from processing.anonymization import anonimizar_cpf

def build_dataset():
    df = load_all_controles()
    df = normalize_columns(df)

    df["cpf_hash"] = df["cpf/cnpj"].apply(anonimizar_cpf)
    df = df.drop(columns=["cpf/cnpj"])

    return df