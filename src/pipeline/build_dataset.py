from ingestion.load_multiple_controles import load_all_controles
#from processing.normalization import normalize_columns
from processing.anonymization import anonimizar_cpf
from cleaning.clean_data import clean_dataset

def build_dataset():
    df_original = load_all_controles()
    df = normalize_columns(df_original)

    return df

def build_dataset_anonymized(dataset):
    df = dataset

    df["cpf_hash"] = df["cpf/cnpj"].apply(anonimizar_cpf)
    df = df.drop(columns=["cpf/cnpj"])

    #Para limpar o dataset antes de retornar o final
    df_final = clean_dataset(df)

    return df_final