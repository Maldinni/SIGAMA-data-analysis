from pathlib import Path

from config.settings import PROCESSED_DIR, RAW_USERS_FILE, RAW_FIRST_ACCESS_FILE, RAW_OPEN_TICKET_FILE, RAW_HISTORY_TICKET_FILE, CLEANED_FIRST_ACCESS_FILE
from utils.load_data import load_csv
from pipeline.build_dataset import build_db_dataset_anonymized, build_db_dataset, build_db_dataset_for_llm

def cleaner(file_path):
    print(file_path)
    df = load_csv(file_path)
    df_normalized = build_db_dataset(df)
    print(df_normalized)

    df = df_normalized

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / f"{Path(file_path).stem}_limpo{Path(file_path).suffix}", index=False)

    print("Dataset limpo com sucesso!")

def cleaner_anonymizer(file_path):
    print(file_path)
    df = load_csv(file_path)
    df_normalized = build_db_dataset(df)
    print(df_normalized)
    df_anonymized = build_db_dataset_anonymized(df_normalized)

    df = df_anonymized

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / f"{Path(file_path).stem}_limpo{Path(file_path).suffix}", index=False)

    print("Dataset limpo com sucesso!")

def cleaner_llm(file_path):
    print(file_path)
    df = load_csv(file_path)
    df_processed = build_db_dataset_for_llm(df)
    print(df_processed)

    df = df_processed

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / f"{Path(file_path).stem}_llm{Path(file_path).suffix}", index=False)

    print("Dataset limpo com sucesso!")

if __name__ == "__main__":
    cleaner_anonymizer(RAW_USERS_FILE)
    cleaner(RAW_FIRST_ACCESS_FILE)
    cleaner(RAW_OPEN_TICKET_FILE)
    cleaner(RAW_HISTORY_TICKET_FILE)
    cleaner_llm(CLEANED_FIRST_ACCESS_FILE)
