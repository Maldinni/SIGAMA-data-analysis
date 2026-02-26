from pipeline.build_dataset import build_dataset, build_dataset_anonymized
from config.settings import OUTPUT_DIR, CSV_INPUT_DIR
from utils.load_data import load_csv

def scraper():
    df = build_dataset()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "dataset_normalizado.csv", index=False)

    print("Dataset gerado com sucesso!")

def anonymizer():
    print(CSV_INPUT_DIR)
    df_normalized = load_csv(CSV_INPUT_DIR)
    print(df_normalized)

    df = build_dataset_anonymized(df_normalized)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "dataset_normalizado_anonimizado.csv", index=False)
    
    print("Dataset anonimizado com sucesso!")

if __name__ == "__main__":
    scraper()
    anonymizer()