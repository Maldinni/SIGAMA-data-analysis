from pipeline.build_dataset import build_dataset
from config.settings import OUTPUT_DIR

def main():
    df = build_dataset()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "dataset_normalizado.csv", index=False)

    print("Dataset gerado com sucesso!")

if __name__ == "__main__":
    main()