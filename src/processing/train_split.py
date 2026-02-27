from utils.load_data import load_csv

from config.settings import RAW_CSV_INPUT_DIR, OUTPUT_DIR, OUTPUT_TRAIN, OUTPUT_TEST

import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_train_test():

    df = load_csv(RAW_CSV_INPUT_DIR)

    df.columns = df.columns.str.strip().str.lower()

    mes_map = {
        "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4,
        "Maio": 5, "Junho": 6, "Julho": 7, "Agosto": 8,
        "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12
    }

    df["mes_num"] = df["mês"].map(mes_map)

    df["data"] = pd.to_datetime(
        dict(year=df["ano"], month=df["mes_num"], day=df["dia"]),
        errors="coerce"
    )

    df["approved"] = (df["status"] == "APROVADO").astype(int)

    df = df.sort_values(["cpf_hash", "data"])

    df["attempt_number"] = df.groupby("cpf_hash").cumcount() + 1

    df["previous_rejections"] = (
        df.groupby("cpf_hash")["approved"]
          .apply(lambda x: (x.shift(1) == 0).cumsum())
          .reset_index(level=0, drop=True)
          .fillna(0)
    )

    df["tipo_pf"] = (df["tipo_de_conta"] == "Pessoa Física").astype(int)

    users = df["cpf_hash"].unique()

    train_users, test_users = train_test_split(
        users,
        test_size=0.3,
        random_state=42
    )

    train_df = df[df["cpf_hash"].isin(train_users)].copy()
    test_df = df[df["cpf_hash"].isin(test_users)].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_TRAIN, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    print("Train/Test gerados com sucesso!")
    print(f"Tamanho treino: {len(train_df)}")
    print(f"Tamanho teste: {len(test_df)}")