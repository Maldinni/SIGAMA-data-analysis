import pandas as pd
from config.settings import CONTROLE_EXCEL

df = pd.read_excel(CONTROLE_EXCEL, header=None)
print(df.head(10))

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    print(df.columns.tolist())

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def normalize_dates(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].str.strip().str.lower()
    return df