import pandas as pd
from config.settings import CONTROLE_EXCEL

def load_controle_excel():
    if not CONTROLE_EXCEL.exists():
        raise FileNotFoundError(f"{CONTROLE_EXCEL} não encontrado")

    df = pd.read_excel(CONTROLE_EXCEL, header=1)
    return df