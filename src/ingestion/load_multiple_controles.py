from pathlib import Path
import pandas as pd
from config.settings import DOCUMENTOS_DIR

def find_controle_files() -> list[Path]:
    arquivos = []

    for file in DOCUMENTOS_DIR.rglob("*Controle de Solicitação*.xls*"):
        if not file.is_file():
            continue

        nome = file.name

        # Ignora arquivos temporários do Excel
        if nome.startswith("~$"):
            continue

        if nome.startswith(".~lock"):
            continue

        arquivos.append(file)

    return arquivos

def extrair_data_da_pasta(file: Path):
    relativo = file.relative_to(DOCUMENTOS_DIR)
    partes = relativo.parts

    if len(partes) >= 4:
        ano, mes, dia = partes[0], partes[1], partes[2]
        return ano, mes, dia

    return None, None, None

def load_all_controles():
    arquivos = DOCUMENTOS_DIR.rglob("*Controle de Solicitação*.xls*")
    dfs = []

    for file in arquivos:
        if not file.is_file():
            continue

        if file.name.startswith(("~$", ".~lock")):
            continue

        df = pd.read_excel(file, header=None)

        # remove primeira linha do arquivo
        df = df.iloc[1:, :3]
        df.columns = ["nome", "cpf/cnpj", "status"]

        # remove headers repetidos internos
        df = df[~df["nome"].str.strip().str.upper().eq("NOME")]

        df["cpf/cnpj"] = (
            df["cpf/cnpj"]
            .astype(str)
            .str.replace(r"\D", "", regex=True)
            .str.strip()
        )

        # cria tipo_de_conta depois
        df["tipo_de_conta"] = df["cpf/cnpj"].apply(
            lambda x: "Empresa" if len(x) == 14 else "Pessoa Física" if len(x) == 11 else "Inválido"
        )

        ano, mes, dia = extrair_data_da_pasta(file)

        df["arquivo_origem"] = file.name
        df["ano"] = ano
        df["mês"] = mes
        df["dia"] = dia

        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame()