import pandas as pd

def load_csv(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega um arquivo CSV e retorna um DataFrame.
    
    :param caminho_arquivo: Caminho do arquivo CSV
    :return: DataFrame com os dados carregados
    """
    try:
        df = pd.read_csv(caminho_arquivo)
        print("CSV carregado com sucesso!")
        return df
    except FileNotFoundError:
        print("Arquivo não encontrado.")
    except pd.errors.EmptyDataError:
        print("O arquivo está vazio.")
    except Exception as e:
        print(f"Erro ao carregar o CSV: {e}")