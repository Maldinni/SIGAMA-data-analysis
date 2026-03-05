import os
from pathlib import Path
from dotenv import load_dotenv # Recebe as variáveis de ambiente

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "keys.env")

#print(teste)
#print(BASE_DIR)

#as configs de app receberão os itens do dicionario para acertar os parametros

#BASEPATH = Path(os.getenv("SIGAMA_BASEPATH")) #caminho padrão do servidor para acesso e manipulação das pastas e arquivos do projeto, criar e alterar o arquivo .env para conter o SIGAMA_BASEPATH(caminho)
LOCAL_BASEPATH = Path(os.getenv("LOCAL_BASEPATH"))
SALT = Path(os.getenv("SALT_HASH_KEY"))

#if not BASEPATH:
#    raise RuntimeError("BASEPATH não definido no .env")

#if not LOCAL_BASEPATH:
#    raise RuntimeError("LOCAL_BASEPATH não definido no .env")

#DOCUMENTOS_DIR = BASEPATH / "Documentos Solicitaçoes de Acesso"
#CONTROLE_EXCEL = BASEPATH / "Controle de Solicitação.xlsx"
OUTPUT_DIR = LOCAL_BASEPATH / "data"
PROCESSED_DIR = OUTPUT_DIR / "processed"
CSV_INPUT_DIR = OUTPUT_DIR / "dataset_normalizado.csv"
OUTPUT_TRAIN = OUTPUT_DIR / "scraped" / "train_dataset.csv"
OUTPUT_TEST = OUTPUT_DIR / "scraped" / "test_dataset.csv"
RAW_CSV_INPUT_DIR = OUTPUT_DIR / "scraped" / "dataset_normalizado_anonimizado.csv"
RAW_USERS_FILE = OUTPUT_DIR / "db" / "usuario_202603050949.csv"
RAW_FIRST_ACCESS_FILE = OUTPUT_DIR / "db" / "primeiro_acesso_historico_202603041232.csv"
RAW_OPEN_TICKET_FILE = OUTPUT_DIR / "db" / "chamado_abrir_202603041306.csv"
RAW_HISTORY_TICKET_FILE = OUTPUT_DIR / "db" / "chamado_historico_202603041306.csv"
CLEANED_USERS_CSV_INPUT_DIR = OUTPUT_DIR / "db" / "usuario_202603050949.csv"

TIMEOUT = 60
