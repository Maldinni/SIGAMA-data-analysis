import os
from pathlib import Path
from dotenv import load_dotenv # Recebe as variáveis de ambiente

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "keys.env")

#print(teste)
#print(BASE_DIR)

#as configs de app receberão os itens do dicionario para acertar os parametros

BASEPATH = Path(os.getenv("SIGAMA_BASEPATH")) #caminho padrão do servidor para acesso e manipulação das pastas e arquivos do projeto, criar e alterar o arquivo .env para conter o SIGAMA_BASEPATH(caminho)
LOCAL_BASEPATH = Path(os.getenv("LOCAL_BASEPATH"))

if not BASEPATH:
    raise RuntimeError("BASEPATH não definido no .env")

if not LOCAL_BASEPATH:
    raise RuntimeError("LOCAL_BASEPATH não definido no .env")

DOCUMENTOS_DIR = BASEPATH / "Documentos Solicitaçoes de Acesso"
CONTROLE_EXCEL = BASEPATH / "Controle de Solicitação.xlsx"
OUTPUT_DIR = LOCAL_BASEPATH / "Documents"

TIMEOUT = 60
