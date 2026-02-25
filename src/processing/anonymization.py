import hashlib
import re
from config.settings import SALT

def anonimizar_cpf(cpf):
    # remove tudo que não é número
    cpf_limpo = re.sub(r"\D", "", str(cpf))
    
    # garante 11 dígitos (caso venha sem zero à esquerda)
    cpf_limpo = cpf_limpo.zfill(11)
    
    return hashlib.sha256((cpf_limpo + str(SALT)).encode()).hexdigest()