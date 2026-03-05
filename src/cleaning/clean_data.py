def clean_dataset(df):
    # Remove linhas de usuários placeholders
    df = df[~df["nome"].str.strip().eq("Italo Marcos De Sousa Barbosa")]
    df = df[~df["nome"].str.strip().eq("F P J Pecuaria Ltda")]

    # Remove a coluna "arquivo_origem" e "nome" ja que nao irei utilizá-las
    df = df.drop(columns=["arquivo_origem"], errors="ignore")
    df = df.drop(columns=["nome"], errors="ignore")

    # Remove as strings que podem vir vazias
    df = df.replace(r"^\s*$", None, regex=True)

    # Remove NULL
    df = df.dropna()

    return df

def clean_users_dataset(df):
    # Remove a coluna "arquivo_origem" e "nome" ja que nao irei utilizá-las
    df = df.drop(columns=["senha"], errors="ignore")
    df = df.drop(columns=["data_expiracao"], errors="ignore")
    df = df.drop(columns=["observacao"], errors="ignore")
    df = df.drop(columns=["alterar_senha"], errors="ignore")
    df = df.drop(columns=["inativo"], errors="ignore")
    df = df.drop(columns=["usuario_interno"], errors="ignore")
    #df = df.drop(columns=["bo_emitentegta"], errors="ignore")
    #df = df.drop(columns=["bo_emitentegtv"], errors="ignore")
    df = df.drop(columns=["nu_osgta"], errors="ignore")
    df = df.drop(columns=["nu_osgtv"], errors="ignore")
    df = df.drop(columns=["dt_validadeosgta"], errors="ignore")
    df = df.drop(columns=["dt_validadeosgtv"], errors="ignore")
    df = df.drop(columns=["tentativas_login"], errors="ignore")
    df = df.drop(columns=["nu_habiliticaogtvexterno"], errors="ignore")
    df = df.drop(columns=["nu_osclassifidorsemente"], errors="ignore")
    df = df.drop(columns=["dt_validadeosclassificacao"], errors="ignore")
    df = df.drop(columns=["bo_bloqueiohorario"], errors="ignore")
    df = df.drop(columns=["id_localidade_usuario_externo"], errors="ignore")
    df = df.drop(columns=["apelido"], errors="ignore")
    df = df.drop(columns=["id_arquivo"], errors="ignore")
    df = df.drop(columns=["id_unidade_vapt_vupt"], errors="ignore")
    df = df.drop(columns=["ts_usuario"], errors="ignore")
    df = df.drop(columns=["en_status_primeiroacesso"], errors="ignore")
    df = df.drop(columns=["id_perfil_primeiroacesso"], errors="ignore")
    df = df.drop(columns=["senha_at"], errors="ignore")
    df = df.drop(columns=["nu_habilitacaogtvexterno"], errors="ignore")
    df = df.drop(columns=["nu_osclassificadorsemente"], errors="ignore")

    # Remove as strings que podem vir vazias
    df = df.replace(r"^\s*$", None, regex=True)

    # Remove NULL
    df = df.dropna(how="all")

    return df

def clean_first_access_dataset(df):
    # Remove a coluna "arquivo_origem" e "nome" ja que nao irei utilizá-las
    df = df.drop(columns=["id_pessoa_responsavel"], errors="ignore")

    # Remove as strings que podem vir vazias
    df = df.replace(r"^\s*$", None, regex=True)

    # Remove NULL
    df = df.dropna(how="all")

    return df

def clean_first_access_llm_dataset(df):
    # Remove linhas de placeholders
    df = df[~df["ds_observacao"].str.strip().eq("Teste")]
    df = df[~df["ds_observacao"].str.strip().eq("teste")]
    df = df[~df["ds_observacao"].str.strip().eq("teste para reativar perfil")]
    df = df[~df["ds_observacao"].str.strip().eq("teste de permissão.")]
    df = df[~df["ds_observacao"].str.strip().eq("Teste para cadastro sem usuário.")]
    df = df[~df["ds_observacao"].str.strip().eq("Teste para regularizar usuário")]
    df = df[~df["ds_observacao"].str.strip().eq("teste para criação do usuário")]
    df = df[~df["ds_observacao"].str.strip().eq("teste para criação do usuário.")]
    df = df[~df["ds_observacao"].str.strip().eq("Teste para criação de usuário")]
    df = df[~df["ds_observacao"].str.strip().eq("teste para criação de usuário. Estava sem CNPJ")]
    df = df[~df["ds_observacao"].str.strip().eq("Solicitação para teste de usuário.")]
    df = df[~df["ds_observacao"].str.strip().eq("Acesso Liberado!")]
    df = df[~df["ds_observacao"].str.strip().eq("Acesso Liberado !")]
    df = df[~df["ds_observacao"].str.strip().eq("acesso Liberado!")]
    df = df[~df["ds_observacao"].str.strip().eq("acesso Liberado.")]
    df = df[~df["ds_observacao"].str.strip().eq("acesso liberado")]
    df = df[~df["ds_observacao"].str.strip().eq("acesso liberado!")]
    df = df[~df["ds_observacao"].str.strip().eq("Acesso LIberado!")]
    df = df[~df["ds_observacao"].str.strip().eq("Acesso Liberado.")]
    df = df[~df["ds_observacao"].str.strip().eq("Acesso Liberado")]
    df = df[~df["ds_observacao"].str.strip().eq("Acesso liberado")]
    df = df[~df["ds_observacao"].str.strip().eq("Acesso liberado !")]
    df = df[~df["ds_observacao"].str.strip().eq("Acesso liberado,")]
    df = df[~df["ds_observacao"].str.strip().eq("Acesso liberado.")]
    df = df[~df["ds_observacao"].str.strip().eq("ACESSO LIBERADO!")]
    df = df[~df["ds_observacao"].str.strip().eq("Acesso liberado!")]
    df = df[~df["ds_observacao"].str.strip().eq("Liberado")]
    df = df[~df["ds_observacao"].str.strip().eq("liberado")]
    df = df[~df["ds_observacao"].str.strip().eq("Aceso liberado.")]
    df = df[~df["ds_observacao"].str.strip().eq("Aceso Liberado.")]
    df = df[~df["ds_observacao"].str.strip().eq("Aceso Liberado!")]
    df = df[~df["ds_observacao"].str.strip().eq("Acessp liberado")]
    df = df[~df["ds_observacao"].str.strip().eq("Aesso liberado")]

    # Remove as strings que podem vir vazias
    df = df.replace(r"^\s*$", None, regex=True)

    # Remove NULL
    df = df.dropna(subset=["ds_observacao"])

    return df