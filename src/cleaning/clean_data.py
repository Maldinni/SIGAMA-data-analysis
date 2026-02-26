def clean_dataset(df):

    # Remove a coluna "arquivo_origem" e "nome" ja que nao irei utilizá-las
    df = df.drop(columns=["arquivo_origem"], errors="ignore")
    df = df.drop(columns=["nome"], errors="ignore")
    # Remove as strings que podem vir vazias
    df = df.replace(r"^\s*$", None, regex=True)

    # Remove NULL
    df = df.dropna()

    return df