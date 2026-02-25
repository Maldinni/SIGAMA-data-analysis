def clean_dataset(df):

    # Remove a coluna "arquivo_origem" ja que nao irei utilizá-la
    df = df.drop(columns=["arquivo_origem"], errors="ignore")

    # Remove as strings que podem vir vazias
    df = df.replace(r"^\s*$", None, regex=True)

    # Remove NULL
    df = df.dropna()

    return df