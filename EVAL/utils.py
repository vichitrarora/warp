import pandas as pd

def clean_mongo_query(query):
    if pd.isna(query):
        return ""
    query = query.lstrip("\n").rstrip(";")
    return query

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df["mongo_query"] = df["mongo_query"].apply(clean_mongo_query)
    df["mongo_query"] = df["mongo_query"].str.strip('"\'')
    return df

def save_dataset(df, file_path):
    df.to_csv(file_path, index=False)

def normalize_query(query):
    query_no_space = ''.join(query.split())
    return query_no_space.replace("'", '"')
