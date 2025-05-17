import pandas as pd
from datasets import Dataset
from config import DATA_PATH

def load_and_prepare_dataset():
    df = pd.read_csv(DATA_PATH)
    df.rename(columns={
        "database_id": "db_id",
        "schema": "schema_info",
        "natural_language_query": "nl_query",
        "mongo_query": "mongo_query"
    }, inplace=True)
    df.drop_duplicates(inplace=True)
    df["mongo_query"] = df["mongo_query"].str.replace("'", '"').str.rstrip(";")

    dataset = Dataset.from_pandas(df)
    return dataset
