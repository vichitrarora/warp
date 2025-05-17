import pandas as pd
from datasets import Dataset
from typing import Tuple

class DataLoader:
    @staticmethod
    def load_datasets(train_path: str, eval_path: str) -> Tuple[Dataset, Dataset]:
        train_df = pd.read_csv(train_path)
        eval_df = pd.read_csv(eval_path)
        
        column_mapping = {
            "schema": "schema_info",
            "natural_language_query": "nl_query",
            "corret_mongo_query": "preferred_query",
            "incorrect_mongo_query": "rejected_query"
        }
        
        train_df.rename(columns=column_mapping, inplace=True)
        eval_df.rename(columns=column_mapping, inplace=True)
        
        return Dataset.from_pandas(train_df), Dataset.from_pandas(eval_df)