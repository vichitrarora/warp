from datasets import Dataset
import pandas as pd
from config import DataConfig

def load_and_preprocess_data(config: DataConfig):
    train_df = pd.read_csv(config.train_path)
    eval_df = pd.read_csv(config.eval_path)
    
    train_df.rename(columns=config.column_mapping, inplace=True)
    eval_df.rename(columns=config.column_mapping, inplace=True)
    
    return Dataset.from_pandas(train_df), Dataset.from_pandas(eval_df)

def format_dpo_dataset(example):
    return {
        "prompt": f"Schema: {example['schema_info']}\nQuery: {example['nl_query']}",
        "chosen": example["preferred_query"],
        "rejected": example["rejected_query"],
    }