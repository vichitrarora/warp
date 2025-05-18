'''import pandas as pd


csv_file_path = "/home/warp/metafusion/train_dpo_data_1600.csv"  
df = pd.read_csv(csv_file_path)


parquet_file_path = "train_dpo_data_1600.parquet"
df.to_parquet(parquet_file_path, engine="pyarrow")

print(f"Parquet file saved at: {parquet_file_path}")'''
import pyarrow.parquet as pq

file_path = "/home/warp/verl/train_dpo_data_1600.parquet"

try:
    table = pq.read_table(file_path)
    print("Parquet file is valid and readable.")
    print("Columns:", table.column_names)
    print("Number of rows:", table.num_rows)
except Exception as e:
    print("Error reading Parquet file:", str(e))

