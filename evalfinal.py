import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def clean_mongo_query(query):
    if pd.isna(query):
        return ""
    query = query.lstrip("\n")
    query = query.rstrip(";")
    return query

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df["mongo_query"] = df["mongo_query"].apply(clean_mongo_query)
    df.columns = df.columns.str.strip()
    df['mongo_query'] = df['mongo_query'].str.strip('"\'')
    return df

def save_dataset(df, file_path):
    df.to_csv(file_path, index=False)

def normalize_query(query):
    # Remove all whitespace characters
    query_no_space = ''.join(query.split())
    # Standardize quotes to double quotes
    query_standard_quotes = query_no_space.replace("'", '"')
    return query_standard_quotes

class MongoQueryGenerator:
    def __init__(self, model_name, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def generate(self, schema, natural_language_query):
        prompt = f"""### Instruction:
You are an AI assistant that generates MongoDB shell queries.
Given the schema, and natural language query, output a valid MongoDB shell query.

### Schema:
{schema}

### Natural Language Query:
{natural_language_query}

### MongoDB Shell Query:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=100)
        mongo_query = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        mongo_query = next((line.strip() for line in mongo_query.split("\n") if line.strip().startswith("db.")), "")
        return mongo_query

def evaluate_queries(df, generator):
    expected_columns = {"schema", "mongo_query", "natural_language_query"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    df["predicted_mongo_query"] = ""
    df["score"] = 0

    correct_count = 0
    total_queries = len(df)

    for i, row in tqdm(df.iterrows(), total=total_queries, desc="Generating & Evaluating Queries"):
        schema = row["schema"].strip()
        nl_query = row["natural_language_query"].strip()
        correct_query = row["mongo_query"].strip()

        if not nl_query or not schema:
            continue

        predicted_query = generator.generate(schema, nl_query)
        df.at[i, "predicted_mongo_query"] = predicted_query

        # Normalize both queries before comparison
        predicted_norm = normalize_query(predicted_query)
        correct_norm = normalize_query(correct_query)

        if predicted_norm == correct_norm:
            df.at[i, "score"] = 1
            correct_count += 1

    accuracy = correct_count / total_queries if total_queries > 0 else 0
    return accuracy, df

def main():
    eval_file_path = "/home/warp/metafusion/WARPxMetafusion/eval/model_output/eval_data_v1.csv"
    cleaned_eval_path = "/home/warp/metafusion/WARPxMetafusion/eval/model_output/eval_data_v1_cleaned.csv"
    output_csv_path = "/home/warp/metafusion/WARPxMetafusion/eval/model_output/eval_data_v4_resultsdpo.csv"
    model_name = "vichitrarora/qwensftfinetuned"
    
    # Load and clean dataset once
    df = load_dataset(eval_file_path)
    save_dataset(df, cleaned_eval_path)
    
    # Initialize generator
    try:
        generator = MongoQueryGenerator(model_name)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Evaluate queries
    try:
        accuracy, evaluated_df = evaluate_queries(df, generator)
        print(f"Accuracy: {accuracy:.2%}")
        
        # Save results
        save_dataset(evaluated_df, output_csv_path)
        print(f"Updated dataset saved to {output_csv_path}")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
