from tqdm import tqdm
from utils import normalize_query

def evaluate_queries(df, generator):
    expected_columns = {"schema", "mongo_query", "natural_language_query"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    df["predicted_mongo_query"] = ""
    df["score"] = 0
    correct_count = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating & Evaluating Queries"):
        schema = row["schema"].strip()
        nl_query = row["natural_language_query"].strip()
        correct_query = row["mongo_query"].strip()

        if not nl_query or not schema:
            continue

        predicted_query = generator.generate(schema, nl_query)
        df.at[i, "predicted_mongo_query"] = predicted_query

        if normalize_query(predicted_query) == normalize_query(correct_query):
            df.at[i, "score"] = 1
            correct_count += 1

    accuracy = correct_count / len(df) if len(df) > 0 else 0
    return accuracy, df
