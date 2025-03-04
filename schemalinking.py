import json
import re
from fuzzywuzzy import fuzz

def extract_json_from_find(query: str):
    try:
        start = query.find("find(")
        if start == -1:
            raise ValueError("⚠️ Error: Invalid MongoDB Query Format. Ensure the query is correctly formatted.")
        brace_start = query.find("{", start)
        if brace_start == -1:
            raise ValueError("⚠️ Error: Could not find a valid JSON object in find().")
        stack = []
        for i in range(brace_start, len(query)):
            if query[i] == "{":
                stack.append(i)
            elif query[i] == "}":
                stack.pop()
                if not stack:
                    json_text = query[brace_start:i+1]
                    json_text = json_text.replace("'", '"')  # Ensure valid JSON
                    return json.loads(json_text)
    except Exception as e:
        raise ValueError(f"⚠️ Error parsing MongoDB query: {e}")
    raise ValueError("⚠️ Error: Mismatched braces in MongoDB query.")

def convert_mongo_shell_to_json(mongo_shell_query: str):
    try:
        return json.loads(mongo_shell_query)
    except json.JSONDecodeError:
        try:
            json_query = extract_json_from_find(mongo_shell_query)
            return json.loads(json_query)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ Error: {e}")
            return None

def extract_fields_from_mongo(query: dict):
    fields = set()
    IGNORED_KEYS = {"collection", "filter", "database"}
    MONGO_OPERATORS = {"$gte", "$lte", "$eq", "$ne", "$in", "$exists", "$regex", "$or", "$and"}

    def extract_keys(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in IGNORED_KEYS or key in MONGO_OPERATORS:
                    extract_keys(value, prefix)
                else:
                    full_key = f"{prefix}.{key}" if prefix else key
                    fields.add(full_key)
                    extract_keys(value, full_key)
        elif isinstance(obj, list):
            for item in obj:
                extract_keys(item, prefix)
    extract_keys(query)
    return fields

def extract_collection_and_db_from_predicted(query: dict):
    return query.get("database"), query.get("collection"), query.get("schema")

def extract_fields_from_schema(schema: dict):
    fields = set()
    def extract_nested_fields(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                extract_nested_fields(value, new_prefix)
        elif isinstance(obj, list):
            for item in obj:
                extract_nested_fields(item, prefix)
    extract_nested_fields(schema)
    return fields

def is_schema_linking_correct(mongo_output: dict, schema: dict, threshold=80):
    query_fields = extract_fields_from_mongo(mongo_output)
    schema_fields = extract_fields_from_schema(schema)
    if not query_fields:
        return 0.0
    total_score = sum(max(fuzz.ratio(field.lower(), valid_field.split(".")[-1].lower()) for valid_field in schema_fields) / 100 for field in query_fields)
    schema_score = total_score / len(query_fields)
    return round(schema_score, 2)

# Accept user inputs
mongo_shell_query = input("Enter MongoDB Output Query (JSON format): ")
mongo_output = convert_mongo_shell_to_json(mongo_shell_query)
if not mongo_output:
    print("Error: Invalid MongoDB Output Query format. Exiting.")
    exit()

predicted_mongo_query = input("Enter Predicted MongoDB Query (JSON format): ")
predicted_query_json = convert_mongo_shell_to_json(predicted_mongo_query)
if not predicted_query_json:
    print("Error: Invalid Predicted MongoDB Query format. Exiting.")
    exit()

database, collection, schema = extract_collection_and_db_from_predicted(predicted_query_json)
if not schema:
    print("Error: Schema missing in Predicted MongoDB Query. Exiting.")
    exit()

schema_score = is_schema_linking_correct(mongo_output, schema)
print(f"\nSchema Linking Score: {schema_score:.2f}")
print("Extracted Query Fields:", extract_fields_from_mongo(mongo_output))
print("Extracted Schema Fields:", extract_fields_from_schema(schema))
