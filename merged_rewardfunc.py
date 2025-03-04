import json
import re
import urllib.parse
from pymongo import MongoClient
from fuzzywuzzy import fuzz

def extract_keywords(query):
    keywords = set()
    def recursive_extract(q):
        for key, value in q.items():
            keywords.add(key)
            if isinstance(value, dict):
                recursive_extract(value)
            elif isinstance(value, str):
                keywords.add(value)
    recursive_extract(query)
    return keywords

def keyword_match_score(input_query, expected_query):
    input_keywords = extract_keywords(input_query)
    expected_keywords = extract_keywords(expected_query)
    common_keywords = input_keywords.intersection(expected_keywords)
    match_score = len(common_keywords) / max(len(expected_keywords), 1)
    structural_similarity = fuzz.ratio(json.dumps(input_query, sort_keys=True), json.dumps(expected_query, sort_keys=True)) / 100
    adjusted_score = (match_score + structural_similarity) / 2
    return round(adjusted_score, 2)

def extract_json_from_find(query: str):
    try:
        start = query.find("find(")
        if start == -1:
            raise ValueError("Invalid MongoDB Query Format.")
        brace_start = query.find("{", start)
        if brace_start == -1:
            raise ValueError("Could not find a valid JSON object in find().")
        stack = []
        for i in range(brace_start, len(query)):
            if query[i] == "{":
                stack.append(i)
            elif query[i] == "}":
                stack.pop()
                if not stack:
                    json_text = query[brace_start:i+1]
                    json_text = json_text.replace("'", '"')
                    return json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Error parsing MongoDB query: {e}")
    raise ValueError("Mismatched braces in MongoDB query.")

def extract_fields_from_mongo(query: dict):
    fields = set()
    def extract_keys(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                fields.add(full_key)
                extract_keys(value, full_key)
        elif isinstance(obj, list):
            for item in obj:
                extract_keys(item, prefix)
    extract_keys(query)
    return fields

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

def is_schema_linking_correct(mongo_output: dict, schema: dict):
    query_fields = extract_fields_from_mongo(mongo_output)
    schema_fields = extract_fields_from_schema(schema)
    if not query_fields:
        return 0.0
    total_score = sum(max(fuzz.ratio(field.lower(), valid_field.split(".")[-1].lower()) for valid_field in schema_fields) / 100 for field in query_fields)
    return round(total_score / len(query_fields), 2)

def check_query_executable(shell_query: str, uri: str) -> int:
    try:
        client = MongoClient(uri)
        db = client["test"]
        match = re.match(r"db\.(\w+)\.find\((.*)\)", shell_query.strip())
        if not match:
            return 0  
        collection_name = match.group(1)
        query_dict = eval(match.group(2))  
        db[collection_name].count_documents(query_dict)
        return 1  
    except Exception:
        return 0  

# Accept user inputs
user = input("Enter the username: ")
passw = input("Enter the password: ")
username = urllib.parse.quote_plus(user)
password = urllib.parse.quote_plus(passw)
uri = f"mongodb+srv://{username}:{password}@cluster0.7j18x.mongodb.net/?retryWrites=true&w=majority"

mongo_shell_query = input("Enter MongoDB Output Query (JSON format): ")
mongo_output = extract_json_from_find(mongo_shell_query)
if not mongo_output:
    print("Error: Invalid MongoDB Output Query format. Exiting.")
    exit()

predicted_mongo_query = input("Enter Predicted MongoDB Query (JSON format): ")
predicted_query_json = extract_json_from_find(predicted_mongo_query)
if not predicted_query_json:
    print("Error: Invalid Predicted MongoDB Query format. Exiting.")
    exit()

database = predicted_query_json.get("database")
collection = predicted_query_json.get("collection")
schema = predicted_query_json.get("schema")
if not schema:
    print("Error: Schema missing in Predicted MongoDB Query. Exiting.")
    exit()

executable = check_query_executable(predicted_mongo_query, uri)
if executable:
    keyword_score = keyword_match_score(mongo_output, predicted_query_json)
    schema_score = is_schema_linking_correct(mongo_output, schema)
    final_score = round((keyword_score + schema_score) / 2, 2)
else:
    final_score = 0.0

print(f"\nFinal Reward Score: {final_score}")
