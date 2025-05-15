import json
import re

def extract_collection_from_mongo_query(mongo_query: str):
    try:
        d = json.loads(mongo_query)
        if "collection" in d:
            return d["collection"]
    except json.JSONDecodeError:
        pass
    match = re.search(r'db\.(\w+)\.find(?:One)?\s*\(', mongo_query)
    return match.group(1) if match else None

def extract_valid_collections_from_schema(schema: dict):
    if not isinstance(schema, dict):
        print("Error: Schema is not a dictionary.")
        return set()
    return set(schema.keys())
