from fuzzywuzzy import fuzz
from extraction import extract_fields_from_mongo, extract_fields_from_schema
from utils import extract_collection_from_mongo_query, extract_valid_collections_from_schema

def is_schema_linking_correct(
    nlp_query: str,
    mongo_output: dict,
    mongo_query: str,
    database: str,
    schema: dict,
    threshold: int = 80
) -> float:
    query_fields = extract_fields_from_mongo(mongo_output)
    schema_fields = extract_fields_from_schema(schema)

    if not query_fields:
        return 0.0

    total_score = 0.0
    for field in query_fields:
        simplified_query = field.lower()
        full_matches = []
        best_partial = 0
        for valid in schema_fields:
            simplified_valid = valid.split(".")[-1].lower()
            score = fuzz.ratio(simplified_query, simplified_valid)
            if score >= threshold:
                full_matches.append(valid)
            if score > best_partial:
                best_partial = score

        if full_matches:
            total_score += 1.0 / len(full_matches)
        elif best_partial >= 50:
            total_score += best_partial / 100.0

    schema_score = total_score / len(query_fields)
    pred_coll = extract_collection_from_mongo_query(mongo_query)
    if pred_coll not in extract_valid_collections_from_schema(schema):
        schema_score *= 0.5

    return round(schema_score, 2)
