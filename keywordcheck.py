import json  # To convert dictionary into a set of string
     

def extract_keywords(query):

    keywords = set()
    query_str = json.dumps(query)

    for key in query.keys():
        keywords.add(key)
        if key.startswith("$"):
            keywords.add(key)

    for value in query.values():
        if isinstance(value, dict):  # Nested queries
            keywords.update(extract_keywords(value))
        elif isinstance(value, str):
            keywords.add(value)

    return keywords
     

def reward_function(input_query, expected_query):

    input_keywords = extract_keywords(input_query)
    expected_keywords = extract_keywords(expected_query)

    # Calculate match percentage
    common_keywords = input_keywords.intersection(expected_keywords)
    match_score = len(common_keywords) / max(len(expected_keywords), 1)  # Avoid division by zero

    # Reward scale: 0 to 10
    reward = round(match_score * 10, 2)  # Scale to 10

    return reward

     

# Testing
input_query = {"name": "John", "age": {"$gte": 25}}
expected_query = {"name": "John", "age": {"$gte": 25, "$lte": 40}}

reward = reward_function(input_query, expected_query)
print(f"Reward Score: {reward}/10")
