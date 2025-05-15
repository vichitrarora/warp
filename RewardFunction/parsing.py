import json
import re

def extract_json_from_find(query: str):
    try:
        start = query.find("find(")
        if start == -1:
            raise ValueError("Error: Invalid MongoDB Query Format. Ensure the query is correctly formatted.")
        brace_start = query.find("{", start)
        if brace_start == -1:
            raise ValueError("Error: Could not find a valid JSON object in find().")
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
        raise ValueError(f"Error parsing MongoDB query: {e}")
    raise ValueError("Error: Mismatched braces in MongoDB query.")

def convert_mongo_shell_to_json(mongo_shell_query: str):
    try:
        return json.loads(mongo_shell_query)
    except json.JSONDecodeError:
        try:
            json_query = extract_json_from_find(mongo_shell_query)
            return json.loads(json_query)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error: {e}")
            return None

def convert_schema_shell_to_json(schema_shell: str):
    try:
        schema_json = schema_shell.replace("'", '"')
        return json.loads(schema_json)
    except json.JSONDecodeError:
        print("Error: Invalid Schema Format. Ensure it is correctly formatted.")
        return None
