from parsing import convert_mongo_shell_to_json, convert_schema_shell_to_json
from extraction import extract_fields_from_mongo, extract_fields_from_schema
from scoring import is_schema_linking_correct

def main():
    nlp_query = input("Enter NLP Query: ")
    mongo_shell_query = input("Enter MongoDB Output Query: ")
    mongo_output = convert_mongo_shell_to_json(mongo_shell_query)
    if mongo_output is None:
        print("Error: Invalid MongoDB query format. Exiting.")
        return

    database = input("Enter Database Name: ")
    schema_shell = input("Enter Schema: ")
    schema = convert_schema_shell_to_json(schema_shell)
    if schema is None:
        print("Error: Invalid Schema format. Exiting.")
        return

    score = is_schema_linking_correct(
        nlp_query, mongo_output, mongo_shell_query, database, schema
    )
    print(f"\nSchema Linking Score: {score:.2f}")
    print("Extracted Query Fields:", extract_fields_from_mongo(mongo_output))
    print("Extracted Schema Fields:", extract_fields_from_schema(schema))

if __name__ == "__main__":
    main()
