def extract_fields_from_mongo(query: dict):
    fields = set()
    IGNORED_KEYS = {"collection", "filter"}
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

def extract_fields_from_schema(schema: dict):
    fields = set()
    is_document_format = any(
        isinstance(val, dict) and "document" in val for val in schema.values()
    )

    if is_document_format:
        # JSON Schema–style
        for coll in schema.values():
            doc = coll.get("document", {})
            props = doc.get("properties", {})
            def extract_leaf_fields(obj):
                leaves = set()
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, dict) and "properties" in v:
                            leaves |= extract_leaf_fields(v["properties"])
                        else:
                            leaves.add(k)
                return leaves
            fields |= extract_leaf_fields(props)

    else:
        # custom nested‐fields style
        def extract_nested_fields(obj, prefix=""):
            if isinstance(obj, dict):
                if "fields" in obj and isinstance(obj["fields"], list):
                    for f in obj["fields"]:
                        full = f"{prefix}.{f}" if prefix else f
                        fields.add(full)
                else:
                    for k, v in obj.items():
                        new_pref = f"{prefix}.{k}" if prefix else k
                        extract_nested_fields(v, new_pref)
            elif isinstance(obj, list):
                for item in obj:
                    extract_nested_fields(item, prefix)

        extract_nested_fields(schema)

    return fields
