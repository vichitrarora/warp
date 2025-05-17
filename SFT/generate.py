def generate_mongo_query(model, tokenizer, nl_query, db_id, schema_info):
    model.eval()
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Generate a MongoDB query for the given database and schema.

### Input:
DB: {db_id}
Schema: {schema_info}
NL Query: {nl_query}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
