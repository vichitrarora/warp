from transformers import AutoTokenizer, AutoModelForCausalLM

class MongoQueryGenerator:
    def __init__(self, model_name, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def generate(self, schema, natural_language_query):
        prompt = f"""### Instruction:
You are an AI assistant that generates MongoDB shell queries.
Given the schema, and natural language query, output a valid MongoDB shell query.

### Schema:
{schema}

### Natural Language Query:
{natural_language_query}

### MongoDB Shell Query:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=100)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return next((line.strip() for line in decoded.split("\n") if line.strip().startswith("db.")), "")
