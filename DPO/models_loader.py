from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model_and_tokenizer(model_name: str, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ),
        device_map="auto",
        **kwargs
    )
    return model, tokenizer