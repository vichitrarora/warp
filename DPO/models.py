from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from config import ModelConfig

def initialize_models(config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.fine_tuned_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        load_in_4bit=True,
        device_map="auto"
    )

    fine_tuned_model = AutoModelForCausalLM.from_pretrained(
        config.fine_tuned_model_path,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        task_type="CAUSAL_LM",
        bias="none"
    )
    
    fine_tuned_model = get_peft_model(fine_tuned_model, lora_config)
    
    return base_model, fine_tuned_model, tokenizer