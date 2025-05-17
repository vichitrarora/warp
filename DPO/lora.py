from peft import LoraConfig, get_peft_model
from configs.model_config import LoRAConfig

def apply_lora(model, config: LoRAConfig = None):
    if config is None:
        config = LoRAConfig()
    return get_peft_model(model, LoraConfig(**config.__dict__))