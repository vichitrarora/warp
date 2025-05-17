from dataclasses import dataclass

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: list = ["q_proj", "v_proj"]
    task_type: str = "CAUSAL_LM"
    bias: str = "none"