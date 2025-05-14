from dataclasses import dataclass

@dataclass
class ModelConfig:
    base_model_name: str = "unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit"
    fine_tuned_model_path: str = "vichitrarora/qwen_finetuned_sft"
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: list = ["q_proj", "v_proj"]

@dataclass
class DataConfig:
    train_path: str = "/content/train_dpo_data_1600.csv"
    eval_path: str = "/content/eval_dpo_data (1).csv"
    column_mapping: dict = {
        "schema": "schema_info",
        "natural_language_query": "nl_query",
        "corret_mongo_query": "preferred_query",
        "incorrect_mongo_query": "rejected_query"
    }

@dataclass
class DPOTrainConfig:
    output_dir: str = "./dpo-output"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    num_train_epochs: int = 3
    logging_steps: int = 10
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    fp16: bool = True