from dataclasses import dataclass

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