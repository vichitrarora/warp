from trl import DPOTrainer, DPOConfig
from config import DPOTrainConfig

def setup_trainer(model, ref_model, tokenizer, train_dataset, eval_dataset, config: DPOTrainConfig):
    dpo_args = DPOConfig(**vars(config))
    
    return DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )