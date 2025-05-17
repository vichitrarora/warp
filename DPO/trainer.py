from trl import DPOTrainer
from configs.dpo_config import DPOTrainConfig

class DPOTrainer:
    def __init__(self, model, ref_model, tokenizer, config: DPOTrainConfig):
        self.trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            **config.__dict__
        )
    
    def train(self, train_dataset, eval_dataset):
        return self.trainer.train()