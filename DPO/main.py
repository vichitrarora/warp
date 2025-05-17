from configs.dpo_config import DPOTrainConfig
from data.dataset import DataLoader
from data.formatter import format_dpo_example
from models.loader import load_model_and_tokenizer
from models.lora import apply_lora
from training.trainer import DPOTrainer

def main():
    # Config
    config = DPOTrainConfig()
    
    # Data
    train_dataset, eval_dataset = DataLoader.load_datasets(
        "/content/train_dpo_data_1600.csv",
        "/content/eval_dpo_data (1).csv"
    )
    train_dataset = train_dataset.map(format_dpo_example)
    eval_dataset = eval_dataset.map(format_dpo_example)
    
    # Models
    base_model, tokenizer = load_model_and_tokenizer("unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit")
    finetuned_model, _ = load_model_and_tokenizer("vichitrarora/qwen_finetuned_sft")
    finetuned_model = apply_lora(finetuned_model)
    
    # Training
    trainer = DPOTrainer(finetuned_model, base_model, tokenizer, config)
    trainer.train(train_dataset, eval_dataset)

if __name__ == "__main__":
    main()