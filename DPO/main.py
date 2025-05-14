from config import ModelConfig, DataConfig, DPOTrainConfig
from models import initialize_models
from data_utils import load_and_preprocess_data, format_dpo_dataset
from train import setup_trainer
from inference import QueryGenerator
from datasets import Dataset

def main():
    # Initialize configuration
    model_config = ModelConfig()
    data_config = DataConfig()
    train_config = DPOTrainConfig()
    
    # Model setup
    base_model, fine_tuned_model, tokenizer = initialize_models(model_config)
    
    # Data preparation
    train_dataset, eval_dataset = load_and_preprocess_data(data_config)
    train_dataset = train_dataset.map(format_dpo_dataset)
    eval_dataset = eval_dataset.map(format_dpo_dataset)
    
    # Training
    trainer = setup_trainer(
        fine_tuned_model, base_model, tokenizer,
        train_dataset, eval_dataset, train_config
    )
    trainer.train()
    
    # Save models
    fine_tuned_model.save_pretrained("/content/lora-adapters")
    tokenizer.save_pretrained("/content/lora-adapters")
    
    # Merge and save final model
    merged_model = fine_tuned_model.merge_and_unload()
    merged_model.save_pretrained("/content/merged-model")
    tokenizer.save_pretrained("/content/merged-model")
    
    # Inference example
    generator = QueryGenerator("/content/merged-model")
    print(generator.generate_response("Schema: user(id,name,email)\nQuery: Find users..."))

if __name__ == "__main__":
    main()