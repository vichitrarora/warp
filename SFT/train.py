from transformers import TrainingArguments
from trl import SFTTrainer
from config import MODEL_DIR

def preprocess_sft(example, tokenizer, eos_token):
    input_text = f"Generate a MongoDB query for the given NL query and schema: NL={example['nl_query']} Schema={example['schema_info']}"
    return {
        "text": input_text + eos_token + example["mongo_query"]
    }

def train_model(model, tokenizer, dataset):
    eos_token = tokenizer.eos_token
    dataset = dataset.map(lambda x: preprocess_sft(x, tokenizer, eos_token))

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        evaluation_strategy="steps",
        eval_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_dir=str(MODEL_DIR / "logs"),
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        evaluation_strategy="no",
        do_eval=False
    )

    trainer.train()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")
