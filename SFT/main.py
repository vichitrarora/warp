from model_loader import load_model
from data_loader import load_and_prepare_dataset
from train import train_model

def main():
    model, tokenizer = load_model()
    dataset = load_and_prepare_dataset()
    train_model(model, tokenizer, dataset)

if __name__ == "__main__":
    main()
