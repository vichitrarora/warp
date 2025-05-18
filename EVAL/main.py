from utils import load_dataset, save_dataset
from generator import MongoQueryGenerator
from evaluator import evaluate_queries

def main():
    eval_file = "/home/warp/metafusion/WARPxMetafusion/eval/model_output/eval_data_v1.csv"
    cleaned_file = "/home/warp/metafusion/WARPxMetafusion/eval/model_output/eval_data_v1_cleaned.csv"
    output_file = "/home/warp/metafusion/WARPxMetafusion/eval/model_output/eval_data_v4_resultsdpo.csv"
    model_name = "vichitrarora/qwensftfinetuned"

    df = load_dataset(eval_file)
    save_dataset(df, cleaned_file)

    try:
        generator = MongoQueryGenerator(model_name)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    try:
        accuracy, results = evaluate_queries(df, generator)
        print(f"Accuracy: {accuracy:.2%}")
        save_dataset(results, output_file)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
