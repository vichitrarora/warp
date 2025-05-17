from pathlib import Path

# Base directory for saving/loading
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "train_data_v4_cleaned.csv"

MODEL_NAME = "unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
