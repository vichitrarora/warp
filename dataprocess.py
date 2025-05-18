# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MongoDB DPO dataset to parquet format.
This script reads the training and evaluation CSV files that contain the following columns:
  - schema: The JSON-like database schema.
  - natural_language_query: The instruction/query for the model.
  - corret_mongo_query: The correct MongoDB shell query.
It then creates a dataset where each example follows the format:
    data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'answer': answer_raw,
            "question": question_raw,
        }
    }
Here, for MongoDB:
  - The "question" is the natural language query combined with the schema and an instruction.
  - The "answer" (and ground_truth) is the correct MongoDB query.
The processed parquet files are saved in /home/warp/metafusion.
"""

import os
import argparse
import datasets
from verl.utils.hdfs_io import copy, makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Set default local_dir to /home/warp/metafusion
    parser.add_argument('--local_dir', default='/home/warp/metafusion', help='Local directory to save the processed dataset')
    parser.add_argument('--hdfs_dir', default=None, help='Optional HDFS directory to copy the processed dataset')
    # Specify the dataset paths as default values
    parser.add_argument('--train_file', type=str, default='/home/warp/metafusion/train_dpo_data_1600.csv', help='Path to the training CSV file')
    parser.add_argument('--eval_file', type=str, default='/home/warp/metafusion/eval_dpo_data (1).csv', help='Path to the evaluation CSV file')
    args = parser.parse_args()

    data_source = 'local_mongodb_dpo_data'

    # Load the CSV files using the Hugging Face datasets library.
    train_dataset = datasets.load_dataset('csv', data_files=args.train_file)['train']
    eval_dataset = datasets.load_dataset('csv', data_files=args.eval_file)['train']

    # Mapping function to convert each example into the desired format.
    def make_map_fn(split):
        def process_fn(example, idx):
            # Extract fields using the exact column names.
            nl_query = example.get('natural_language_query', '').strip()    # Will be used as question
            schema_text = example.get('schema', '').strip()
            correct_query = example.get('corret_mongo_query', '').strip()     # Will be used as answer/solution

            # Compose the prompt with the additional instruction.
            question = (
                "Instruction: Convert this natural language query to MongoDB shell query.\n"
                f"Query: {nl_query}\n"
                f"Schema: {schema_text}"
            )

            # Maintain the format you provided.
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",  # Keeping "math" as in your original example; change if desired.
                "reward_model": {
                    "style": "rule",
                    "ground_truth": correct_query
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": correct_query,
                    "question": nl_query,
                }
            }
            return data
        return process_fn

    # Remove original columns by passing remove_columns parameter.
    orig_train_columns = train_dataset.column_names
    orig_eval_columns = eval_dataset.column_names

    train_dataset = train_dataset.map(
        function=make_map_fn('train'),
        with_indices=True,
        remove_columns=orig_train_columns
    )
    eval_dataset = eval_dataset.map(
        function=make_map_fn('eval'),
        with_indices=True,
        remove_columns=orig_eval_columns
    )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Save the processed datasets as parquet files in /home/warp/metafusion.
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    eval_dataset.to_parquet(os.path.join(local_dir, 'eval.parquet'))

    # If an HDFS directory is provided, copy the data there.
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
