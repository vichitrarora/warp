def format_dpo_example(example: dict) -> dict:
    return {
        "prompt": f"Schema: {example['schema_info']}\nQuery: {example['nl_query']}",
        "chosen": example["preferred_query"],
        "rejected": example["rejected_query"],
    }