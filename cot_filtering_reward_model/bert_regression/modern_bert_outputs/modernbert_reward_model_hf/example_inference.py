import os

import torch
from transformers import AutoTokenizer

from modeling_reward import load_finetuned_model


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    tokenizer = AutoTokenizer.from_pretrained(repo_root)
    model = load_finetuned_model(repo_root)

    sql = "SELECT COUNT(*) FROM orders WHERE status = 'complete';"
    reasoning = "think: Count rows in orders filtered by status 'complete'."
    nl = "How many completed orders exist?"
    text = f"SQL: {sql}\nReasoning: {reasoning}\nNL: {nl}"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        score = model(**inputs)["scores"].item()

    print(f"Reward score: {score:.3f}")


if __name__ == "__main__":
    main()

