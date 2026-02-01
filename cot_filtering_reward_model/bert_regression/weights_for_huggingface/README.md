---
language:
- en
tags:
- regression
- similarity
- sql
- natural-language
- reward-model
license: mit
datasets:
- custom
metrics:
- mse
- mae
- rmse
model-index:
- name: BERT Reward Model for CoT Filtering
  results:
  - task:
      type: regression
      name: Similarity Score Prediction
    dataset:
      name: Custom CoT Dataset
      type: custom
    metrics:
    - type: mse
      value: 0.0238
    - type: mae
      value: 0.1229
    - type: rmse
      value: 0.1543
---

# BERT Reward Model for CoT Filtering

A BERT-based regression model fine-tuned to predict similarity scores between SQL queries, reasoning chains (Chain-of-Thought), and natural language descriptions.

## Model Description

This model is based on `bert-base-uncased` and has been fine-tuned for regression to predict similarity scores in the range [0, 1]. The model takes as input a concatenation of:
- SQL query
- Reasoning/Chain-of-Thought explanation
- Predicted natural language description

And outputs a similarity score indicating how well the predicted NL matches the ground truth.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_path = "path/to/weights_for_huggingface"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,
    problem_type="regression"
)
model.eval()

# Prepare input
sql = "SELECT movie_title FROM movies WHERE movie_release_year = 1945"
reasoning = "think: The SQL selects the movie title..."
predicted_nl = "What was the most popular movie released in 1945?"

input_text = f"SQL: {sql}\nReasoning: {reasoning}\nNL: {predicted_nl}"

# Tokenize and predict
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)
    # Apply sigmoid to get probability
    similarity_score = torch.sigmoid(outputs.logits).item()

print(f"Predicted similarity: {similarity_score:.3f}")
```

## Training Details

- **Base Model**: bert-base-uncased
- **Training Dataset**: Custom CoT dataset with corruptions (7,342 examples)
- **Train/Val/Test Split**: 75% / 12.5% / 12.5%
- **Training Loss**: MSE (Mean Squared Error)
- **Evaluation Metrics**:
  - MSE: 0.0238
  - MAE: 0.1229
  - RMSE: 0.1543

## Limitations

- Maximum input length: 512 tokens (BERT's limit)
- Trained on a specific domain (SQL to NL translation with CoT)
- Performance may vary on out-of-domain data

## Citation

If you use this model, please cite:

```bibtex
@misc{bert_cot_reward_model,
  title={BERT Reward Model for Chain-of-Thought Filtering},
  author={Your Name},
  year={2024},
}
```

