# ModernBERT Reward Model (CoT SQL/NL Alignment)

Finetuned `answerdotai/ModernBERT-base` to score how well a generated natural-language description (NL) and chain-of-thought reasoning align with a SQL query. The model is trained as a regression head (sigmoid output in `[0, 1]`) to predict `similarity_with_penalty` scores derived from human preference data plus corruption heuristics.

## Repo Layout (ready for Hugging Face)

```
modernbert_reward_model_hf/
├── config.json
├── model.safetensors
├── tokenizer.json / tokenizer_config.json / special_tokens_map.json
├── training_args.bin
├── training_results.json
├── test_metrics.json
├── train_set.csv / eval_set.csv / test_set.csv
├── modernbert_reward_model_predictions_scatter.png
├── modeling_reward.py
└── README.md  ← (this file)
```

Upload the folder to a new Hugging Face repo (e.g. `daeilee/modernbert-cot-reward`) and run `huggingface-cli upload` or push via git. No extra conversion steps are required.

## Training Summary

- **Base model:** `answerdotai/ModernBERT-base`
- **Context length:** truncated to 2,048 tokens (model supports 8,192)
- **Dataset:** 7,633 SQL + chain-of-thought + NL examples from `data/cot_dataset_with_corruptions.csv`
  - Invalid rows dropped (missing fields, score outside `[0, 1]`)
  - Token-length filter at 2,048
  - Split: 75% train / 12.5% eval / 12.5% test (random state 12)
- **Training setup:** batch size 2, gradient accumulation 4, AdamW (lr `2e-5`), early stopping patience 5, ModernBERT encoder + mean-pooling + linear + sigmoid, MSE loss scaled by 100

See `training_args.bin` + `training_results.json` for the exact `transformers.TrainingArguments` and eval metrics (`eval_mse = 0.0197`, `eval_mae = 0.1128`, `eval_rmse = 0.1402`).

## Test Metrics (955 examples)

- **MSE:** 0.0203
- **MAE:** 0.1129
- **RMSE:** 0.1425
- **Ground-truth mean/std:** 0.673 / 0.215
- **Prediction mean/std:** 0.691 / 0.170

Full details live in `test_metrics.json`, and the scatter plot is stored as `modernbert_reward_model_predictions_scatter.png`.

## Usage

```python
import torch
from transformers import AutoTokenizer
from modeling_reward import BERTRewardModel

repo_dir = "daeilee/modernbert-cot-reward"  # once uploaded
tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
model = BERTRewardModel(model_name="answerdotai/ModernBERT-base")
state_dict = torch.load("model.safetensors")  # or use safetensors.torch.load_file
model.load_state_dict(state_dict)
model.eval()

sql = "SELECT COUNT(*) FROM orders WHERE status = 'complete';"
reasoning = "think: Count rows in orders filtered by status 'complete'."
nl = "How many completed orders exist?"
text = f"SQL: {sql}\nReasoning: {reasoning}\nNL: {nl}"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
score = model(**inputs)["scores"].item()
print(f"Reward: {score:.3f}")
```

For convenience, `modeling_reward.py` exposes `load_finetuned_model(model_dir)` which handles loading `model.safetensors` or `pytorch_model.bin` and moves the module to GPU if available (falling back to CPU on OOM).

## Reproducing Evaluation

1. `pip install -r requirements.txt` (same stack as training: `transformers`, `torch`, `safetensors`, `pandas`, `scikit-learn`, `matplotlib`).
2. Run `python test_mosaic_bert_on_test_set.py` from `cot_filtering_reward_model/bert_regression/` to regenerate metrics + scatter plot (uses the saved `test_set.csv` by default).

## Notes

- The reward target is bounded `[0, 1]` and already penalizes copied NL or incorrect reasoning.
- The model uses mean pooling instead of CLS to better leverage long ModernBERT contexts.
- Tokenizer files are saved from the finetuned run; no extra special tokens were introduced.
- If you upload to the Hub and want `AutoModel.from_pretrained(..., trust_remote_code=True)` support, keep `modeling_reward.py` in the repo root.

