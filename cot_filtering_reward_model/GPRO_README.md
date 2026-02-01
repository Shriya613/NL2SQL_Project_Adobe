# GPRO (Generalized Preference Optimization) for CoT Filtering

This implementation uses TRL's GPRO (Generalized Preference Optimization) to train a reward model for filtering.


### 1. **Data Flow**
```
SQL + CoT + NL → Prompt → Model → Multiple Generations → Reward Function → GPRO Update
```

### 2. **Reward Function**
The `reward_func` in `reward_functions.py`:
- Expects model output in format: `<think>...</think><score>X.XX</score>`
- Compares predicted score to `sbert_similarity` (ground truth)
- Returns reward in range [-1, 1] based on accuracy
- Perfect match = +1.0, complete mismatch = -1.0

### 3. **Training Process**
1. **Generate**: Model generates multiple completions per prompt
2. **Score**: Reward function evaluates each completion
3. **Optimize**: GPRO updates model to prefer higher-rewarded completions
4. **Repeat**: Process continues for multiple epochs

## Key Components

### `CoTDataset`
- Loads SQL, CoT, NL, and similarity scores
- Creates evaluation prompts using `VAL_PROMPT_TEMPLATE`
- Tokenizes prompts for model input

### `reward_func`
- Parses model output for score prediction
- Validates format and score range [0, 1]
- Completes content reward based on MSE, which via some complex exponent equation is always between -1 and 1

### `GRPOTrainer`
- Manages the GPRO training loop
- Handles multiple generations per prompt
- Applies reward-based policy updates

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
# Create the dataset if not already done
python3 data_preprocessing.py
python3 cot_dataset_creation.py
```

### 3. Run GPRO Training
```bash
python3 training_reward_function.py
```

## Configuration

### Key Parameters in `GRPOConfig`:
- `num_generations=4`: Number of completions per prompt
- `max_completion_length=256`: Max tokens for model output
- `per_device_train_batch_size=1`: Batch size for training
- `gradient_accumulation_steps=4`: Effective batch size
- `learning_rate=2e-5`: Learning rate for optimization

### LoRA Configuration:
- `r=16`: LoRA rank
- `lora_alpha=32`: LoRA alpha (2x rank)
- `target_modules`: All attention and MLP layers

## Expected Behavior

### Training:
- Model learns to generate scores closer to `sbert_similarity`
- Reward increases as predictions become more accurate
- Multiple generations allow exploration of different scoring approaches

### Output Format:
The model should learn to output:
```
<think>
Step 1: The SQL query selects count of records from a table.
Step 2: The reasoning correctly identifies this as a counting operation.
Step 3: The NL question asks for total number of records.
Step 4: The reasoning and NL are well-aligned.
Step 5: This is a high-quality reasoning chain.
</think>
<score>
0.85
</score>
