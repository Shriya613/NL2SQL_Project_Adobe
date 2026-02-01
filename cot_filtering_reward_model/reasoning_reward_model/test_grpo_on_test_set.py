"""
Test script to evaluate trained GRPO model on test set.
This loads the trained Qwen model, generates predictions on test examples,
extracts scores, and compares them to ground truth.
"""
import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re

def load_grpo_model():
    """Load the trained GRPO model with LoRA weights"""
    model_path = "outputs/gpro_reward_function_multi_full"
    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    print("🔄 Loading GRPO model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, tokenizer

def generate_prediction(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    """Generate a completion for a given prompt"""
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the completion (remove the prompt)
    prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    completion = generated_text[len(prompt_text):].strip()
    
    return completion

def extract_score(completion):
    """Extract score from generated completion"""
    # Try multiple patterns to extract score
    patterns = [
        r'<score>\s*(.*?)\s*</score>',
        r'<think>.*?</think>\s*<score>\s*(.*?)\s*</score>',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, completion, re.DOTALL | re.IGNORECASE)
        if match:
            score_str = match.group(1).strip()
            try:
                score = float(score_str)
                # Validate range
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
    
    return None

def test_grpo_on_test_set():
    """Test GRPO model on test set examples"""
    print("🚀 Testing GRPO model on test set...")
    
    # Load model
    model, tokenizer = load_grpo_model()
    
    # Load dataset
    data_csv = "../../data/cot_dataset_with_corruptions.csv"
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"❌ Dataset not found at {data_csv}")
    
    df = pd.read_csv(data_csv)
    print(f"📂 Loaded {len(df)} examples from dataset")
    
    # Validate dataset (same as training)
    required_cols = ['sql', 'reasoning', 'predicted_nl', 'true_nl', 'similarity_with_penalty', 'prompt']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    df = df.dropna(subset=required_cols)
    df = df[df['prompt'].str.len() > 0]
    print(f"✅ Valid examples: {len(df)}")
    
    # Split into train/test (same as training - random_state=42, test_size=0.2)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"📊 Train examples: {len(df_train)}, Test examples: {len(df_test)}")
    print(f"📊 Using {len(df_test)} examples for testing")
    
    print("\n🔍 Testing GRPO predictions vs ground truth...")
    print("=" * 80)
    
    predictions = []
    ground_truth = []
    valid_predictions = []  # Track which predictions were successfully extracted
    invalid_count = 0
    
    for i, (_, row) in enumerate(df_test.iterrows(), 1):
        prompt = row['prompt']
        true_score = row['similarity_with_penalty']
        
        # Generate prediction
        completion = generate_prediction(model, tokenizer, prompt)
        
        # Extract score
        pred_score = extract_score(completion)
        
        ground_truth.append(true_score)
        
        if pred_score is not None:
            predictions.append(pred_score)
            valid_predictions.append(True)
        else:
            # If we can't extract a valid score, skip this example for metrics
            predictions.append(None)
            valid_predictions.append(False)
            invalid_count += 1
        
        # Print first 10 examples in detail
        if i <= 10:
            print(f"\n📝 Example {i}:")
            print(f"SQL: {row['sql'][:100]}...")
            print(f"Reasoning: {row['reasoning'][:100]}...")
            print(f"NL: {row['predicted_nl'][:100]}...")
            print(f"Ground Truth: {true_score:.3f}")
            if pred_score is not None:
                print(f"GRPO Prediction: {pred_score:.3f}")
                print(f"Difference: {abs(pred_score - true_score):.3f}")
            else:
                print(f"GRPO Prediction: ❌ Could not extract valid score")
                print(f"Completion: {completion[:200]}...")
            print("-" * 60)
        
        # Progress indicator
        if i % 50 == 0:
            print(f"Processed {i}/{len(df_test)} examples...")
    
    # Filter to valid predictions only
    valid_gt = [gt for gt, valid in zip(ground_truth, valid_predictions) if valid]
    valid_preds = [pred for pred, valid in zip(predictions, valid_predictions) if valid]
    
    print(f"\n📊 Prediction Statistics:")
    print(f"Total examples: {len(df_test)}")
    print(f"Valid predictions: {len(valid_preds)} ({len(valid_preds)/len(df_test)*100:.1f}%)")
    print(f"Invalid predictions: {invalid_count} ({invalid_count/len(df_test)*100:.1f}%)")
    
    if len(valid_preds) == 0:
        print("❌ No valid predictions found! Cannot calculate metrics.")
        return None, None
    
    # Calculate metrics
    mse = mean_squared_error(valid_gt, valid_preds)
    mae = mean_absolute_error(valid_gt, valid_preds)
    rmse = np.sqrt(mse)
    
    print(f"\n📊 Performance Metrics on {len(valid_preds)} Test Examples (with valid predictions):")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Show statistics
    print(f"\n📈 Statistics:")
    print(f"Ground Truth - Mean: {np.mean(valid_gt):.3f}, Std: {np.std(valid_gt):.3f}")
    print(f"Predictions - Mean: {np.mean(valid_preds):.3f}, Std: {np.std(valid_preds):.3f}")
    print(f"Ground Truth - Min: {np.min(valid_gt):.3f}, Max: {np.max(valid_gt):.3f}")
    print(f"Predictions - Min: {np.min(valid_preds):.3f}, Max: {np.max(valid_preds):.3f}")
    
    # Show examples with largest errors
    errors = [abs(p - t) for p, t in zip(valid_preds, valid_gt)]
    worst_indices = np.argsort(errors)[-5:]  # Top 5 worst predictions
    
    print(f"\n❌ Worst Predictions (Top 5):")
    valid_indices = [i for i, valid in enumerate(valid_predictions) if valid]
    for rank, idx in enumerate(worst_indices, 1):
        original_idx = valid_indices[idx]
        print(f"{rank}. Example {original_idx+1}: GT={valid_gt[idx]:.3f}, Pred={valid_preds[idx]:.3f}, Error={errors[idx]:.3f}")
    
    # Show examples with best predictions
    best_indices = np.argsort(errors)[:5]  # Top 5 best predictions
    
    print(f"\n✅ Best Predictions (Top 5):")
    for rank, idx in enumerate(best_indices, 1):
        original_idx = valid_indices[idx]
        print(f"{rank}. Example {original_idx+1}: GT={valid_gt[idx]:.3f}, Pred={valid_preds[idx]:.3f}, Error={errors[idx]:.3f}")
    
    # Create scatter plot
    try:
        plt.figure(figsize=(10, 8))
        plt.scatter(valid_gt, valid_preds, alpha=0.4, s=50)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction', linewidth=2)
        plt.xlabel('Ground Truth Similarity (similarity_with_penalty)', fontsize=12)
        plt.ylabel('GRPO Predicted Score', fontsize=12)
        plt.title(f'GRPO Predictions vs Ground Truth (Test Set)\n(N={len(valid_preds)} valid predictions, {invalid_count} invalid)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add metrics text box
        textstr = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plot_path = 'grpo_predictions_scatter_test_set.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Scatter plot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not create plot: {e}")
    
    return valid_preds, valid_gt

if __name__ == "__main__":
    predictions, ground_truth = test_grpo_on_test_set()

