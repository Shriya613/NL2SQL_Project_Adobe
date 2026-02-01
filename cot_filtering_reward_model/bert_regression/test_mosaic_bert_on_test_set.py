import os
import pandas as pd
import torch
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import the custom model class if needed
sys.path.append(os.path.dirname(__file__))
from mosaic_bert_training import BERTRewardModel


MODEL_NAME = 'answerdotai/ModernBERT-base'


def _resolve_model_output_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    
    candidate_paths = [
        os.path.join(script_dir, 'modern_bert_outputs', 'modernbert_reward_model'),
        os.path.join(project_root, 'modern_bert_outputs', 'modernbert_reward_model'),
        os.path.join(project_root, 'outputs', 'modernbert_reward_model'),
    ]
    
    for path in candidate_paths:
        if os.path.isdir(path):
            return path
    
    raise FileNotFoundError(
        "Could not locate ModernBERT model artifacts. Checked:\n" +
        "\n".join(candidate_paths)
    )


def load_mosaic_bert_model():
    """Load the trained ModernBERT reward model"""
    model_path = _resolve_model_output_dir()
    
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    print("🔄 Loading ModernBERT reward model...")
    
    model = BERTRewardModel(model_name=MODEL_NAME)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = model.to(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        message = str(e).lower()
        if isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in message:
            print("   ⚠️  GPU out of memory, falling back to CPU for evaluation")
            device = torch.device('cpu')
            model = model.to(device)
        else:
            raise
    
    state_dict_paths = [
        os.path.join(model_path, 'model.safetensors'),
        os.path.join(model_path, 'pytorch_model.bin'),
    ]
    loaded = False
    for state_dict_path in state_dict_paths:
        if not os.path.exists(state_dict_path):
            continue
        if state_dict_path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(state_dict_path)
            except ImportError as e:
                print(f"   ⚠️  safetensors unavailable ({e}), skipping")
                continue
        else:
            state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
        loaded = True
        print(f"   ✓ Loaded state dict from {state_dict_path}")
        break
    
    if not loaded:
        raise FileNotFoundError("Could not find model state dict (.safetensors or .bin)")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("   ✓ Loaded tokenizer from model path")
    except Exception as e:
        print(f"   ⚠️  Could not load tokenizer from model path: {e}")
        print("   🔄 Falling back to pretrained tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"   ✓ Loaded tokenizer from {MODEL_NAME}")
    
    model.eval()
    
    return model, tokenizer, True, device


def predict_similarity(model, tokenizer, sql, reasoning, predicted_nl, max_length=2048, use_custom_model=False, device=torch.device('cpu')):
    """Predict similarity score for a given example"""
    # Create input text
    input_text = f"SQL: {sql}\nReasoning: {reasoning}\nNL: {predicted_nl}"
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Handle different output formats
        if use_custom_model:
            # Custom BERTRewardModel returns dict with 'scores' (already has sigmoid applied)
            if isinstance(outputs, dict) and 'scores' in outputs:
                prediction = outputs['scores'].item()
            elif isinstance(outputs, dict) and 'logits' in outputs:
                # If logits are returned, apply sigmoid
                prediction = torch.sigmoid(outputs['logits']).item()
            else:
                # Fallback
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                prediction = torch.sigmoid(logits).item()
        else:
            # Standard AutoModelForSequenceClassification returns logits
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            # Apply sigmoid to get probability
            prediction = torch.sigmoid(logits).item()
    
    return prediction


def test_mosaic_bert_on_test_set():
    """Test Mosaic BERT model on test set examples"""
    print("🚀 Testing ModernBERT reward model on test set...")
    
    # Load model
    model, tokenizer, use_custom_model, device = load_mosaic_bert_model()
    
    # Try to load saved test set first (for consistency)
    test_file = os.path.join(_resolve_model_output_dir(), 'test_set.csv')
    
    if os.path.exists(test_file):
        print(f"📂 Loading saved test set from training...")
        test_df = pd.read_csv(test_file)
        print(f"   Loaded {len(test_df)} examples from saved test set")
    else:
        # Fall back to recalculating split if saved set doesn't exist
        print("⚠️  Saved test set not found, recalculating split...")
        # Resolve path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../../'))
        data_csv = os.path.join(project_root, 'data', 'cot_dataset_with_corruptions.csv')
        df = pd.read_csv(data_csv)
        
        # Filter to valid examples
        required_cols = ['sql', 'reasoning', 'predicted_nl', 'similarity_with_penalty']
        df = df.dropna(subset=required_cols).copy()
        df = df[df['similarity_with_penalty'].between(0, 1)].copy()
        
        # Filter by token length (max 2048 for ModernBERT)
        temp_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("🔍 Filtering examples by token length (max 2048)...")
        valid_indices = []
        for idx, row in df.iterrows():
            input_text = f"SQL: {row['sql']}\nReasoning: {row['reasoning']}\nNL: {row['predicted_nl']}"
            tokens = temp_tokenizer.encode(input_text, add_special_tokens=True)
            if len(tokens) <= 2048:
                valid_indices.append(idx)
        
        df = df.loc[valid_indices].copy()
        print(f"   {len(df)} examples after token length filtering")
        
        # Split into train/val/test (same as training - 75/12.5/12.5 split)
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(df, test_size=0.25, random_state=12, shuffle=True)
        eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=12, shuffle=True)
        print(f"📊 Test set size: {len(test_df)} examples (recalculated)")
    
    # Use all test examples
    test_sample = test_df
    print(f"📊 Using {len(test_sample)} examples for testing")
    
    print("\n🔍 Testing Mosaic BERT predictions vs ground truth...")
    print("=" * 80)
    
    predictions = []
    ground_truth = []
    
    for i, (_, row) in enumerate(test_sample.iterrows(), 1):
        # Get prediction
        pred_score = predict_similarity(
            model, tokenizer, 
            row['sql'], 
            row['reasoning'], 
            row['predicted_nl'],
            max_length=2048,
            use_custom_model=use_custom_model,
            device=device
        )
        
        true_score = row['similarity_with_penalty']
        
        predictions.append(pred_score)
        ground_truth.append(true_score)
        
        # Print first 10 examples in detail
        if i <= 10:
            print(f"\n📝 Example {i}:")
            print(f"SQL: {row['sql'][:100]}...")
            print(f"Reasoning: {row['reasoning'][:100]}...")
            print(f"NL: {row['predicted_nl'][:100]}...")
            print(f"Ground Truth: {true_score:.3f}")
            print(f"Mosaic BERT Prediction: {pred_score:.3f}")
            print(f"Difference: {abs(pred_score - true_score):.3f}")
            print("-" * 60)
    
    # Calculate metrics
    mse = mean_squared_error(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\n📊 Performance Metrics on Test Set (N={len(predictions)} examples):")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Show statistics
    print(f"\n📈 Statistics:")
    print(f"Ground Truth - Mean: {np.mean(ground_truth):.3f}, Std: {np.std(ground_truth):.3f}")
    print(f"Predictions - Mean: {np.mean(predictions):.3f}, Std: {np.std(predictions):.3f}")
    print(f"Ground Truth - Min: {np.min(ground_truth):.3f}, Max: {np.max(ground_truth):.3f}")
    print(f"Predictions - Min: {np.min(predictions):.3f}, Max: {np.max(predictions):.3f}")
    
    # Show examples with largest errors
    errors = [abs(p - t) for p, t in zip(predictions, ground_truth)]
    worst_indices = np.argsort(errors)[-5:]  # Top 5 worst predictions
    
    print(f"\n❌ Worst Predictions (Top 5):")
    for i, idx in enumerate(worst_indices, 1):
        print(f"{i}. Example {idx+1}: GT={ground_truth[idx]:.3f}, Pred={predictions[idx]:.3f}, Error={errors[idx]:.3f}")
    
    # Show examples with best predictions
    best_indices = np.argsort(errors)[:5]  # Top 5 best predictions
    
    print(f"\n✅ Best Predictions (Top 5):")
    for i, idx in enumerate(best_indices, 1):
        print(f"{i}. Example {idx+1}: GT={ground_truth[idx]:.3f}, Pred={predictions[idx]:.3f}, Error={errors[idx]:.3f}")
    
    # Create a simple scatter plot
    try:
        plt.figure(figsize=(10, 8))
        plt.scatter(ground_truth, predictions, alpha=0.4)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
        plt.xlabel('Ground Truth Similarity (similarity_with_penalty)')
        plt.ylabel('Mosaic BERT Prediction')
        plt.title(f'Mosaic BERT Predictions vs Ground Truth (Test Set)\n(N={len(predictions)} examples, max_length=2048)')
        plt.legend(fontsize=18)
        plt.grid(True, alpha=0.3)
        
        # Add MSE and MAE as text on the plot
        metrics_text = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}'
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        output_path = os.path.join(
            _resolve_model_output_dir(),
            'modernbert_reward_model_predictions_scatter.png'
        )
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Scatter plot saved to: {output_path}")
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    return predictions, ground_truth


if __name__ == "__main__":
    predictions, ground_truth = test_mosaic_bert_on_test_set()

