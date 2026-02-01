import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_bert_model():
    """Load the trained BERT model"""
    model_path = "/home/daeilee/nl2sql/cot_filtering_reward_model/bert_regression/outputs/bert_reward_model_large_dataset"
    
    print("🔄 Loading BERT model...")
    # Use the model as saved by transformers
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,  # Regression task
        problem_type="regression"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.eval()
    
    return model, tokenizer

def predict_similarity(model, tokenizer, sql, reasoning, predicted_nl):
    """Predict similarity score for a given example"""
    # Create input text
    input_text = f"SQL: {sql}\nReasoning: {reasoning}\nNL: {predicted_nl}"
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply sigmoid to get probability
        prediction = torch.sigmoid(outputs.logits).item()
    
    return prediction

def test_bert_on_test_set():
    """Test BERT model on test set examples"""
    print("🚀 Testing BERT model on test set...")
    
    # Load model
    model, tokenizer = load_bert_model()
    
    # Try to load saved test set first (for consistency)
    test_file = "/home/daeilee/nl2sql/cot_filtering_reward_model/bert_regression/outputs/bert_reward_model_large_dataset/test_set_large_dataset.csv"
    
    if os.path.exists(test_file):
        print(f"📂 Loading saved test set from training...")
        test_df = pd.read_csv(test_file)
        print(f"   Loaded {len(test_df)} examples from saved test set")
    else:
        # Fall back to recalculating split if saved set doesn't exist
        print("⚠️  Saved test set not found, recalculating split...")
        df = pd.read_csv("/home/daeilee/nl2sql/data/cot_dataset_with_corruptions.csv")
        
        # Filter to valid examples
        required_cols = ['sql', 'reasoning', 'predicted_nl', 'similarity_with_penalty']
        df = df.dropna(subset=required_cols).copy()
        df = df[df['similarity_with_penalty'].between(0, 1)].copy()
        
        # Split into train/val/test (same as training - 75/12.5/12.5 split)
        train_df, temp_df = train_test_split(df, test_size=0.25, random_state=12, shuffle=True)
        eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=12, shuffle=True)
        print(f"📊 Test set size: {len(test_df)} examples (recalculated)")
    
    # Use all test examples
    test_sample = test_df
    print(f"📊 Using {len(test_sample)} examples for testing")
    
    print("\n🔍 Testing BERT predictions vs ground truth...")
    print("=" * 80)
    
    predictions = []
    ground_truth = []
    
    for i, (_, row) in enumerate(test_sample.iterrows(), 1):
        # Get prediction
        pred_score = predict_similarity(
            model, tokenizer, 
            row['sql'], 
            row['reasoning'], 
            row['predicted_nl']
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
            print(f"BERT Prediction: {pred_score:.3f}")
            print(f"Difference: {abs(pred_score - true_score):.3f}")
            print("-" * 60)
    
    # Calculate metrics
    mse = mean_squared_error(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\n📊 Performance Metrics on 40 Test Examples:")
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
        plt.ylabel('BERT Prediction')
        plt.title(f'BERT Predictions vs Ground Truth (Test Set)\n(N={len(predictions)} examples)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add MSE and MAE as text on the plot
        metrics_text = f'MSE: {mse:.4f}\nMAE: {mae:.4f}'
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig('/home/daeilee/nl2sql/cot_filtering_reward_model/bert_regression/bert_predictions_scatter_w_corruptions_large_dataset.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Scatter plot saved to: /home/daeilee/nl2sql/cot_filtering_reward_model/bert_regression/bert_predictions_scatter_w_corruptions_large_dataset.png")
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    return predictions, ground_truth

if __name__ == "__main__":
    predictions, ground_truth = test_bert_on_test_set()
