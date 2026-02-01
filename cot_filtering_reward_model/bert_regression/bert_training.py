"""
BERT-based regression model with sigmoid head for reward modeling.

This is a simpler alternative to the RL-trained reward model that uses
a BERT encoder with a sigmoid head to directly predict similarity scores.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    IntervalStrategy
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


class RewardDataset(Dataset):
    """Dataset for BERT reward model training."""
    
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Combine SQL, reasoning, and predicted NL into input text
        input_text = f"SQL: {row['sql']}\nReasoning: {row['reasoning']}\nNL: {row['predicted_nl']}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(row['similarity_with_penalty'], dtype=torch.float)
        }


class BERTRewardModel(torch.nn.Module):
    """BERT model with sigmoid head for reward prediction."""
    
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        scores = self.sigmoid(logits).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(scores, labels) * 100
        
        return {
            'loss': loss,
            'scores': scores,
            'logits': logits
        }


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    
    # Handle case where predictions is a tuple (when model returns dict)
    if isinstance(predictions, tuple):
        # Extract the actual predictions from the tuple
        predictions = predictions[0] if len(predictions) > 0 else predictions
    
    # If predictions is still a dict, extract the scores
    if isinstance(predictions, dict):
        predictions = predictions.get('scores', predictions.get('logits', predictions))
    
    # Convert to numpy array and flatten
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    elif hasattr(predictions, 'numpy'):
        predictions = predictions.numpy()
    
    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()
    
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse)
    }


def main():
    # Configuration
    model_name = 'bert-base-uncased'
    output_dir = 'outputs/bert_reward_model_mae_loss'
    batch_size = 16  
    num_epochs = 30
    learning_rate = 2e-5
    gradient_accumulation_steps = 4  # Effective batch size = 16 * 4 = 64
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load dataset
    data_csv = '../../data/cot_dataset_with_corruptions.csv'
    if not os.path.exists(data_csv):
        print(f"❌ Dataset not found: {data_csv}")
        print("Please run cot_dataset_creation.py first")
        return
    
    df = pd.read_csv(data_csv)
    print(f"📊 Loaded {len(df)} examples")
    
    # Validate dataset
    required_cols = ['sql', 'reasoning', 'predicted_nl', 'similarity_with_penalty']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return
    
    # Filter out invalid examples
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean = df_clean[df_clean['similarity_with_penalty'].between(0, 1)].copy()
    print(f"✅ Clean dataset: {len(df_clean)} examples")
    
    if len(df_clean) == 0:
        print("❌ No valid examples found!")
        return
    
    # Calculate token length statistics and filter out long examples
    print("🔍 Analyzing token length distribution...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"📏 Filtering examples over 512 tokens (BERT's limit)...")
    initial_count = len(df_clean)
    valid_indices = []
    token_lengths = []
    
    for idx, row in df_clean.iterrows():
        input_text = f"SQL: {row['sql']}\nReasoning: {row['reasoning']}\nNL: {row['predicted_nl']}"
        tokens = tokenizer.encode(input_text, add_special_tokens=True)
        length = len(tokens)
        
        if length <= 512:
            valid_indices.append(idx)
            token_lengths.append(length)
    
    # Filter to valid examples only
    df_clean = df_clean.loc[valid_indices].copy()
    token_lengths = np.array(token_lengths)
    
    # Print statistics
    excluded_count = initial_count - len(df_clean)
    print(f"✅ Excluded {excluded_count} examples ({excluded_count/initial_count*100:.1f}%)")
    print(f"📊 Remaining examples: {len(df_clean)} ({len(df_clean)/initial_count*100:.1f}% of valid data)")
    print(f"\n📊 Token length statistics (after filtering):")
    print(f"   Mean: {token_lengths.mean():.1f}")
    print(f"   Median: {np.median(token_lengths):.1f}")
    print(f"   Std: {token_lengths.std():.1f}")
    print(f"   Min: {token_lengths.min()}")
    print(f"   Max: {token_lengths.max()}")
    print(f"   95th percentile: {np.percentile(token_lengths, 95):.1f}")
    
    # Set max_length to 512 (BERT's maximum)
    max_length = 512
    print(f"\n📏 Using max_length: {max_length} tokens (BERT's maximum)")
    
    if len(df_clean) == 0:
        print("❌ No valid examples remaining after filtering!")
        return
    
    # First split: 75% train, 25% holdout (for val + test)
    train_df, temp_df = train_test_split(
        df_clean, 
        test_size=0.25, 
        random_state=12,
        shuffle=True
    )
    
    # Second split: take the 25% and split into 12.5% val, 12.5% test
    eval_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=12,
        shuffle=True
    )
    
    print(f"📊 Train: {len(train_df)}, Val: {len(eval_df)}, Test: {len(test_df)}")
    print(f"   Random state: 12 (for reproducibility)")
    
    # Save train, eval, and test sets for consistency
    train_file = os.path.join(output_dir, 'train_set_mae_loss.csv')
    eval_file = os.path.join(output_dir, 'eval_set_mae_loss.csv')
    test_file = os.path.join(output_dir, 'test_set_mae_loss.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(train_file, index=False)
    eval_df.to_csv(eval_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"💾 Saved train set: {train_file} ({len(train_df)} examples)")
    print(f"💾 Saved eval set: {eval_file} ({len(eval_df)} examples)")
    print(f"💾 Saved test set: {test_file} ({len(test_df)} examples)")
    
    # Load model (tokenizer already loaded above)
    model = BERTRewardModel(model_name)
    
    # Create datasets
    train_dataset = RewardDataset(train_df, tokenizer, max_length)
    eval_dataset = RewardDataset(eval_df, tokenizer, max_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=32,
        save_strategy='steps',
        save_steps=32,
        load_best_model_at_end=True,
        metric_for_best_model='mse',
        greater_is_better=False,
        save_total_limit=3,  # Keep only last 3 checkpoints
        report_to=[],
        dataloader_drop_last=False,
        gradient_accumulation_steps=4,  # Effective batch size = 16 * 4 = 64
        remove_unused_columns=False,
        dataloader_num_workers=4,  # Parallel data loading
        seed=42,  # For reproducibility
    )
    
    # Create trainer with callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
        ]
    )
    
    # Train
    print("🚀 Starting BERT reward model training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Final evaluation
    print("\n📊 Final evaluation:")
    eval_results = trainer.evaluate()
    print(f"   MSE: {eval_results['eval_mse']:.4f}")
    print(f"   MAE: {eval_results['eval_mae']:.4f}")
    print(f"   RMSE: {eval_results['eval_rmse']:.4f}")
    
    # Save results
    results = {
        'model_name': model_name,
        'dataset_size': len(df_clean),
        'train_size': len(train_df),
        'eval_size': len(eval_df),
        'final_metrics': eval_results,
        'similarity_stats': {
            'mean': float(df_clean['similarity_with_penalty'].mean()),
            'std': float(df_clean['similarity_with_penalty'].std()),
            'min': float(df_clean['similarity_with_penalty'].min()),
            'max': float(df_clean['similarity_with_penalty'].max())
        }
    }
    
    results_file = os.path.join(output_dir, 'training_results_mae_loss.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Training complete! Model saved to: {output_dir}")
    print(f"📁 Results saved to: {results_file}")


if __name__ == "__main__":
    main()