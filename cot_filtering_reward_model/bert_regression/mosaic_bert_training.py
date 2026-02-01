"""
ModernBERT-based regression model with sigmoid head for reward modeling.

This is a version using ModernBERT which supports sequences up to 8,192 tokens,
allowing for longer SQL queries and reasoning chains.
"""

import os
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
import json

# Disable torch.compile to avoid conflicts with DataParallel
# ModernBERT has compiled layers that cause issues with multi-GPU training
os.environ['TORCH_COMPILE_DISABLE'] = '1'
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.suppress_errors = True


class RewardDataset(Dataset):
    """Dataset for BERT reward model training."""
    
    def __init__(self, df, tokenizer, max_length=2048):
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
    
    def __init__(self, model_name='answerdotai/ModernBERT-base'):
        super().__init__()
        # ModernBERT has compiled layers - compilation is disabled at module level
        self.bert = AutoModel.from_pretrained(
            model_name,
            reference_compile=False,      
            attn_implementation="eager",  
        )
        # Simple linear layer to map embeddings to single score, then sigmoid
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # ModernBERT doesn't have pooler_output, so we use mean pooling over the sequence
        # Mean pooling: average the hidden states, weighted by attention mask
        last_hidden_state = outputs.last_hidden_state
        # Expand attention mask to match hidden state dimensions
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # Sum hidden states and divide by sum of attention mask (mean pooling)
        sum_hidden = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_hidden / sum_mask
        
        # Apply linear layer then sigmoid to get score between 0 and 1
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
    model_name = 'answerdotai/ModernBERT-base'
    output_dir = 'outputs/modernbert_reward_model'
    
    # Check GPU availability and use the last GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        # Use the last GPU (highest index)
        device_id = num_gpus - 1
        device = f'cuda:{device_id}'
        # Set the default CUDA device to the last GPU
        torch.cuda.set_device(device_id)
        print(f"🔧 Device: {device} (GPU {device_id} of {num_gpus})")
        print(f"   GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
        print(f"   Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
        # Check current memory usage
        memory_allocated = torch.cuda.memory_allocated(device_id) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device_id) / 1e9
        print(f"   Memory allocated: {memory_allocated:.2f} GB")
        print(f"   Memory reserved: {memory_reserved:.2f} GB")
    else:
        device = 'cpu'
        num_gpus = 0
        print(f"🔧 Device: {device} (CPU)")
    
    # Reduce batch size to avoid OOM with longer sequences (2048 tokens)
    batch_size = 2  # Reduced for longer sequences to avoid OOM
    
    num_epochs = 30
    learning_rate = 2e-5
    gradient_accumulation_steps = 4  # Effective batch size = batch_size * num_gpus * gradient_accumulation_steps
    
    # Load dataset - resolve path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    data_csv = os.path.join(project_root, 'data', 'cot_dataset_with_corruptions.csv')
    
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
    
    # Set max_length to 8192 (ModernBERT supports up to 8,192 tokens natively)
    # Using 2048 as a reasonable default for training efficiency
    max_length = 2048
    print(f"📏 Filtering examples over {max_length} tokens (ModernBERT supports up to 8,192 tokens)...")
    initial_count = len(df_clean)
    valid_indices = []
    token_lengths = []
    
    for idx, row in df_clean.iterrows():
        input_text = f"SQL: {row['sql']}\nReasoning: {row['reasoning']}\nNL: {row['predicted_nl']}"
        tokens = tokenizer.encode(input_text, add_special_tokens=True)
        length = len(tokens)
        
        if length <= max_length:
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
    
    print(f"\n📏 Using max_length: {max_length} tokens (ModernBERT supports up to 8,192 tokens)")
    
    if len(df_clean) == 0:
        print("❌ No valid examples remaining after filtering!")
        return
    
    # Split data randomly: 75% train, 12.5% validation, 12.5% test
    from sklearn.model_selection import train_test_split
    
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
    train_file = os.path.join(output_dir, 'train_set.csv')
    eval_file = os.path.join(output_dir, 'eval_set.csv')
    test_file = os.path.join(output_dir, 'test_set.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(train_file, index=False)
    eval_df.to_csv(eval_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"💾 Saved train set: {train_file} ({len(train_df)} examples)")
    print(f"💾 Saved eval set: {eval_file} ({len(eval_df)} examples)")
    print(f"💾 Saved test set: {test_file} ({len(test_df)} examples)")
    
    # Disable torch.compile before loading model (ModernBERT has compiled layers that conflict with DataParallel)
    # This is already done at module level, but ensure it's set
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    if hasattr(torch, '_dynamo'):
        torch._dynamo.config.suppress_errors = True
    
    # Load model (tokenizer already loaded above)
    model = BERTRewardModel(model_name)
    
    # Move model to the last GPU (no DataParallel)
    # Ensure we're using the correct device
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        model = model.to(device)
        print(f"   Model moved to {device}")
    
    # Create datasets
    train_dataset = RewardDataset(train_df, tokenizer, max_length)
    eval_dataset = RewardDataset(eval_df, tokenizer, max_length)
    
    # Training arguments
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"   Batch size: {batch_size}")
    print(f"   With gradient accumulation: {effective_batch_size}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # Explicitly set the device
        no_cuda=not torch.cuda.is_available(),
        local_rank=-1,  # Single GPU, no distributed training
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
        gradient_accumulation_steps=gradient_accumulation_steps,
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
    print("🚀 Starting ModernBERT reward model training...")
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
    
    results_file = os.path.join(output_dir, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Training complete! Model saved to: {output_dir}")
    print(f"📁 Results saved to: {results_file}")


if __name__ == "__main__":
    main()

