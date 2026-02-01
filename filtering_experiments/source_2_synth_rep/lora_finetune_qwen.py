"""
LoRA Fine-tuning script for Qwen/Qwen2.5-3B-Instruct
Fine-tunes the model to generate SQL queries from natural language questions given table information.
"""
import os
import json
import ast
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def format_table_info(table_info_str: str) -> str:
    """Format table_info string into a readable format for the prompt."""
    try:
        table_info = ast.literal_eval(table_info_str)  # Parse the string representation of dict
        if isinstance(table_info, dict):
            # Extract key information
            header = table_info.get('header', [])
            name = table_info.get('name', 'table')
            
            # Format header
            header_str = ", ".join(header) if header else "N/A"
            
            # Create formatted table info
            formatted = f"Table: {name}\n"
            formatted += f"Columns: {header_str}\n"
            
            # Optionally include sample rows (first 3)
            if 'rows' in table_info and table_info['rows']:
                formatted += "\nSample rows:\n"
                for i, row in enumerate(table_info['rows'][:3]):
                    formatted += f"  {dict(zip(header, row))}\n"
            
            return formatted
    except Exception as e:
        print(f"Warning: Could not parse table_info: {e}")
        return table_info_str
    return table_info_str


def create_prompt(table_info: str, nl_question: str) -> str:
    """Create a prompt for SQL generation task."""
    formatted_table = format_table_info(table_info)
    prompt = f"""You are an expert SQL developer. Given a table schema and a natural language question, generate the corresponding SQLite query.

{formatted_table}

Question: {nl_question}

Generate a SQLite query that answers this question. Use backticks for column names with spaces. Only return the SQL query, no explanations."""
    return prompt


def format_chat_prompt(prompt: str, sql: str, tokenizer) -> str:
    """Format prompt and response in Qwen2.5-Instruct chat format."""
    # Qwen2.5-Instruct uses a specific chat template
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": sql}
    ]
    
    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # Fallback format for Qwen2.5-Instruct
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{sql}<|im_end|>"
    
    return formatted


def preprocess_dataset(df: pd.DataFrame, tokenizer, max_length: int = 2048):
    """Preprocess the dataset for training."""
    print("📝 Preprocessing dataset...")
    
    examples = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Formatting examples"):
        table_info = row['table_info']
        nl = row['nl']
        sql = row['sql'].strip()
        
        # Create prompt
        prompt = create_prompt(table_info, nl)
        
        # Format as chat
        formatted_text = format_chat_prompt(prompt, sql, tokenizer)
        
        examples.append(formatted_text)
    
    # Tokenize all examples
    print("🔤 Tokenizing dataset...")
    tokenized = tokenizer(
        examples,
        truncation=True,
        max_length=max_length,
        padding=False,  # Let data collator handle padding
        return_tensors=None,
    )
    
    # Return as dict of lists - Dataset.from_dict expects this format
    return tokenized


def main():
    # Configuration
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    data_path = "filtering_experiments/source_2_synth_rep/generated_data.csv"
    output_dir = "outputs/qwen_lora_nl2sql"
    
    # Training hyperparameters
    max_length = 2048
    batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    num_epochs = 3
    save_steps = 500
    logging_steps = 50
    
    # ---- Load data ----
    print(f"📂 Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} examples")
    
    # Validate required columns
    required_cols = ['table_info', 'nl', 'sql']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Remove rows with missing data
    df = df.dropna(subset=required_cols)
    df = df[df['sql'].str.len() > 0]
    df = df[df['nl'].str.len() > 0]
    print(f"✅ Valid examples: {len(df)}")
    
    # Split train/val
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
    print(f"📊 Train: {len(df_train)}, Val: {len(df_val)}")
    
    # ---- GPU summary ----
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"🧠 Detected {n} GPU(s)")
        for i in range(n):
            free_gb = torch.cuda.mem_get_info(i)[0] / 1024**3
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} — {free_gb:.2f} GB free")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # ---- Load model and tokenizer ----
    print(f"🧩 Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side to right for causal LM
    tokenizer.padding_side = "right"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # ---- Setup LoRA ----
    print("🔧 Setting up LoRA...")
    peft_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
        bias="none",
        lora_dropout=0.05,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Ensure model config has pad_token_id set
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # ---- Preprocess datasets ----
    print("📝 Preprocessing training set...")
    train_tokenized = preprocess_dataset(df_train, tokenizer, max_length)
    train_dataset = Dataset.from_dict(train_tokenized)
    
    print("📝 Preprocessing validation set...")
    val_tokenized = preprocess_dataset(df_val, tokenizer, max_length)
    val_dataset = Dataset.from_dict(val_tokenized)
    
    # ---- Data collator ----
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=save_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer (deprecated)
    )
    
    # ---- Train ----
    print("🚀 Starting training...")
    trainer.train()
    
    # ---- Save ----
    print(f"💾 Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model and tokenizer saved to {output_dir}")
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "num_train_examples": len(df_train),
        "num_val_examples": len(df_val),
        "max_length": max_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
    }
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("✅ Training complete!")


if __name__ == "__main__":
    main()

