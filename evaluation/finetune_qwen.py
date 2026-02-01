"""
LoRA Fine-tuning script for Qwen from HuggingFace
Fine-tunes Qwen to generate SQL queries from natural language questions using data from wip_pipeline.

Version Requirements:
    peft 0.10.0 requires transformers >= 4.47.0 (for modeling_layers module)
    
    Recommended combinations:
    - transformers==4.48.0 or 4.49.0 with peft==0.10.0 (upgrade path)
    - transformers==4.46.3 with peft==0.9.0 or 0.8.0 (downgrade path)
    
    Install compatible versions:
    pip install transformers==4.48.0 peft==0.10.0
    OR
    pip install transformers==4.46.3 peft==0.9.0
"""
import sys
import os
import json
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

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_table_data(jsonl_path: str) -> dict:
    """Load all table data from JSONL file into a dictionary keyed by table ID."""
    table_data = {}
    if not os.path.exists(jsonl_path):
        print(f"⚠️  Table JSONL file not found at {jsonl_path}, proceeding without table context")
        return table_data
    
    print(f"📂 Loading table data from {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading tables"):
            try:
                table = json.loads(line.strip())
                table_id = table.get('id')
                if table_id:
                    # Normalize table ID to match format used in pipeline
                    normalized_id = "table_" + table_id.replace("-", "_")
                    table_data[normalized_id] = table
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing line: {e}")
                continue
    
    print(f"✅ Loaded {len(table_data)} tables")
    return table_data


def format_table_info(table: dict) -> str:
    """Format table dict into a readable format for the prompt."""
    header = table.get('header', [])
    name = table.get('name', table.get('id', 'table'))
    
    # Format header
    header_str = ", ".join(header) if header else "N/A"
    
    # Create formatted table info
    formatted = f"Table: {name}\n"
    formatted += f"Columns: {header_str}\n"
    
    # Optionally include sample rows (first 3)
    if 'rows' in table and table['rows']:
        formatted += "\nSample rows:\n"
        for i, row in enumerate(table['rows'][:3]):
            formatted += f"  {dict(zip(header, row))}\n"
    
    return formatted


def create_prompt(nl_question: str, table_info: str = None) -> str:
    """Create a prompt for SQL generation task."""
    if table_info:
        prompt = f"""You are an expert SQL developer. Given a table schema and a natural language question, generate the corresponding SQLite query.

{table_info}

Question: {nl_question}

Generate a SQLite query that answers this question. Use backticks for column names with spaces. Only return the SQL query, no explanations."""
    else:
        prompt = f"""You are an expert SQL developer. Given a natural language question, generate the corresponding SQLite query.

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


def preprocess_dataset(df: pd.DataFrame, tokenizer, table_data: dict = None, max_length: int = 2048):
    """Preprocess the dataset for training."""
    print("📝 Preprocessing dataset...")
    
    examples = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Formatting examples"):
        nl = str(row['nl']).strip()
        sql = str(row['sql']).strip()
        
        # Get table info if available
        table_info_str = None
        if table_data is not None and 'table_id' in row:
            table_id = str(row['table_id']).strip()
            if table_id in table_data:
                table = table_data[table_id]
                table_info_str = format_table_info(table)
        
        # Create prompt
        prompt = create_prompt(nl, table_info_str)
        
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
    data_path = os.path.join(project_root, "pipeline/intermediate_jack/filtered.csv")
    table_jsonl_path = os.path.join(project_root, "data/train.tables.modified.jsonl")
    output_dir = os.path.join(project_root, "evaluation/outputs/qwen_lora_nl2sql")
    
    # Training hyperparameters
    max_length = 2048
    batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    num_epochs = 20
    save_steps = 500
    logging_steps = 1
    
    # ---- Load data ----
    print(f"📂 Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found at {data_path}. Please run pipeline/run.py wip_pipeline() first to generate data.")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} examples")
    
    # Validate required columns
    required_cols = ['nl', 'sql']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Remove rows with missing data
    df = df.dropna(subset=required_cols)
    df = df[df['sql'].astype(str).str.len() > 0]
    df = df[df['nl'].astype(str).str.len() > 0]
    print(f"✅ Valid examples: {len(df)}")
    
    # Load table data if available
    table_data = load_table_data(table_jsonl_path)
    
    # Split train/val
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
    print(f"📊 Train: {len(df_train)}, Val: {len(df_val)}")
    
    # ---- GPU summary ----
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(2)
        print(f"🧠 Detected {gpu_name}")
        print(f"🧠 Memory: {torch.cuda.get_device_properties(2).total_memory / 1024**3:.2f} GB")
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
    train_tokenized = preprocess_dataset(df_train, tokenizer, table_data, max_length)
    train_dataset = Dataset.from_dict(train_tokenized)
    
    print("📝 Preprocessing validation set...")
    val_tokenized = preprocess_dataset(df_val, tokenizer, table_data, max_length)
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
        tokenizer=tokenizer,
    )
    
    # ---- Train ----
    print("🚀 Starting training...")
    trainer.train()
    
    # ---- Save ----
    print(f"💾 Saving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model and tokenizer saved to {output_dir}")
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "data_path": data_path,
        "num_train_examples": len(df_train),
        "num_val_examples": len(df_val),
        "max_length": max_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "used_table_context": len(table_data) > 0,
    }
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("✅ Training complete!")


if __name__ == "__main__":
    main()

