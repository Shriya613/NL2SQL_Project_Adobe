"""
Compare OpenAI and fine-tuned Qwen models for SQL generation from NL questions.
Uses both models to predict SQL queries and saves comparison results.
"""
import sys
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from openai import OpenAI

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load .env file
import dotenv
env_path = os.path.join(project_root, '.env')
dotenv.load_dotenv(dotenv_path=env_path)


def load_finetuned_qwen(model_path: str = None):
    """Load the fine-tuned Qwen model with LoRA adapters."""
    if model_path is None:
        model_path = os.path.join(project_root, "evaluation/outputs/qwen_lora_nl2sql")
    
    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    print(f"🔄 Loading fine-tuned Qwen model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at {model_path}. Please run finetune_qwen.py first.")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
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
    
    print("✅ Fine-tuned Qwen model loaded successfully!")
    return model, tokenizer


def format_table_info(table: dict) -> str:
    """Format table dict into a readable format for the prompt."""
    header = table.get('header', [])
    name = table.get('name', table.get('id', 'table'))
    
    header_str = ", ".join(header) if header else "N/A"
    
    formatted = f"Table: {name}\n"
    formatted += f"Columns: {header_str}\n"
    
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


def format_chat_prompt(prompt: str, tokenizer) -> str:
    """Format prompt in Qwen2.5-Instruct chat format."""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    return formatted


def generate_sql_qwen(model, tokenizer, nl_question: str, table_info: str = None, max_new_tokens: int = 512, temperature: float = 0.7):
    """Generate SQL using fine-tuned Qwen model."""
    prompt = create_prompt(nl_question, table_info)
    formatted_prompt = format_chat_prompt(prompt, tokenizer)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
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
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract SQL from response (remove the prompt part)
    # The model generates the full conversation, so we need to extract just the assistant's response
    if "<|im_start|>assistant" in generated_text:
        sql = generated_text.split("<|im_start|>assistant")[-1].strip()
    elif "assistant" in generated_text.lower():
        # Fallback extraction
        parts = generated_text.split("assistant")
        if len(parts) > 1:
            sql = parts[-1].strip()
        else:
            sql = generated_text
    else:
        # If no markers found, try to extract from the end (assistant response should be at the end)
        sql = generated_text
    
    # Clean up SQL - remove chat template markers
    sql = sql.replace("<|im_end|>", "").strip()
    sql = sql.replace("<|im_start|>", "").strip()
    
    # Try to extract SQL from code blocks if present
    if "```sql" in sql:
        start = sql.find("```sql") + 6
        end = sql.find("```", start)
        if end != -1:
            sql = sql[start:end].strip()
    elif "```" in sql:
        start = sql.find("```") + 3
        end = sql.find("```", start)
        if end != -1:
            sql = sql[start:end].strip()
    
    return sql


def generate_sql_openai(client: OpenAI, nl_question: str, table_info: str = None, model: str = "gpt-4o-mini", temperature: float = 0.7):
    """Generate SQL using OpenAI API."""
    prompt = create_prompt(nl_question, table_info)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert SQL developer. Generate SQLite queries from natural language questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=512,
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Try to extract SQL from code blocks if present
        if "```sql" in sql:
            start = sql.find("```sql") + 6
            end = sql.find("```", start)
            if end != -1:
                sql = sql[start:end].strip()
        elif "```" in sql:
            start = sql.find("```") + 3
            end = sql.find("```", start)
            if end != -1:
                sql = sql[start:end].strip()
        
        return sql
    except Exception as e:
        print(f"⚠️  OpenAI API error: {e}")
        return None


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
                    normalized_id = "table_" + table_id.replace("-", "_")
                    table_data[normalized_id] = table
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing line: {e}")
                continue
    
    print(f"✅ Loaded {len(table_data)} tables")
    return table_data


def main():
    # Configuration
    data_path = os.path.join(project_root, "pipeline/intermediate/filtered.csv")
    table_jsonl_path = os.path.join(project_root, "data/train.tables.modified.jsonl")
    output_path = os.path.join(project_root, "evaluation/comparison_results.csv")
    model_path = os.path.join(project_root, "evaluation/outputs/qwen_lora_nl2sql")
    
    # OpenAI configuration
    openai_model = "gpt-4o-mini"  # Can be changed to "gpt-4", "gpt-3.5-turbo", etc.
    max_examples = None  # Set to a number to limit examples, None for all
    
    print("=" * 60)
    print("Model Comparison: OpenAI vs Fine-tuned Qwen")
    print("=" * 60)
    
    # Load data
    print(f"\n📂 Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run pipeline/run.py first.")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} examples")
    
    # Limit examples if specified
    if max_examples:
        df = df.head(max_examples)
        print(f"📊 Using first {len(df)} examples")
    
    # Load table data
    table_data = load_table_data(table_jsonl_path)
    
    # Load models
    print("\n🔄 Loading models...")
    openai_client = OpenAI()
    qwen_model, qwen_tokenizer = load_finetuned_qwen(model_path)
    
    # Generate predictions
    results = []
    print(f"\n🚀 Generating predictions for {len(df)} examples...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        nl = str(row['nl']).strip()
        ground_truth_sql = str(row['sql']).strip()
        table_id = str(row.get('table_id', '')).strip() if 'table_id' in row else None
        
        # Get table info if available
        table_info = None
        if table_id and table_id in table_data:
            table = table_data[table_id]
            table_info = format_table_info(table)
        
        # Generate with OpenAI
        print(f"\n[{idx+1}/{len(df)}] Generating with OpenAI...")
        openai_sql = generate_sql_openai(openai_client, nl, table_info, model=openai_model)
        
        # Generate with Qwen
        print(f"[{idx+1}/{len(df)}] Generating with fine-tuned Qwen...")
        qwen_sql = generate_sql_qwen(qwen_model, qwen_tokenizer, nl, table_info)
        
        results.append({
            'index': idx,
            'nl_question': nl,
            'ground_truth_sql': ground_truth_sql,
            'openai_model': openai_model,
            'openai_sql': openai_sql,
            'qwen_sql': qwen_sql,
            'table_id': table_id,
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")
    print(f"   Total examples: {len(results_df)}")
    print(f"   OpenAI predictions: {results_df['openai_sql'].notna().sum()}")
    print(f"   Qwen predictions: {results_df['qwen_sql'].notna().sum()}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"OpenAI model: {openai_model}")
    print(f"Fine-tuned Qwen: {model_path}")
    print(f"Total examples processed: {len(results_df)}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

