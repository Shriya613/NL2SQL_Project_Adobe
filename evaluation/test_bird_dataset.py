"""
Test fine-tuned Qwen model against base Qwen model on BIRD dataset.
Evaluates SQL generation performance using multiple metrics.
"""
import sys
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fuzzywuzzy import fuzz

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_bird_dataset(json_path: str = "data/train_bird.json", max_examples: int = None):
    """Load BIRD dataset from JSON file."""
    abs_path = os.path.join(project_root, json_path) if not os.path.isabs(json_path) else json_path
    
    print(f"📂 Loading BIRD dataset from {abs_path}...")
    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if max_examples:
        data = data[:max_examples]
        print(f"   Limited to {max_examples} examples")
    
    print(f"✅ Loaded {len(data)} examples")
    return data


def load_base_qwen():
    """Load base Qwen model (without fine-tuning)."""
    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    print(f"🔄 Loading base Qwen model: {base_model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    
    print("✅ Base Qwen model loaded successfully!")
    return model, tokenizer


def load_finetuned_qwen(model_path: str = None):
    """Load fine-tuned Qwen model with LoRA adapters."""
    if model_path is None:
        model_path = os.path.join(project_root, "evaluation/outputs/qwen_lora_nl2sql")
    
    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    print(f"🔄 Loading fine-tuned Qwen model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at {model_path}. Please run finetune_qwen.py first.")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    print("✅ Fine-tuned Qwen model loaded successfully!")
    return model, tokenizer


def create_prompt(nl_question: str, evidence: str = None) -> str:
    """Create a prompt for SQL generation task."""
    prompt = f"""You are an expert SQL developer. Given a natural language question, generate the corresponding SQLite query.

Question: {nl_question}"""
    
    if evidence:
        prompt += f"\n\nEvidence: {evidence}"
    
    prompt += "\n\nGenerate a SQLite query that answers this question. Use backticks for column names with spaces. Only return the SQL query, no explanations."
    
    return prompt


def format_chat_prompt(prompt: str, tokenizer) -> str:
    """Format prompt in Qwen2.5-Instruct chat format."""
    messages = [{"role": "user", "content": prompt}]
    
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    return formatted


def generate_sql(model, tokenizer, nl_question: str, evidence: str = None, max_new_tokens: int = 512, temperature: float = 0.7):
    """Generate SQL using Qwen model."""
    prompt = create_prompt(nl_question, evidence)
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
    
    # Extract SQL from response
    if "<|im_start|>assistant" in generated_text:
        sql = generated_text.split("<|im_start|>assistant")[-1].strip()
    else:
        sql = generated_text
    
    # Clean up
    sql = sql.replace("<|im_end|>", "").strip()
    sql = sql.replace("<|im_start|>", "").strip()
    
    # Extract from code blocks if present
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


def evaluate_exact_match(gold_sql: str, predicted_sql: str) -> float:
    """Exact string match evaluation."""
    return 1.0 if gold_sql.strip() == predicted_sql.strip() else 0.0


def evaluate_fuzzy_match(gold_sql: str, predicted_sql: str, threshold: float = 0.9) -> float:
    """Fuzzy string matching evaluation."""
    similarity = fuzz.ratio(gold_sql.strip(), predicted_sql.strip()) / 100.0
    return 1.0 if similarity >= threshold else 0.0


def evaluate_fuzzy_score(gold_sql: str, predicted_sql: str) -> float:
    """Return fuzzy similarity score (0-1)."""
    return fuzz.ratio(gold_sql.strip(), predicted_sql.strip()) / 100.0


def main():
    # Configuration
    bird_json_path = os.path.join(project_root, "data/train_bird.json")
    output_path = os.path.join(project_root, "evaluation/bird_evaluation_results.csv")
    model_path = os.path.join(project_root, "evaluation/outputs/qwen_lora_nl2sql")
    
    max_examples = 25  # Set to a number to limit examples, None for all
    temperature = 0.7
    max_new_tokens = 512
    
    print("=" * 70)
    print("BIRD Dataset Evaluation: Base Qwen vs Fine-tuned Qwen")
    print("=" * 70)
    
    # Load dataset
    bird_data = load_bird_dataset(bird_json_path, max_examples=max_examples)
    
    # Load models
    print("\n🔄 Loading models...")
    base_model, base_tokenizer = load_base_qwen()
    finetuned_model, finetuned_tokenizer = load_finetuned_qwen(model_path)
    
    # Generate predictions
    results = []
    print(f"\n🚀 Generating predictions for {len(bird_data)} examples...")
    
    base_exact_matches = 0
    finetuned_exact_matches = 0
    base_fuzzy_matches = 0
    finetuned_fuzzy_matches = 0
    base_fuzzy_scores = []
    finetuned_fuzzy_scores = []
    
    for idx, item in enumerate(tqdm(bird_data, desc="Processing")):
        nl_question = item.get("question", "").strip()
        gold_sql = item.get("SQL", "").strip()
        evidence = item.get("evidence", "").strip()
        db_id = item.get("db_id", "")
        
        if not nl_question or not gold_sql:
            continue
        
        # Generate with base model
        try:
            base_sql = generate_sql(
                base_model, base_tokenizer, nl_question, evidence,
                max_new_tokens=max_new_tokens, temperature=temperature
            )
        except Exception as e:
            print(f"\n⚠️  Error generating with base model for example {idx}: {e}")
            base_sql = ""
        
        # Generate with fine-tuned model
        try:
            finetuned_sql = generate_sql(
                finetuned_model, finetuned_tokenizer, nl_question, evidence,
                max_new_tokens=max_new_tokens, temperature=temperature
            )
        except Exception as e:
            print(f"\n⚠️  Error generating with fine-tuned model for example {idx}: {e}")
            finetuned_sql = ""
        
        # Evaluate
        base_exact = evaluate_exact_match(gold_sql, base_sql)
        finetuned_exact = evaluate_exact_match(gold_sql, finetuned_sql)
        base_fuzzy = evaluate_fuzzy_match(gold_sql, base_sql)
        finetuned_fuzzy = evaluate_fuzzy_match(gold_sql, finetuned_sql)
        base_fuzzy_score = evaluate_fuzzy_score(gold_sql, base_sql)
        finetuned_fuzzy_score = evaluate_fuzzy_score(gold_sql, finetuned_sql)
        
        base_exact_matches += base_exact
        finetuned_exact_matches += finetuned_exact
        base_fuzzy_matches += base_fuzzy
        finetuned_fuzzy_matches += finetuned_fuzzy
        base_fuzzy_scores.append(base_fuzzy_score)
        finetuned_fuzzy_scores.append(finetuned_fuzzy_score)
        
        results.append({
            'index': idx,
            'db_id': db_id,
            'question': nl_question,
            'evidence': evidence,
            'gold_sql': gold_sql,
            'base_sql': base_sql,
            'finetuned_sql': finetuned_sql,
            'base_exact_match': base_exact,
            'finetuned_exact_match': finetuned_exact,
            'base_fuzzy_match': base_fuzzy,
            'finetuned_fuzzy_match': finetuned_fuzzy,
            'base_fuzzy_score': base_fuzzy_score,
            'finetuned_fuzzy_score': finetuned_fuzzy_score,
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")
    
    # Print summary statistics
    total = len(results)
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Total examples: {total}")
    print(f"\n{'Metric':<30} {'Base Qwen':<20} {'Fine-tuned Qwen':<20} {'Improvement':<15}")
    print("-" * 70)
    
    base_exact_pct = (base_exact_matches / total * 100) if total > 0 else 0
    finetuned_exact_pct = (finetuned_exact_matches / total * 100) if total > 0 else 0
    exact_improvement = finetuned_exact_pct - base_exact_pct
    
    base_fuzzy_pct = (base_fuzzy_matches / total * 100) if total > 0 else 0
    finetuned_fuzzy_pct = (finetuned_fuzzy_matches / total * 100) if total > 0 else 0
    fuzzy_improvement = finetuned_fuzzy_pct - base_fuzzy_pct
    
    base_avg_fuzzy = sum(base_fuzzy_scores) / len(base_fuzzy_scores) if base_fuzzy_scores else 0
    finetuned_avg_fuzzy = sum(finetuned_fuzzy_scores) / len(finetuned_fuzzy_scores) if finetuned_fuzzy_scores else 0
    avg_fuzzy_improvement = finetuned_avg_fuzzy - base_avg_fuzzy
    
    print(f"{'Exact Match':<30} {base_exact_matches}/{total} ({base_exact_pct:.2f}%){'':<5} {finetuned_exact_matches}/{total} ({finetuned_exact_pct:.2f}%){'':<5} {exact_improvement:+.2f}%")
    print(f"{'Fuzzy Match (>0.9)':<30} {base_fuzzy_matches}/{total} ({base_fuzzy_pct:.2f}%){'':<5} {finetuned_fuzzy_matches}/{total} ({finetuned_fuzzy_pct:.2f}%){'':<5} {fuzzy_improvement:+.2f}%")
    print(f"{'Avg Fuzzy Score':<30} {base_avg_fuzzy:.4f}{'':<15} {finetuned_avg_fuzzy:.4f}{'':<15} {avg_fuzzy_improvement:+.4f}")
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

