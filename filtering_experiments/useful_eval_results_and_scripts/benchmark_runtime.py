"""
Benchmark the runtime of each filtering method on a single example.
"""
import os
import sys
import json
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

# Import filtering functions (lazy imports to avoid loading models at import time)
import groq

def benchmark_finetuned_qwen(nl_question, sql_query, table_data_str):
    """Benchmark finetuned Qwen filter - same as filter.py. Uses base model if LoRA doesn't exist."""
    model_path = os.path.join(project_root, "filtering_experiments/source_2_synth_rep/outputs/qwen_lora_nl2sql")
    use_lora = os.path.exists(model_path)
    
    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model exactly as in filter.py
    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
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
    
    # Load LoRA weights if they exist, otherwise use base model (runtime should be the same)
    if use_lora:
        model = PeftModel.from_pretrained(model, model_path)
    
    model.eval()
    
    # Compile model if available (same as filter.py)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass
    
    device = next(model.parameters()).device
    
    # Format prompt exactly as in filter.py
    table_info_str = ""
    if table_data_str:
        try:
            table_data = json.loads(table_data_str)
            header = table_data.get('header', [])
            name = table_data.get('name', 'table')
            header_str = ", ".join(header) if header else "N/A"
            table_info_str = f"Table: {name}\nColumns: {header_str}\n"
            if 'rows' in table_data and table_data['rows']:
                table_info_str += "\nSample rows:\n"
                for i, row_data in enumerate(table_data['rows'][:2]):
                    table_info_str += f"  {dict(zip(header, row_data))}\n"
        except:
            pass
    
    if table_info_str:
        prompt_text = f"""You are an expert SQL developer. Given a table schema and a natural language question, generate the corresponding SQLite query.

{table_info_str}
Question: {nl_question}

Generate a SQLite query that answers this question. Use backticks for column names with spaces. Only return the SQL query, no explanations."""
    else:
        prompt_text = f"""You are an expert SQL developer. Given a natural language question, generate the corresponding SQLite query.

Question: {nl_question}

Generate a SQLite query that answers this question. Use backticks for column names with spaces. Only return the SQL query, no explanations."""
    
    # Format using chat template (same as filter.py)
    messages = [{"role": "user", "content": prompt_text}]
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    
    # Benchmark - same as filter.py (greedy, first try)
    start = time.time()
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,  # Greedy on first try (same as filter.py)
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed = time.time() - start
    
    # Clean up
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return elapsed

def benchmark_llama_teacher(nl_question, sql_query, table_data_str):
    """Benchmark Llama teacher filter via Groq API - same as filter.py."""
    from dotenv import load_dotenv
    load_dotenv()
    
    GROQ_TOKEN = os.getenv("GROQ_TOKEN")  # Same as filter.py
    if not GROQ_TOKEN:
        return None
    
    client = groq.Groq(api_key=GROQ_TOKEN)
    
    # Format prompt
    if table_data_str:
        try:
            table_data = json.loads(table_data_str)
            table_info_str = f"Table: {table_data.get('name', '')}\n"
            if 'header' in table_data:
                table_info_str += f"Columns: {', '.join(table_data['header'])}\n"
            if 'rows' in table_data and table_data['rows']:
                table_info_str += "\nAll rows:\n"
                for row_data in table_data['rows']:
                    table_info_str += f"  {dict(zip(table_data['header'], row_data))}\n"
        except:
            table_info_str = ""
    else:
        table_info_str = ""
    
    prompt_text = f"""You are an expert SQL developer. Given a table schema and a natural language question, generate the corresponding SQLite query.

{table_info_str}
Question: {nl_question}

Generate a SQLite query that answers this question. Use backticks for column names with spaces. Only return the SQL query, no explanations."""
    
    # Benchmark
    start = time.time()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.0,
        max_tokens=200,
    )
    elapsed = time.time() - start
    
    return elapsed

def benchmark_prompt_filter(nl_question, sql_query, table_data_str):
    """Benchmark prompt filter - same as filter.py."""
    # Lazy import to avoid loading model at import time
    from data_gen.data_gen import run_llm
    
    # Use the same prompt definition as in filter.py
    PROMPT_FILTER_PROMPT_NO_REASONING = """
    You are an expert natural language expert tasked with judging whether a natural language question is valid based on the SQLite query and table information.

    Your judgement should be based on the following criteria:
    1. The natural language question should produce a question. It does not include the SQLite query or table information.
    2. The natural language question should include all of the information from the SQLite query. There should be no missing information. 
    Penalize natural language questions heavily if they are missing information from the SQLite query.
    3. The natural language question should not include details that the SQLite query does not contain.
    4. The natural language question should not include any information besides the question.
    5. The natural language question should only be one singular natural language question.
    6. Align with the SQLite query as closely as possible. They should be as similar as possible.

    Natural Language Question: {nl_question}
    SQLite Query: {sql_query}
    Table Information: {table}

    Format your response as a JSON object with the following keys:
    - "score": the score of the natural language question, between 0 and 1

    Example response:
    ```json
    {{
        "score": ...,
    }}
    ```

    Do not include any other text in your response besides the JSON object.
    """
    
    # Format table data exactly as in filter.py
    if table_data_str:
        try:
            table_data = json.loads(table_data_str)
        except:
            table_data = {}
    else:
        table_data = {}
    
    # Format prompt exactly as in filter.py
    prompt_filter = PROMPT_FILTER_PROMPT_NO_REASONING.format(
        sql_query=sql_query,
        nl_question=nl_question,
        table=table_data
    )
    
    # Benchmark - same as filter.py
    start = time.time()
    response = run_llm(prompt_filter)
    elapsed = time.time() - start
    
    return elapsed

def benchmark_reward_model(nl_question, sql_query, table_data_str):
    """Benchmark reward model filter - just the reward calculation (without reasoning generation)."""
    # Lazy import to avoid loading model at import time
    from cot_filtering_reward_model.testing_final_model.load_model_from_hf import load_model_from_huggingface, get_reward
    
    # Load model once (like in pipetest)
    model, tokenizer, device = load_model_from_huggingface()
    
    # Use empty reasoning (shouldnt affect runtime)
    reasoning = ""
    
    # Benchmark just the reward calculation (same format as pipetest but without reasoning)
    start = time.time()
    score = get_reward(model, tokenizer, sql_query, reasoning, nl_question, device)
    elapsed = time.time() - start
    
    # Clean up model from memory
    del model
    del tokenizer
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return elapsed

def benchmark_reward_threshold(nl_question, sql_query, table_data_str):
    """Benchmark reward threshold (just reading from CSV, so very fast)."""
    # This is just reading a pre-computed value, so it's essentially instant
    return 0.001  # Very fast, just reading from CSV

def get_sample_example():
    """Get a sample example from the test set."""
    test_path = os.path.join(project_root, "filtering_experiments/source_2_synth_rep/finetuned_filtered_scores.csv")
    df = pd.read_csv(test_path)
    # Get first row with all required data
    for idx, row in df.iterrows():
        if pd.notna(row.get("nl_questions")) and pd.notna(row.get("sql_queries")):
            return {
                "nl_question": str(row["nl_questions"]),
                "sql_query": str(row["sql_queries"]),
                "table_data": str(row.get("table_data", "")) if pd.notna(row.get("table_data")) else ""
            }
    return None

def benchmark_all_methods():
    """Benchmark all filtering methods."""
    example = get_sample_example()
    if not example:
        print("❌ Could not find sample example")
        return {}
    
    print("🔄 Benchmarking filtering methods on a single example...")
    print(f"   Example: {example['nl_question'][:50]}...")
    
    results = {}
    
    # Benchmark each method
    methods = [
        ("reward_model", benchmark_reward_model),  # Do reward model first since it needs to load
        ("finetuned_qwen", benchmark_finetuned_qwen),
        ("llama_teacher", benchmark_llama_teacher),
        ("prompt_filter", benchmark_prompt_filter),
        ("reward_threshold", benchmark_reward_threshold),
    ]
    
    for method_name, benchmark_func in methods:
        try:
            print(f"   Testing {method_name}...", end=" ")
            elapsed = benchmark_func(
                example["nl_question"],
                example["sql_query"],
                example["table_data"]
            )
            if elapsed is not None:
                results[method_name] = elapsed
                print(f"{elapsed:.3f}s")
            else:
                print("skipped (not available)")
        except Exception as e:
            print(f"error: {e}")
            results[method_name] = None
    
    return results

if __name__ == "__main__":
    results = benchmark_all_methods()
    print("\n📊 Runtime Results (seconds per example):")
    for method, time_taken in results.items():
        if time_taken is not None:
            print(f"  {method}: {time_taken:.3f}s")
        else:
            print(f"  {method}: N/A")

