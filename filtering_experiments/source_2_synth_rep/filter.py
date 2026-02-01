"""
Filter script to test the fine-tuned Qwen model on the test set.
Generates SQL queries from natural language questions and compares with ground truth.
"""
import os
import sys
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fuzzywuzzy import fuzz
from tqdm import tqdm

# Add project root to path for relative imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

RETRY_PROMPT = """You are an expert SQL developer. Given a table schema and a natural language question, generate the corresponding SQLite query.

{table_info_str}
Question: {nl_question}

Generate a SQLite query that answers this question. Use backticks for column names with spaces. Only return the SQL query, no explanations.
The query you generated most recently was: {recent_query}
This query was incorrect! Please generate a new query that answers the question. Do not repeat the incorrect query."""


def load_finetuned_model(model_path: str = "outputs/qwen_lora_nl2sql"):
    """Load the fine-tuned LoRA model."""
    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    print(f"🔄 Loading fine-tuned model from {model_path}...")
    
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
    
    # Compile model for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("⚡ Model compiled for faster inference")
        except Exception as e:
            print(f"⚠️  Could not compile model: {e}")
    
    print("✅ Model loaded successfully!")
    return model, tokenizer


def format_chat_prompt(prompt: str, tokenizer):
    """Format prompt in Qwen2.5-Instruct chat format."""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback format for Qwen2.5-Instruct
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    return formatted


def extract_sql_from_response(response_text: str) -> str:
    """Extract SQL query from model response."""
    # Remove the prompt part if it's still there
    response_text = response_text.strip()
    
    # Try to extract SQL from code blocks
    if "```sql" in response_text:
        start = response_text.find("```sql") + 6
        end = response_text.find("```", start)
        if end != -1:
            return response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        if end != -1:
            return response_text[start:end].strip()
    
    # If no code blocks, try to find SQL-like patterns
    # Remove common prefixes/suffixes
    response_text = response_text.split("SQL query:")[-1]
    response_text = response_text.split("Query:")[-1]
    response_text = response_text.split("\n\n")[0]  # Take first paragraph
    
    # Clean up
    response_text = response_text.strip()
    if response_text.endswith(";"):
        return response_text
    elif ";" in response_text:
        return response_text.split(";")[0] + ";"
    
    return response_text


def filter_data(model, tokenizer, data, device, k=5, similarity_threshold=90, batch_size=1):
    """Filter data by generating SQL and comparing with ground truth."""
    finetuned_filtered_scores = []
    
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Filtering"):
        tries = 0
        correct = False
        response_text = None
        
        # Parse table_data if it exists and is not empty
        table_info_str = ""
        if pd.notna(row.get("table_data")) and str(row["table_data"]).strip():
            try:
                import json
                table_data = json.loads(row["table_data"])
                # Format table info similar to training
                header = table_data.get('header', [])
                name = table_data.get('name', 'table')
                header_str = ", ".join(header) if header else "N/A"
                table_info_str = f"Table: {name}\nColumns: {header_str}\n"
                # Optionally include sample rows (first 2 for speed)
                if 'rows' in table_data and table_data['rows']:
                    table_info_str += "\nSample rows:\n"
                    for i, row_data in enumerate(table_data['rows'][:2]):
                        table_info_str += f"  {dict(zip(header, row_data))}\n"
            except:
                pass
        
        # Create base prompt
        if table_info_str:
            base_prompt_text = f"""You are an expert SQL developer. Given a table schema and a natural language question, generate the corresponding SQLite query.

{table_info_str}
Question: {row["nl_questions"]}

Generate a SQLite query that answers this question. Use backticks for column names with spaces. Only return the SQL query, no explanations."""
        else:
            base_prompt_text = f"""You are an expert SQL developer. Given a natural language question, generate the corresponding SQLite query.

Question: {row["nl_questions"]}

Generate a SQLite query that answers this question. Use backticks for column names with spaces. Only return the SQL query, no explanations."""
        
        previous_sql_query = None
        
        while tries < k:
            tries += 1
            
            # Use retry prompt if this is a retry and we have a previous query
            if tries > 1 and previous_sql_query:
                prompt_text = RETRY_PROMPT.format(
                    table_info_str=table_info_str,
                    nl_question=row["nl_questions"],
                    recent_query=previous_sql_query
                )
            else:
                prompt_text = base_prompt_text
            
            # Format as chat prompt
            prompt = format_chat_prompt(prompt_text, tokenizer)
            
            # Tokenize
            input_enc = tokenizer(
                prompt,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )
            input_enc = {k: v.to(device) for k, v in input_enc.items()}
            prompt_length = input_enc["input_ids"].shape[1]
            
            try:
                # Generate with optimized parameters
                with torch.no_grad():
                    response = model.generate(
                        input_enc["input_ids"],
                        attention_mask=input_enc["attention_mask"],
                        max_new_tokens=200,  # Reduced from 256 for speed
                        do_sample=(tries > 1),  # Greedy on first try (faster), sample on retries
                        temperature=0.7 if tries > 1 else None,
                        top_p=0.9 if tries > 1 else None,
                        pad_token_id=tokenizer.pad_token_id,
                        num_return_sequences=1,
                    )
                
                # Extract just the generated part (more efficient)
                generated_tokens = response[0][prompt_length:]
                response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Extract SQL from response
                sql_query = extract_sql_from_response(response_text)
                previous_sql_query = sql_query  # Store for potential retry
                
                # Compare with ground truth
                ground_truth_sql = row["sql_queries"].strip()
                similarity = fuzz.partial_ratio(sql_query.lower(), ground_truth_sql.lower())
                
                if similarity >= similarity_threshold:
                    if tries == 1:
                        print(f"✅ Correct on first try! (similarity: {similarity}%)")
                    else:
                        print(f"✅ Correct answer! (similarity: {similarity}%, tries: {tries})")
                    correct = True
                    break
                elif tries == 1 and similarity >= similarity_threshold * 0.85:  # Early exit if very close
                    # Close enough, might be acceptable
                    pass
                    
            except Exception as e:
                if tries == 1:  # Only print error on first try to reduce noise
                    print(f"⚠️  Error: {e}")
                continue
        
        if not correct:
            if tries == k:  # Only print if all tries failed
                print(f"❌ Failed after {k} tries: {row['nl_questions'][:50]}...")
        
        finetuned_filtered_scores.append(1 if correct else 0)
    
    # Create results dataframe with new column
    return_df = data.copy()
    return_df["finetuned_filtered_scores"] = finetuned_filtered_scores
    
    return return_df


if __name__ == "__main__":
    # Configuration - use paths relative to project root
    test_set_path = os.path.join(project_root, "data/human_eval_filtering_test.csv")
    model_path = os.path.join(project_root, "outputs/qwen_lora_nl2sql")
    output_path = os.path.join(project_root, "filtering_experiments/source_2_synth_rep/finetuned_filtered_scores.csv")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load test set
    print(f"📂 Loading test set from {test_set_path}...")
    test_set = pd.read_csv(test_set_path)
    print(f"✅ Loaded {len(test_set)} examples")
    
    # Load model
    model, tokenizer = load_finetuned_model(model_path)
    
    # Filter data
    print("🚀 Starting filtering...")
    # Reduce k from 5 to 3 for faster processing (can adjust based on needs)
    filtered_df = filter_data(model, tokenizer, test_set, device, k=3, similarity_threshold=90)
    
    # Save results
    filtered_df.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path}")
    
    # Print summary
    total = len(filtered_df)
    correct = filtered_df["finetuned_filtered_scores"].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n📊 Summary:")
    print(f"   Total examples: {total}")
    print(f"   Correct: {correct}")
    print(f"   Failed: {total - correct}")
    print(f"   Accuracy: {accuracy:.2f}%")
