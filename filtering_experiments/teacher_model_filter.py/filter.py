"""
Filter script to test Llama 3.3 70B (via Groq) on the test set.
Generates SQL queries from natural language questions and compares with ground truth.
"""
import os
import sys
import json
import pandas as pd
import time
from groq import Groq
from fuzzywuzzy import fuzz
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path for relative imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

# Load environment variables
load_dotenv()
GROQ_TOKEN = os.getenv("GROQ_TOKEN")

if not GROQ_TOKEN:
    raise ValueError("GROQ_TOKEN not found in environment variables. Please set it in .env file.")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_TOKEN)

RETRY_PROMPT = """You are an expert SQL developer. Given a table schema and a natural language question, generate the corresponding SQLite query.

{table_info_str}
Question: {nl_question}

Generate a SQLite query that answers this question. Use backticks for column names with spaces. Only return the SQL query, no explanations.
The query you generated most recently was: {recent_query}
This query was incorrect! Please generate a new query that answers the question. Do not repeat the incorrect query."""


def ask_llama(prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    """Ask the Llama model a question and return the response."""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️  Groq API error: {e}")
        raise


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


def filter_data(data, k=3, similarity_threshold=90):
    """Filter data by generating SQL using Llama and comparing with ground truth."""
    filtered_scores = []
    
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Filtering"):
        tries = 0
        correct = False
        response_text = None
        
        # Parse table_data if it exists and is not empty
        table_info_str = ""
        if pd.notna(row.get("table_data")) and str(row["table_data"]).strip():
            try:
                table_data = json.loads(row["table_data"])
                # Format table info with full data
                header = table_data.get('header', [])
                name = table_data.get('name', 'table')
                header_str = ", ".join(header) if header else "N/A"
                table_info_str = f"Table: {name}\nColumns: {header_str}\n"
                # Include ALL rows
                if 'rows' in table_data and table_data['rows']:
                    table_info_str += "\nRows:\n"
                    for row_data in table_data['rows']:
                        table_info_str += f"  {dict(zip(header, row_data))}\n"
            except Exception as e:
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
            
            try:
                # Use greedy decoding (temperature=0) on first try for speed, sampling on retries
                temperature = 0.0 if tries == 1 else 0.7
                
                # Call Llama via Groq API
                response_text = ask_llama(prompt_text, temperature=temperature, max_tokens=256)
                
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
                    
                # Small delay between retries to avoid rate limiting
                if tries < k:
                    time.sleep(0.1)
                    
            except Exception as e:
                if tries == 1:  # Only print error on first try to reduce noise
                    print(f"⚠️  Error: {e}")
                # Wait a bit longer on error before retry
                if tries < k:
                    time.sleep(1.0)
                continue
        
        if not correct:
            if tries == k:  # Only print if all tries failed
                print(f"❌ Failed after {k} tries: {row['nl_questions'][:50]}...")
        
        filtered_scores.append(1 if correct else 0)
    
    # Create results dataframe with new column
    return_df = data.copy()
    return_df["llama_filtered_scores"] = filtered_scores
    
    return return_df


if __name__ == "__main__":
    # Configuration - use paths relative to project root
    test_set_path = os.path.join(project_root, "data/human_eval_filtering_test.csv")
    output_path = os.path.join(project_root, "filtering_experiments/teacher_model_filter.py/llama_filtered_scores.csv")
    
    print(f"🤖 Using Llama 3.3 70B via Groq API")
    
    # Load test set
    print(f"📂 Loading test set from {test_set_path}...")
    test_set = pd.read_csv(test_set_path)
    print(f"✅ Loaded {len(test_set)} examples")
    
    # Filter data
    print("🚀 Starting filtering...")
    filtered_df = filter_data(test_set, k=3, similarity_threshold=90)
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path}")
    
    # Print summary
    total = len(filtered_df)
    correct = filtered_df["llama_filtered_scores"].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n📊 Summary:")
    print(f"   Total examples: {total}")
    print(f"   Correct: {correct}")
    print(f"   Failed: {total - correct}")
    print(f"   Accuracy: {accuracy:.2f}%")
