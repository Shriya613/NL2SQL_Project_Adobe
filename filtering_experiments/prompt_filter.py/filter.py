"""
Prompt-based filter script using PROMPT_FILTER_PROMPT from pipe_test.py
Judges whether natural language questions are valid based on SQL query, table information, and reasoning.
Uses the same model as pipe_test.py: Qwen/Qwen2.5-3B-Instruct via run_llm
"""
import os
import sys
import json
import pandas as pd
from tqdm import tqdm

# Add project root to path for relative imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

# Import the prompt and run_llm function from pipe_test.py (same as used in pipe_test.py)
from data_gen.pipelines.length_pipeline.prompts import PROMPT_FILTER_PROMPT
from data_gen.data_gen import run_llm

# I forgot to save the reasoning from the generated data, since it would take significant time to regenerate and rehuman evaluate, I created a version of the prompt that works without reasoning
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




def parse_score_from_response(response: str) -> float:
    """Parse score from LLM response (JSON format). More robust to handle multiple JSON objects."""
    try:
        if not response or len(response.strip()) == 0:
            return 0.0
        
        cleaned_response = response.strip()
        
        # Remove all markdown code block markers
        cleaned_response = cleaned_response.replace("```json", "").replace("```", "")
        cleaned_response = cleaned_response.strip()
        
        # If there are multiple JSON objects, extract all of them and use the first valid one
        # Find all JSON objects in the response
        json_objects = []
        start_idx = 0
        
        while True:
            # Find next opening brace
            first_brace = cleaned_response.find('{', start_idx)
            if first_brace == -1:
                break
            
            # Find matching closing brace
            brace_count = 0
            last_brace = -1
            for i in range(first_brace, len(cleaned_response)):
                if cleaned_response[i] == '{':
                    brace_count += 1
                elif cleaned_response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_brace = i
                        break
            
            if last_brace != -1:
                json_str = cleaned_response[first_brace:last_brace + 1]
                json_objects.append(json_str)
                start_idx = last_brace + 1
            else:
                break
        
        # Try to parse each JSON object until we find a valid one with a score
        for json_str in json_objects:
            try:
                score_data = json.loads(json_str)
                score = score_data.get("score", None)
                
                if score is not None:
                    # Ensure score is a number between 0 and 1
                    try:
                        score = float(score)
                        score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
                        return score
                    except (ValueError, TypeError):
                        continue
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, try the original method (extract first complete JSON)
        first_brace = cleaned_response.find('{')
        if first_brace != -1:
            # Find matching closing brace
            brace_count = 0
            last_brace = -1
            for i in range(first_brace, len(cleaned_response)):
                if cleaned_response[i] == '{':
                    brace_count += 1
                elif cleaned_response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_brace = i
                        break
            
            if last_brace != -1:
                json_str = cleaned_response[first_brace:last_brace + 1]
                score_data = json.loads(json_str)
                score = score_data.get("score", 0.0)
                score = float(score)
                score = max(0.0, min(1.0, score))
                return score
        
        return 0.0
        
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        # Silently return 0.0 instead of printing (to reduce noise)
        return 0.0


def format_table_data(table_data_json: str) -> dict:
    """Format table data from JSON string to dict format expected by prompt."""
    try:
        table_data = json.loads(table_data_json)
        return table_data
    except:
        return {}


def filter_data(data, score_threshold=0.9):
    """Filter data using prompt-based scoring."""
    filtered_scores = []
    prompt_scores = []
    
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Filtering"):
        try:
            # Get table data
            table_data = {}
            if pd.notna(row.get("table_data")) and str(row["table_data"]).strip():
                table_data = format_table_data(row["table_data"])
            
            # Get required fields
            nl_question = str(row["nl_questions"]) if pd.notna(row.get("nl_questions")) else ""
            sql_query = str(row["sql_queries"]) if pd.notna(row.get("sql_queries")) else ""
            
            # Format the prompt filter (using version without reasoning requirement)
            try:
                prompt_filter = PROMPT_FILTER_PROMPT_NO_REASONING.format(
                    sql_query=sql_query,
                    nl_question=nl_question,
                    table=table_data
                )
            except (KeyError, ValueError, AttributeError) as e:
                print(f"  Error formatting prompt filter for row {index}: {e}")
                filtered_scores.append(0)
                prompt_scores.append(0.0)
                continue
            
            # Call LLM to get score (using same run_llm as pipe_test.py)
            try:
                response = run_llm(prompt_filter)
                if response is None:
                    raise ValueError("LLM returned None response")
                if not isinstance(response, str):
                    response = str(response)
                    
                score = parse_score_from_response(response)
                prompt_scores.append(score)
                
                # Filter based on score threshold (same as pipe_test.py uses 0.8)
                if score >= score_threshold:
                    filtered_scores.append(1)
                else:
                    filtered_scores.append(0)
                    
            except Exception as e:
                print(f"  Error calling LLM for row {index}: {e}")
                filtered_scores.append(0)
                prompt_scores.append(0.0)
                continue
            
        except Exception as e:
            print(f"  Unexpected error for row {index}: {e}")
            filtered_scores.append(0)
            prompt_scores.append(0.0)
            continue
    
    # Create results dataframe with new columns
    return_df = data.copy()
    return_df["prompt_filtered_scores"] = filtered_scores
    return_df["prompt_scores"] = prompt_scores
    
    return return_df


if __name__ == "__main__":
    # Configuration - use paths relative to project root
    test_set_path = os.path.join(project_root, "data/human_eval_filtering_test.csv")
    output_path = os.path.join(project_root, "filtering_experiments/prompt_filter.py/prompt_filtered_scores.csv")
    
    print(f"🤖 Using Prompt-based Filtering")
    print(f"📋 Using PROMPT_FILTER_PROMPT from pipe_test.py")
    print(f"🤖 Using Qwen/Qwen2.5-3B-Instruct (same as pipe_test.py)")
    
    # Load test set
    print(f"📂 Loading test set from {test_set_path}...")
    test_set = pd.read_csv(test_set_path)
    print(f"✅ Loaded {len(test_set)} examples")
    
    # Filter data (using 0.8 threshold like pipe_test.py)
    print("🚀 Starting prompt-based filtering...")
    print(f"   Score threshold: 0.8 (same as pipe_test.py)")
    filtered_df = filter_data(test_set, score_threshold=0.8)
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path}")
    
    # Print summary
    total = len(filtered_df)
    correct = filtered_df["prompt_filtered_scores"].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_score = filtered_df["prompt_scores"].mean() if total > 0 else 0
    
    print(f"\n📊 Summary:")
    print(f"   Total examples: {total}")
    print(f"   Filtered (accepted): {correct}")
    print(f"   Rejected: {total - correct}")
    print(f"   Acceptance rate: {accuracy:.2f}%")
    print(f"   Average prompt score: {avg_score:.4f}")

