import pandas as pd
from fuzzywuzzy import fuzz
import sqlite3
import re
from typing import Dict, List, Tuple, Set
import sqlglot
import ast
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def evaluate_sql_query_exact_match(gold_sql_query: str, generated_sql_query: str) -> float:
    """
    Evaluate the gold SQL query against the generated SQL query, this is a simple exact match evaluation
    Input - Gold SQL query and the generated SQL query
    Output - Evaluation of the SQL query produced by the model
    """
    if gold_sql_query == generated_sql_query:
        return 1.0
    else:
        return 0.0

def evaluate_sql_query_relaxed_match(gold_sql_query: str, generated_sql_query: str, table: str = "") -> float:
    """
    FROM PAPER: https://arxiv.org/html/2312.10321v2
    The one shared by Kun
    Modivation: You can change parts of a SQL query which will affect what it is retriving, but won't have a strong semantic impact
    For example, changing the order of the columns in the SELECT clause will not change the result of the query, but will change the order of the columns in the result. 
    Also, changing from LIMIT 10 to LIMIT 20 will not change the result of the query, but will change the number of rows in the result.
    """
    prompt = """
    Instruction:
    You are evaluating SQL queries for "relaxed equivalence".

    DEFINITION: Two SQL queries Q1 and Q2 are relaxed equivalent if one may edit a TRIVIAL portion 
    of the SQL queries while preserving the same logic to make Q1 and Q2 semantically equivalent.

    CRITICAL: Relaxed equivalence asks "CAN these queries be made equivalent with trivial edits?"
    NOT "ARE these queries currently semantically equivalent?"

    If the queries share the SAME LOGIC but differ only in how they reference tables or columns,
    and those differences can be resolved with trivial edits (like changing which table a column
    is selected from when a foreign key relationship exists), then they ARE relaxed equivalent.

    TRIVIAL EDITS:
    - Column order in SELECT
    - Formatting / whitespace
    - LIMIT or OFFSET values
    - Table aliases
    - FROM A, B vs FROM A JOIN B with same join condition

    SIGNIFICANT DIFFERENCES:
    - Different WHERE predicates
    - Different GROUP BY affecting grouping
    - JOIN condition changes logic
    - Missing/additional tables changing logic
    - Different aggregates
    - HAVING clause differences

    EXAMPLE (relaxed equivalent):
    Q1: SELECT Supply.pnum FROM Supply WHERE shipdate < 10 GROUP BY pnum
    Q2: SELECT Parts.pnum FROM Parts, Supply WHERE shipdate < 10 AND Parts.pnum = Supply.pnum

    SQL Query 1:
    {gold_sql_query}

    SQL Query 2:
    {generated_sql_query}

    Database Schema (primary/foreign keys):
    {table}

    Return the result strictly as JSON:
    {{
        "explanation_query_1": "...",
        "explanation_query_2": "...",
        "comparison_logical_differences": "...",
        "relaxed_equivalent": 0 or 1
    }}

    """
    
    prompt_formatted = prompt.format(gold_sql_query=gold_sql_query, generated_sql_query=generated_sql_query, table=table)
    
    # Get OpenAI API token from environment
    openai_token = os.getenv("OPENAI_API_KEY")
    if not openai_token:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please set OPENAI_API_KEY in your .env file.")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_token)
    
    try:
        # Make API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o, can be changed to other models
            messages=[
                {"role": "system", "content": "You are an expert SQL evaluator. Analyze SQL queries and determine if they are semantically equivalent, ignoring trivial differences."},
                {"role": "user", "content": prompt_formatted}
            ],
            response_format={"type": "json_object"},
            temperature=0.0  # Use deterministic output
        )
        
        # Extract response content
        response_content = response.choices[0].message.content.strip()
        
        # Robust JSON parsing with multiple fallback strategies
        result = None
        
        # Strategy 1: Try direct JSON parsing
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            # Strategy 2: Extract JSON from markdown code blocks
            import re
            # Match ```json ... ``` or ``` ... ``` with JSON inside
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Strategy 3: Find JSON object in the text (handles nested objects)
            if result is None:
                # More robust nested JSON extraction
                brace_count = 0
                start_idx = -1
                for i, char in enumerate(response_content):
                    if char == '{':
                        if brace_count == 0:
                            start_idx = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_idx != -1:
                            try:
                                json_str = response_content[start_idx:i+1]
                                result = json.loads(json_str)
                                break
                            except json.JSONDecodeError:
                                continue
            
            # Strategy 4: Try to extract just the relaxed_equivalent value directly
            if result is None:
                # Look for "relaxed_equivalent": 0 or 1 or "relaxed_equivalent": "0" or "1"
                relaxed_match = re.search(r'"relaxed_equivalent"\s*:\s*([01]|"[01]")', response_content)
                if relaxed_match:
                    value = relaxed_match.group(1).strip('"')
                    result = {"relaxed_equivalent": int(value)}
                else:
                    # Try without quotes
                    relaxed_match = re.search(r'relaxed_equivalent\s*[:=]\s*([01])', response_content, re.IGNORECASE)
                    if relaxed_match:
                        result = {"relaxed_equivalent": int(relaxed_match.group(1))}
        
        # Extract the relaxed_equivalent value with multiple fallback strategies
        if result:
            # Try direct key access
            relaxed_equivalent = result.get("relaxed_equivalent")
            
            # If not found, try case-insensitive key matching
            if relaxed_equivalent is None:
                for key, value in result.items():
                    if key.lower() == "relaxed_equivalent":
                        relaxed_equivalent = value
                        break
            
            # Convert to int/float if it's a string or boolean
            if isinstance(relaxed_equivalent, bool):
                relaxed_equivalent = 1 if relaxed_equivalent else 0
            elif isinstance(relaxed_equivalent, str):
                relaxed_equivalent = relaxed_equivalent.strip()
                if relaxed_equivalent.lower() in ['true', '1', 'yes', 'equivalent', 'same']:
                    relaxed_equivalent = 1
                elif relaxed_equivalent.lower() in ['false', '0', 'no', 'not equivalent', 'different']:
                    relaxed_equivalent = 0
                else:
                    try:
                        relaxed_equivalent = int(relaxed_equivalent)
                    except ValueError:
                        relaxed_equivalent = 0
            elif isinstance(relaxed_equivalent, (int, float)):
                # Already numeric, just ensure it's 0 or 1
                relaxed_equivalent = 1 if relaxed_equivalent else 0
            
            # Ensure it's 0 or 1 (final check)
            if relaxed_equivalent is None:
                relaxed_equivalent = 0
            else:
                try:
                    relaxed_equivalent = 1 if int(relaxed_equivalent) else 0
                except (ValueError, TypeError):
                    relaxed_equivalent = 0
            
            return float(relaxed_equivalent)
        else:
            # If all parsing strategies failed, try to infer from response text
            response_lower = response_content.lower()
            # Look for positive indicators
            positive_indicators = ['equivalent', 'same', 'similar', 'no significant difference', 'semantically equivalent', 'relaxed equivalent']
            negative_indicators = ['not equivalent', 'different', 'significant difference', 'not the same', 'not similar']
            
            has_positive = any(word in response_lower for word in positive_indicators)
            has_negative = any(word in response_lower for word in negative_indicators)
            
            if has_positive and not has_negative:
                print(f"⚠️  Could not parse JSON, but inferred equivalent from text. Response: {response_content[:300]}")
                return 1.0
            elif has_negative:
                print(f"⚠️  Could not parse JSON, but inferred not equivalent from text. Response: {response_content[:300]}")
                return 0.0
            else:
                print(f"⚠️  Could not parse JSON response and could not infer from text. Response: {response_content[:300]}")
                return None
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        if 'response_content' in locals():
            print(f"Response content: {response_content[:500]}")
        return None
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        if 'response_content' in locals():
            print(f"Response content: {response_content[:500] if 'response_content' in locals() else 'N/A'}")
        return None
        
def evaluate_sql_query_fuzzy_match(gold_sql_query: str, generated_sql_query: str) -> float:
    """
    Evaluate the gold SQL query against the generated SQL query, with a fuzzy match evaluation
    Input - Gold SQL query and the generated SQL query
    Output - Evaluation of the SQL query produced by the model
    """
    return fuzz.partial_ratio(gold_sql_query, generated_sql_query) / 100

def extract_field_names(sql_query: str) -> Set[str]:
    """
    Extract field names from SQL query using sqlglot
    """
    parsed_query = sqlglot.parse_one(sql_query, read="sqlite")
    return {col.name for col in parsed_query.find_all(sqlglot.exp.Column)}

def fuzzy_field_match(field1: str, field2: str) -> bool:
    """
    Implement fuzzy matching for field names based on the MultiSQL paper criteria:
    1. One field name is contained within the other
    2. Parts split by underscores are found in the other field
    """
    field1_lower = field1.lower()
    field2_lower = field2.lower()
    
    # Exact match
    if field1_lower == field2_lower:
        return True
    
    # One field contains the other
    if field1_lower in field2_lower or field2_lower in field1_lower:
        return True
    
    # Split by underscores and check if parts match
    field1_parts = field1_lower.split('_')
    field2_parts = field2_lower.split('_')
    
    # Check if any part of field1 is in field2
    for part in field1_parts:
        if part and part in field2_lower:
            return True
    
    # Check if any part of field2 is in field1
    for part in field2_parts:
        if part and part in field1_lower:
            return True
    
    return False

def create_field_mapping(gold_fields: Set[str], predicted_fields: Set[str]) -> Dict[str, str]:
    """
    Create a mapping dictionary between predicted and actual field names
    """
    mapping = {}
    used_gold_fields = set()
    
    # First pass: exact matches
    for pred_field in predicted_fields:
        for gold_field in gold_fields:
            if pred_field.lower() == gold_field.lower() and gold_field not in used_gold_fields:
                mapping[pred_field] = gold_field
                used_gold_fields.add(gold_field)
                break
    
    # Second pass: fuzzy matches
    for pred_field in predicted_fields:
        if pred_field not in mapping:  # Not already mapped
            for gold_field in gold_fields:
                if gold_field not in used_gold_fields and fuzzy_field_match(pred_field, gold_field):
                    mapping[pred_field] = gold_field
                    used_gold_fields.add(gold_field)
                    break
    
    return mapping

def evaluate_sql_query_context_aware(gold_sql_query: str, generated_sql_query: str) -> float:
    """
    Evaluate SQL queries using context-aware matching as described in MultiSQL paper
    """
    # Extract field names from both queries
    gold_fields = extract_field_names(gold_sql_query)
    predicted_fields = extract_field_names(generated_sql_query)
    
    if not gold_fields and not predicted_fields:
        # No fields to compare, check if queries are structurally similar
        return 1.0 if gold_sql_query.lower().strip() == generated_sql_query.lower().strip() else 0.0
    
    if not gold_fields or not predicted_fields:
        return 0.0
    
    # Create field mapping
    field_mapping = create_field_mapping(gold_fields, predicted_fields)
    
    # Check if all gold fields are mapped and all predicted fields are mapped
    if len(field_mapping) == len(gold_fields) and len(field_mapping) == len(predicted_fields):
        # All fields are mapped, check if mappings are valid (fuzzy matches are acceptable)
        return 1.0
    else:
        return 0.0

def evaluate_sql_query_with_database(gold_sql_query: str, generated_sql_query: str, table: str) -> float:
    """
    Evaluate the gold SQL query against the generated SQL query, by comparing the result of the query against the database
    Input - Gold SQL query and the generated SQL query and the database
    Output - Evaluation of the SQL query produced by the model
    """

    # Convert the SQL query to the database format
    print("Gold SQL query: ", gold_sql_query)
    print("Generated SQL query: ", generated_sql_query)
    gold_sql_clean = convert_sql_query(table, gold_sql_query)
    generated_sql_clean = convert_sql_query(table, generated_sql_query)

    try:
        conn = sqlite3.connect(f"data/train.db")
        cur = conn.cursor()
        
        # Execute gold query
        cur.execute(gold_sql_clean)
        gold_result = cur.fetchall()
        
        # Execute generated query
        cur.execute(generated_sql_clean)
        generated_result = cur.fetchall()
        
        conn.close()
        if gold_result == generated_result:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        print(f"Database evaluation error: {e}")
        return 0.0

def convert_sql_query(table: str, sql_query: str) -> str:
    conversion = {table["header"][i]: "col" + str(i) for i in range(0, len(table["header"]))}
    for key in sorted(conversion, key=len, reverse=True):
        try:
            matches = re.findall(r"`([^`]*)`", sql_query)
            for m in matches:
                for original, replacement in conversion.items():
                    if m.strip().lower() == original.strip().lower():
                        sql_query = sql_query.replace(f"`{m}`", replacement)
            
        except re.error as e:
            print(f"Regex error for key '{key}': {e}")
            continue  # Skip this pattern and continue with the next one
    return sql_query

if __name__ == "__main__":
    data = pd.read_csv("data_gen/order_experiment/generated_sql_nl.csv")

    exact_matches = 0
    fuzzy_matches = 0
    context_aware_matches = 0
    total_samples = len(data)
    database_matches = 0

    print(f"Evaluating {total_samples} samples...")
    
    for index, row in data.iterrows():
        gold_sql_query = row["actual_sql"]
        generated_sql_query = row["sql"]
        table = ast.literal_eval(row["table_info"])
        
        # Exact match evaluation
        exact_matches += evaluate_sql_query_exact_match(gold_sql_query, generated_sql_query)
        
        # Fuzzy match evaluation (threshold > 0.9)
        if evaluate_sql_query_fuzzy_match(gold_sql_query, generated_sql_query) > 0.9:
            fuzzy_matches += 1
        
        # Context-aware match evaluation
        context_aware_matches += evaluate_sql_query_context_aware(gold_sql_query, generated_sql_query)

        # Database match evaluation
        database_matches += evaluate_sql_query_with_database(gold_sql_query, generated_sql_query, table) == 1.0
        
        # Print progress every 100 samples
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{total_samples} samples...")

    print("\n--- Evaluation Results ---")
    print(f"Total samples: {total_samples}")
    print(f"Exact matches: {exact_matches} ({exact_matches/total_samples*100:.2f}%)")
    print(f"Fuzzy matches (>0.9): {fuzzy_matches} ({fuzzy_matches/total_samples*100:.2f}%)")
    print(f"Context-aware matches: {context_aware_matches} ({context_aware_matches/total_samples*100:.2f}%)")
    print(f"Database matches: {database_matches} ({database_matches/total_samples*100:.2f}%)")
