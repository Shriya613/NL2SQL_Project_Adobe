import os
import re
import random
import sqlite3
import torch
import json
import dotenv
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from huggingface_hub import login, HfFolder
from data_gen.final_gen.reserved import reserved_word_patterns
from data_gen.final_gen.reserved_probs import get_length
from data_gen.final_gen.prompts import (
    SQL_GEN_RESERVED_PROMPT, 
    SQL_CORRECT_RESERVED_PROMPT, 
    SQL_GEN_PERSONA_PROMPT,
    PERSONA_PROMPT_COMMON,
    PERSONA_PROMPT_OUT_OF_THE_BOX,
    SQL_CORRECT_PERSONA_PROMPT,
    MULTIPLE_SQL_GEN_PROMPT,
    MULTIPLE_SQL_CORRECT_PROMPT
)
from data_gen.final_gen.utils.llm import run_llm_gemini
import google.genai as genai
dotenv.load_dotenv()

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not hf_token:
    try:
        if not HfFolder.get_token():
            login()
    except Exception:
        login()

pipe = genai.Client()

def length_seed(SMOOTHING_FACTOR: int = 10) -> int:
    """Sample a number of reserved words seed from the probabilities."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    path = os.path.join(project_root, "data/train_bird.json")
    length_stats = get_length(path, reserved_word_patterns, SMOOTHING_FACTOR)
    
    lengths = list(length_stats.keys())
    probs = list(length_stats.values())
    
    return random.choices(lengths, weights=probs, k=1)[0]

def get_persona_prompt(table: str) -> str:
    """Get the persona prompt based on the table"""
    prob = random.random()
    if prob < 0.3:
        return PERSONA_PROMPT_COMMON.format(table=table)
    else:
        return PERSONA_PROMPT_OUT_OF_THE_BOX.format(table=table)

def generate_sql_reserved_query(table: str, sampled_length: int, reserved_word_patterns: dict) -> str:
    """Generate SQL query based on table"""
    reserved_words_str = ", ".join(reserved_word_patterns.keys())
    return SQL_GEN_RESERVED_PROMPT.format(
        table=table,
        length_seed=sampled_length,
        reserved_word_patterns=reserved_words_str
        )

def generate_sql_persona_query(table: str, persona_desc: str, goals: str, example_queries: str) -> str:
    """Generate SQL query based on table"""
    prompt_formatted = SQL_GEN_PERSONA_PROMPT.format(
        table=table, 
        persona=persona_desc, 
        goals=goals, 
        example_queries=example_queries,
        )
    return prompt_formatted

def extract_persona_info(persona_response: str) -> tuple:
    """Extract persona information from the model response"""
    import json
    import re
    
    try:
        # First, try to extract JSON from code blocks
        json_block = None
        if "```json" in persona_response:
            json_block = persona_response.split("```json")[1].split("```")[0].strip()
        elif "```" in persona_response:
            # Try to find JSON in any code block
            code_blocks = re.findall(r'```(?:json)?\s*(.*?)```', persona_response, re.DOTALL)
            for block in code_blocks:
                block = block.strip()
                if block.startswith('{') and block.endswith('}'):
                    json_block = block
                    break
        
        # If we found a JSON block, parse it
        if json_block:
            # Remove trailing commas before closing braces/brackets (common LLM error)
            json_block = re.sub(r',(\s*[}\]])', r'\1', json_block)
            try:
                data = json.loads(json_block)
                # Extract fields with various possible key names based on updated prompts
                persona_desc = data.get("persona_description", "") or data.get("SHORT Persona description", "") or data.get("Persona description", "") or data.get("persona", "")
                goals = data.get("goals", "") or data.get("Goals", "")
                example_queries = data.get("example_queries", "") or data.get("Example queries", "")
                
                # Handle goals and example_queries if they're lists
                if isinstance(goals, list):
                    goals = " ".join(str(g) for g in goals)
                if isinstance(example_queries, list):
                    example_queries = " ".join(str(q) for q in example_queries)
                
                if persona_desc or goals:
                    return persona_desc.strip(), str(goals).strip(), str(example_queries).strip()
            except json.JSONDecodeError as e:
                print(f"⚠️  Failed to parse JSON: {e}")
                print(f"   JSON block: {json_block[:200]}...")
        
        # Fallback: Try to parse markdown-style headers (legacy support)
        lines = persona_response.split('\n')
        persona_desc = ""
        goals = ""
        example_queries = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('**Persona description**') or line.startswith('1. **Persona description**') or 'SHORT Persona description' in line or 'persona_description' in line:
                current_section = "persona"
                continue
            elif line.startswith('**Goals**') or line.startswith('2. **Goals**') or 'goals' in line:
                current_section = "goals"
                continue
            elif line.startswith('**Example queries**') or line.startswith('3. **Example queries**') or 'example_queries' in line:
                current_section = "queries"
                continue

            if current_section and line and not line.startswith('**'):
                if current_section == "persona":
                    persona_desc += line + " "
                elif current_section == "goals":
                    goals += line + " "
                elif current_section == "queries":
                    example_queries += line + " "

        # If structured parsing failed, use the full response
        if not persona_desc and not goals:
            persona_desc = persona_response[:500]  # First 500 chars as fallback
        
        return persona_desc.strip(), goals.strip(), example_queries.strip()
    except Exception as e:
        print(f"⚠️  Error extracting persona info: {e}")
        return persona_response[:500], "", ""

def filter_sql_query(table: str, sql_query: str, cur: sqlite3.Cursor) -> tuple:
    """Filter SQL query based on execution against the database"""
    """ Note: Darian made some edits cause we are now allowing for table name changes, which would cause the tables names to need to be changed here too"""
    # Get the actual database table name - check what table actually exists in the DB
    # Try the name from table dict first, but also check for variations
    potential_names = [
        table.get("name", table.get("id", "unknown")),
        table.get("id", "unknown"),
        table.get("name", "").replace("table_", "table_table_") if table.get("name", "").startswith("table_") else "",
        table.get("id", "").replace("table_", "table_table_") if table.get("id", "").startswith("table_") else "",
    ]
    
    # Check which table actually exists in the database
    db_table_name = potential_names[0]  # Default to first option
    for name in potential_names:
        if name:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
            if cur.fetchone():
                db_table_name = name
                break
    
    # Convert column names to placeholders
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
    
    # Extract CTE names to avoid replacing them
    cte_names = set()
    # Match: WITH cte_name AS ( or WITH cte1 AS (...), cte2 AS (...)
    with_matches = re.finditer(r'WITH\s+([^()]+?)\s+AS\s*\(', sql_query, re.IGNORECASE | re.DOTALL)
    for match in with_matches:
        cte_def = match.group(1).strip()
        # Handle multiple CTEs: "cte1 AS (...), cte2 AS (...)"
        for cte_part in cte_def.split(','):
            cte_part = cte_part.strip()
            # Extract CTE name (before AS if present, or the whole part)
            if ' AS ' in cte_part.upper():
                cte_name = cte_part.split(' AS ', 1)[0].strip().strip('`')
            else:
                cte_name = cte_part.strip().strip('`')
            if cte_name:
                cte_names.add(cte_name.lower())
    
    # Convert table names - replace table references in FROM/JOIN clauses with actual DB table name
    # Match: FROM `table_name` or FROM table_name or JOIN `table_name` etc.
    # Only replace if it's not already the correct table name, not a column name, and not a CTE name
    def replace_table_name(match):
        keyword = match.group(1)  # FROM, JOIN, etc.
        table_ref = match.group(2).strip('`')  # Remove backticks if present
        
        # Don't replace if it's already the correct table name, a column name, or a CTE name
        if table_ref == db_table_name or table_ref in conversion or table_ref.lower() in cte_names:
            return match.group(0)
        return f"{keyword} `{db_table_name}`"
    
    sql_query = re.sub(r"(FROM|JOIN)\s+`?([^`\s,()]+)`?", replace_table_name, sql_query, flags=re.IGNORECASE)
    print("SQL query after table name conversion: ", sql_query)
    try:
        res = cur.execute(sql_query)
        result = res.fetchall()
        if not result:
            print("Query returned empty result")
            return sql_query, None, "No result found"

        return sql_query, result, None
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return sql_query, None, e

def correct_sql_reserved_query(table: str, sql_query: str, error: str, sampled_length: int) -> str:
    return SQL_CORRECT_RESERVED_PROMPT.format(
        sql_query=sql_query,
        error=error,
        table=table,
        length_seed=sampled_length
        )

def correct_sql_persona_query(table: str, sql_query: str, error: str, persona: str, goals: str, example_queries: str) -> str:
    prompt_formatted = SQL_CORRECT_PERSONA_PROMPT.format(
        sql_query=sql_query, 
        error=error, 
        table=table, 
        persona=persona,
        goals=goals,
        example_queries=example_queries
        )
    return prompt_formatted

def clean_output(output: str, tag: str) -> str:
    """Clean the output of the LLM"""
    print(f"Output: {output}")
    if f"```{tag}" in output:
        cleaned = output.split(f"```{tag}", 1)[1].split("```")[0]
    elif f"```text" in output:
        cleaned = output.split(f"```text", 1)[1].split("```")[0]
    elif f"```json" in output:
        cleaned = output.split(f"```json", 1)[1].split("```")[0].strip()
        # Try to parse structured JSON first
        try:
            import json
            parsed = json.loads(cleaned)
            # New format with explicit fields
            if isinstance(parsed, dict):
                if "sql_query" in parsed:
                    return str(parsed["sql_query"]).strip()
                # Backwards compatibility if older key name is used
                if "query" in parsed:
                    return str(parsed["query"]).strip()
        except Exception:
            # Fall back to the old heuristic parsing below
            pass

        # Legacy fallback: look for a "query:" field in the JSON-like text
        if "query:" in cleaned:
            cleaned = cleaned.split("query:", 1)[1].strip().strip('"')
    else:
        cleaned = output
    return cleaned

def sql_gen_reserved_pipeline(table: str, cur: sqlite3.Cursor) -> tuple[str, str]:
    """Full NL to SQL generation, cleaning, correction, and validation"""
    sampled_num = length_seed()
    sql_prompt = generate_sql_reserved_query(table, sampled_num, reserved_word_patterns)
    raw_sql = run_llm_gemini(pipe, sql_prompt)

    cleaned_sql = clean_output(raw_sql, "sql")
    run_sql_query, db_ans, error = filter_sql_query(table, cleaned_sql, cur)
    
    tries = 0
    while (db_ans == None or db_ans == []) and tries < 3:
        print(f"Error: {error}, Tries: {tries}")
        tries += 1
        sql_prompt_retry = correct_sql_reserved_query(table, cleaned_sql, error, sampled_num)
        raw_sql_retry = run_llm_gemini(pipe, sql_prompt_retry)
        cleaned_sql_retry = clean_output(raw_sql_retry, "sql")
        run_sql_query, db_ans, error = filter_sql_query(table, cleaned_sql_retry, cur)
    
    # Persist runnable SQL query for evaluation
    if db_ans != None:
        return cleaned_sql, run_sql_query, sampled_num, db_ans
    else:
        return None, None, None, None

def sql_gen_persona_pipeline(table: str, cur: sqlite3.Cursor, persona: str, goals: str, example_queries: str) -> tuple[str, str]:
    """Full NL to SQL generation, cleaning, correction, and validation"""
    sql_prompt = generate_sql_persona_query(table, persona, goals, example_queries)
    raw_sql = run_llm_gemini(pipe, sql_prompt)

    cleaned_sql = clean_output(raw_sql, "sql")
    run_sql_query, db_ans, error = filter_sql_query(table, cleaned_sql, cur)
    
    tries = 0
    while (db_ans == None or db_ans == []) and tries < 3:
        print(f"Error: {error}, Tries: {tries}")
        tries += 1
        sql_prompt_retry = correct_sql_persona_query(table, cleaned_sql, error, persona, goals, example_queries)
        raw_sql_retry = run_llm_gemini(pipe, sql_prompt_retry)
        cleaned_sql_retry = clean_output(raw_sql_retry, "sql")
        run_sql_query, db_ans, error = filter_sql_query(table, cleaned_sql_retry, cur)
    
    if db_ans != None:
        return cleaned_sql, run_sql_query, db_ans
    else:
        return None, None, None

def sql_gen_multiple_pipeline(table: str, cur: sqlite3.Cursor, nl_question: str, first_sql_query: str):
    """Generate multiple SQL query based on table"""
    sql_prompt = generate_multiple_sql_query(table, nl_question, first_sql_query)
    raw_sql = run_llm_gemini(pipe, sql_prompt)

    # Try to parse reasoning + sql_query from JSON, but fall back gracefully
    reasoning = None
    cleaned_sql = None
    if "```json" in raw_sql:
        json_block = raw_sql.split("```json", 1)[1].split("```")[0].strip()
        try:
            parsed = json.loads(json_block)
            if isinstance(parsed, dict):
                reasoning = parsed.get("reasoning")
                sql_from_json = parsed.get("sql_query") or parsed.get("query")
                if sql_from_json:
                    cleaned_sql = str(sql_from_json).strip()
        except Exception:
            pass

    # If JSON parsing failed or no sql found, fall back to generic cleaner
    if not cleaned_sql:
        cleaned_sql = clean_output(raw_sql, "sql")

    run_sql_query, db_ans, error = filter_sql_query(table, cleaned_sql, cur)

    tries = 0
    while (db_ans == None or db_ans == []) and tries < 3:
        print(f"Error: {error}, Tries: {tries}")
        tries += 1
        sql_prompt_retry = correct_multiple_sql_query(table, cleaned_sql, nl_question, error, first_sql_query)
        raw_sql_retry = run_llm_gemini(pipe, sql_prompt_retry)
        cleaned_sql_retry = clean_output(raw_sql_retry, "sql")
        run_sql_query, db_ans, error = filter_sql_query(table, cleaned_sql_retry, cur)
        # We keep the original reasoning for simplicity; retries are just fixes of the same idea.

    if db_ans != None and first_sql_query != cleaned_sql:
        return cleaned_sql, run_sql_query, reasoning, db_ans
    else:
        return None, None, None, None

def generate_multiple_sql_query(table: str, nl_question: str, first_sql_query: str) -> str:
    """Generate SQL query based on table"""
    prompt_formatted = MULTIPLE_SQL_GEN_PROMPT.format(table=table, nl_question=nl_question, first_sql_query=first_sql_query)
    return prompt_formatted

def correct_multiple_sql_query(table: str, sql_query: str, nl_question: str, error: str, first_sql_query: str) -> str:
    prompt_formatted = MULTIPLE_SQL_CORRECT_PROMPT.format(sql_query=sql_query, nl_question=nl_question, error=error, table=table, first_sql_query=first_sql_query)
    return prompt_formatted