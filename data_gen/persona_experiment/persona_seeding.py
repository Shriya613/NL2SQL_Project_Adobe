import sqlite3
import json
from tqdm import tqdm
from transformers import pipeline
import pandas as pd
import torch
import re
from huggingface_hub import login
import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from evaluation.evaluate import calculate_complexity
from persona_prompts import PERSONA_PROMPT_COMMON, PERSONA_PROMPT_OUT_OF_THE_BOX, SQL_GEN_PROMPT, NL_GEN_PROMPT, SQL_CORRECT_PROMPT

login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Qwen/Qwen2.5-3B-Instruct"
pipe = pipeline(model=model_name, device=device)

def get_persona_prompt(table: str) -> str:
    """Get the persona prompt based on the table"""
    prob = random.random()
    if prob < 0.3:
        return PERSONA_PROMPT_COMMON.format(table=table)
    else:
        return PERSONA_PROMPT_OUT_OF_THE_BOX.format(table=table)

def generate_sql_query(table: str, persona_desc: str, goals: str, example_queries: str) -> str:
    """Generate SQL query based on table"""
    prompt_formatted = SQL_GEN_PROMPT.format(table=table, persona=persona_desc, goals=goals, example_queries=example_queries)
    return prompt_formatted

def extract_persona_info(persona_response: str) -> tuple:
    """Extract persona information from the model response"""
    try:
        # Look for structured persona information
        lines = persona_response.split('\n')
        persona_desc = ""
        goals = ""
        example_queries = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('**Persona description**') or line.startswith('1. **Persona description**'):
                current_section = "persona"
                continue
            elif line.startswith('**Goals**') or line.startswith('2. **Goals**'):
                current_section = "goals"
                continue
            elif line.startswith('**Example queries**') or line.startswith('3. **Example queries**'):
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
        return persona_response[:500], "", ""

def filter_sql_query(table: str, sql_query: str, cur: sqlite3.Cursor) -> tuple:
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
    print("SQL query after conversion: ", sql_query)
    try:
        res = cur.execute(sql_query)
        result = res.fetchall()
        print("Result: ", result)
        if not result:
            print("Query returned empty result")
            return sql_query, None, "No result found"

        return sql_query, result, None
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return sql_query, None, e

def correct_sql_query(table: str, sql_query: str, error: str, persona: str) -> str:
    prompt_formatted = SQL_CORRECT_PROMPT.format(sql_query=sql_query, error=error, table=table, persona=persona)
    return prompt_formatted


def generate_nl(table: str, sql_query: str, persona: str) -> str:
    prompt_formatted = NL_GEN_PROMPT.format(table=table, sql_query=sql_query, persona=persona)
    return prompt_formatted


def run_llm(prompt: str) -> str:
    # Run the LLM using prompt and return the generated text
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, top_k=10, return_full_text=False)
    for output in outputs:
        gen_text = output["generated_text"]
    return gen_text

def clean_output(output: str, tag: str) -> str:
    if len(output.split(f"```{tag}")) > 1:
        return output.split(f"```{tag}")[-1].split("```")[0]
    elif len(output.split(f"```text")) > 1:
        return output.split(f"```text")[-1].split("```")[0]
    else:
        return output

def run_pipeline():
    table_data = []
    with open("data/train.tables.jsonl", "r") as f:
        for line in f:
            table = json.loads(line)
            table["id"] = "table_" + table["id"].replace("-", "_")
            table["name"] = table["id"]
            table_data.append(table)

        generated_sql = []
        generated_nl = []
        database_ans = []
        table_info = []
        actual_sql = []
        personas = []

        conn = sqlite3.connect("data/train.db")
        cur = conn.cursor()

        for table in tqdm(table_data[:1000]):
            tries = 0

            # Generate persona
            persona_prompt = get_persona_prompt(table)
            persona_response = run_llm(persona_prompt)
            persona_desc, goals, example_queries = extract_persona_info(persona_response)

            sql_prompt = generate_sql_query(table, persona_desc, goals, example_queries)
            sql_query = run_llm(sql_prompt)

            cleaned_sql = clean_output(sql_query, "sql")
            run_sql_query, db_ans, error = filter_sql_query(table, cleaned_sql, cur)

            while db_ans == None and tries < 3:
                print(f"Error: {error}, Tries: {tries}")
                tries += 1
                sql_prompt = correct_sql_query(table, cleaned_sql, error, persona_desc)
                sql_query = run_llm(sql_prompt)
                cleaned_sql = clean_output(sql_query, "sql")
                run_sql_query, db_ans, error = filter_sql_query(table, cleaned_sql, cur)
            
            if db_ans != None:
                table_info.append(table)
                database_ans.append(db_ans)
                actual_sql.append(run_sql_query)
                personas.append(persona_desc)
                print("Persona: ", persona_desc)

                # Generate NL
                nl_prompt = generate_nl(table, cleaned_sql, persona_desc)
                nl_ques = run_llm(nl_prompt)
                cleaned_nl = clean_output(nl_ques, "question")
                print("Final Natural Language Question: ", cleaned_nl)

                print("Final SQL Query: ", cleaned_sql)
                generated_nl.append(cleaned_nl)
                generated_sql.append(cleaned_sql)
        
        pd.DataFrame({"table_info": table_info, "nl": generated_nl, "sql": generated_sql, "actual_sql": actual_sql, "ans": database_ans, "personas": personas}).to_csv("data_gen/persona_experiment/persona_seeding.csv", index=False)

def evaluate_pipeline():
    calculate_complexity("data_gen/persona_experiment/persona_seeding.csv", "actual_sql", "nl")

run_pipeline()
evaluate_pipeline()