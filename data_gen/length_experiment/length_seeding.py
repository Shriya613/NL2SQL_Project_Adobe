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
from reserved import reserved_word_patterns
from reserved_probs import get_length
from length_prompts import SQL_GEN_PROMPT, SQL_CORRECT_PROMPT, NL_GEN_PROMPT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from evaluation.evaluate import calculate_complexity

login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Qwen/Qwen2.5-3B-Instruct"
pipe = pipeline(model=model_name, device=device)
SMOOTHING_FACTOR = 10

def length_seed() -> int:
    """Sample a length seed from the length probabilities."""
    path = "data/train_bird.json"
    
    length_stats = get_length(path, reserved_word_patterns, SMOOTHING_FACTOR)
    
    # Extract probabilities and lengths for random.choices
    lengths = list(length_stats.keys())
    probabilities = list(length_stats.values())
    
    sampled_length = random.choices(lengths, weights=probabilities, k=1)[0]
    
    return sampled_length

def generate_sql_query(table: str, sampled_length: int, reserved_word_patterns: dict) -> str:
    """Generate SQL query based on table"""
    reserved_words_str = ", ".join(reserved_word_patterns.keys())
    prompt_formatted = SQL_GEN_PROMPT.format(table=table, length_seed=sampled_length, reserved_word_patterns=reserved_words_str)
    return prompt_formatted

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

def correct_sql_query(table: str, sql_query: str, error: str, sampled_length: int) -> str:
    prompt_formatted = SQL_CORRECT_PROMPT.format(sql_query=sql_query, error=error, table=table, length_seed=sampled_length)
    return prompt_formatted


def generate_nl(table: str, sql_query: str) -> str:
    prompt_formatted = NL_GEN_PROMPT.format(table=table, sql_query=sql_query)
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
        sampled_lengths = []
        conn = sqlite3.connect("data/train.db")
        cur = conn.cursor()

        for table in tqdm(table_data[:1000]):
            tries = 0

            # Generate length seed and examples
            sampled_length = length_seed()
            print("Sampled length: ", sampled_length)
            sql_prompt = generate_sql_query(table, sampled_length, reserved_word_patterns)
            sql_query = run_llm(sql_prompt)
            cleaned_sql = clean_output(sql_query, "sql")
            print("Cleaned SQL query: ", cleaned_sql)
            run_sql_query, db_ans, error = filter_sql_query(table, cleaned_sql, cur)

            while db_ans == None and tries < 3:
                print(f"Error: {error}, Tries: {tries}")
                tries += 1
                sql_prompt = correct_sql_query(table, cleaned_sql, error, sampled_length)
                sql_query = run_llm(sql_prompt)
                cleaned_sql = clean_output(sql_query, "sql")
                run_sql_query, db_ans, error = filter_sql_query(table, cleaned_sql, cur)
            
            if db_ans != None:
                table_info.append(table)
                database_ans.append(db_ans)
                actual_sql.append(run_sql_query)
                sampled_lengths.append(sampled_length)

                # Generate NL
                nl_prompt = generate_nl(table, cleaned_sql)
                nl_ques = run_llm(nl_prompt)
                cleaned_nl = clean_output(nl_ques, "question")
                print("Final Natural Language Question: ", cleaned_nl)

                print("Final SQL Query: ", cleaned_sql)
                generated_nl.append(cleaned_nl)
                generated_sql.append(cleaned_sql)
        
        pd.DataFrame({"table_info": table_info, "nl": generated_nl, "sql": generated_sql, "actual_sql": actual_sql, "ans": database_ans, "example_length": sampled_lengths}).to_csv("data_gen/length_experiment/length_seeding.csv", index=False)

def evaluate_pipeline():
    calculate_complexity("data_gen/length_experiment/length_seeding.csv", "actual_sql", "nl")

run_pipeline()
evaluate_pipeline()