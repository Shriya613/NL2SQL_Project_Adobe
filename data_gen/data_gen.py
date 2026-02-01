import sqlite3
import json
from tqdm import tqdm
from transformers import pipeline
import pandas as pd
import torch
import numpy as np
import os
from huggingface_hub import login
from dotenv import load_dotenv
from data_gen.sql_nl_gen import generate_sql_query, filter_sql_query, correct_sql_query, generate_nl

# Load environment variables
load_dotenv()

# Login to HuggingFace if token is available
# If HF_TOKEN is set in environment, HuggingFace Hub will use it automatically
# Only call login() if token is not set (will prompt interactively)
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not hf_token:
    # Check if already logged in by trying to access token file
    try:
        from huggingface_hub.utils import HfFolder
        if not HfFolder.get_token():
            login()  # Only prompt if not logged in
    except Exception:
        login()  # Fallback to interactive login
# If token is set, no need to call login() - HuggingFace Hub uses it automatically

def _select_best_device():
    """Select the CUDA device with the most free memory (fallback to CPU)."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(idx)
        except RuntimeError:
            free_mem, total_mem = 0, 0
        if free_mem > best_free:
            best_idx = idx
            best_free = free_mem
            best_total = total_mem

    gb = 1024 ** 3
    print(
        f"📈 Selected cuda:{best_idx} "
        f"({best_free / gb:.2f} / {best_total / gb:.2f} GiB free) for generation LLM"
    )
    return torch.device(f"cuda:{best_idx}")


device = _select_best_device()

model_name = "Qwen/Qwen2.5-3B-Instruct"
pipe = pipeline(model=model_name, device=device)

def run_llm(prompt: str) -> str:
    """Run the LLM using prompt and return the generated text"""
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, top_k=10, return_full_text=False)
    for output in outputs:
        gen_text = output["generated_text"]
    return gen_text

def clean_output(output: str, tag: str) -> str:
    """Clean the output of the LLM"""
    if len(output.split(f"```{tag}")) > 1:
        return output.split(f"```{tag}")[-1].split("```")[0]
    elif len(output.split(f"```text")) > 1:
        return output.split(f"```text")[-1].split("```")[0]
    elif "```json" in output:
        # Handle JSON outputs that may contain reasoning + sql_query
        json_block = output.split("```json", 1)[1].split("```")[0].strip()
        try:
            parsed = json.loads(json_block)
            if isinstance(parsed, dict):
                if "sql_query" in parsed:
                    return str(parsed["sql_query"]).strip()
                if "query" in parsed:
                    return str(parsed["query"]).strip()
        except Exception:
            pass
        # Fallback: just return the raw JSON block if parsing fails
        return json_block
    else:
        return output

def get_table_description(table):
    """Convert table dictionary to pandas DataFrame"""
    df = pd.DataFrame(table["rows"], columns=table["header"])
    print(df.columns)
    return df.describe()

def run_multiple_pipeline(table: str, nl_question: str, first_sql_query: str):

    table["id"] = "table_" + table["id"].replace("-", "_")
    table["name"] = table["id"]

    conn = sqlite3.connect("data/train.db")
    cur = conn.cursor()

    tries = 0

    sql_prompt = generate_multiple_sql_query(table, nl_question, first_sql_query)
    sql_query = run_llm(sql_prompt)

    cleaned_sql = clean_output(sql_query, "sql")
    db_ans, error = filter_sql_query(table, cleaned_sql, cur)

    while db_ans == None and tries < 3:
        tries += 1
        sql_prompt = correct_multiple_sql_query(table, cleaned_sql, nl_question, error, first_sql_query)
        sql_query = run_llm(sql_prompt)
        cleaned_sql = clean_output(sql_query, "sql")
        db_ans, error = filter_sql_query(table, cleaned_sql, cur)
        
    if db_ans != None and first_sql_query != cleaned_sql:
        return cleaned_sql
    else:
        return None

def main():
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
        conn = sqlite3.connect("data/train.db")
        cur = conn.cursor()

        for table in tqdm(table_data[:100]):
            table_description = get_table_description(table)
            tries = 0
            sql_prompt = generate_sql_query(table, table_description)
            sql_query = run_llm(sql_prompt)

            cleaned_sql = clean_output(sql_query, "sql")
            db_ans, error = filter_sql_query(table, cleaned_sql, cur)

            while db_ans == None and tries < 3:
                print(f"Error: {error}, Tries: {tries}")
                tries += 1
                sql_prompt = correct_sql_query(table, table_description, cleaned_sql, error)
                sql_query = run_llm(sql_prompt)
                cleaned_sql = clean_output(sql_query, "sql")
                db_ans, error = filter_sql_query(table, cleaned_sql, cur)
            
            if db_ans != None:
                table_info.append(table)
                database_ans.append(db_ans)

                # Generate NL
                nl_prompt = generate_nl(table, table_description, cleaned_sql)
                nl_ques = run_llm(nl_prompt)
                cleaned_nl = clean_output(nl_ques, "question")
                print("Final Natural Language Question: ", cleaned_nl)

                print("Final SQL Query: ", cleaned_sql)
                generated_nl.append(cleaned_nl)
                generated_sql.append(cleaned_sql)
        
        pd.DataFrame({"table_info": table_info, "nl": generated_nl, "sql": generated_sql, "ans": database_ans}).to_csv("data/generated_data.csv", index=False)

if __name__ == "__main__":
    main()