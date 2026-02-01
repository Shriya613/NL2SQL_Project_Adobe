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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from evaluation.evaluate import calculate_complexity

login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Qwen/Qwen2.5-3B-Instruct"
pipe = pipeline(model=model_name, device=device)

SQL_GEN_PROMPT = """
    You are an expert SQL developer tasked with generating a complex, 
    creative, and non-trivial SQLite query. 

    Your query must: 
    1. Be constructed using only the information from the table provided: {table} 
    2. Be complex and challenging.
    3. Have sound logic and be executable (it is okay if the variables names have spaces)
    4. The intended result from the query must produce actionable insights.
    5. Use the column and row values verbatim to the table information but encapsulate them in backticks. 
    This means that underscores are not used to replace spaces. For example, 
    if the column name is "State/territory", then the SQLite query should use "`State/territory`".
    If the row value is "New South Wales", then the SQLite query should use "`New South Wales`".
    6. Produce one SQLite query, and format it inside of ```sql tags.
    7. End with a semicolon.
    
    Do not include any other text in your response besides the SQLite query. 
    Absolutely NO explanations or natural language. 
    """

SQL_CORRECT_PROMPT = """
    Correct the following SQLite query based on the error message,
    and table information. ONLY return the corrected SQLite query.
    Make sure to enclose the column names in backticks.
    
    SQLite Query: {sql_query}

    Error: {error}

    Table Information: {table}
    """

NL_GEN_PROMPT = """
    You are a natural language expert tasked with translating a SQLite 
    query into a question that it answers.

    Given the table information and the SQLite query, generate a natural 
    language question that would logically lead to this query. 

    Your natural language question should:
    1. Only produce a question. Do not include the table or SQLite query in 
    your response.
    2. Be specific enough to be answered by the SQLite query.
    3. Not include details that the SQLite query does not contain.
    4. Not include any information besides the question.
    5. Produce one natural language question, and format it inside of ```question tags.

    Table information: {table}

    SQLite Query: {sql_query}
    """

def generate_sql_query(table: str) -> str:
    """Generate SQL query based on table"""
    prompt_formatted = SQL_GEN_PROMPT.format(table=table)
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

def correct_sql_query(table: str, sql_query: str, error: str) -> str:
    prompt_formatted = SQL_CORRECT_PROMPT.format(sql_query=sql_query, error=error, table=table)
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
        conn = sqlite3.connect("data/train.db")
        cur = conn.cursor()

        for table in tqdm(table_data[:1000]):
            tries = 0
            sql_prompt = generate_sql_query(table)
            sql_query = run_llm(sql_prompt)

            cleaned_sql = clean_output(sql_query, "sql")
            run_sql_query, db_ans, error = filter_sql_query(table, cleaned_sql, cur)

            while db_ans == None and tries < 3:
                print(f"Error: {error}, Tries: {tries}")
                tries += 1
                sql_prompt = correct_sql_query(table, cleaned_sql, error)
                sql_query = run_llm(sql_prompt)
                cleaned_sql = clean_output(sql_query, "sql")
                run_sql_query, db_ans, error = filter_sql_query(table, cleaned_sql, cur)
            
            if db_ans != None:
                table_info.append(table)
                database_ans.append(db_ans)
                actual_sql.append(run_sql_query)

                # Generate NL
                nl_prompt = generate_nl(table, cleaned_sql)
                nl_ques = run_llm(nl_prompt)
                cleaned_nl = clean_output(nl_ques, "question")
                print("Final Natural Language Question: ", cleaned_nl)

                print("Final SQL Query: ", cleaned_sql)
                generated_nl.append(cleaned_nl)
                generated_sql.append(cleaned_sql)
        
        pd.DataFrame({"table_info": table_info, "nl": generated_nl, "sql": generated_sql, "actual_sql": actual_sql, "ans": database_ans}).to_csv("data_gen/order_experiment/generated_sql_nl.csv", index=False)

def evaluate_pipeline():
    calculate_complexity("data_gen/order_experiment/generated_sql_nl.csv", "actual_sql", "nl")

run_pipeline()
evaluate_pipeline()