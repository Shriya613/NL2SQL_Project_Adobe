import sqlite3
import json
from tqdm import tqdm
from transformers import pipeline
import pandas as pd
import torch
import numpy as np
from huggingface_hub import login
import re
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

LLM_AS_A_JUDGE_PROMPT = """
    You are an expert natural language expert tasked with correcting a natural language question.
    Check whether or not the natural language question is valid based on the SQLite query and table information and provide a 
    score between 0 and 100.

    Your score should be based on the following criteria:
    1. The natural language question should produce a question. It does not include the SQLite query or table information.
    2. The natural language question should be specific enough to be answered by the SQLite query.
    3. The natural language question should not include details that the SQLite query does not contain.
    4. The natural language question should not include any information besides the question.
    5. The natural language question should only be one singular natural language question.

    Natural Language Question: {nl_question}

    SQLite Query: {sql_query}

    Table Information: {table}

    Format your response as a JSON object with the following keys:
    - "score": the score between 0 and 100
    - "reasoning": the reasoning for the score

    Example response:
    ```json
    {{
        "score": 80,
        "reasoning": "The natural language question is valid based on the SQLite query and table information but isn't specific enough to be answered by the SQLite query."
    }}
    ```

    Do not include any other text in your response besides the JSON object.
    """

NL_CORRECT_PROMPT = """
    Correct the following natural language question based on the judge's score and reasoning.
    ONLY return the corrected natural language question. Do not include any other information besides the natural language question.
    Make sure that the question is enclosed in ```question tags.

    Table Information: {table}
    SQLite Query: {sql_query}

    Judge's Score: {score}
    Judge's Reasoning: {reasoning}

    Natural Language Question: {nl_question}

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
    """Correct SQL query based on error"""
    prompt_formatted = SQL_CORRECT_PROMPT.format(sql_query=sql_query, error=error, table=table)
    return prompt_formatted


def generate_nl(table: str, sql_query: str) -> str:
    """Generate NL query based on table and SQL query"""
    prompt_formatted = NL_GEN_PROMPT.format(table=table, sql_query=sql_query)
    return prompt_formatted

def filter_nl_question(nl_question: str, sql_query: str, table: str) -> str:
    """Filter NL query based on execution against the database"""
    prompt_formatted = LLM_AS_A_JUDGE_PROMPT.format(nl_question=nl_question, sql_query=sql_query, table=table)
    return prompt_formatted

def correct_nl_question(table: str, sql_query: str, nl_question: str, score: int, reasoning: str) -> str:
    """Correct NL query based on score and reasoning"""
    prompt_formatted = NL_CORRECT_PROMPT.format(nl_question=nl_question, sql_query=sql_query, table=table, score=score, reasoning=reasoning)
    return prompt_formatted


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
            tries_nl = 0
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
                # Generate NL
                nl_prompt = generate_nl(table, cleaned_sql)
                nl_ques = run_llm(nl_prompt)

                # LLM-as-a-judge
                cleaned_nl = clean_output(nl_ques, "question")
                judge_prompt = filter_nl_question(cleaned_nl, cleaned_sql, table)
                judge_response = run_llm(judge_prompt)

                try:
                    cleaned_judge_response = clean_output(judge_response, "json")
                    judge_data = json.loads(cleaned_judge_response)
                    judge_score = int(judge_data["score"])
                    judge_reasoning = judge_data["reasoning"]
                except:
                    judge_score = 0
                    judge_reasoning = "Invalid JSON"


                while judge_score < 90 and tries_nl < 3:
                    print(f"SCORE TOO LOW: Judge score: {judge_score}, Reasoning: {judge_reasoning}")
                    print("NL Question That Failed: ", cleaned_nl)
                    tries_nl += 1
                    correct_nl_prompt = correct_nl_question(table, cleaned_sql, cleaned_nl, judge_score, judge_reasoning)
                    nl_ques = run_llm(correct_nl_prompt)
                    cleaned_nl = clean_output(nl_ques, "question")
                  
                    # LLM-as-a-judge
                    judge_prompt = filter_nl_question(cleaned_nl, cleaned_sql, table)
                    judge_response = run_llm(judge_prompt)
                    try:
                        # Clean the judge response to extract JSON from markdown code blocks
                        cleaned_judge_response = clean_output(judge_response, "json")
                        judge_data = json.loads(cleaned_judge_response)
                        judge_score = int(judge_data["score"])
                        judge_reasoning = judge_data["reasoning"]
                    except:
                        judge_score = 0
                        judge_reasoning = "Invalid JSON"
                
                # Only add to dataset if score meets threshold
                if judge_score >= 90:
                    table_info.append(table)
                    database_ans.append(db_ans)
                    generated_nl.append(cleaned_nl)
                    generated_sql.append(cleaned_sql)
                    actual_sql.append(run_sql_query)
                    print("Added to dataset - score meets threshold")
                    print("Final Natural Language Question: ", cleaned_nl)
                    print("Judge score: ", judge_score)
                    print("Judge reasoning: ", judge_reasoning)
                    print("Final SQL Query: ", cleaned_sql)
                else:
                    print(f"Skipped - final score {judge_score} below threshold of 90")
        
        pd.DataFrame({"table_info": table_info, "nl": generated_nl, "sql": generated_sql, "ans": database_ans, "actual_sql": actual_sql}).to_csv("data_gen/judge_experiment/score_judge.csv", index=False)

def evaluate_pipeline():
    calculate_complexity("data_gen/judge_experiment/score_judge.csv", "actual_sql", "nl")

if __name__ == "__main__":
    run_pipeline()
    evaluate_pipeline()