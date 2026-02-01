import re
import sqlite3
from data_gen.prompts import SQL_GEN_PROMPT, SQL_CORRECT_PROMPT, NL_GEN_PROMPT, NL_CORRECT_PROMPT, LLM_AS_A_JUDGE_PROMPT, MULTIPLE_SQL_GEN_PROMPT, MULTIPLE_SQL_CORRECT_PROMPT

def generate_sql_query(table: str, table_description: str) -> str:
    """Generate SQL query based on table"""
    prompt_formatted = SQL_GEN_PROMPT.format(table=table, table_description=table_description)
    return prompt_formatted

def filter_sql_query(table: str, sql_query: str, cur: sqlite3.Cursor) -> tuple:
    """Filter SQL query based on execution against the database"""
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
            return None, "No result found"

        return result, None
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return None, e

def correct_sql_query(table: str, table_description: str, sql_query: str, error: str) -> str:
    """Correct SQL query based on error"""
    prompt_formatted = SQL_CORRECT_PROMPT.format(sql_query=sql_query, error=error, table=table, table_description=table_description)
    return prompt_formatted


def generate_nl(table: str, table_description: str, sql_query: str) -> str:
    """Generate NL query based on table and SQL query"""
    prompt_formatted = NL_GEN_PROMPT.format(table=table, table_description=table_description, sql_query=sql_query)
    return prompt_formatted

def filter_nl_question(nl_question: str, sql_query: str, table: str) -> str:
    """Filter NL query based on execution against the database"""
    prompt_formatted = LLM_AS_A_JUDGE_PROMPT.format(nl_question=nl_question, sql_query=sql_query, table=table)
    return prompt_formatted

def correct_nl_question(table: str, sql_query: str, nl_question: str, score: int, reasoning: str) -> str:
    """Correct NL query based on score and reasoning"""
    prompt_formatted = NL_CORRECT_PROMPT.format(nl_question=nl_question, sql_query=sql_query, table=table, score=score, reasoning=reasoning)
    return prompt_formatted

def generate_multiple_sql_query(table: str, nl_question: str, first_sql_query: str) -> str:
    """Generate SQL query based on table"""
    prompt_formatted = MULTIPLE_SQL_GEN_PROMPT.format(table=table, nl_question=nl_question, first_sql_query=first_sql_query)
    return prompt_formatted

def correct_multiple_sql_query(table: str, sql_query: str, nl_question: str, error: str, first_sql_query: str) -> str:
    prompt_formatted = MULTIPLE_SQL_CORRECT_PROMPT.format(sql_query=sql_query, nl_question=nl_question, error=error, table=table, first_sql_query=first_sql_query)
    return prompt_formatted