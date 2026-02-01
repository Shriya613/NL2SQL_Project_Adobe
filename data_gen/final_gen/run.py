import json, os, sys, signal, sqlite3, re
from dataclasses import dataclass, field
import pandas as pd
import dotenv
from tqdm import tqdm
from transformers import pipeline
from typing import Any
import random
import google.genai as genai
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from cot_filtering_reward_model.testing_final_model.load_model_from_hf import load_model_from_huggingface, get_reward
from data_gen.final_gen.sql_gen import (
    sql_gen_reserved_pipeline,
    sql_gen_persona_pipeline,
    extract_persona_info,
    get_persona_prompt
)
from data_gen.final_gen.nl_gen import (
    nl_gen_reserved_pipeline,
    nl_gen_persona_pipeline,
)
from data_gen.final_gen.prompts import PROMPT_FILTER_PROMPT
from data_gen.final_gen.utils.llm import run_llm_gemini

dotenv.load_dotenv()

@dataclass
class ResultBuffer:
    nl: list = field(default_factory=list)
    sql: list = field(default_factory=list)
    db_sql: list = field(default_factory=list)
    db_result: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    table_id: list = field(default_factory=list)
    reasoning: list = field(default_factory=list)
    seeding_type: list = field(default_factory=list)
    seeding_value: list = field(default_factory=list)
    filtered: list = field(default_factory=list)
    schema_change_type: list = field(default_factory=list)
    db_result_after_change: list = field(default_factory=list)
    db_query_match: list = field(default_factory=list)

# Set up signal handlers to allow graceful shutdown
def signal_handler(sig, frame):
    """Handle signals to allow graceful shutdown"""
    print("\nReceived interrupt signal. Progress has been saved incrementally.")
    if 'conn' in globals() and conn:
        try:
            conn.close()
        except:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_tables() -> list:
    tables = []
    jsonl_path = os.path.join(project_root, "data/train.tables.jsonl")
    with open(jsonl_path, "r") as f:
        for line in f:
            table = json.loads(line)
            table["id"] = "table_" + table["id"].replace("-", "_")
            table["name"] = table["id"]
            tables.append(table)
    return tables

def save_results(option: str, accepted: ResultBuffer, rejected: ResultBuffer) -> None:
    """Save results to CSV files"""
    pd.DataFrame(accepted.__dict__).to_csv(f"data_gen/final_gen/accepted_results_{option}.csv", index=False)
    pd.DataFrame(rejected.__dict__).to_csv(f"data_gen/final_gen/rejected_results_{option}.csv", index=False)

def reward_filter(sql: str, nl: str, reasoning: str) -> tuple[float, bool]:
    """Filter based on reward model"""
    # Load model
    model, tokenizer, device = load_model_from_huggingface()  
    score = get_reward(model, tokenizer, sql, reasoning, nl, device)
    return score, score >= 0.5

def extract_score(resp: str) -> float | None:
    """Extract a score even if the LLM response is not strict JSON."""
    try:
        parsed = json.loads(resp)
        if isinstance(parsed, dict) and "score" in parsed:
            return float(parsed["score"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    match = re.search(r'"?score"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', resp, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))

    fallback = re.search(r'([0-9]+(?:\.[0-9]+)?)', resp)
    if fallback:
        candidate = float(fallback.group(1))
        if 0.0 <= candidate <= 1.0:
            return candidate
    return None


def prompt_filter(pipe, sql: str, nl: str, table: str, reasoning: str) -> tuple[float, bool]:
    """Filter based on prompt model"""
    prompt = PROMPT_FILTER_PROMPT.format(sql_query=sql, nl_question=nl, table=table, reasoning=reasoning)
    raw_response = run_llm_gemini(pipe, prompt).strip()

    score = extract_score(raw_response)
    if score is None:
        print(f"[prompt_filter] Unable to parse score from response: {raw_response!r}")
        return 0.0, False

    return score, score >= 0.8
    

def gen_pipeline(pipe: Any, table: str, idx: int, cur: sqlite3.Cursor) -> tuple[str, str, str, str, str, str, any]:
    
    
    # Generate SQL query based on seeding method
    if idx % 2 == 0:
        # Generate reserved SQL
        print("Generating reserved SQL")
        sql, dq_sql, sampled_num, db_result = sql_gen_reserved_pipeline(table, cur)
        seeding_type = "reserved"
        seeding_value = sampled_num
        print("Seeding value: ", seeding_value)
    else:
        # Generate persona
        print("Generating persona SQL")
        persona_prompt = get_persona_prompt(table)
        persona_response = run_llm_gemini(pipe, persona_prompt)
        persona, goals, example_queries = extract_persona_info(persona_response)
        seeding_type = "persona"
        seeding_value = persona + " Goals: " + goals + " Example Queries: " + example_queries
        print("Seeding value: ", seeding_value)
        sql, dq_sql, db_result = sql_gen_persona_pipeline(table, cur, persona, goals, example_queries)


    if not sql:
        return None, None, None, None, None, None, None
    
    # Generate NL question
    if idx % 2 == 0:
        print("Generating reserved NL")
        nl, reasoning = nl_gen_reserved_pipeline(pipe, table, sql)
        print("NL: ", nl)

    else:
        print("Generating persona NL")
        nl, reasoning = nl_gen_persona_pipeline(pipe, table, sql, persona, goals)
        print("NL: ", nl)
    if not nl:
        return None, None, None, None, None, None, None

    return nl, sql, dq_sql, reasoning, seeding_type, seeding_value, db_result


def filtering(option: str, pipe: pipeline, table: str, sql: str, nl: str, dq_sql: str, reasoning: str, db_result: any = None) -> tuple[float, bool] | None:
    accepted = ResultBuffer()
    rejected = ResultBuffer()
    print("Starting filtering: ", option)

    # Filter 
    if option == "reward":
        score, keep = reward_filter(sql, nl, reasoning)
        print("Score: ", score)
        print("Keep: ", keep)
            
    elif option == "prompt":
        score, keep = prompt_filter(pipe, sql, nl, table, reasoning)
        print("Score: ", score)
        print("Keep: ", keep)
    else:
        score, keep = 0.0, False

    if keep:
        accepted.nl.append(nl)
        accepted.sql.append(sql)
        accepted.db_sql.append(dq_sql)
        accepted.db_result.append(db_result)
        accepted.rewards.append(score)
        accepted.table_id.append(table["id"])
        return accepted
    else:
        rejected.nl.append(nl)
        rejected.sql.append(sql)
        rejected.db_sql.append(dq_sql)
        rejected.db_result.append(db_result)
        rejected.rewards.append(score)
        rejected.table_id.append(table["id"])
        return None


if __name__ == "__main__":
    # Connect to database
    db_path = os.path.join(project_root, "data/train_modified.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    pipe = genai.Client()
    tables = load_tables()

    # Process each example completely (generate → filter → save) before moving to next
    for idx, table in enumerate(
        tqdm(tables[:5], desc="Processing tables", unit="table")
    ):
        # Generation
        nl, sql, dq_sql, reasoning, seeding_type, seeding_value, db_result = gen_pipeline(pipe, table, idx, cur)

        # Filtering
        if (nl is not None and sql is not None):
            result_reward = filtering("reward", pipe, table, sql, nl, dq_sql, reasoning, db_result)
            if result_reward is not None:
                result = result_reward
                save_results("reward", result_reward)

