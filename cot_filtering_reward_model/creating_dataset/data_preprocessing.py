import json
import os
from typing import List, Tuple
import pandas as pd


def load_bird_pairs(json_path: str = "data/train_bird.json") -> Tuple[List[str], List[str], List[str]]:
    """Load SQL and NL question pairs from the BIRD dataset JSON file.

    Args:
        json_path: Path to the BIRD JSON file (list of objects with keys
                   'db_id', 'question', 'SQL', optionally 'evidence').

    Returns:
        A tuple (sqls, questions, db_ids) where:
          - sqls is a list of SQL strings extracted from the 'SQL' field
          - questions is a list of NL questions extracted from the 'question' field
          - db_ids is a list of database IDs extracted from the 'db_id' field
    """
    # Resolve to absolute path relative to repo root to be robust to CWD
    # Script is at: cot_filtering_reward_model/creating_dataset/data_preprocessing.py
    # Need to go up two levels to get to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    abs_path = json_path
    if not os.path.isabs(json_path):
        abs_path = os.path.join(project_root, json_path)

    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sqls: List[str] = []
    questions: List[str] = []
    db_ids: List[str] = []

    for item in data:
        # Some rows may be missing fields; skip safely
        sql_val = item.get("SQL")
        q_val = item.get("question")
        db_id_val = item.get("db_id")
        if not isinstance(sql_val, str) or not isinstance(q_val, str):
            continue
        sqls.append(sql_val.strip())
        questions.append(q_val.strip())
        # db_id can be None, so use empty string if missing
        db_ids.append(db_id_val if isinstance(db_id_val, str) else "")

    return sqls, questions, db_ids


# Eagerly load on import for convenience in downstream modules
sqls, gold_nl_questions, db_ids = load_bird_pairs()

# Resolve CSV path relative to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
csv_path = os.path.join(project_root, "data", "bird_sqls_and_nl_questions.csv")
df = pd.DataFrame({"sql": sqls, "nl_question": gold_nl_questions, "db_id": db_ids})
df.to_csv(csv_path, index=False)

