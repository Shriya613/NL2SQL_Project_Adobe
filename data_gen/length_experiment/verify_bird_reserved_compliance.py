#!/usr/bin/env python3
"""
BIRD-only reserved-word seeding compliance (no DB)

- Uses data/train_bird.json questions + db_id as lightweight context
- Varies requested reserved-word counts in the prompt
- Generates SQL with Qwen (via existing utils.initialize_model/run_llm)
- Counts actual reserved words in outputs
- Prints terminal stats; saves CSV; optional matplotlib visuals

Requirements:
- .env with HUGGINGFACE_HUB_TOKEN
- transformers, torch, pandas, tqdm, (optional) matplotlib, seaborn
"""

import os
import re
import json
import random
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load env for HuggingFace token
load_dotenv()

# Import model helpers from repo (no DB needed)
import sys
BASE_DIR = os.path.dirname(__file__)
SHARED_DIR = os.path.join(BASE_DIR, "data_gen", "diversity_experiments", "shared")
sys.path.insert(0, SHARED_DIR)
from utils import initialize_model, run_llm, extract_sql_and_reasoning_from_json  # noqa: E402

# Reserved word patterns from repo (fall back if unavailable)
DATA_GEN_DIR = os.path.join(BASE_DIR, "data_gen")
sys.path.insert(0, DATA_GEN_DIR)
try:
    from reserved import reserved_word_patterns  # noqa: E402
except Exception:
    BASIC = [
        "SELECT","FROM","WHERE","GROUP","BY","ORDER","HAVING","LIMIT","JOIN","INNER","LEFT","RIGHT",
        "OUTER","ON","AS","AND","OR","NOT","IN","LIKE","BETWEEN","IS","NULL","COUNT","SUM","AVG",
        "MIN","MAX","DISTINCT","CASE","WHEN","THEN","ELSE","END"
    ]
    reserved_word_patterns = {w: rf"\b{w}\b" for w in BASIC}


def load_bird(path="data/train_bird.json", limit=None):
    with open(path, "r") as f:
        data = json.load(f)
    return data if limit is None else data[:limit]


def count_reserved_words(sql_query: str) -> int:
    if not isinstance(sql_query, str) or not sql_query:
        return 0
    s = sql_query.upper()
    return sum(1 for _, pat in reserved_word_patterns.items() if re.search(pat, s))


def build_prompt(requested_count: int, db_id: str, question: str) -> str:
    # Enhanced prompt with better guidance for reserved word usage
    reserved_examples = {
        3: "SELECT, FROM, WHERE",
        5: "SELECT, FROM, WHERE, ORDER, BY", 
        8: "SELECT, FROM, WHERE, GROUP, BY, ORDER, BY, LIMIT",
        12: "SELECT, FROM, WHERE, GROUP, BY, HAVING, ORDER, BY, LIMIT, DISTINCT, COUNT",
        15: "SELECT, FROM, WHERE, GROUP, BY, HAVING, ORDER, BY, LIMIT, DISTINCT, COUNT, SUM, AVG, MAX",
        18: "SELECT, FROM, WHERE, GROUP, BY, HAVING, ORDER, BY, LIMIT, DISTINCT, COUNT, SUM, AVG, MAX, MIN, CASE, WHEN, THEN"
    }
    
    closest_example = min(reserved_examples.keys(), key=lambda x: abs(x - requested_count))
    example_words = reserved_examples[closest_example]
    
    return f"""
You are an expert SQL generator. Your task is to create a SQL query that answers the question while using approximately {requested_count} SQL reserved keywords.

IMPORTANT: Count these reserved keywords carefully: SELECT, FROM, WHERE, JOIN, INNER, LEFT, RIGHT, OUTER, ON, AS, AND, OR, NOT, IN, LIKE, BETWEEN, IS, NULL, COUNT, SUM, AVG, MIN, MAX, DISTINCT, CASE, WHEN, THEN, ELSE, END, GROUP, BY, HAVING, ORDER, LIMIT, UNION, INTERSECT, EXCEPT, EXISTS, ALL, ANY, SOME.

Target: Use around {requested_count} reserved keywords (like: {example_words})

Strategy:
- Use complex WHERE conditions with AND/OR
- Add GROUP BY with HAVING clauses
- Include ORDER BY with multiple columns
- Use aggregate functions (COUNT, SUM, AVG, MAX, MIN)
- Add DISTINCT when appropriate
- Use CASE statements for conditional logic
- Add LIMIT clauses
- Use subqueries if needed

Context:
- Database: {db_id}
- Question: {question}

Return ONLY a JSON object with "reasoning" and "final_sql_query" fields:
{{
  "reasoning": "Explain how you'll use approximately {requested_count} reserved keywords",
  "final_sql_query": "```sql\\nYOUR_SQL_QUERY_HERE\\n```"
}}
""".strip()


def choose_requested_counts(n, strategy="balanced"):
    """
    Strategy:
    - 'balanced': spread requests across a range so we can see the effect vs. count
    - 'random': random in a range
    """
    rng = list(range(3, 21))  # Expanded range: 3..20 reserved words for better coverage
    if strategy == "balanced":
        seq = []
        while len(seq) < n:
            seq.extend(rng)
        return seq[:n]
    return [random.randint(3, 20) for _ in range(n)]


def main():
    print("🔬 BIRD-only Reserved-Word Seeding Compliance (no DB)")
    print("=====================================================")

    # Optional: remove stray DB if present and user doesn't want it
    stray_db = os.path.join("data", "data", "train.db")
    if os.path.exists(stray_db):
        print(f"ℹ️ Removing unused database: {stray_db}")
        try:
            os.remove(stray_db)
        except Exception as e:
            print(f"⚠️ Couldn't remove {stray_db}: {e}")

    # Load BIRD
    bird_path = "data/train_bird.json"
    if not os.path.exists(bird_path):
        print("❌ Missing data/train_bird.json")
        return

    num_examples = int(os.getenv("BIRD_EXAMPLES", "20"))  # Increased from 3 to 20
    data = load_bird(bird_path, limit=num_examples)
    requested_counts = choose_requested_counts(len(data), strategy="balanced")

    # Initialize Qwen
    print("Initializing Qwen model...")
    pipe = initialize_model()
    print("✅ Model ready")

    results = []
    print(f"Running {len(data)} generations...")
    print(f"Sample data: {data[0] if data else 'No data'}")
    
    for i, (item, req) in enumerate(tqdm(zip(data, requested_counts), total=len(data))):
        db_id = item.get("db_id", f"bird_{i}")
        question = item.get("question", "Write a SQL query.")

        prompt = build_prompt(req, db_id, question)
        
        try:
            raw = run_llm(prompt, pipe, max_tokens=256)  # Increased token limit
            sql, reasoning = extract_sql_and_reasoning_from_json(raw)
            
            # Debug: print first few examples
            if i < 3:
                print(f"\n--- Example {i+1} ---")
                print(f"Requested: {req} reserved words")
                print(f"Raw output: {raw[:200]}...")
                print(f"Extracted SQL: {sql[:100]}...")
            
            actual = count_reserved_words(sql)
            diff = actual - req
            compliance = "exact" if diff == 0 else ("close" if abs(diff) <= 1 else "way_off")

            results.append({
                "index": i,
                "db_id": db_id,
                "requested_count": req,
                "actual_count": actual,
                "difference": diff,
                "compliance_type": compliance,
                "question": question,
                "sql": sql,
                "reasoning": reasoning,
                "raw_output": raw[:500]  # Store first 500 chars for debugging
            })

            print(f"✅ {i+1}/{len(data)} req={req:2d} got={actual:2d} diff={diff:+d} type={compliance}")
            
        except Exception as e:
            print(f"❌ Error processing example {i+1}: {e}")
            # Add error entry to maintain data structure
            results.append({
                "index": i,
                "db_id": db_id,
                "requested_count": req,
                "actual_count": 0,
                "difference": -req,
                "compliance_type": "error",
                "question": question,
                "sql": "",
                "reasoning": f"Error: {str(e)}",
                "raw_output": ""
            })

    df = pd.DataFrame(results)
    out_csv = "bird_only_reserved_compliance.csv"
    df.to_csv(out_csv, index=False)

    # Terminal summary
    exact = (df["compliance_type"] == "exact").sum()
    close_any = df["compliance_type"].isin(["exact", "close"]).sum()
    way_off = (df["compliance_type"] == "way_off").sum()
    total = len(df)
    rate = (close_any / total) * 100 if total else 0.0

    print("\n=====================================================")
    print("📊 Summary")
    print("=====================================================")
    print(f"Total: {total}")
    print(f"Exact matches: {exact}")
    print(f"Close (±1): {close_any - exact}")
    print(f"Way off (>±2): {way_off}")
    print(f"Compliance rate (exact+close): {rate:.1f}%")
    print(f"📄 Saved: {out_csv}")

    # Enhanced visuals with better error handling and debugging
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print(f"\n📊 Creating visualizations with {len(df)} data points...")
        print(f"Requested range: {df['requested_count'].min()}-{df['requested_count'].max()}")
        print(f"Actual range: {df['actual_count'].min()}-{df['actual_count'].max()}")
        print(f"Difference range: {df['difference'].min()}-{df['difference'].max()}")

        plt.style.use("default")

        # 1) Requested vs Actual (scatter) - Enhanced
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="requested_count", y="actual_count", alpha=0.7, s=60)
        
        # Add perfect line
        mn, mx = df["requested_count"].min(), df["requested_count"].max()
        plt.plot([mn, mx], [mn, mx], "r--", label="Perfect match", linewidth=2)
        
        # Add trend line
        if len(df) > 1:
            sns.regplot(data=df, x="requested_count", y="actual_count", scatter=False, color="blue", label="Trend")
        
        plt.title(f"Requested vs Actual Reserved Words\n(n={len(df)} samples)")
        plt.xlabel("Requested Count")
        plt.ylabel("Actual Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("viz_requested_vs_actual.png", dpi=200, bbox_inches='tight')

        # 2) Difference distribution (hist) - Enhanced
        plt.figure(figsize=(8, 5))
        diff_min, diff_max = int(df["difference"].min()), int(df["difference"].max())
        bins = range(diff_min-1, diff_max+2) if diff_max > diff_min else [diff_min-1, diff_min, diff_min+1]
        
        sns.histplot(df["difference"], bins=bins, kde=True, alpha=0.7)
        plt.axvline(0, color="r", linestyle="--", linewidth=2, label="Perfect")
        plt.title(f"Distribution of Differences (Actual - Requested)\n(n={len(df)} samples)")
        plt.xlabel("Difference")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("viz_difference_hist.png", dpi=200, bbox_inches='tight')

        # 3) Compliance by requested bucket - Enhanced
        df["_bucket"] = pd.cut(df["requested_count"], bins=[2,6,10,14,18,22], labels=["3-6","7-10","11-14","15-18","19-22"])
        comp_by_bucket = df.groupby("_bucket")["compliance_type"].apply(lambda s: (s.isin(["exact","close"]).mean()*100))
        
        plt.figure(figsize=(8, 5))
        bars = comp_by_bucket.plot(kind="bar", color="#4C72B0", alpha=0.8)
        plt.ylim(0, 100)
        plt.title(f"Compliance Rate by Requested Count Bucket\n(n={len(df)} samples)")
        plt.ylabel("Compliance Rate (%)")
        plt.xlabel("Requested Count Bucket")
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(comp_by_bucket.values):
            if not pd.isna(v):
                plt.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("viz_compliance_by_bucket.png", dpi=200, bbox_inches='tight')

        # 4) Additional: SQL length vs compliance
        df["sql_length"] = df["sql"].str.len()
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="sql_length", y="actual_count", hue="compliance_type", alpha=0.7)
        plt.title(f"SQL Length vs Actual Reserved Word Count\n(n={len(df)} samples)")
        plt.xlabel("SQL Query Length (characters)")
        plt.ylabel("Actual Reserved Word Count")
        plt.legend(title="Compliance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("viz_sql_length_vs_count.png", dpi=200, bbox_inches='tight')

        print("🖼️ Saved enhanced visuals:")
        print(" - viz_requested_vs_actual.png")
        print(" - viz_difference_hist.png") 
        print(" - viz_compliance_by_bucket.png")
        print(" - viz_sql_length_vs_count.png")
        
    except Exception as e:
        print(f"⚠️ Skipping visuals (matplotlib/seaborn not available or failed): {e}")
        print(f"Debug info: df shape={df.shape if 'df' in locals() else 'N/A'}")
        if 'df' in locals():
            print(f"Sample data:\n{df[['requested_count', 'actual_count', 'difference']].head()}")


if __name__ == "__main__":
    main()
    
    
'''

How to run:
- Ensure `.env` has your `HUGGINGFACE_HUB_TOKEN`
- Then:
```bash
source venv/bin/activate
python verify_bird_reserved_compliance.py
```

You’ll see:
- Per-example terminal lines like: req=12 got=11 diff=-1 type=close
- A CSV `bird_only_reserved_compliance.csv`
- Visuals (if matplotlib/seaborn available):
  - `viz_requested_vs_actual.png`
  - `viz_difference_hist.png`
  - `viz_compliance_by_bucket.png`

Note:
- This removes any `train.tables.jsonl` dependency.
- No database is touched. The script also removes a stray `data/data/train.db` if present.

'''