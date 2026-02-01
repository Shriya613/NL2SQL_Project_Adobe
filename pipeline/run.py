import os
import sys
import sqlite3
import csv
import argparse
import logging
from tqdm import tqdm
import pandas as pd
import torch
import random
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress verbose logging from third-party libraries
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)
logging.getLogger("accelerate.accelerator").setLevel(logging.WARNING)
# Suppress all INFO level messages
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s', force=True)

# google.generativeai is imported where needed in machine_translation functions
import google.genai as genai
from typing import Any
from openai import OpenAI
from transformers import AutoTokenizer, pipeline
from torch.utils.data import DataLoader

from data_gen.final_gen.run import (
    load_tables,
    gen_pipeline,
    filtering,
    ResultBuffer
)
from data_gen.final_gen.sql_gen import sql_gen_multiple_pipeline
from multilingual_evoschema import apply_schema_change, apply_schema_change_translation
from ambi_detect.ambiguity_trainer import (
    pick_text_columns,
    NLSQLDataset,
    predict,
    MeanPoolerRegressor
)
from machine_translation.mt_improved_ner_semantic import (
    ImprovedNER,
    EntityTranslator,
    TranslationQualityGate,
    LANG_NAMES,
    run_translation_gemini
)
try:
    from machine_translation.translation_quality_filter import TranslationQualityFilter
    QUALITY_FILTER_AVAILABLE = True
except ImportError:
    QUALITY_FILTER_AVAILABLE = False


def ambiguity_detection(generated_data: str, tokenizer: AutoTokenizer, device: torch.device) -> list[float]:
    """Detect ambiguity in generated data"""
    df = pd.read_csv(generated_data)

    try:
        _ = pick_text_columns(df)
    except ValueError:
        if not {"nl", "sql"}.issubset(df.columns):
            raise

    dataset = NLSQLDataset(df, tokenizer, max_len=256, with_labels=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = MeanPoolerRegressor("distilbert-base-uncased")
    best_model_path = os.path.join("ambig_model", "pytorch_model.bin")
    if os.path.exists(best_model_path):
        # Load checkpoint to CPU first to avoid OOM
        ckpt = torch.load(best_model_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
    
    # Try to move model to device, try other GPUs if CUDA OOM
    original_device = device
    model_loaded = False
    
    if device.type == "cuda" and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        current_gpu = device.index if device.index is not None else 0
        
        # Try current GPU first
        try:
            model = model.to(device)
            print(f"✅Model loaded on {device}")
            model_loaded = True
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(type(e).__name__):
                print(f"CUDA out of memory on GPU {current_gpu}, trying other GPUs... ")
                
                # Try other GPUs
                for gpu_idx in range(num_gpus):
                    if gpu_idx == current_gpu:
                        continue
                    try:
                        new_device = torch.device(f"cuda:{gpu_idx}")
                        print(f"Trying GPU {gpu_idx}...")
                        model = model.to(new_device)
                        device = new_device
                        print(f"Model loaded on {device}")
                        model_loaded = True
                        break
                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        print(f"   GPU {gpu_idx} also out of memory, trying next...")
                        continue
                
                if not model_loaded:
                    print(f"⚠️  All GPUs out of memory, falling back to CPU")
                    device = torch.device("cpu")
                    model = model.to(device)
                    print(f"✅ Model loaded on CPU instead")
                    model_loaded = True
            else:
                raise
    else:
        # Not CUDA or CUDA not available, just try the device
        try:
            model = model.to(device)
            print(f"✅ Model loaded on {device}")
            model_loaded = True
        except Exception as e:
            print(f"❌ ERROR MOVING MODEL TO DEVICE")
            print(f"   Device: {device}")
            print(f"   Error: {e}")
            # Fallback to CPU
            device = torch.device("cpu")
            model = model.to(device)
            print(f"✅ Falling back to CPU ")
            model_loaded = True
    
    if not model_loaded:
        raise RuntimeError("Failed to load model on any device")
    best_model_path = os.path.join("ambig_model", "pytorch_model.bin")
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    preds = predict(model, loader, device)
    return preds

def machine_translation(result_buffer: ResultBuffer, lang: str = "ru") -> ResultBuffer:
    ner = ImprovedNER()
    translator = EntityTranslator(target_lang=lang)

    tgt_lang_name = LANG_NAMES.get(lang)
    if not tgt_lang_name:
        print(f"Unsupported language: {lang}")
        return result_buffer

    quality_filter = TranslationQualityFilter(lang) if QUALITY_FILTER_AVAILABLE else None
    quality_gate = TranslationQualityGate(lang, quality_filter)
    
    result = ResultBuffer()
    
    for i in tqdm(range(len(result_buffer.nl)), desc=f"Translating to {lang}"):
        original = result_buffer.nl[i]
        entities = ner.extract_entities(original)
        prepared = translator.process_entities(original, entities)
        original_text = result_buffer.nl[i]

        # Translate and retry until we get a translation that passes quality checks
        max_retries = 3
        retry_count = 0
        translation_accepted = False
        translations = None
        
        while retry_count < max_retries and not translation_accepted:
            # Get translation from Gemini
            translation = run_translation_gemini(prepared, target_lang=tgt_lang_name)
            
            if not translation:
                # No translation returned - retry
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Translation API failed for item {i+1}. Retry {retry_count}/{max_retries}...")
                    import time
                    time.sleep(1)
                else:
                    print(f"❌ Translation API failed for item {i+1} after {max_retries} retries. Skipping item.")
            
            # Evaluate translation quality (back-translation, semantic similarity, etc.)
            keep, reason, cleaned, _ = quality_gate.evaluate(original_text, translation)
            
            if keep:
                # Translation passed all quality checks - accept it
                # Skip if cleaned translation is blank
                if not cleaned or not cleaned.strip():
                    print(f"⚠️  Skipping translation with blank NL for item {i+1}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying translation for item {i+1}...")
                        import time
                        time.sleep(1)
                    else:
                        print(f"❌ Translation resulted in blank NL for item {i+1} after {max_retries} retries. Skipping item.")
                    continue
                
                translation_accepted = True
                result.nl.append(cleaned)
                result.sql.append(result_buffer.sql[i])
                result.db_sql.append(result_buffer.db_sql[i])
                # Preserve db_result if it exists
                if hasattr(result_buffer, 'db_result') and i < len(result_buffer.db_result):
                    result.db_result.append(result_buffer.db_result[i])
                else:
                    result.db_result.append(None)
                if hasattr(result_buffer, 'rewards') and i < len(result_buffer.rewards):
                    result.rewards.append(result_buffer.rewards[i])
                if hasattr(result_buffer, 'table_id') and i < len(result_buffer.table_id):
                    result.table_id.append(result_buffer.table_id[i])
                # Preserve other attributes if they exist
                for attr in ['reasoning', 'seeding_type', 'seeding_value', 'filtered', 'schema_change_type', 
                             'db_result_after_change', 'db_query_match']:
                    if hasattr(result_buffer, attr) and i < len(getattr(result_buffer, attr)):
                        if not hasattr(result, attr):
                            setattr(result, attr, [])
                        getattr(result, attr).append(getattr(result_buffer, attr)[i])
            else:
                # Translation rejected by quality gate - retry
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Translation rejected ({reason}) for item {i+1}. Retry {retry_count}/{max_retries}...")
                    import time
                    time.sleep(1)
                else:
                    print(f"❌ Translation rejected ({reason}) for item {i+1} after {max_retries} retries. Skipping item.")
    
    return result

def generate_rows(client: Any, tables: list[dict], cur: sqlite3.Cursor, output_csv_path: str | None = None) -> list[dict]:
    """Generate NL and SQL for row"""
    rows = []
    csv_file = None
    writer = None
    fieldnames = [
        "table_id",
        "nl",
        "sql",
        "db_sql",
        "db_result",
        "reasoning",
        "seeding_type",
        "seeding_value",
    ]

    # Determine starting index based on existing CSV
    start_idx = 0
    csv_exists = output_csv_path and os.path.exists(output_csv_path)
    if csv_exists:
        try:
            df_existing = pd.read_csv(output_csv_path)
            if len(df_existing) > 0:
                # Find the last table_id that doesn't contain "m" (modified table indicator)
                last_original_table_id = None
                for table_id in reversed(df_existing["table_id"].tolist()):
                    if table_id and "m" not in str(table_id):
                        last_original_table_id = table_id
                        break
                
                if last_original_table_id:
                    # Find the index of this table in the tables list
                    for idx, table in enumerate(tables):
                        if table.get("id") == last_original_table_id:
                            start_idx = idx + 1
                            print(f"Resuming from table index {start_idx} (table_id: {last_original_table_id})")
                            break
                    else:
                        print(f"Warning: Last table_id {last_original_table_id} not found in tables list. Starting from beginning.")
                else:
                    # All rows are modified tables, so we've already processed everything
                    print("All rows in CSV are modified tables. Stopping - no original tables to process.")
                    return []
        except Exception as e:
            print(f"Error reading existing CSV: {e}. Starting from beginning.")

    try:
        if output_csv_path:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            # Open in append mode if resuming, write mode if starting fresh
            mode = "a" if start_idx > 0 and csv_exists else "w"
            csv_file = open(output_csv_path, mode, newline="", encoding="utf-8")
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            # Only write header if starting fresh
            if mode == "w":
                writer.writeheader()

        for idx, table in enumerate(
            tqdm(tables, desc="Processing tables", unit="table", initial=start_idx, total=len(tables))
        ):
            # Skip tables before the resume point
            if idx < start_idx:
                continue
            
            # Safety check: Stop if we encounter a modified table 
            table_id = table.get("id", "")
            if table_id and "m" in str(table_id):
                print(f"Stopping at first modified table in JSONL: {table_id}")
                break
            # NOTE: this is for debugging purposes. Doesnt effect the generation process
            # just so we can see the table schema and rows before generation happens
            print(f"\n{'='*80}")
            print("starting generation for table: ")
            print(f"Table {idx + 1}: {table.get('id', 'Unknown ID')}")
            print(f"{'='*80}")
            
            headers = table.get('header', [])
            table_rows = table.get('rows', [])
            
            if headers:
                print(f"\nColumns ({len(headers)}):")
                print("  " + " | ".join(f"{i+1}. {col}" for i, col in enumerate(headers)))
            
            if table_rows:
                print(f"\nFirst row values:")
                first_row = table_rows[0]
                max_col_width = max(len(str(h)) for h in headers) if headers else 20
                for i, (col_name, value) in enumerate(zip(headers, first_row)):
                    value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    print(f"  {col_name:>{max_col_width}} : {value_str}")
                
                if len(table_rows) > 1:
                    print(f"\n(Total rows: {len(table_rows)})")
            else:
                print("\n(No rows in table)")
            
            print(f"{'='*80}\n")

            # Generate Data
            nl, sql, dq_sql, reasoning, seeding_type, seeding_value, db_result = gen_pipeline(client, table, idx, cur)
            print(nl, sql, dq_sql, reasoning)

            if not all([nl, sql, dq_sql, reasoning]):
                continue
            
            # Skip if NL is blank
            if not nl or not str(nl).strip():
                print(f"⚠️  Skipping generated row with blank NL: table_id={table.get('id')}")
                continue

            row_data = {
                "table_id": table.get("id"),
                "nl": nl,
                "sql": sql,
                "db_sql": dq_sql,
                "db_result": db_result,
                "reasoning": reasoning,
                "seeding_type": seeding_type,
                "seeding_value": seeding_value,
            }
            rows.append(row_data)
            if writer:
                writer.writerow(row_data)
                csv_file.flush()
    finally:
        if csv_file:
            csv_file.close()

    return rows

def filter_and_schema_change(df: pd.DataFrame, preds: list[float], client: Any, table: dict, db_path: str, cur: sqlite3.Cursor, filtered_file: str = None) -> list[dict]:
    """Apply filtering and schema change"""
    print("================================================")
    print("Filtering and schema changing...")
    print("================================================")
    original_results = []
    schema_changed_results = []
    
    # Prepare CSV writer if file path is provided
    csv_file = None
    csv_writer = None
    fieldnames = None
    if filtered_file:
        # Determine fieldnames from expected structure
        fieldnames = ["nl", "sql", "db_sql", "db_result", "table_id", "filtered", "score", 
                     "reasoning", "seeding_type", "seeding_value", 
                     "predicted_ambiguity", "schema_change_type",
                     "db_result_after_change", "db_query_match"]
        
        # Check if file exists to determine mode
        file_exists = os.path.exists(filtered_file)
        mode = "a" if file_exists else "w"
        
        csv_file = open(filtered_file, mode, newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write header only if file is new
        if not file_exists:
            csv_writer.writeheader()
    
    for i, pred in enumerate(preds):
        row = df.iloc[i]
        nl, sql, dq_sql, reasoning, seeding_type, seeding_value = row["nl"], row["sql"], row["db_sql"], row["reasoning"], row["seeding_type"], row["seeding_value"]
        db_result = row.get("db_result", None)
        # Parse db_result if it's a string (from CSV)
        if db_result is not None and isinstance(db_result, str):
            try:
                import ast
                db_result = ast.literal_eval(db_result)
            except (ValueError, SyntaxError):
                # If parsing fails, try json
                try:
                    import json
                    db_result = json.loads(db_result)
                except (json.JSONDecodeError, ValueError):
                    print(f"⚠️  Warning: Could not parse db_result: {db_result[:100] if len(str(db_result)) > 100 else db_result}")
                    db_result = None
        print("☀️☀️ PREDICTED AMBIGUITY: ", pred)
        method = "prompt" if pred >= 0.25 else "reward"
        result = filtering(method, client, table, sql, nl, dq_sql, reasoning, db_result)

        # If filtering rejected the result (returned None), skip this example
        if result is None:
            continue

        # Track SQL queries for prompt filtering (may have multiple)
        sql_queries = [sql]  # Start with original SQL
        db_sql_queries = [dq_sql]  # Start with original db_sql
        
        if method == "prompt":
            print("☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️")
            print("❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️")
            print("BY GOLLY GOSH, THE QUERY IS AMBIGUOUS. WE NEED TO GENERATE AN ALTERNATIVE SQL QUERY.")
            print("❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️")
            print("☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️")
            # Generate an alternative SQL (and reasoning) for ambiguous queries
            new_sql, new_run_sql_query, new_reason, new_db_result = sql_gen_multiple_pipeline(table, cur, nl, sql)
            if new_sql:
                alt_reason = new_reason or reasoning
                result_2 = filtering(method, client, table, new_sql, nl, new_run_sql_query, alt_reason, new_db_result)
                if result_2:
                    # result_2 is a ResultBuffer with lists containing one element each (single example)
                    result.sql.append(result_2.sql[0])
                    result.db_sql.append(result_2.db_sql[0])
                    result.db_result.append(result_2.db_result[0] if result_2.db_result else None)
                    result.rewards.append(result_2.rewards[0])
                    # Add alternative SQL to the list
                    sql_queries.append(result_2.sql[0])
                    db_sql_queries.append(result_2.db_sql[0])
            schema_change = False
        else:
            print(f"Schema change true")
            schema_change = True
        
        # Skip if NL is blank
        if not nl or not str(nl).strip():
            print(f"⚠️  Skipping filtered row with blank NL: table_id={row.get('table_id')}")
            continue
        
        # Get score from result.rewards (list with one element per example)
        # Since filtering is called on one example at a time, rewards[0] is the score for this example
        score = result.rewards[0] if result.rewards else None
        
        # Add original row to original_results (not schema_changed_results)
        # For prompt filtering, sql and db_sql will be lists; for reward filtering, single values
        original_result = {
            "nl": nl,
            "sql": sql_queries if method == "prompt" and len(sql_queries) > 1 else sql,
            "db_sql": db_sql_queries if method == "prompt" and len(db_sql_queries) > 1 else dq_sql,
            "db_result": db_result,
            "table_id": row["table_id"],
            "filtered": method,
            "score": score,
            "reasoning": reasoning,
            "seeding_type": seeding_type,
            "seeding_value": seeding_value,
            "predicted_ambiguity": pred,
            "schema_change_type": None,
            "db_result_after_change": None,  # Not applicable for original rows
            "db_query_match": None,  # Not applicable for original rows
        }
        original_results.append(original_result)
        
        # Save original row immediately to CSV
        if csv_writer:
            # Skip if NL is blank
            if not original_result.get("nl") or not str(original_result.get("nl", "")).strip():
                print(f"⚠️  Skipping row with blank NL: table_id={original_result.get('table_id')}")
                continue
            
            # Convert db_result and lists to string for CSV
            csv_row = original_result.copy()
            if csv_row["db_result"] is not None:
                csv_row["db_result"] = str(csv_row["db_result"])
            # Convert sql and db_sql lists to string representation if they are lists
            if isinstance(csv_row["sql"], list):
                csv_row["sql"] = str(csv_row["sql"])
            if isinstance(csv_row["db_sql"], list):
                csv_row["db_sql"] = str(csv_row["db_sql"])
            csv_writer.writerow(csv_row)
            csv_file.flush()  # Ensure it's written to disk immediately
        
        # we change the schema for the same NL question, and append it to the results also
        # only done with non ambiguous queries
        if schema_change:
            print("Applying schema change...")
            # Pass table_id and db_query directly instead of extracting from query
            schema_changes = apply_schema_change(db_path, nl, sql, dq_sql, row["table_id"], db_result)
            for sc in schema_changes:
                # Compare original db_query (dq_sql) with new db_query (sc["db_sql"])
                new_db_sql = sc.get("db_sql", "")
                # Normalize both queries for comparison (remove whitespace differences)
                import re
                dq_sql_normalized = re.sub(r'\s+', ' ', str(dq_sql).strip()) if dq_sql else ""
                new_db_sql_normalized = re.sub(r'\s+', ' ', str(new_db_sql).strip()) if new_db_sql else ""
                db_query_match = dq_sql_normalized == new_db_sql_normalized
                
                # If they match, store True, otherwise store the new db_query
                db_query_match_value = True if db_query_match else new_db_sql
                
                # Skip if NL is blank (shouldn't happen since we check earlier, but double-check)
                if not nl or not str(nl).strip():
                    print(f"⚠️  Skipping schema change result with blank NL: table_id={sc.get('new_table_id')}")
                    continue
                
                schema_result = {
                    "nl": nl,
                    "sql": sc["new_query"],
                    "db_sql": sc["db_sql"],
                    "db_result": sc.get("query_result"),  # Database query result from schema change
                    "table_id": sc["new_table_id"],
                    "filtered": "N/A",
                    "score": score,
                    "reasoning": reasoning,
                    "seeding_type": seeding_type,
                    "seeding_value": seeding_value,
                    "schema_change_type": sc.get("change_type"),
                    "db_result_after_change": sc.get("query_result"),  # DB answer after the change
                    "db_query_match": db_query_match_value,  # True if matches, otherwise the new db_query
                }
                schema_changed_results.append(schema_result)
                
                # Save schema change row immediately to CSV
                # this is cause db results cant save directly to csv so we need to stringify them
                if csv_writer:
                    # Skip if NL is blank
                    if not schema_result.get("nl") or not str(schema_result.get("nl", "")).strip():
                        print(f"⚠️  Skipping schema change row with blank NL: table_id={schema_result.get('table_id')}")
                        continue
                    
                    # Convert db_result and db_result_after_change to string for CSV
                    csv_row = schema_result.copy()
                    if csv_row["db_result"] is not None:
                        csv_row["db_result"] = str(csv_row["db_result"])
                    if csv_row["db_result_after_change"] is not None:
                        csv_row["db_result_after_change"] = str(csv_row["db_result_after_change"])
                    # db_query_match is already a string (new db_query) or True, so no conversion needed
                    csv_writer.writerow(csv_row)
                    csv_file.flush()  # Ensure it's written to disk immediately
        else:
            print("No schema change applied")

    # Close CSV file if opened
    if csv_file:
        csv_file.close()
    
    # Return all original results first, then all schema-changed results
    return original_results + schema_changed_results

def run_pipeline(step=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Connect to database
    db_path = os.path.join(project_root, "data/train.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    client = genai.Client()
    tables = load_tables()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Get step from argument or prompt
    if step is None:
        while step != "1" and step != "2":
            step = input("Enter the step you want to run: 1. Generate Data, 2. Filter and Schema Change")
    elif step not in ["1", "2"]:
        raise ValueError(f"Invalid step: {step}. Must be '1' or '2'")
    
    # Define file paths (used by both steps)
    gen_file = "pipeline/final_dataset/gen_data.csv"
    
    # Ensure output directories exist
    os.makedirs("pipeline/final_dataset", exist_ok=True)
    os.makedirs("pipeline/final_translations", exist_ok=True)
           
    if step == "1":
        # Generate Data
        print("Generating data...")
        if os.path.exists(gen_file):
            print(f"Found existing generation file at {gen_file}. Will resume from last position.")
        generated_rows = generate_rows(client, tables, cur, output_csv_path=gen_file)
        # Only return if no rows were generated AND file doesn't exist (fresh start with no data)
        if not generated_rows and not os.path.exists(gen_file):
            print("No rows generated")
            return
    if step == "2":
        # Ambiguity Detection & Filtering
        filtered_file = "pipeline/final_dataset/filtered.csv"
        
        # Read the generated data
        df = pd.read_csv(gen_file)
        
        # Looking for last processed table in existing filtered.csv
        start_idx = 0
        existing_results = []
        if os.path.exists(filtered_file):
            print(f"existing filtered file at {filtered_file}. looking for last table_id")
            try:
                existing_df = pd.read_csv(filtered_file)
                if len(existing_df) > 0:
                    existing_results = existing_df.to_dict('records')
                    # Find the last table_id in filtered.csv (only original table_ids, not the m ones from schema changes)
                    last_table_id = None
                    for result in reversed(existing_results):
                        table_id = result.get("table_id", "")
                        if table_id and "m" not in str(table_id):
                            last_table_id = table_id
                            break
                    
                    if last_table_id:
                        # Find the last row index in gen_data.csv that has this table_id
                        # (only match original table_ids, not m ones)
                        # Then continue from the next row (which should be the next table)
                        last_idx_for_table = -1
                        for idx, row in df.iterrows():
                            table_id = row.get("table_id", "")
                            # Only match original table_ids (not schema-changed ones with m)
                            if table_id == last_table_id and "m" not in str(table_id):
                                last_idx_for_table = idx
                        
                        # Verify we found the table
                        if last_idx_for_table >= 0:
                            start_idx = last_idx_for_table + 1
                            print(f"Found last processed table_id: {last_table_id}")
                            print(f"Last occurrence at row index {last_idx_for_table} in gen_data.csv")
                            print(f"Resuming from row index {start_idx} (out of {len(df)} total rows)")
                            # Count how many tables have been processed
                            processed_table_ids = set()
                            for result in existing_results:
                                tid = result.get("table_id", "")
                                if tid and "m" not in str(tid):
                                    processed_table_ids.add(tid)
                            print(f"Already processed {len(processed_table_ids)} original tables")
                        else:
                            print(f"Warning: Last table_id {last_table_id} not found in gen_data.csv. Starting from beginning.")
                            start_idx = 0
                    else:
                        print("No original table_ids found in filtered.csv. Starting from beginning.")
                        start_idx = 0
            except Exception as e:
                print(f"Error reading existing filtered file: {e}. Starting from beginning.")
                start_idx = 0
                existing_results = []
        
        # Process remaining rows if any
        if start_idx < len(df):
            print(f"Processing remaining {len(df) - start_idx} rows...")
            print("Detecting ambiguity...")

            preds = ambiguity_detection(gen_file, tokenizer, device)
            
            # Slice to only process remaining rows
            df_remaining = df.iloc[start_idx:].copy()
            preds_remaining = preds[start_idx:]
            
            # Process remaining rows (pass filtered_file for incremental saving)
            new_results = filter_and_schema_change(df_remaining, preds_remaining, client, tables[0], db_path, cur, filtered_file=filtered_file)
            
            # Combine existing and new results
            results = existing_results + new_results
            
            # Results are already saved incrementally, just print summary
            print(f"Processed {len(new_results)} new results. Total: {len(results)} results in {filtered_file}")
        else:
            print("All rows have already been processed.")
            results = existing_results

    # Prepare Translation Buffer
    print("Preparing Translation Buffer...")
    translation_buffer = ResultBuffer()
    translation_buffer.nl = [r["nl"] for r in results]
    translation_buffer.sql = [r["sql"] for r in results]
    translation_buffer.db_sql = [r["db_sql"] for r in results]
    translation_buffer.db_result = [r.get("db_result") for r in results]  # Preserve original db_result for translation
    translation_buffer.table_id = [r["table_id"] for r in results]
    translation_buffer.reasoning = [r["reasoning"] for r in results]
    translation_buffer.seeding_type = [r["seeding_type"] for r in results]
    translation_buffer.seeding_value = [r["seeding_value"] for r in results]
    translation_buffer.filtered = [r["filtered"] for r in results]
    translation_buffer.schema_change_type = [r["schema_change_type"] for r in results]
    translation_buffer.db_result_after_change = [r["db_result_after_change"] for r in results]
    translation_buffer.db_query_match = [r["db_query_match"] for r in results]
    # Initialize rewards if not present in results (may not exist in filtered.csv)
    translation_buffer.rewards = [r.get("rewards") if "rewards" in r else None for r in results] if results and "rewards" in results[0] else [None] * len(results)
    # Save English (original) - includes ALL results: original queries + schema-changed variants
    en_file = "pipeline/final_translations/translated_en.csv"
    if os.path.exists(en_file):
        print(f"English file exists at {en_file}. Skipping save.")
    else:
        print("\nSaving English (original)...")
        # Filter out results with blank NL before creating DataFrame
        filtered_results = [r for r in results if r.get("nl") and str(r.get("nl", "")).strip()]
        if len(filtered_results) < len(results):
            print(f"⚠️  Filtered out {len(results) - len(filtered_results)} rows with blank NL")
        
        # Include all fields from results - original queries AND schema-changed variants
        english_data = {
            "nl": [r["nl"] for r in filtered_results],
            "sql": [r["sql"] for r in filtered_results],
            "db_sql": [r["db_sql"] for r in filtered_results],
            "db_result": [r.get("db_result") for r in filtered_results],
            "table_id": [r["table_id"] for r in filtered_results],
            "reasoning": [r.get("reasoning", "") for r in filtered_results],
            "seeding_type": [r.get("seeding_type", "") for r in filtered_results],
            "seeding_value": [r.get("seeding_value", "") for r in filtered_results],
            "filtered": [r.get("filtered", "") for r in filtered_results],
            "schema_change_type": [r.get("schema_change_type") for r in filtered_results],
            "db_result_after_change": [r.get("db_result_after_change") for r in filtered_results],
            "db_query_match": [r.get("db_query_match") for r in filtered_results],
        }
        pd.DataFrame(english_data).to_csv(en_file, index=False)
        print(f"Saved {len(filtered_results)} English rows to {en_file} (including schema-changed variants)")

    # Machine Translation - schema changes are ONLY applied to translated data, not English
    languages = ["es", "tr", "ru", "zh", "ja", "hi", "de", "uk", "kk", "ro"]
    
    # Get translation preference from argument or prompt

    for lang in languages:
        output_file = f"pipeline/final_translations/translated_{lang}.csv"
        
        # Check for existing translations and resume from last table_id
        start_idx = 0
        existing_translations = []
        is_resume = False
        
        if os.path.exists(output_file):
            print(f"📋 Translation file exists for {lang}. Checking for resume point...")
            try:
                existing_df = pd.read_csv(output_file)
                if len(existing_df) > 0:
                    existing_translations = existing_df.to_dict('records')
                    # Find the last table_id in the existing translations
                    last_table_id = None
                    for result in reversed(existing_translations):
                        table_id = result.get("table_id", "")
                        if table_id:
                            last_table_id = table_id
                            break
                    
                    if last_table_id:
                        # Find the last occurrence of this table_id in translation_buffer
                        last_idx_for_table = -1
                        for idx in range(len(translation_buffer.table_id)):
                            if translation_buffer.table_id[idx] == last_table_id:
                                last_idx_for_table = idx
                        
                        if last_idx_for_table >= 0:
                            start_idx = last_idx_for_table + 1
                            print(f"  Found last translated table_id: {last_table_id}")
                            print(f"  Last occurrence at index {last_idx_for_table} in translation buffer")
                            print(f"  Resuming from index {start_idx} (out of {len(translation_buffer.table_id)} total items)")
                            print(f"  Already translated: {len(existing_translations)} items")
                            print(f"  Remaining to translate: {len(translation_buffer.table_id) - start_idx} items")
                        else:
                            print(f"  Warning: Last table_id {last_table_id} not found in translation buffer. Starting from beginning.")
                            start_idx = 0
                    else:
                        print(f"  No table_ids found in existing translations. Starting from beginning.")
                        start_idx = 0
            except Exception as e:
                print(f"  Error reading existing translation file: {e}. Starting from beginning.")
                start_idx = 0
                existing_translations = []
        
        # Check if all items are already translated
        if start_idx >= len(translation_buffer.table_id):
            print(f"📋 All items already translated for {lang}. Skipping.")
            continue
        
        # If we're resuming, create a subset of the translation buffer
        if start_idx > 0:
            print(f"\n🌍 Resuming Machine Translation for {lang} from index {start_idx}...")
            # Create a new buffer with only the remaining items
            remaining_buffer = ResultBuffer()
            remaining_buffer.nl = translation_buffer.nl[start_idx:]
            remaining_buffer.sql = translation_buffer.sql[start_idx:]
            remaining_buffer.db_sql = translation_buffer.db_sql[start_idx:]
            remaining_buffer.db_result = translation_buffer.db_result[start_idx:] if hasattr(translation_buffer, 'db_result') else [None] * (len(translation_buffer.nl) - start_idx)
            remaining_buffer.table_id = translation_buffer.table_id[start_idx:]
            remaining_buffer.reasoning = translation_buffer.reasoning[start_idx:] if hasattr(translation_buffer, 'reasoning') else [None] * (len(translation_buffer.nl) - start_idx)
            remaining_buffer.seeding_type = translation_buffer.seeding_type[start_idx:] if hasattr(translation_buffer, 'seeding_type') else [None] * (len(translation_buffer.nl) - start_idx)
            remaining_buffer.seeding_value = translation_buffer.seeding_value[start_idx:] if hasattr(translation_buffer, 'seeding_value') else [None] * (len(translation_buffer.nl) - start_idx)
            remaining_buffer.filtered = translation_buffer.filtered[start_idx:] if hasattr(translation_buffer, 'filtered') else [None] * (len(translation_buffer.nl) - start_idx)
            remaining_buffer.schema_change_type = translation_buffer.schema_change_type[start_idx:] if hasattr(translation_buffer, 'schema_change_type') else [None] * (len(translation_buffer.nl) - start_idx)
            remaining_buffer.db_result_after_change = translation_buffer.db_result_after_change[start_idx:] if hasattr(translation_buffer, 'db_result_after_change') else [None] * (len(translation_buffer.nl) - start_idx)
            remaining_buffer.db_query_match = translation_buffer.db_query_match[start_idx:] if hasattr(translation_buffer, 'db_query_match') else [None] * (len(translation_buffer.nl) - start_idx)
            remaining_buffer.rewards = translation_buffer.rewards[start_idx:] if hasattr(translation_buffer, 'rewards') else [None] * (len(translation_buffer.nl) - start_idx)
            
            translated_buffer = machine_translation(remaining_buffer, lang=lang)
            
            # If we have new translations, we'll append them to existing ones later
            is_resume = True
        else:
            print(f"\n🌍 Starting Machine Translation for {lang}...")
            translated_buffer = machine_translation(translation_buffer, lang=lang)
            is_resume = False
        print(f"✅ Translation complete for {lang}: {len(translated_buffer.nl) if translated_buffer.nl else 0} items translated")

        if translated_buffer.nl:
            translated_data = {
                "nl": translated_buffer.nl,
                "sql": translated_buffer.sql,
                "db_sql": translated_buffer.db_sql,
                "db_result": translated_buffer.db_result if hasattr(translated_buffer, 'db_result') else [None] * len(translated_buffer.nl),
                "table_id": translated_buffer.table_id,
                "reasoning": translated_buffer.reasoning,
                "seeding_type": translated_buffer.seeding_type,
                "seeding_value": translated_buffer.seeding_value,
                "filtered": translated_buffer.filtered,
                "schema_change_type": translated_buffer.schema_change_type,
                "db_result_after_change": translated_buffer.db_result_after_change,
                "db_query_match": translated_buffer.db_query_match,
                "schema_translation_type": None,

            }
            
            # Schema change to translate the schema sometimes - ONLY for translated data, not English
            # Find acceptable schema translation languages for the target language. Based on geopolitical and business influence in countries where these languages are present 
            schema_change_languages = {"tr": {"tr" : .8, "de" : .2}, "ru" : {"ru" : .8, "zh" : .2}, "zh" :{"zh" : .8, "ru" : .2}, "ja" : {"ja" : .8, "zh" : .2}, "hi" : {"hi": .8, "hi_typed" : .2}, "de" : {"de": .8, "fr": .2}, "uk" : {"uk" : .5 , "ru" : .5}, "kk" : {"kk": .5, "ru": .5}, "ro" : {"ro": .5, "ru": .25, "fr": .25}}
            
            # Only apply schema changes if this language has schema change options defined
            # This applies schema translation to the translated NL/SQL pairs, creating variants with translated table/column names
            if lang in schema_change_languages:
                print(f"🔄 Starting schema changes for {lang} (applying to {len(translated_buffer.nl)} items)...")
                # note! not doing schema changes for prompt filtered ones! 
                # Convert dict to list for iteration
                db_results = translated_buffer.db_result if hasattr(translated_buffer, 'db_result') else [None] * len(translated_buffer.nl)
                filtered_list = translated_buffer.filtered if hasattr(translated_buffer, 'filtered') else [None] * len(translated_buffer.nl)
                translated_data_list = [{"nl": nl, "sql": sql, "db_sql": db_sql, "db_result": db_result, "table_id": tid, "filtered": filtered} 
                                       for nl, sql, db_sql, db_result, tid, filtered in zip(translated_buffer.nl, translated_buffer.sql, 
                                                                      translated_buffer.db_sql, db_results, translated_buffer.table_id, filtered_list)]
                
                schema_changes_applied = 0
                schema_changes_failed = 0
                for idx, i in enumerate(translated_data_list):
                    if (idx + 1) % 10 == 0:
                        print(f"  📊 Schema change progress: {idx + 1}/{len(translated_data_list)} items processed, {schema_changes_applied} applied, {schema_changes_failed} failed")
                    
                    # Skip if this was prompt filtered (note: not doing schema changes for prompt filtered ones!)
                    filtered_value = i.get("filtered", "")
                    if filtered_value == "prompt":
                        print(f"  ⏭️  Skipping schema change for prompt-filtered item: {i.get('table_id')}")
                        continue
                    
                    change = random.choice([0, 1])
                    if change == 0:
                        change_lang = random.choices(list(schema_change_languages[lang].keys()), 
                                                    weights=list(schema_change_languages[lang].values()))[0]
                        print(f"  🔧 Applying schema change to item {idx + 1}/{len(translated_data_list)}: translating schema to {change_lang}...")
                        schema_changes = apply_schema_change_translation(db_path, i["nl"], i["sql"], change_lang)
                        if schema_changes:
                            schema_change = schema_changes[0]  # Get first result
                            translated_data_list.append({
                                "nl": i["nl"],
                                "sql": schema_change["new_query"],
                                "db_sql": schema_change["new_query"],
                                "db_result": schema_change.get("query_result"),  # Database query result from schema translation
                                "table_id": schema_change["new_table_id"],
                                "reasoning": i["reasoning"],
                                "seeding_type": i["seeding_type"],
                                "seeding_value": i["seeding_value"],
                                "filtered": i["filtered"],
                                "schema_change_type": i["schema_change_type"],
                                "db_result_after_change": schema_change.get("query_result"),
                                "db_query_match": schema_change["new_query"],
                                "schema_translation_type": change_lang

                            })
                            schema_changes_applied += 1
                            print(f"  ✅ Schema change applied successfully for item {idx + 1}")
                        else:
                            schema_changes_failed += 1
                            print(f"  ⚠️  Schema change failed for item {idx + 1} (no results returned)")
                
                print(f"✅ Schema changes complete for {lang}: {schema_changes_applied} applied, {schema_changes_failed} failed")
                
                # Filter out items with blank NL before creating DataFrame
                filtered_translated_list = [item for item in translated_data_list if item.get("nl") and str(item.get("nl", "")).strip()]
                if len(filtered_translated_list) < len(translated_data_list):
                    print(f"⚠️  Filtered out {len(translated_data_list) - len(filtered_translated_list)} translated items with blank NL")
                
                # Convert back to dict format for DataFrame
                translated_data = {
                    "nl": [item["nl"] for item in filtered_translated_list],
                    "sql": [item["sql"] for item in filtered_translated_list],
                    "db_sql": [item["db_sql"] for item in filtered_translated_list],
                    "db_result": [item.get("db_result") for item in filtered_translated_list],
                    "table_id": [item["table_id"] for item in filtered_translated_list],
                    "schema_change_type": [item.get("schema_change_type") for item in filtered_translated_list],
                    "reasoning": [item.get("reasoning") for item in filtered_translated_list],
                    "seeding_type": [item.get("seeding_type") for item in filtered_translated_list],
                    "seeding_value": [item.get("seeding_value") for item in filtered_translated_list],
                    "filtered": [item.get("filtered") for item in filtered_translated_list],
                    "db_result_after_change": [item.get("db_result_after_change") for item in filtered_translated_list],
                    "db_query_match": [item.get("db_query_match") for item in filtered_translated_list],
                    "schema_translation_type": [item.get("schema_translation_type") for item in filtered_translated_list],
                }
            else:
                print(f"ℹ️  No schema changes configured for {lang}, skipping schema translation step")
                # Filter out items with blank NL before creating DataFrame
                filtered_nl = [nl for nl in translated_data["nl"] if nl and str(nl).strip()]
                if len(filtered_nl) < len(translated_data["nl"]):
                    print(f"⚠️  Filtered out {len(translated_data['nl']) - len(filtered_nl)} translated items with blank NL")
                    # Filter all fields to match
                    indices_to_keep = [i for i, nl in enumerate(translated_data["nl"]) if nl and str(nl).strip()]
                    translated_data = {key: [values[i] for i in indices_to_keep] for key, values in translated_data.items()}
            
            # If resuming, append to existing translations; otherwise create new file
            if is_resume and existing_translations:
                print(f"💾 Appending {len(translated_data['nl'])} new translated rows to existing {len(existing_translations)} rows...")
                # Combine existing and new translations
                combined_data = {}
                for key in translated_data.keys():
                    existing_values = [item.get(key) for item in existing_translations if key in item]
                    combined_data[key] = existing_values + translated_data[key]
                
                pd.DataFrame(combined_data).to_csv(output_file, index=False)
                print(f"✅ Saved {len(combined_data['nl'])} total translated rows to {output_file} (appended {len(translated_data['nl'])} new rows)")
            else:
                print(f"💾 Saving {len(translated_data['nl'])} translated rows to {output_file}...")
                pd.DataFrame(translated_data).to_csv(output_file, index=False)
                print(f"✅ Saved {len(translated_data['nl'])} translated rows to {output_file}")
        else:
            print(f"⚠️  No translated rows generated for {lang}.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data generation pipeline")
    parser.add_argument("--step", type=str, choices=["1", "2"], 
                        help="Step to run: 1. Generate Data, 2. Filter and Schema Change")
    
    args = parser.parse_args()
    
    run_pipeline(step=args.step)