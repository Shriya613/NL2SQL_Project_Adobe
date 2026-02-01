"""
Script to add table_data column to human_eval_filtering_test.csv
Extracts table IDs from SQL queries and matches them with table data from train.tables.jsonl
"""
import os
import re
import json
import pandas as pd
from tqdm import tqdm


def extract_table_ids_from_sql(sql_query: str) -> list:
    """Extract all table IDs from SQL query (e.g., table_1_11250_4 -> 1-11250-4)."""
    # Pattern to match table_X_Y_Z where X, Y, Z are numbers
    pattern = r'`?table_(\d+)_(\d+)_(\d+)`?'
    matches = re.findall(pattern, sql_query, re.IGNORECASE)
    
    # Convert to format used in JSONL: X-Y-Z
    table_ids = [f"{m[0]}-{m[1]}-{m[2]}" for m in matches]
    return list(set(table_ids))  # Remove duplicates


def load_table_data(jsonl_path: str) -> dict:
    """Load all table data from JSONL file into a dictionary keyed by table ID."""
    table_data = {}
    print(f"📂 Loading table data from {jsonl_path}...")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading tables"):
            try:
                table = json.loads(line.strip())
                table_id = table.get('id')
                if table_id:
                    table_data[table_id] = table
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing line: {e}")
                continue
    
    print(f"✅ Loaded {len(table_data)} tables")
    return table_data


def find_table_data_for_sql(sql_query: str, table_data_dict: dict) -> dict:
    """Find table data for a SQL query. Returns the first matching table or None."""
    table_ids = extract_table_ids_from_sql(sql_query)
    
    if not table_ids:
        return None
    
    # Return the first matching table (most queries use one table)
    for table_id in table_ids:
        if table_id in table_data_dict:
            return table_data_dict[table_id]
    
    # If no exact match, try to find any table with similar ID
    # (sometimes the format might be slightly different)
    return None


def main():
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    test_csv_path = os.path.join(project_root, "data/human_eval_filtering_test.csv")
    jsonl_path = os.path.join(project_root, "data/train.tables.jsonl")
    output_path = os.path.join(project_root, "data/human_eval_filtering_test.csv")
    
    # Load table data
    table_data_dict = load_table_data(jsonl_path)
    
    # Load test CSV
    print(f"📂 Loading test CSV from {test_csv_path}...")
    df = pd.read_csv(test_csv_path)
    print(f"✅ Loaded {len(df)} rows")
    
    # Extract table data for each row
    print("🔍 Extracting table data from SQL queries...")
    table_data_list = []
    found_count = 0
    not_found_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        sql_query = str(row['sql_queries'])
        table_info = find_table_data_for_sql(sql_query, table_data_dict)
        
        if table_info:
            # Convert to JSON string for storage
            table_data_list.append(json.dumps(table_info))
            found_count += 1
        else:
            table_data_list.append(None)
            not_found_count += 1
            # Print warning for first few not found
            if not_found_count <= 5:
                table_ids = extract_table_ids_from_sql(sql_query)
                print(f"⚠️  No table data found for SQL (table IDs: {table_ids})")
    
    # Add table_data column as first column
    df.insert(0, 'table_data', table_data_list)
    
    # Save updated CSV
    print(f"💾 Saving updated CSV to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} rows")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   Total rows: {len(df)}")
    print(f"   Table data found: {found_count} ({found_count/len(df)*100:.2f}%)")
    print(f"   Table data not found: {not_found_count} ({not_found_count/len(df)*100:.2f}%)")


if __name__ == "__main__":
    main()

