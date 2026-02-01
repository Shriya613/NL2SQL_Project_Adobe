"""
Main route for schema change operations.
This file can be called with a database path and a list of schema changes to apply.
It then returns the updated database information in a JSON format.
"""
import os
import json
import random
import sqlite3
import shutil
from typing import List, Dict, Any
import tqdm
import sys

from multilingual_evoschema.schema_change import (
    convert_table_to_schema,
    add_column,
    remove_column,
    rename_column,
    rename_table,
    split_column,
    merge_column_in_query,
    merge_column_outside_query,
    parse_schema_change,
    repair_query_rename,
    repair_query_column_split

)
from multilingual_evoschema.db_manager import add_new_table_to_db
from multilingual_evoschema.llm import run_llm
from multilingual_evoschema.llm import run_gpt
from openai import OpenAI
import re


def has_operation_on_multiple_columns(query: str) -> bool:
    """
    Check if the SQL query has operations that combine 2 or more columns together.
    This means columns used together in the same operation, not separate operations on different columns.
    This is to let us know if we can merge 2 columns in the query nicely
    Examples that SHOULD return True:
    - GROUP BY col1, col2 (multiple columns grouped together)
    - ORDER BY col1, col2 (multiple columns ordered together)
    - WHERE col1 = X AND col2 = Y (multiple columns in conditions)
    - SELECT col1, col2 (multiple columns selected together, not in aggregations)
    
    Examples that SHOULD return False:
    - SELECT COUNT(col1), AVG(col2) (separate operations on different columns)
    - SELECT col1, COUNT(col2) (one plain column, one aggregation - different operations)
    """
    query_upper = query.upper()
    
    # Check for GROUP BY with multiple columns (comma-separated)
    # This means columns are grouped together
    group_by_match = re.search(r'GROUP\s+BY\s+([^,\s]+(?:\s*,\s*[^,\s]+)+)', query_upper)
    if group_by_match:
        columns = [col.strip() for col in group_by_match.group(1).split(',')]
        # Filter out function calls and keep only column references
        column_refs = [col for col in columns if not re.search(r'\(|\)', col)]
        if len(column_refs) >= 2:
            return True
    
    # Check for ORDER BY with multiple columns
    # This means columns are ordered together
    order_by_match = re.search(r'ORDER\s+BY\s+([^,\s]+(?:\s*,\s*[^,\s]+)+)', query_upper)
    if order_by_match:
        columns = [col.strip() for col in order_by_match.group(1).split(',')]
        column_refs = [col for col in columns if not re.search(r'\(|\)', col)]
        if len(column_refs) >= 2:
            return True
    
    # Check for SELECT with multiple non-aggregation columns (columns selected together, not in functions)
    # This means plain columns like "SELECT col1, col2" not "SELECT COUNT(col1), AVG(col2)"
    # Also ignore queries that only use COUNT(*) or similar with * (no actual column references)
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query_upper, re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        # Split by comma to get individual select items
        select_items = [item.strip() for item in select_clause.split(',')]
        
        # Check if all items are subqueries or aggregations with only * (like COUNT(*))
        all_use_only_asterisk = True
        for item in select_items:
            # Remove AS aliases
            item_clean = re.sub(r'\s+AS\s+[^,\s]+', '', item, flags=re.IGNORECASE).strip()
            # Check if it's a subquery
            if re.search(r'\(SELECT', item_clean, re.IGNORECASE):
                # Check if subquery only uses COUNT(*) or similar with *
                if not re.search(r'COUNT\s*\(\s*\*\s*\)', item_clean, re.IGNORECASE):
                    all_use_only_asterisk = False
                    break
            # Check if it's an aggregation with * (like COUNT(*))
            elif re.search(r'\b(COUNT|SUM|AVG|MAX|MIN|GROUP_CONCAT)\s*\(\s*\*\s*\)', item_clean, re.IGNORECASE):
                continue  # This uses only *, skip it
            else:
                # Not a COUNT(*) pattern, might have actual column references
                all_use_only_asterisk = False
                break
        
        # If all items only use *, no actual column references, return False
        if all_use_only_asterisk:
            return False
        
        # Count items that are plain column references (not aggregations)
        plain_column_count = 0
        for item in select_items:
            # Remove AS aliases
            item = re.sub(r'\s+AS\s+[^,\s]+', '', item, flags=re.IGNORECASE)
            item = item.strip()
            
            # Skip subqueries - they're handled separately
            if re.search(r'\(SELECT', item, re.IGNORECASE):
                continue
            
            # Skip aggregations with only * (like COUNT(*))
            if re.search(r'\b(COUNT|SUM|AVG|MAX|MIN|GROUP_CONCAT)\s*\(\s*\*\s*\)', item, re.IGNORECASE):
                continue
            
            # Check if it's a plain column (backticked or plain name) without function calls
            # Pattern: `column_name` or column_name (not inside function parentheses)
            if re.match(r'^`[^`]+`$', item) or re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', item):
                plain_column_count += 1
            # Check if it's a column reference that's not inside an aggregation function
            # Look for patterns like "MAX(col) - MIN(col)" which are operations on the same column
            # vs "col1, col2" which are multiple columns together
            elif not re.search(r'\b(COUNT|SUM|AVG|MAX|MIN|GROUP_CONCAT)\s*\(', item, re.IGNORECASE):
                # Check if it contains multiple column references (like arithmetic operations)
                column_refs = re.findall(r'`([^`]+)`|(\b[A-Za-z_][A-Za-z0-9_]*\b)', item)
                # Filter out keywords
                keywords = ['AS', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL']
                unique_cols = set()
                for match in column_refs:
                    col_name = (match[0] if match[0] else match[1]).strip()
                    if col_name and col_name.upper() not in keywords:
                        unique_cols.add(col_name.lower())
                if len(unique_cols) >= 2:
                    # Multiple columns in the same expression (like arithmetic)
                    return True
        
        # If we have 2+ plain columns selected together (not in aggregations)
        if plain_column_count >= 2:
            return True
    
    # Check for WHERE/HAVING with multiple column conditions connected by AND/OR
    # This means conditions on different columns combined together
    where_match = re.search(r'(WHERE|HAVING)\s+(.*?)(?:\s+(?:GROUP|ORDER|LIMIT|$))', query_upper, re.DOTALL | re.IGNORECASE)
    if where_match:
        where_clause = where_match.group(2)
        # Count distinct column references in conditions
        # Look for patterns like "col1 = X AND col2 = Y" or "col1 IN (...) AND col2 > Z"
        column_pattern = r'`([^`]+)`|(\b[A-Za-z_][A-Za-z0-9_]*\b)'
        matches = re.findall(column_pattern, where_clause)
        # Filter out operators and keywords
        keywords = ['AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL', 'TRUE', 'FALSE', 'SELECT', 'FROM']
        column_refs = []
        for match in matches:
            col_name = match[0] if match[0] else match[1]
            if col_name and col_name.upper() not in keywords:
                column_refs.append(col_name)
        unique_cols = set([col.lower() for col in column_refs])
        if len(unique_cols) >= 2:
            return True
    
    return False


def extract_table_name_from_query(query: str) -> str:
    """Extract table name from SQL query."""
    # Try to find table name in FROM clause
    from_match = re.search(r'FROM\s+`?([^`\s,]+)`?', query, re.IGNORECASE)
    if from_match:
        return from_match.group(1)
    # Try JOIN clause
    join_match = re.search(r'JOIN\s+`?([^`\s,]+)`?', query, re.IGNORECASE)
    if join_match:
        return join_match.group(1)
    return None


def extract_columns_from_query(query: str) -> set:
    """
    Extract all column names referenced in a SQL query.
    Returns a set of column names (without backticks, case-insensitive).
    """
    columns = set()
    
    # Pattern to match column names: `column_name` or column_name
    # Exclude SQL keywords
    keywords = {
        'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'ORDER', 'HAVING', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
        'ON', 'AS', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL', 'TRUE', 'FALSE',
        'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP_CONCAT', 'DISTINCT', 'LIMIT', 'OFFSET'
    }
    
    # Find all backticked column names: `column_name`
    backticked = re.findall(r'`([^`]+)`', query)
    for col in backticked:
        if col.upper() not in keywords:
            columns.add(col)
    
    # Find column names in SELECT, WHERE, GROUP BY, ORDER BY clauses
    # Match word boundaries but exclude keywords
    word_pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\b'
    matches = re.findall(word_pattern, query, re.IGNORECASE)
    for match in matches:
        if match.upper() not in keywords:
            columns.add(match)
    
    return columns


def filter_schema_exclude_columns(schema: str, exclude_columns: set) -> str:
    """
    Filter a schema string to exclude specified columns.
    
    Args:
        schema: Schema string in format "Table: ...\nColumns:\n- col1 (TYPE)\n- col2 (TYPE)..."
        exclude_columns: Set of column names to exclude (case-insensitive)
    
    Returns:
        Filtered schema string with excluded columns removed
    """
    if not exclude_columns:
        return schema
    
    # Normalize exclude_columns to lowercase for comparison
    exclude_lower = {col.lower() for col in exclude_columns}
    
    lines = schema.strip().split('\n')
    filtered_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        # Check if this is a column line: "- column_name (TYPE)"
        if line_stripped.startswith('-') and '(' in line_stripped:
            match = re.match(r'\s*-\s*([^(]+)\s*\(', line_stripped)
            if match:
                col_name = match.group(1).strip()
                # Check if this column should be excluded
                if col_name.lower() not in exclude_lower:
                    filtered_lines.append(line)
                continue
        # Keep non-column lines (Table:, Columns:, etc.)
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def filter_table_sample_exclude_columns(table_sample: dict, exclude_columns: set) -> dict:
    """
    Filter table sample to exclude specified columns.
    
    Args:
        table_sample: Table dict with 'header', 'types', 'rows' OR JSON string
        exclude_columns: Set of column names to exclude (case-insensitive)
    
    Returns:
        Filtered table dict with excluded columns removed
    """
    if not exclude_columns or not table_sample:
        return table_sample
    
    # Parse JSON string if needed
    if isinstance(table_sample, str):
        try:
            table_sample = json.loads(table_sample)
        except json.JSONDecodeError:
            return table_sample
    
    # Normalize exclude_columns to lowercase for comparison
    exclude_lower = {col.lower() for col in exclude_columns}
    
    # Create a copy to avoid modifying original
    filtered_table = json.loads(json.dumps(table_sample)) if isinstance(table_sample, dict) else table_sample
    
    headers = filtered_table.get('header', [])
    types = filtered_table.get('types', [])
    rows = filtered_table.get('rows', [])
    
    # Find indices to keep
    indices_to_keep = []
    for i, header in enumerate(headers):
        if header.lower() not in exclude_lower:
            indices_to_keep.append(i)
    
    # Filter headers and types
    filtered_table['header'] = [headers[i] for i in indices_to_keep]
    filtered_table['types'] = [types[i] if i < len(types) else 'text' for i in indices_to_keep]
    
    # Filter rows
    filtered_table['rows'] = []
    for row in rows:
        filtered_row = [row[i] if i < len(row) else None for i in indices_to_keep]
        filtered_table['rows'].append(filtered_row)
    
    return filtered_table


def get_table_from_jsonl(table_id: str, tables_jsonl_path: str = "data/train.tables.jsonl") -> Dict[str, Any]:
    """Get table from JSONL file by ID."""
    if not os.path.exists(tables_jsonl_path):
        return None
    
    with open(tables_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    table = json.loads(line)
                    # Check if ID matches (handle both with and without table_ prefix)
                    table_id_normalized = table_id.replace('table_', '').replace('_', '-')
                    current_id_normalized = table.get('id', '').replace('_', '-')
                    if current_id_normalized == table_id_normalized:
                        return table
                except json.JSONDecodeError:
                    continue
    return None


def apply_single_schema_change(
    db_path: str,
    original_schema: str,
    original_table: Dict[str, Any],
    nl: str,
    query: str,
    change_type: str,
    tables_jsonl_path: str = "data/train.tables.jsonl"
) -> Dict[str, Any]:
    """
    Apply a single schema change and create a new modified table.
    
    Args:
        db_path: Path to SQLite database
        original_schema: Schema string from convert_table_to_schema
        original_table: Original table dict from JSONL
        nl: Natural language question
        query: SQL query string
        change_type: Type of schema change to apply
        tables_jsonl_path: Path to train.tables.jsonl file
        
    Returns:
        Dict with keys: 'db_id', 'question', 'SQL' (matching train_bird.json format) or None if failed
    """
    try:
        # Start with original schema and query (each change starts fresh)
        current_schema = original_schema
        current_query = query
        current_table = json.loads(json.dumps(original_table))  # Deep copy
        print("🐁" * 80)
        print(f"  SCHEMA CHANGE: {change_type}")

        
        # Generate prompt for schema change
        table_sample_json = json.dumps(current_table)
        change_info = None
        skip_llm = False
        
        if change_type == "add_column":
            prompt = add_column(current_schema, table_sample_json)
        elif change_type == "remove_column":
            # For remove_column, we need to check if column is in query first
            # Get the column to remove (randomly selected in remove_column function)
            # But we need to know which column before calling LLM
            # So we'll extract it from the prompt generation
            
            # Parse schema to extract column names (same logic as remove_column)
            column_names = []
            schema_lines = current_schema.split('\n')
            for line in schema_lines:
                match = re.match(r'\s*-\s*(.+?)\s*\(', line)
                if match:
                    column_names.append(match.group(1).strip())
            
            if not column_names and current_table.get('header'):
                column_names = current_table['header']
            
            if not column_names:
                print(f"    ⚠️  Could not extract column names, skipping")
                return None
            
            # Randomly choose a column to remove (without LLM)
            column_to_remove = random.choice(column_names)
            print(f"    🎲 Randomly selected column to remove: {column_to_remove}")
            
            # Manually check if the column is used in the query (without LLM)
            escaped_col = re.escape(column_to_remove)
            column_in_query = bool(re.search(rf'`?{escaped_col}`?', current_query, re.IGNORECASE))
            
            if not column_in_query:
                # Column NOT in query - skip LLM, handle removal manually
                print(f"    ✅ Column '{column_to_remove}' is NOT in query - skipping LLM call")
                skip_llm = True
                # Convert to db_sql format (with generic names)
                conversion = {current_table["header"][i]: f"col{i}" for i in range(len(current_table["header"]))}
                db_sql = current_query
                for original, replacement in sorted(conversion.items(), key=lambda x: len(x[0]), reverse=True):
                    db_sql = re.sub(rf'`{re.escape(original)}`', replacement, db_sql, flags=re.IGNORECASE)
                change_info = {
                    "removed_column": column_to_remove,
                    "new_query": current_query,  # Query with JSONL column names
                    "db_sql": db_sql  # Query with database placeholder names
                }
            else:
                # Column IS in query - need LLM to check if NL can still be answered
                if not nl:
                    print(f"    ⚠️  Column '{column_to_remove}' IS in query but NL is not provided - cannot determine if question can still be answered, skipping")
                    return None
                print(f"    ⚠️  Column '{column_to_remove}' IS in query - will ask LLM if NL can still be answered")
                # Pass removed_column in change_info so it's available after LLM call
                change_info = {"removed_column": column_to_remove}
                prompt = remove_column(current_schema, current_query, nl, table_sample_json, column_to_remove=column_to_remove)
        elif change_type == "rename_column":
            prompt = rename_column(current_schema, table_sample_json)
        elif change_type == "split_column":
            prompt = split_column(current_schema, current_query, table_sample_json)
        elif change_type == "merge_column":
            # Check if query has operations on 2+ columns
            if has_operation_on_multiple_columns(current_query):
                print(f"    🔍 Query has operations on 2+ columns, using merge_column_in_query")
                prompt = merge_column_in_query(current_schema, current_query, table_sample_json)
            else:
                print(f"    🔍 Query does not have operations on 2+ columns, using merge_column_outside_query")
                # Extract columns used in query and filter schema/table_sample to exclude them
                columns_in_query = extract_columns_from_query(current_query)
                print(f"    📋 Columns used in query (will be excluded from merge): {sorted(columns_in_query)}")
                
                # Filter schema to exclude columns used in query
                filtered_schema = filter_schema_exclude_columns(current_schema, columns_in_query)
                
                # Filter table_sample to exclude columns used in query
                filtered_table_sample = filter_table_sample_exclude_columns(table_sample_json, columns_in_query)
                
                prompt = merge_column_outside_query(filtered_schema, filtered_table_sample)
        else:
            print(f"    ⚠️  Unknown change type: {change_type}")
            return None
        
        # Call LLM only if needed
        if not skip_llm:
            # Debug: Print schema being sent to LLM
            print(f"    📋 Schema being sent to LLM:")
            print(f"       {current_schema.replace(chr(10), ' | ')}")
            
            # Print full prompt for remove_column operations
            if change_type == "remove_column":
                print(f"\n    {'='*80}")
                print(f"    FULL LLM PROMPT FOR REMOVE_COLUMN:")
                print(f"    {'='*80}")
                print(f"    {prompt.replace(chr(10), chr(10) + '    ')}")
                print(f"    {'='*80}\n")
            
            # Call LLM to get schema change
            # Use OpenAI for split/merge/remove_column operations, Qwen for others
            if change_type in ["split_column", "merge_column", "remove_column"]:
                
                print(f"    🦙 Using OpenAI for {change_type}")
                response = run_gpt(prompt)
    
                print(f"    🤖 OpenAI Response: {response}")
            else:
                response = run_llm(prompt)
                print(f"    🤖 Qwen Response: {response}")
            parsed_response = parse_schema_change(response)
            
            if not parsed_response:
                print(f"    ⚠️  Failed to parse {change_type} response, skipping")
                return None
            
            # Merge parsed response with any pre-set change_info (like removed_column)
            if change_info:
                change_info.update(parsed_response)
            else:
                change_info = parsed_response
        
        # Apply the schema change to the table structure using refactored functions
        from multilingual_evoschema.schema_change import (
            apply_add_column_change,
            apply_remove_column_change,
            apply_rename_column_change,
            apply_split_column_change,
            apply_merge_column_change
        )
        from multilingual_evoschema.db_manager import convert_query_to_db_format, update_query_table_name
        
        if change_type == "add_column":
            current_table, current_schema = apply_add_column_change(current_table, change_info)
            
        elif change_type == "remove_column":
            result = apply_remove_column_change(current_table, change_info, current_query)
            if result[0] is None:
                return None
            current_table, current_schema, current_query = result
            # Generate db_sql if not already set
            if "db_sql" not in change_info:
                conversion = {current_table["header"][i]: f"col{i}" for i in range(len(current_table["header"]))}
                db_sql = current_query
                for original, replacement in sorted(conversion.items(), key=lambda x: len(x[0]), reverse=True):
                    db_sql = re.sub(rf'`{re.escape(original)}`', replacement, db_sql, flags=re.IGNORECASE)
                change_info["db_sql"] = db_sql
                
        elif change_type == "rename_column":
            current_table, current_schema, current_query = apply_rename_column_change(current_table, change_info, current_query)
                
        
                
        elif change_type == "split_column":
            current_table, current_schema, current_query = apply_split_column_change(current_table, change_info, current_query, extract_table_name_from_query)
                    
        elif change_type == "merge_column":
            current_table, current_schema, current_query = apply_merge_column_change(current_table, change_info, query, has_operation_on_multiple_columns)
        
        # Add modified table to database
        # Ensure table name is normalized
        current_table["name"] = f"table_{current_table.get('id', 'unknown').replace('-', '_')}"
        success, modified_table_id, modified_table_name = add_new_table_to_db(
            schema=current_schema,
            table_sample=current_table,
            modification_info=None,
            db_path=db_path,
            tables_jsonl_path=tables_jsonl_path
        )
        
        if not success or not modified_table_id:
            print(f"    ❌ Failed to add modified table to database")
            return None
        
        # Test SQL query execution
        conn = sqlite3.connect(db_path, timeout=5.0)
        cur = conn.cursor()
        
        # Use the modified table name returned from add_new_table_to_db
        # This ensures we use the exact name that was created in the database
        db_table_name = modified_table_name
        
        # Initialize flag for column matching (will be set during verification)
        original_columns_match = False
        
        # Verify the table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (db_table_name,))
        result = cur.fetchone()
        if not result:
            print(f"    ⚠️  Table '{db_table_name}' not found in database")
            conn.close()
            return None
        
        # Compare ALL rows of original and modified tables
        print(f"\n{'🥭' * 80}")
        print(f"{'🥭' * 80}")
        print(f"    📊 COMPARING ORIGINAL vs MODIFIED TABLE (ALL Rows)")
        print(f"{'🥭' * 80}")
        print(f"{'🥭' * 80}\n")
        
        # Get original table data (ALL rows)
        original_headers = original_table.get("header", [])
        original_rows = original_table.get("rows", [])
        
        # Get modified table data from database (ALL rows)
        cur.execute(f"SELECT * FROM `{db_table_name}`")
        modified_db_rows = cur.fetchall()
        modified_db_columns = [description[0] for description in cur.description]  # col0, col1, etc.
        
        # Get modified table headers (from current_table after schema change)
        modified_headers = current_table.get("header", [])
        modified_rows = current_table.get("rows", [])
        
        # Determine which columns existed in original (for comparison)
        original_col_count = len(original_headers)
        
        print(f"    📋 ORIGINAL TABLE:")
        print(f"       Headers: {original_headers}")
        print(f"       Total rows: {len(original_rows)}")
        print(f"       First 2 rows:")
        for i, row in enumerate(original_rows[:2], 1):
            print(f"          Row {i}: {row}")
        if len(original_rows) > 2:
            print(f"       ... ({len(original_rows) - 2} more rows)")
        
        print(f"\n    📋 MODIFIED TABLE:")
        print(f"       Headers: {modified_headers}")
        print(f"       Total rows: {len(modified_rows)}")
        print(f"       First 2 rows:")
        for i, row in enumerate(modified_rows[:2], 1):
            print(f"          Row {i}: {row}")
        if len(modified_rows) > 2:
            print(f"       ... ({len(modified_rows) - 2} more rows)")
        
        # Verify row counts match
        if len(original_rows) != len(modified_rows):
            print(f"\n    ❌ ERROR: Row counts don't match!")
            print(f"       Original: {len(original_rows)} rows")
            print(f"       Modified: {len(modified_rows)} rows")
            print(f"\n{'🥭' * 80}")
            print(f"{'🥭' * 80}\n")
        else:
            print(f"\n    ✅ Row counts match: {len(original_rows)} rows")
        
        # For merge_column, compare by column name (not index) since structure changes
        if change_type == "merge_column":
            # Get merged columns from change_info
            merged_cols = change_info.get("original_columns", []) if change_info else []
            new_merged_col = change_info.get("new_column", "") if change_info else ""
            
            print(f"\n    🔍 VERIFICATION (comparing non-merged columns by name, ALL rows):")
            print(f"       Merged columns: {merged_cols} → {new_merged_col}")
            print(f"       Comparing non-merged columns only")
            
            all_match = True
            mismatched_rows = []
            
            # Create mapping of original column names to indices
            original_col_to_idx = {col: idx for idx, col in enumerate(original_headers)}
            # Create mapping of modified column names to indices
            modified_col_to_idx = {col: idx for idx, col in enumerate(modified_headers)}
            
            # Get non-merged columns to compare
            non_merged_cols = [col for col in original_headers if col not in merged_cols]
            
            for row_idx in range(len(original_rows)):
                original_row = original_rows[row_idx]
                modified_row = modified_rows[row_idx] if row_idx < len(modified_rows) else None
                
                if modified_row is None:
                    print(f"       ❌ Row {row_idx + 1}: MISSING in modified table!")
                    all_match = False
                    mismatched_rows.append(row_idx + 1)
                    continue
                
                # Compare non-merged columns by name
                row_matches = True
                for col_name in non_merged_cols:
                    orig_idx = original_col_to_idx.get(col_name)
                    mod_idx = modified_col_to_idx.get(col_name)
                    
                    if orig_idx is None or mod_idx is None:
                        continue  # Column doesn't exist in one of the tables
                    
                    orig_val = original_row[orig_idx] if orig_idx < len(original_row) else None
                    mod_val = modified_row[mod_idx] if mod_idx < len(modified_row) else None
                    
                    if orig_val != mod_val:
                        row_matches = False
                        if row_idx < 2:  # Only print first 2 rows
                            print(f"       ❌ Row {row_idx + 1}, Column '{col_name}': MISMATCH!")
                            print(f"          Original: {orig_val}")
                            print(f"          Modified: {mod_val}")
                        break
                
                if row_matches:
                    if row_idx < 2:
                        print(f"       ✅ Row {row_idx + 1}: Non-merged column values match")
                else:
                    all_match = False
                    mismatched_rows.append(row_idx + 1)
        else:
            # For other operations, compare by index (original behavior)
            print(f"\n    🔍 VERIFICATION (comparing first {original_col_count} columns - original columns only, ALL rows):")
            all_match = True
            mismatched_rows = []
            
            for row_idx in range(len(original_rows)):
                original_row = original_rows[row_idx]
                modified_row = modified_rows[row_idx] if row_idx < len(modified_rows) else None
                
                if modified_row is None:
                    print(f"       ❌ Row {row_idx + 1}: MISSING in modified table!")
                    all_match = False
                    mismatched_rows.append(row_idx + 1)
                    continue
                
                # Compare only the original columns (ignore any added columns)
                original_values = original_row[:original_col_count] if len(original_row) >= original_col_count else original_row
                modified_values = modified_row[:original_col_count] if len(modified_row) >= original_col_count else modified_row
                
                if original_values == modified_values:
                    # Only print first 2 rows for brevity
                    if row_idx < 2:
                        print(f"       ✅ Row {row_idx + 1}: ORIGINAL VALUES MATCH")
                        print(f"          Original: {original_values}")
                        print(f"          Modified: {modified_values}")
                else:
                    print(f"       ❌ Row {row_idx + 1}: ORIGINAL VALUES DO NOT MATCH!")
                    print(f"          Original: {original_values}")
                    print(f"          Modified: {modified_values}")
                    all_match = False
                    mismatched_rows.append(row_idx + 1)
        
        if all_match:
            print(f"\n    ✅ SUCCESS: All original column values match in ALL {len(original_rows)} row(s)!")
        else:
            print(f"\n    ⚠️  WARNING: Some original column values do not match!")
            print(f"       Mismatched rows: {mismatched_rows[:10]}{'...' if len(mismatched_rows) > 10 else ''} (out of {len(original_rows)} total)")
        
        # Store all_match flag for validation later
        original_columns_match = all_match
        
        # Show added columns if any
        if len(modified_headers) > original_col_count:
            added_columns = modified_headers[original_col_count:]
            print(f"\n    ➕ ADDED COLUMNS (not in original): {added_columns}")
            print(f"       First 2 rows with added values:")
            for row_idx in range(min(len(modified_rows), 2)):
                added_values = modified_rows[row_idx][original_col_count:] if len(modified_rows[row_idx]) > original_col_count else []
                print(f"          Row {row_idx + 1}: {added_values}")
        
        print(f"\n{'🥭' * 80}")
        print(f"{'🥭' * 80}\n")
        
        # Update query to use modified table name and convert to DB format
        current_query = update_query_table_name(current_query, db_table_name)
        sql, db_sql = convert_query_to_db_format(current_query, current_table, db_table_name)
        
        # Skip query execution if query is "unanswerable"
        if current_query.strip().lower() == "unanswerable":
            print(f"    ⚠️  Query is 'unanswerable' - skipping execution")
            modified_result = None
            conn.close()
        else:
            # Try to execute query using db_sql (with generic names)
            try:
                cur.execute(db_sql)
                modified_result = cur.fetchall()
                if modified_result is None:
                    print(f"    ⚠️  Query returned no results")
                    conn.close()
                    return None
                else:
                    print(f"    ✅ Query executed successfully, returned {len(modified_result)} rows")
            except Exception as e:
                print(f"    ❌ Query execution failed: {e}")
                print(f"       Query: {db_sql}")
                
                # For merge_column operations, try LLM repair if manual repair failed
                if change_type == "merge_column":
                    print(f"    🔧 Attempting LLM repair for failed merge_column query...")
                    try:
                        # Create repair prompt with old query, schema change info, new query, and error
                        original_cols = change_info.get("original_columns", [])
                        new_col = change_info.get("new_column", "")
                        repair_prompt = f"""You are an expert SQL developer. A SQL query was modified due to a schema change (merge_column), but the modified query has a syntax error.

Schema Change:
- Original columns merged: {', '.join(original_cols)}
- New merged column name: {new_col}

Original Query:
{query}

Modified Query (with error):
{current_query}

Error Message:
{str(e)}

Please repair the modified query to fix the syntax error while maintaining the same logical intent. The query should work with the new schema where the original columns have been merged into a single column.

Return ONLY the repaired SQL query, no explanations."""
                        
                        repair_response = run_gpt(repair_prompt, max_tokens=1024)  # Use default temperature (1) for gpt-5-nano
                        
                        # Extract SQL from response (handle code blocks)
                        repaired_sql = repair_response.strip()
                        if "```sql" in repaired_sql:
                            repaired_sql = repaired_sql.split("```sql")[1].split("```")[0].strip()
                        elif "```" in repaired_sql:
                            repaired_sql = repaired_sql.split("```")[1].split("```")[0].strip()
                        
                        # Update current_query with repaired version
                        current_query = repaired_sql
                        sql = current_query
                        
                        # Regenerate db_sql with repaired query using convert_query_to_db_format
                        sql, db_sql = convert_query_to_db_format(current_query, current_table, db_table_name)
                        
                        # Try executing the repaired query
                        try:
                            cur.execute(db_sql)
                            modified_result = cur.fetchall()
                            if modified_result is None:
                                print(f"    ⚠️  Repaired query returned no results")
                                conn.close()
                                return None
                            else:
                                print(f"    ✅ LLM-repaired query executed successfully, returned {len(modified_result)} rows")
                        except Exception as e2:
                            print(f"    ❌ LLM-repaired query also failed: {e2}")
                            print(f"       Repaired Query: {db_sql}")
                            conn.close()
                            return None
                    except Exception as repair_error:
                        print(f"    ❌ LLM repair attempt failed: {repair_error}")
                        conn.close()
                        return None
                else:
                    conn.close()
                    return None
            
            conn.close()
        
        # For merge_column operations, validate that results match the original query
        # Skip validation if query is "unanswerable" (no modified_result to compare)
        if change_type == "merge_column" and current_query.strip().lower() != "unanswerable" and modified_result is not None:
            # Execute original query on original table to get baseline
            # Get original table ID from the table name
            table_name_from_query = extract_table_name_from_query(query)
            if table_name_from_query:
                original_table_id = table_name_from_query.replace('table_', '').replace('_', '-')
                original_table_name_db = f"table_{original_table_id.replace('-', '_')}"
                original_result, original_error = execute_query_on_table(query, original_table, db_path, original_table_name_db)
                
                if original_error:
                    print(f"    ⚠️  Could not execute original query for validation: {original_error}")
                else:
                    # Normalize results for comparison (case-insensitive for strings)
                    def normalize_value(val):
                        """Normalize a value for comparison (case-insensitive for strings)."""
                        if val is None:
                            return None
                        # Handle bytes
                        if isinstance(val, bytes):
                            val = val.decode('utf-8', errors='ignore')
                        # Convert to string and lowercase if it's a string
                        if isinstance(val, str):
                            return val.lower()
                        # For numbers, keep as is
                        return val
                    
                    def normalize_row(row):
                        """Normalize a row for comparison."""
                        return tuple(normalize_value(x) for x in row)
                    
                    original_normalized = sorted([normalize_row(row) for row in (original_result or [])])
                    modified_normalized = sorted([normalize_row(row) for row in (modified_result or [])])
                    
                    if original_normalized == modified_normalized:
                        print(f"    ✅ Validation passed: Results match original query ({len(modified_result) if modified_result is not None else 0} rows)")
                    else:
                        # Check if the only difference is case (for strings)
                        original_str = str(original_normalized).lower()
                        modified_str = str(modified_normalized).lower()
                        
                        if original_str == modified_str:
                            print(f"    ✅ Validation passed: Results match (case-insensitive, {len(modified_result) if modified_result is not None else 0} rows)")
                        else:
                            print(f"    ⚠️  Validation warning: Results differ!")
                            print(f"       Original result ({len(original_result) if original_result is not None else 0} rows): {original_result[:3] if original_result and len(original_result) > 0 else '[]'}...")
                            print(f"       Modified result ({len(modified_result) if modified_result is not None else 0} rows): {modified_result[:3] if modified_result and len(modified_result) > 0 else '[]'}...")
                            print(f"       This may be expected for merge operations with extraction logic")
        
        # Return result with both sql (JSONL column names) and db_sql (database placeholder names)
        return {
            "new_query": sql,  # Query with JSONL column names
            "new_table_id": modified_table_id,
            "change_type": change_type,
            "db_sql": db_sql,  # Query with database placeholder names (col0, col1, etc.)
            "query_result": modified_result,  # Store result for validation
            "change_info": change_info,  # Store change details for verbose output
            "original_query": query,  # Store original query to show what changed
            "original_columns_match": original_columns_match  # Flag indicating if all original column values match
        }
        
    except Exception as e:
        print(f"    ❌ Error applying schema change: {e}")
        import traceback
        traceback.print_exc()
        return None


def execute_query_on_table(query: str, table: Dict[str, Any], db_path: str, table_name_in_db: str) -> tuple:
    """Execute SQL query on a table and return results."""
    conn = sqlite3.connect(db_path, timeout=5.0)
    cur = conn.cursor()
    
    # Convert column names to placeholders (using same approach as final_gen)
    conversion = {table["header"][i]: f"col{i}" for i in range(len(table["header"]))}
    test_query = query
    
    # Use the same approach as final_gen: find all backticked strings and replace them
    for key in sorted(conversion, key=len, reverse=True):
        try:
            matches = re.findall(r"`([^`]*)`", test_query)
            for m in matches:
                original = key
                replacement = conversion[original]
                # Case-insensitive comparison (same as final_gen)
                if m.strip().lower() == original.strip().lower():
                    test_query = test_query.replace(f"`{m}`", replacement)
        except re.error as e:
            print(f"Regex error for key '{key}': {e}")
            continue  # Skip this pattern and continue with the next one
    
    # Debug: Print conversion mapping and converted query
    print(f"🔍 DEBUG execute_query_on_table:")
    print(f"   Table header: {table.get('header', [])}")
    print(f"   Conversion map: {conversion}")
    print(f"   Converted query: {test_query}")
    
    # Extract CTE names to avoid replacing them when replacing table names
    cte_names = set()
    # Match: WITH cte_name AS ( or WITH cte1 AS (...), cte2 AS (...)
    with_matches = re.finditer(r'WITH\s+([^()]+?)\s+AS\s*\(', test_query, re.IGNORECASE | re.DOTALL)
    for match in with_matches:
        cte_def = match.group(1).strip()
        # Handle multiple CTEs: "cte1 AS (...), cte2 AS (...)"
        # Split by comma, but be careful of nested parentheses
        cte_parts = []
        current_part = ""
        paren_depth = 0
        for char in cte_def:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                cte_parts.append(current_part.strip())
                current_part = ""
                continue
            current_part += char
        if current_part:
            cte_parts.append(current_part.strip())
        
        # Extract CTE name from each part (before "AS")
        for part in cte_parts:
            if ' AS ' in part.upper():
                cte_name = part.split(' AS ', 1)[0].strip().strip('`')
                cte_names.add(cte_name.lower())
    
    # Replace table name in query, but skip CTE names
    def replace_table_name(match):
        keyword = match.group(1)  # FROM, JOIN, etc.
        table_ref = match.group(2).strip('`')  # Remove backticks if present
        
        # Don't replace if it's already the correct table name, a column name, or a CTE name
        if table_ref == table_name_in_db or table_ref in conversion or table_ref.lower() in cte_names:
            return match.group(0)
        return f"{keyword} `{table_name_in_db}`"
    
    test_query = re.sub(r"(FROM|JOIN)\s+`?([^`\s,()]+)`?", replace_table_name, test_query, flags=re.IGNORECASE)
    
    try:
        cur.execute(test_query)
        result = cur.fetchall()
        conn.close()
        return result, None
    except Exception as e:
        conn.close()
        return None, str(e)


def apply_schema_change(db_path: str, nl: str, query: str, db_query, table_id: str, db_result: list, tables_jsonl_path: str = "data/train.tables.jsonl") -> List[Dict[str, Any]]:
    '''
    Takes in a NL SQL and schema. The goal is to change the schema (and thus also the columns mentioned in the query) without changing the NL
    This creates difficulty because it ensures the model cant overfit to one schema and is flexible to schema evolutions.
    For each NL->SQL pair, we will generate several different schema change versions in the same db.
    Each change creates a new table and returns a separate result.
    
    Args:
        db_path: Path to SQLite database
        nl: Natural language question
        query: SQL query string (with JSONL column names)
        db_query: SQL query string (with database placeholder names, ready to execute)
        table_id: Table ID (e.g., "1-1000181-1" or "table_1_1000181_1")
        db_result: Expected result from executing db_query (for validation)
        tables_jsonl_path: Path to train.tables.jsonl file
        
    Returns:
        List of Dicts, each with keys: 'new_query', 'new_table_id'
        One result per schema change applied
    '''
    print(f"Applying schema change for NL: {nl} and query: {query}")
    print(f"Table ID: {table_id}")
    try:
        # Normalize table_id (remove table_ prefix if present and convert underscores to dashes)
        normalized_table_id = table_id.replace('table_', '').replace('_', '-')
        
        # Get original table from JSONL
        original_table = get_table_from_jsonl(normalized_table_id, tables_jsonl_path)
        
        if not original_table:
            print(f"❌ Could not find table '{normalized_table_id}' in JSONL")
            return []
        
        # Construct original_schema from the table if not provided
        
        from multilingual_evoschema.schema_change import convert_table_to_schema
        original_schema = convert_table_to_schema(json.dumps(original_table))
        print(f"📋 Constructed schema from table:")
        print(f"   {original_schema.replace(chr(10), ' | ')}")
        
        # Execute original query to get baseline result
        # IMPORTANT: filter_sql_query dynamically finds the table name by checking what exists in DB
        # So we should extract the actual table name from db_query, not construct it
        original_table_name_db = f"table_{normalized_table_id.replace('-', '_')}"
        
        # Extract table name from db_query (this is what filter_sql_query actually used)
        table_in_query_match = re.search(r'FROM\s+`?([^`\s,()]+)`?', db_query, re.IGNORECASE)
        if table_in_query_match:
            table_in_query = table_in_query_match.group(1)
            # Use the table name from the query (what filter_sql_query actually found)
            original_table_name_db = table_in_query
            print(f"🔍 DEBUG: Using table name from db_query: '{original_table_name_db}'")
        else:
            print(f"🔍 DEBUG: Could not extract table name from db_query, using constructed name: '{original_table_name_db}'")
        
        # Check if original table exists in database - THIS SHOULD NEVER HAPPEN
        # Use timeout to handle database locks (wait up to 5 seconds)
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (original_table_name_db,))
            table_exists = cur.fetchone() is not None
        finally:
            conn.close()
        
        if not table_exists:
            print(f"\n{'='*80}")
            print(f"{'!'*80}")
            print(f"{'!'*80}")
            print(f"  ⚠️  ⚠️  ⚠️  CRITICAL ERROR: ORIGINAL TABLE DOES NOT EXIST IN DATABASE  ⚠️  ⚠️  ⚠️")
            print(f"  Table name: {original_table_name_db}")
            print(f"  Table ID: {normalized_table_id}")
            print(f"  Database: {db_path}")
            print(f"  THIS SHOULD NEVER HAPPEN!")
            print(f"  The original table must exist in the database for validation to work correctly.")
            print(f"{'!'*80}")
            print(f"{'!'*80}")
            print(f"{'='*80}\n")
        
        # Execute db_query directly on the original table (no conversion needed)
        # db_query already has the correct table name (from filter_sql_query)
        print(f"🔍 DEBUG: Executing db_query directly on original table '{original_table_name_db}'")
        print(f"🔍 DEBUG: db_query: {db_query}")
        
        conn = sqlite3.connect(db_path, timeout=5.0)
        cur = conn.cursor()
        try:
            cur.execute(db_query)
            original_result = cur.fetchall()
            original_error = None
            print(f"🔍 DEBUG: Original query result: {original_result} ({(len(original_result) if original_result else 0)} rows)")
            
            # Verify that original_result matches db_result
            print(f"\n{'🔍' * 80}")
            print(f"    🔍 VERIFYING ORIGINAL QUERY RESULT MATCHES db_result")
            print(f"{'🔍' * 80}\n")
            
            # Parse db_result if it's a string (from CSV serialization)
            if db_result is not None:
                if isinstance(db_result, str):
                    try:
                        import ast
                        db_result = ast.literal_eval(db_result)
                    except (ValueError, SyntaxError):
                        try:
                            db_result = json.loads(db_result)
                        except (json.JSONDecodeError, ValueError):
                            print(f"    ⚠️  Warning: Could not parse db_result string")
                            db_result = None
                
                # Check if db_result is incorrectly formatted (list of strings instead of list of tuples)
                if db_result and len(db_result) > 0 and isinstance(db_result[0], str):
                    # Try to parse each string element
                    try:
                        import ast
                        parsed_db_result = []
                        for item in db_result:
                            if isinstance(item, str):
                                parsed = ast.literal_eval(item)
                                if isinstance(parsed, list):
                                    parsed_db_result.extend(parsed)
                                else:
                                    parsed_db_result.append(parsed)
                            else:
                                parsed_db_result.append(item)
                        db_result = parsed_db_result
                        print(f"    ℹ️  Parsed db_result from string format: {len(db_result)} rows after parsing")
                    except (ValueError, SyntaxError):
                        print(f"    ⚠️  Warning: Could not parse db_result elements")
            
            # Normalize results for comparison (convert to tuples, handle None values)
            def normalize_result(result):
                if result is None:
                    return None
                normalized = []
                for row in result:
                    if isinstance(row, (list, tuple)):
                        # Convert to tuple and handle None values
                        normalized_row = tuple(None if x is None else x for x in row)
                        normalized.append(normalized_row)
                    else:
                        normalized.append((row,))
                return normalized
            
            original_normalized = normalize_result(original_result)
            db_result_normalized = normalize_result(db_result)
            
            # Debug: Print table names being used
            print(f"    🔍 DEBUG: Query execution details:")
            print(f"       Executing on table: {original_table_name_db}")
            print(f"       db_query: {db_query[:200]}..." if len(db_query) > 200 else f"       db_query: {db_query}")
            print(f"       Original result type: {type(original_result)}, length: {len(original_result) if original_result else 0}")
            print(f"       db_result type: {type(db_result)}, length: {len(db_result) if db_result else 0}")
            if db_result and len(db_result) > 0:
                print(f"       db_result[0] type: {type(db_result[0])}")
                print(f"       db_result[0] value: {db_result[0]}")
            
            # Sort both for comparison (in case order differs)
            original_sorted = sorted(original_normalized) if original_normalized else []
            db_result_sorted = sorted(db_result_normalized) if db_result_normalized else []
            
            if original_sorted == db_result_sorted:
                print(f"    ✅ SUCCESS: Original query result matches db_result!")
                print(f"       Both have {len(original_sorted)} rows")
            else:
                print(f"    ❌ ERROR: Original query result does NOT match db_result!")
                print(f"       Original query returned: {len(original_sorted)} rows")
                print(f"       db_result has: {len(db_result_sorted)} rows")
                print(f"       Original result (first 3): {original_sorted[:3] if len(original_sorted) > 0 else '[]'}")
                print(f"       db_result (first 3): {db_result_sorted[:3] if len(db_result_sorted) > 0 else '[]'}")
                print(f"    ⚠️  WARNING: Results don't match - checking if query execution differs...")
                
                # Check if they're executing on the same table
                print(f"       Checking table name in db_query...")
                if original_table_name_db in db_query:
                    print(f"       ✅ db_query references correct table: {original_table_name_db}")
                else:
                    print(f"       ⚠️  db_query may not reference table {original_table_name_db}")
                    # Try to find what table it references
                    table_match = re.search(r'FROM\s+`?([^`\s,()]+)`?', db_query, re.IGNORECASE)
                    if table_match:
                        print(f"       Found table reference in query: {table_match.group(1)}")
            
            print(f"{'🔍' * 80}\n")
            
        except Exception as e:
            original_error = str(e)
            print(f"❌ Could not execute original query: {original_error}")
            return []
        finally:
            conn.close()
        
        # apply one schema change
        number_of_changes = 1
        #change_types = ["add_column", "remove_column", "rename_column", "rename_table", "split_column", "merge_column"]
        change_types = ["merge_column", "split_column", "add_column", "remove_column", "rename_column"] # deleted rename_table since we technically are renaming the table every time anyways
        
        results = []

        
        for change_num in range(number_of_changes):
            print(f"Change {change_num + 1}/{number_of_changes}:")
            change_type = random.choice(change_types)
            print(f"Change type: {change_type}")
            
            # Apply single change (starts from original table each time)
            result = apply_single_schema_change(
                db_path=db_path,
                original_schema=original_schema,
                original_table=original_table,
                nl=nl,
                query=query,
                change_type=change_type,
                tables_jsonl_path=tables_jsonl_path
            )
            
            if result:
                # Validate that results match only for add_column and rename_column
                change_type_applied = result.get("change_type")
                modified_result = result.get("query_result")
                
                # For add_column and rename_column, accept if results are returned (even if not exactly the same)
                # For other operations (remove_column, split_column, merge_column), accept even if results differ, as this is expected for these operations.
                if change_type_applied in ["add_column", "rename_column"]:
                    # Check if modified result exists and is not empty
                    if not modified_result or len(modified_result) == 0:
                        print(f"  ❌ No results returned! Rejecting this change.")
                        print(f"     Original query: {query}")
                        print(f"     Modified query: {result.get('new_query', 'N/A')}")
                        # Don't add to results
                        continue
                    
                    # Results are returned, so accept the change
                    # Compare results for informational purposes (but don't reject if they differ)
                    def normalize_value(val):
                        """Normalize a value for comparison (case-insensitive for strings)."""
                        if val is None:
                            return None
                        # Handle bytes
                        if isinstance(val, bytes):
                            val = val.decode('utf-8', errors='ignore')
                        # Convert to string and lowercase if it's a string
                        if isinstance(val, str):
                            return val.lower()
                        # For numbers, keep as is
                        return val
                    
                    def normalize_row(row):
                        """Normalize a row for comparison."""
                        return tuple(normalize_value(x) for x in row)
                    
                    original_normalized = sorted([normalize_row(row) for row in (original_result or [])])
                    modified_normalized = sorted([normalize_row(row) for row in (modified_result or [])])
                    
                    # Flag whether database answers are the same or different
                    results_match = original_normalized == modified_normalized
                    if results_match:
                        print(f"  🟢 FLAG: DB ANSWERS ARE THE SAME (exact match)")
                        print(f"  ✅ Results match original query exactly")
                    else:
                        print(f"  🟡 FLAG: DB ANSWERS ARE DIFFERENT")
                        print(f"  ✅ Results returned (accepting even though they differ from original)")
                        print(f"     Original result: {len(original_normalized)} rows")
                        print(f"     Modified result: {len(modified_normalized)} rows")
                        print(f"     Original result (first 3): {original_normalized[:3] if len(original_normalized) > 0 else '[]'}")
                        print(f"     Modified result (first 3): {modified_normalized[:3] if len(modified_normalized) > 0 else '[]'}")
                else:
                    # For other operations, accept the result even if it differs
                    # Compare results for informational purposes
                    if original_result and modified_result:
                        def normalize_value(val):
                            """Normalize a value for comparison (case-insensitive for strings)."""
                            if val is None:
                                return None
                            if isinstance(val, bytes):
                                val = val.decode('utf-8', errors='ignore')
                            if isinstance(val, str):
                                return val.lower()
                            return val
                        
                        def normalize_row(row):
                            """Normalize a row for comparison."""
                            return tuple(normalize_value(x) for x in row)
                        
                        original_normalized = sorted([normalize_row(row) for row in (original_result or [])])
                        modified_normalized = sorted([normalize_row(row) for row in (modified_result or [])])
                        results_match = original_normalized == modified_normalized
                        
                        if results_match:
                            print(f"  🟢 FLAG: DB ANSWERS ARE THE SAME (exact match)")
                        else:
                            print(f"  🟡 FLAG: DB ANSWERS ARE DIFFERENT")
                    else:
                        print(f"  ⚪ FLAG: Cannot compare DB answers (one or both results are empty)")
                    print(f"  ✅ Change applied (validation skipped for {change_type_applied})")
                
                # Return only new query and new table id
                new_table_id = result.get("new_table_id")
                # Construct database table name from table ID (convert dashes to underscores, add table_ prefix)
                db_table_name = f"table_{new_table_id.replace('-', '_')}" if new_table_id else "unknown"
                
                # Get table from JSONL to print column names
                jsonl_table = get_table_from_jsonl(new_table_id, tables_jsonl_path)
                jsonl_columns = jsonl_table.get("header", []) if jsonl_table else []
                
                # Get table columns from database
                db_columns = []
                try:
                    conn = sqlite3.connect(db_path, timeout=5.0)
                    cur = conn.cursor()
                    cur.execute(f"PRAGMA table_info(`{db_table_name}`)")
                    db_info = cur.fetchall()
                    # Extract column names (col0, col1, etc.)
                    db_columns = [col[1] for col in db_info]  # col[1] is the column name
                    conn.close()
                except Exception as e:
                    print(f"    ⚠️  Could not get DB table info: {e}")
                
                results.append({
                    "new_query": result.get("new_query"),  # Query with JSONL column names
                    "new_table_id": new_table_id,
                    "change_type": change_type,
                    "db_sql": result.get("db_sql"),  # Query with database placeholder names
                    "query_result": result.get("query_result")  # Database query result
                })
                print(f"json for new table ({new_table_id}): {jsonl_columns}")
                print(f"db for new table ({db_table_name}): {db_columns}")
            else:
                print(f"  ⚠️  Change {change_num + 1} failed, skipping")

        return results
        
    except Exception as e:
        print(f"❌ Error in apply_schema_change: {e}")
        import traceback
        traceback.print_exc()
        return []


def apply_schema_change_translation(
    db_path: str,
    nl: str,
    query: str,
    target_language: str,
    tables_jsonl_path: str = "data/train.tables.jsonl"
) -> List[Dict[str, Any]]:
    """
    Apply schema translation to a table. Translates table and column names to target language.
    If target_language is "hi_typed", first translates to Hindi then latinizes.
    
    Args:
        db_path: Path to SQLite database
        nl: Natural language question
        query: SQL query string
        target_language: Target language code (e.g., "hi", "hi_typed", "ru", "zh", etc.)
        tables_jsonl_path: Path to train.tables.jsonl file
        
    Returns:
        List of Dicts, each with keys: 'new_query', 'new_table_id', 'change_type'
    """
    print(f"Applying schema translation for NL: {nl}")
    print(f"Query: {query}")
    print(f"Target language: {target_language}")
    
    try:
        # Extract table name from query
        table_name = extract_table_name_from_query(query)
        if not table_name:
            print(f"❌ Could not extract table name from query: {query}")
            return []
        
        # Get original table from JSONL
        table_id = table_name.replace('table_', '').replace('_', '-')
        original_table = get_table_from_jsonl(table_id, tables_jsonl_path)
        
        if not original_table:
            print(f"❌ Could not find table '{table_id}' in JSONL")
            return []
        
        # Construct original_schema from the table
        original_schema = convert_table_to_schema(json.dumps(original_table))
        print(f"📋 Original schema:")
        print(f"   {original_schema.replace(chr(10), ' | ')}")
        
        # Determine if we need latinization
        needs_latinization = target_language == "hi_typed"
        translation_lang = "hi" if needs_latinization else target_language
        
        # Get translation prompt using translate_schema
        from multilingual_evoschema.schema_change import translate_schema
        translation_prompt = translate_schema(original_schema, translation_lang)
        
        # Call OpenAI to translate
        print(f"🌐 Translating schema to {translation_lang}...")
        client = OpenAI()
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert database translator. Translate database schemas accurately."},
                    {"role": "user", "content": translation_prompt}
                ],
                temperature=0.0
            )
            translation_response = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error calling OpenAI for translation: {e}")
            return []
        
        # Parse JSON response (handle trailing commas and other common LLM JSON errors)
        try:
            # Try to extract JSON block if wrapped in code blocks
            if "```json" in translation_response:
                json_block = translation_response.split("```json", 1)[1].split("```")[0].strip()
            elif "```" in translation_response:
                json_block = translation_response.split("```", 1)[1].split("```")[0].strip()
            else:
                json_block = translation_response
            
            # Remove trailing commas before closing braces/brackets
            json_block = re.sub(r',(\s*[}\]])', r'\1', json_block)
            
            translation_data = json.loads(json_block)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse translation response: {e}")
            print(f"   Response: {translation_response}")
            return []
        
        # Extract translation mappings
        if "translated_schema" not in translation_data or not translation_data["translated_schema"]:
            print(f"❌ Invalid translation response format")
            return []
        
        trans_info = translation_data["translated_schema"][0]
        table_rename_map = {trans_info["original_table"]: trans_info["translated_table"]}
        column_rename_map = {}
        for col_info in trans_info.get("columns", []):
            column_rename_map[col_info["old_name"]] = col_info["new_name"]
        
        # If hi_typed, latinize the translated names
        if needs_latinization:
            print(f"🔤 Latinizing Hindi text...")
            try:
                from indic_transliteration import sanscript
                from indic_transliteration.sanscript import transliterate
                
                # Collect all names to latinize
                names_to_latinize = [trans_info["translated_table"]] + [col["new_name"] for col in trans_info.get("columns", [])]
                latinized_names = []
                
                for name in names_to_latinize:
                    try:
                        # Transliterate from Devanagari (Hindi) to IAST (Latin)
                        latinized = transliterate(name, sanscript.DEVANAGARI, sanscript.IAST)
                        # Convert to lowercase and replace spaces with underscores for database naming
                        latinized = latinized.lower().replace(' ', '_')
                        latinized_names.append(latinized)
                    except Exception as e:
                        print(f"⚠️  Error latinizing '{name}': {e}, using original")
                        latinized_names.append(name)
                
                if len(latinized_names) == len(names_to_latinize):
                    # Update rename maps with latinized names
                    table_rename_map[trans_info["original_table"]] = latinized_names[0]
                    for i, col_info in enumerate(trans_info.get("columns", []), 1):
                        if i < len(latinized_names):
                            column_rename_map[col_info["old_name"]] = latinized_names[i]
                else:
                    print(f"⚠️  Latinization count mismatch, using original translation")
            except ImportError:
                print(f"⚠️  indic-transliteration not installed. Install with: pip install indic-transliteration")
                print(f"   Falling back to OpenAI for latinization...")
                # Fallback to OpenAI if library not available
                names_to_latinize = [trans_info["translated_table"]] + [col["new_name"] for col in trans_info.get("columns", [])]
                
                latinization_prompt = f"""
Translate the following Hindi database names to Latin script (transliteration).
Use standard transliteration (e.g., "नाम" -> "naam", "उम्र" -> "umr").

Names to transliterate:
{json.dumps(names_to_latinize, ensure_ascii=False, indent=2)}

Return ONLY valid JSON in the format:
{{
    "latinized_names": ["name1", "name2", ...]
}}

The order must match the input order. Only JSON, no explanations.
"""
                try:
                    latinize_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an expert in Hindi transliteration to Latin script."},
                            {"role": "user", "content": latinization_prompt}
                        ],
                        temperature=0.0
                    )
                    latinize_text = latinize_response.choices[0].message.content.strip()
                    
                    # Parse latinization response
                    if "```json" in latinize_text:
                        latinize_json = latinize_text.split("```json", 1)[1].split("```")[0].strip()
                    elif "```" in latinize_text:
                        latinize_json = latinize_text.split("```", 1)[1].split("```")[0].strip()
                    else:
                        latinize_json = latinize_text
                    
                    latinize_json = re.sub(r',(\s*[}\]])', r'\1', latinize_json)
                    latinize_data = json.loads(latinize_json)
                    latinized_names = latinize_data.get("latinized_names", [])
                    
                    if len(latinized_names) == len(names_to_latinize):
                        # Update rename maps with latinized names
                        table_rename_map[trans_info["original_table"]] = latinized_names[0]
                        for i, col_info in enumerate(trans_info.get("columns", []), 1):
                            if i < len(latinized_names):
                                column_rename_map[col_info["old_name"]] = latinized_names[i]
                    else:
                        print(f"⚠️  Latinization count mismatch, using original translation")
                except Exception as e:
                    print(f"⚠️  Error latinizing with OpenAI, using original translation: {e}")
            except Exception as e:
                print(f"⚠️  Error latinizing, using original translation: {e}")
        
        # Apply translation to schema
        current_schema = original_schema
        current_table = json.loads(json.dumps(original_table))  # Deep copy
        
        # Update table name in schema
        for old_table, new_table in table_rename_map.items():
            current_schema = current_schema.replace(f"Table: {old_table}", f"Table: {new_table}")
            current_table["name"] = new_table
            current_table["id"] = new_table.replace(' ', '_').replace('-', '_')
        
        # Update column names in schema and table
        schema_lines = current_schema.split('\n')
        updated_schema_lines = []
        for line in schema_lines:
            if line.strip().startswith('-') and '(' in line:
                # Extract column name
                match = re.match(r'(\s*-\s*)([^(]+)(\s*\([^)]+\))', line)
                if match:
                    prefix = match.group(1)
                    col_name = match.group(2).strip()
                    suffix = match.group(3)
                    # Replace if in rename map
                    if col_name in column_rename_map:
                        new_col_name = column_rename_map[col_name]
                        updated_schema_lines.append(f"{prefix}{new_col_name}{suffix}")
                    else:
                        updated_schema_lines.append(line)
                else:
                    updated_schema_lines.append(line)
            else:
                updated_schema_lines.append(line)
        current_schema = '\n'.join(updated_schema_lines)
        
        # IMPORTANT: 
        # - JSONL should use translated/renamed column names (as they appear conceptually)
        # - Database should use placeholder names (col0, col1, etc.)
        # - Query should use the SAME names as JSONL (translated names), NOT generic names
        
        # Update table header with translated names for JSONL
        original_headers = original_table.get("header", [])
        if "header" in current_table:
            # Use translated names in header for JSONL
            current_table["header"] = [column_rename_map.get(col, col) for col in original_headers]
        
        # Repair query to use translated names (same as JSONL)
        current_query = query
        # Build rename map for repair_query_rename
        all_renames = {}
        all_renames.update(table_rename_map)
        all_renames.update(column_rename_map)
        
        # Use repair_query_rename for each rename (this updates query to use translated names)
        for old_name, new_name in all_renames.items():
            rename_info = {"renames": [{"old_name": old_name, "new_name": new_name}]}
            current_query = repair_query_rename(current_query, json.dumps(rename_info), json.dumps(rename_info))
        
        print(f"✅ Updated query (with translated names, matching JSONL): {current_query[:100]}...")
        
        # Add modified table to database
        # Note: add_new_table_to_db creates tables with generic column names (col0, col1, etc.)
        # But the schema and table_sample header have translated names, which will be used in JSONL
        current_table["name"] = f"table_{current_table.get('id', 'unknown').replace('-', '_')}"
        success, modified_table_id, modified_table_name = add_new_table_to_db(
            schema=current_schema,  # Schema has translated names (for JSONL)
            table_sample=current_table,  # Header has translated names (for JSONL)
            modification_info=None,
            db_path=db_path,
            tables_jsonl_path=tables_jsonl_path
        )
        
        if not success or not modified_table_id:
            print(f"❌ Failed to add translated table to database")
            return []
        
        print(f"✅ Created translated table: {modified_table_name}")
        
        # For query execution, we need to convert translated names to generic names
        # Map translated column names to generic names (col0, col1, etc.) based on position
        translated_to_generic = {}
        for i, orig_col in enumerate(original_headers):
            if orig_col in column_rename_map:
                translated_name = column_rename_map[orig_col]
                generic_name = f"col{i}"
                translated_to_generic[translated_name] = generic_name
            else:
                # Column not translated, map original name to generic
                generic_name = f"col{i}"
                translated_to_generic[orig_col] = generic_name
        
        # Create a version of the query with generic names for database execution
        query_for_db = current_query
        for translated_name, generic_name in translated_to_generic.items():
            escaped_translated = re.escape(translated_name)
            # Replace backticked: `translated_name` -> `col0`
            pattern = rf'`{escaped_translated}`'
            query_for_db = re.sub(pattern, f'`{generic_name}`', query_for_db, flags=re.IGNORECASE)
            # Replace plain (with word boundaries)
            pattern2 = rf'\b{escaped_translated}\b'
            query_for_db = re.sub(pattern2, generic_name, query_for_db, flags=re.IGNORECASE)
        
        # Execute original query on original table to get baseline result
        original_table_name = extract_table_name_from_query(query)
        if original_table_name:
            original_table_id = original_table_name.replace('table_', '').replace('_', '-')
            original_table_name_db = f"table_{original_table_id.replace('-', '_')}"
            original_result, original_error = execute_query_on_table(query, original_table, db_path, original_table_name_db)
            
            if original_error:
                print(f"❌ Could not execute original query for validation: {original_error}")
                print(f"❌ Rejecting translation - original query execution failed")
                return []
        else:
            print(f"❌ Could not extract table name from original query")
            print(f"❌ Rejecting translation - cannot validate")
            return []
        
        # Execute query on new table to validate (using generic names for DB)
        new_result, new_error = execute_query_on_table(query_for_db, current_table, db_path, modified_table_name)
        if new_error:
            print(f"❌ Query execution error on translated table: {new_error}")
            print(f"❌ Rejecting translation - translated query execution failed")
            return []
        
        # Compare results - must match exactly
        if original_result is None or new_result is None:
            print(f"❌ Rejecting translation - one or both queries returned None")
            return []
        
        # Normalize results for comparison (case-insensitive for strings)
        def normalize_value(val):
            """Normalize a value for comparison (case-insensitive for strings)."""
            if val is None:
                return None
            # Handle bytes
            if isinstance(val, bytes):
                val = val.decode('utf-8', errors='ignore')
            # Convert to string and lowercase if it's a string
            if isinstance(val, str):
                return val.lower()
            # For numbers, keep as is
            return val
        
        def normalize_row(row):
            """Normalize a row for comparison."""
            return tuple(normalize_value(x) for x in row)
        
        original_normalized = sorted([normalize_row(row) for row in original_result])
        new_normalized = sorted([normalize_row(row) for row in new_result])
        
        if original_normalized == new_normalized:
            print(f"✅ Validation passed: Translated query results match original ({len(new_result)} rows)")
        else:
            # Check if the only difference is case (for strings)
            original_str = str(original_normalized).lower()
            new_str = str(new_normalized).lower()
            
            if original_str == new_str:
                print(f"✅ Validation passed: Results match (case-insensitive, {len(new_result)} rows)")
            else:
                print(f"❌ Rejecting translation - results don't match!")
                print(f"   Original result ({len(original_result)} rows): {original_result[:3] if len(original_result) > 0 else '[]'}...")
                print(f"   Translated result ({len(new_result)} rows): {new_result[:3] if len(new_result) > 0 else '[]'}...")
                return []
        
        # Create db_sql version (convert translated names to generic names for database)
        db_sql = query_for_db  # Already converted to generic names above
        
        return [{
            "new_query": current_query,  # Query with JSONL column names (translated)
            "new_table_id": modified_table_id,
            "change_type": target_language,
            "db_sql": db_sql,  # Query with database placeholder names (col0, col1, etc.)
            "query_result": new_result  # Database query result from translated schema
        }]
        
    except Exception as e:
        print(f"❌ Error in apply_schema_change_translation: {e}")
        import traceback
        traceback.print_exc()
        return []
