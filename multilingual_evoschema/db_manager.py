"""
Database management functions for creating and managing tables in SQLite database.
"""
import sqlite3
import json
import random
import os
import re
import time
import uuid

from multilingual_evoschema.formatting_helpers import (
    denormalize_sql_type
)


def add_new_table_to_db(schema: str, table_sample: str = None, modification_info: dict = None, db_path: str = "data/train.db", tables_jsonl_path: str = "data/train.tables.jsonl") -> tuple:
    """
    Append a modified table to the database with a unique "m" prefix identifier.
    Keeps all original tables intact. Also appends the corresponding entry to train.tables.jsonl.
    
    Args:
        schema: Schema string from convert_table_to_schema showing table structure
        table_sample: Optional original table data (JSON string or dict) to use as base
        modification_info: Optional dict with modification details:
            - For add_column: {"column_name": "...", "sql_type": "...", "values": [...]}
            - For remove_column: {"removed_column": "..."}
            - For rename_column: {"old_name": "...", "new_name": "..."}
        db_path: Path to SQLite database file
        tables_jsonl_path: Path to train.tables.jsonl file
        
    Returns:
        tuple: (success: bool, modified_table_id: str, modified_table_name: str) or (False, None, None) on error
    """
    try:
        # Parse schema to extract table name and columns
        schema_lines = schema.strip().split('\n')
        table_name = None
        columns = []  # List of (name, type) tuples
        
        for line in schema_lines:
            line = line.strip()
            if line.startswith('Table:'):
                table_name = line.split('Table:')[1].strip()
            elif line.startswith('-') and '(' in line:
                # Extract column name and type: "  - column_name (TYPE)"
                # Handle column names that may contain parentheses like "Height (in cm) (REAL)"
                # Find the last occurrence of " (TYPE)" pattern by matching from the end
                # Use greedy match to capture everything up to the last parentheses
                match = re.search(r'\s*-\s*(.+)\s*\(([^)]+)\)\s*$', line)
                if match:
                    col_name = match.group(1).strip()
                    col_type = match.group(2).strip()
                    columns.append((col_name, col_type))
        
        if not table_name or not columns:
            print(f"Error: Could not parse schema. Table name: {table_name}, Columns: {len(columns)}")
            return (False, None, None)
        
        # Extract original table ID from the table name
        # Table name might be like "table_1_1000181_1" or "1-1000181-1"
        original_table_id = table_name
        if table_name.startswith('table_'):
            original_table_id = table_name.replace('table_', '').replace('_', '-')
        else:
            original_table_id = table_name.replace('_', '-')
        
        # Generate unique identifier for modified table
        # Format: m_<timestamp>_<short_uuid>
        timestamp = int(time.time() * 1000)  # milliseconds for better uniqueness
        short_uuid = str(uuid.uuid4())[:8]  # First 8 chars of UUID
        unique_id = f"{timestamp}_{short_uuid}"
        
        # Create modified table ID with "m" prefix
        modified_table_id = f"m_{unique_id}_{original_table_id}"
        
        # Create modified table name for database (with table_ prefix and underscores)
        modified_table_name = f"table_m_{unique_id}_{original_table_id.replace('-', '_')}"
        
        # Connect to database
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Check if this exact modified table already exists (shouldn't happen, but safety check)
        cur.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (modified_table_name,))
        
        if cur.fetchone() is not None:
            # This shouldn't happen, but if it does, append another unique suffix
            modified_table_name = f"{modified_table_name}_{int(time.time())}"
            modified_table_id = f"{modified_table_id}_{int(time.time())}"
        
        # Build CREATE TABLE statement
        # Use placeholder column names (col0, col1, col2, etc.) to match the database format
        col_defs = []
        for i, (col_name, col_type) in enumerate(columns):
            # Use placeholder name: col0, col1, col2, etc.
            placeholder_name = f"col{i}"
            col_defs.append(f"`{placeholder_name}` {col_type}")
        
        create_sql = f"CREATE TABLE `{modified_table_name}` ({', '.join(col_defs)})"
        cur.execute(create_sql)
        
        # Prepare data for insertion
        rows_to_insert = []
        
        if table_sample:
            # Parse original table data
            if isinstance(table_sample, str):
                original_table = json.loads(table_sample)
            else:
                original_table = table_sample
            
            original_headers = original_table.get('header', [])
            original_rows = original_table.get('rows', [])
            
            # Map original columns to new columns
            col_mapping = {}  # Maps new column index to (original_col_index, is_new)
            new_col_indices = []
            
            for i, (new_col_name, new_col_type) in enumerate(columns):
                if modification_info and 'old_name' in modification_info and new_col_name == modification_info.get('new_name'):
                    # This is a renamed column
                    old_name = modification_info['old_name']
                    if old_name in original_headers:
                        col_mapping[i] = (original_headers.index(old_name), False)
                elif modification_info and 'removed_column' in modification_info and new_col_name == modification_info.get('removed_column'):
                    # This column should be removed, but it's in the schema - skip it
                    continue
                elif new_col_name in original_headers:
                    # Existing column
                    col_mapping[i] = (original_headers.index(new_col_name), False)
                else:
                    # New column
                    new_col_indices.append(i)
                    col_mapping[i] = (None, True)
            
            # Process each row from original data
            for row in original_rows:
                # Check if row already has the correct number of columns (e.g., from add_column where values were already appended)
                # AND verify that column order matches (for rename_column, we need to ensure correct mapping)
                use_direct = (len(row) == len(columns) and 
                             len(col_mapping) == len(columns) and
                             all(not is_new and orig_idx == i for i, (orig_idx, is_new) in enumerate(col_mapping.values()) if orig_idx is not None))
                
                if use_direct:
                    # Row already has all columns in the correct order, use it directly
                    rows_to_insert.append(row[:])  # Make a copy
                else:
                    # Need to map and reconstruct the row to ensure correct order
                    new_row = [None] * len(columns)
                    
                    # Copy existing column values using the mapping
                    for new_idx, (orig_idx, is_new) in col_mapping.items():
                        if not is_new and orig_idx is not None and orig_idx < len(row):
                            new_row[new_idx] = row[orig_idx]
                    
                    # Fill new columns with random sample values if provided
                    if modification_info and 'column_name' in modification_info and 'values' in modification_info:
                        new_col_name = modification_info['column_name']
                        sample_values = modification_info['values']
                        
                        # Find the index of the new column
                        for i, (col_name, _) in enumerate(columns):
                            if col_name == new_col_name:
                                if sample_values and len(sample_values) > 0:
                                    # Randomly select from sample values
                                    new_row[i] = random.choice(sample_values)
                                break
                    
                    rows_to_insert.append(new_row)
        else:
            # No original data, create empty table
            rows_to_insert = []
        
        # Insert rows
        if rows_to_insert:
            # Build INSERT statement
            # Use placeholder column names (col0, col1, col2, etc.)
            col_names = [f"`col{i}`" for i in range(len(columns))]
            placeholders = ', '.join(['?' for _ in columns])
            insert_sql = f"INSERT INTO `{modified_table_name}` ({', '.join(col_names)}) VALUES ({placeholders})"
            
            # Convert values to appropriate types
            typed_rows = []
            for row in rows_to_insert:
                typed_row = []
                for i, (col_name, col_type) in enumerate(columns):
                    value = row[i] if i < len(row) else None
                    # Convert to appropriate type
                    if value is None:
                        typed_row.append(None)
                    elif col_type.upper() in ['INTEGER', 'INT']:
                        try:
                            typed_row.append(int(value))
                        except (ValueError, TypeError):
                            typed_row.append(None)
                    elif col_type.upper() in ['REAL', 'FLOAT', 'DOUBLE', 'NUMERIC']:
                        try:
                            typed_row.append(float(value))
                        except (ValueError, TypeError):
                            typed_row.append(None)
                    else:
                        typed_row.append(str(value))
                typed_rows.append(typed_row)
            
            cur.executemany(insert_sql, typed_rows)
        
        conn.commit()
        
        # Query the table to verify it was created correctly and print column names and first 2 rows
        try:
            cur.execute(f"SELECT * FROM `{modified_table_name}` LIMIT 2")
            rows = cur.fetchall()
            column_names = [description[0] for description in cur.description]
            
            print(f"\n📊 Table '{modified_table_name}' verification:")
            print(f"   Column names: {', '.join(column_names)}")
            if rows:
                print(f"   First {len(rows)} row(s):")
                for i, row in enumerate(rows, 1):
                    print(f"      Row {i}: {row}")
            else:
                print(f"   (Table is empty)")
        except Exception as e:
            print(f"   ⚠️  Could not query table for verification: {e}")
        
        conn.close()
        
        # Also append to train.tables.jsonl (keep all original entries, append modified one)
        try:
            # Prepare rows in the same format as train.tables.jsonl (list of lists, with string values)
            jsonl_rows = []
            for row in rows_to_insert:
                # Convert row to list of strings (matching train.tables.jsonl format)
                jsonl_row = []
                for i, (col_name, col_type) in enumerate(columns):
                    value = row[i] if i < len(row) else None
                    # Convert to string for JSONL format (matching original format)
                    if value is None:
                        jsonl_row.append("")
                    else:
                        jsonl_row.append(str(value))
                jsonl_rows.append(jsonl_row)
            
            # Create table entry in the same format as train.tables.jsonl
            # Use modified_table_id (with "m" prefix) and real column names
            table_entry = {
                "id": modified_table_id,
                "header": [col_name for col_name, _ in columns],  # Real column names
                "types": [denormalize_sql_type(col_type) for _, col_type in columns],
                "rows": jsonl_rows
            }
            
            # Append to JSONL file (keep all original entries)
            os.makedirs(os.path.dirname(tables_jsonl_path) if os.path.dirname(tables_jsonl_path) else '.', exist_ok=True)
            
            # Always append the modified table entry
            with open(tables_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(table_entry, ensure_ascii=False) + '\n')
            
            print(f"✅ Successfully appended modified table '{modified_table_id}' to '{tables_jsonl_path}'")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not append to train.tables.jsonl: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"✅ Successfully created modified table '{modified_table_name}' (ID: {modified_table_id}) with {len(rows_to_insert)} rows")
        return (True, modified_table_id, modified_table_name)
        
    except Exception as e:
        print(f"❌ Error adding table to database: {e}")
        import traceback
        traceback.print_exc()
        return (False, None, None)


def convert_query_to_db_format(query: str, table: dict, db_table_name: str) -> tuple[str, str]:
    """
    Convert a SQL query to use database placeholder names (col0, col1, etc.) and correct table name.
    
    Args:
        query: SQL query with JSONL column names
        table: Table dict with 'header' list
        db_table_name: Database table name to use
        
    Returns:
        Tuple of (sql, db_sql) where:
        - sql: Query with JSONL column names and correct table name
        - db_sql: Query with database placeholder names (col0, col1, etc.) and correct table name
    """
    import re
    
    # Generate conversion map: JSONL column names -> col0, col1, etc.
    conversion = {table["header"][i]: f"col{i}" for i in range(len(table["header"]))}
    
    # Start with original query
    sql = query
    db_sql = query
    
    # Convert column names to placeholders in db_sql
    for original, replacement in sorted(conversion.items(), key=lambda x: len(x[0]), reverse=True):
        db_sql = re.sub(rf'`{re.escape(original)}`', replacement, db_sql, flags=re.IGNORECASE)
    
    # Extract CTE names to avoid replacing them (CTEs are defined in WITH clauses)
    cte_names = set()
    with_matches = re.finditer(r'WITH\s+(\w+)\s+AS', query, re.IGNORECASE)
    for match in with_matches:
        cte_names.add(match.group(1).lower())
    
    # Also find CTEs in comma-separated lists: WITH cte1 AS (...), cte2 AS (...)
    with_comma_matches = re.finditer(r',\s*(\w+)\s+AS\s*\(', query, re.IGNORECASE)
    for match in with_comma_matches:
        cte_names.add(match.group(1).lower())
    
    # Extract original table name from the query (the actual table, not CTEs)
    # Look for FROM clauses that are NOT part of CTE definitions
    # We need to find the actual table name that appears in FROM/JOIN clauses outside of CTEs
    original_table_name = None
    # Find the first FROM that's not inside a CTE (rough heuristic: after the first closing paren of CTEs)
    # For simplicity, find table names that match the pattern table_* or have backticks
    table_name_pattern = r'FROM\s+`?([^`\s,()]+)`?'
    for match in re.finditer(table_name_pattern, query, re.IGNORECASE):
        potential_name = match.group(1)
        # Skip if it's a CTE name
        if potential_name.lower() not in cte_names and ('table_' in potential_name.lower() or '`' in match.group(0)):
            original_table_name = potential_name
            break
    
    # If we found the original table name, replace it (but not CTE names)
    if original_table_name:
        escaped_old = re.escape(original_table_name)
        # Replace in FROM clauses (but not if it's a CTE name)
        def replace_from(match):
            table_ref = match.group(1)
            if table_ref.lower() in cte_names:
                return match.group(0)  # Keep CTE reference unchanged
            if table_ref.lower() == original_table_name.lower():
                return f'FROM `{db_table_name}`'
            return match.group(0)
        
        # Replace table name in FROM clauses (careful not to replace CTE names)
        db_sql = re.sub(r'FROM\s+`?([^`\s,()]+)`?', replace_from, db_sql, flags=re.IGNORECASE)
        
        # Replace in JOIN clauses similarly
        def replace_join(match):
            table_ref = match.group(1)
            if table_ref.lower() in cte_names:
                return match.group(0)  # Keep CTE reference unchanged
            if table_ref.lower() == original_table_name.lower():
                return f'JOIN `{db_table_name}`'
            return match.group(0)
        
        db_sql = re.sub(r'JOIN\s+`?([^`\s,()]+)`?', replace_join, db_sql, flags=re.IGNORECASE)
    else:
        # Fallback: use simpler replacement but exclude CTE names
        def safe_replace_from(match):
            table_ref = match.group(1) if match.lastindex >= 1 else match.group(0)
            if table_ref.lower() in cte_names:
                return match.group(0)
            return f'FROM `{db_table_name}`'
        
        db_sql = re.sub(r'FROM\s+`?([^`\s,()]+)`?', safe_replace_from, db_sql, flags=re.IGNORECASE)
        db_sql = re.sub(r'JOIN\s+`?([^`\s,()]+)`?', safe_replace_from, db_sql, flags=re.IGNORECASE)
    
    # Also update sql to use correct table name (if it doesn't already)
    if db_table_name not in sql and original_table_name:
        escaped_old = re.escape(original_table_name)
        pattern = rf'`{escaped_old}`'
        sql = re.sub(pattern, f'`{db_table_name}`', sql, flags=re.IGNORECASE)
        
        def replace_from_sql(match):
            table_ref = match.group(1) if match.lastindex >= 1 else match.group(0)
            if table_ref.lower() in cte_names:
                return match.group(0)
            if table_ref.lower() == original_table_name.lower():
                return f'FROM `{db_table_name}`'
            return match.group(0)
        
        sql = re.sub(r'FROM\s+`?([^`\s,()]+)`?', replace_from_sql, sql, flags=re.IGNORECASE)
        sql = re.sub(r'JOIN\s+`?([^`\s,()]+)`?', replace_from_sql, sql, flags=re.IGNORECASE)
    
    return sql, db_sql


def execute_query_on_modified_table(db_path: str, db_sql: str, db_table_name: str) -> tuple[list, str]:
    """
    Execute a SQL query on a modified table in the database.
    
    Args:
        db_path: Path to SQLite database
        db_sql: SQL query with database placeholder names (col0, col1, etc.)
        db_table_name: Database table name
        
    Returns:
        Tuple of (result, error) where result is list of rows or None, error is error message or None
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    try:
        cur.execute(db_sql)
        result = cur.fetchall()
        conn.close()
        return result, None
    except Exception as e:
        conn.close()
        return None, str(e)


def update_query_table_name(query: str, new_table_name: str) -> str:
    """
    Update the table name in a SQL query.
    
    Args:
        query: SQL query string
        new_table_name: New table name to use
        
    Returns:
        Updated SQL query with new table name
    """
    import re
    
    # Extract original table name
    from_match = re.search(r'FROM\s+`?([^`\s,]+)`?', query, re.IGNORECASE)
    if from_match:
        original_table_name = from_match.group(1)
        if original_table_name != new_table_name:
            escaped_old = re.escape(original_table_name)
            pattern = rf'`{escaped_old}`'
            query = re.sub(pattern, f'`{new_table_name}`', query, flags=re.IGNORECASE)
            pattern2 = rf'\bFROM\s+`?{escaped_old}`?'
            query = re.sub(pattern2, f'FROM `{new_table_name}`', query, flags=re.IGNORECASE)
            pattern3 = rf'\bJOIN\s+`?{escaped_old}`?'
            query = re.sub(pattern3, f'JOIN `{new_table_name}`', query, flags=re.IGNORECASE)
    
    return query

