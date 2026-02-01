"""
Helper functions for formatting and type conversion in schema operations.
"""
import sqlite3
import json
import os
import time


def normalize_sql_type(type_str: str) -> str:
    """
    Normalize type strings to SQLite-compatible types.
    
    Args:
        type_str: Type string from table data (e.g., 'text', 'number', etc.)
        
    Returns:
        SQLite-compatible type name
    """
    type_lower = type_str.lower() if type_str else 'text'
    
    type_mapping = {
        'text': 'TEXT',
        'string': 'TEXT',
        'varchar': 'TEXT',
        'char': 'TEXT',
        'number': 'INTEGER',
        'integer': 'INTEGER',
        'int': 'INTEGER',
        'real': 'REAL',
        'float': 'REAL',
        'double': 'REAL',
        'numeric': 'NUMERIC',
        'date': 'TEXT',  # SQLite doesn't have native DATE type
        'datetime': 'TEXT',
        'timestamp': 'TEXT',
        'boolean': 'INTEGER',  # SQLite uses INTEGER for booleans
        'bool': 'INTEGER',
    }
    
    return type_mapping.get(type_lower, 'TEXT')  # Default to TEXT if unknown


def denormalize_sql_type(sql_type: str) -> str:
    """
    Convert SQLite type back to the original type format used in train.tables.jsonl.
    
    Args:
        sql_type: SQLite type (e.g., 'TEXT', 'INTEGER', 'REAL')
        
    Returns:
        Original type string (e.g., 'text', 'number')
    """
    sql_type_upper = sql_type.upper() if sql_type else 'TEXT'
    
    reverse_mapping = {
        'TEXT': 'text',
        'INTEGER': 'number',
        'INT': 'number',
        'REAL': 'number',
        'FLOAT': 'number',
        'DOUBLE': 'number',
        'NUMERIC': 'number',
    }
    
    return reverse_mapping.get(sql_type_upper, 'text')  # Default to 'text'


def get_unique_table_name(cur: sqlite3.Cursor, base_name: str) -> str:
    """
    Get a unique table name by checking existing tables and appending a suffix if needed.
    
    Args:
        cur: SQLite cursor
        base_name: Base table name to check
        
    Returns:
        Unique table name (may have suffix like _1, _2, etc.)
    """
    # Check if base name exists
    cur.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
    """, (base_name,))
    
    if cur.fetchone() is None:
        # Base name is available
        return base_name
    
    # Base name exists, try with suffix
    suffix = 1
    while True:
        candidate_name = f"{base_name}_{suffix}"
        cur.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (candidate_name,))
        
        if cur.fetchone() is None:
            # Found unique name
            print(f"⚠️  Table '{base_name}' already exists, using '{candidate_name}' instead")
            return candidate_name
        
        suffix += 1
        # Safety limit to prevent infinite loop
        if suffix > 10000:
            # Use timestamp as fallback
            timestamp = int(time.time())
            candidate_name = f"{base_name}_{timestamp}"
            print(f"⚠️  Many variants exist, using '{candidate_name}' instead")
            return candidate_name


def get_unique_table_id(tables_jsonl_path: str, base_id: str) -> str:
    """
    Get a unique table ID by checking existing IDs in train.tables.jsonl.
    
    Args:
        tables_jsonl_path: Path to train.tables.jsonl file
        base_id: Base table ID to check
        
    Returns:
        Unique table ID (may have suffix like _1, _2, etc.)
    """
    if not os.path.exists(tables_jsonl_path):
        return base_id
    
    # Read existing IDs
    existing_ids = set()
    try:
        with open(tables_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        table_entry = json.loads(line)
                        existing_ids.add(table_entry.get('id', ''))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    
    # Check if base ID exists
    if base_id not in existing_ids:
        return base_id
    
    # Base ID exists, try with suffix
    suffix = 1
    while True:
        candidate_id = f"{base_id}_{suffix}"
        if candidate_id not in existing_ids:
            return candidate_id
        
        suffix += 1
        # Safety limit
        if suffix > 10000:
            timestamp = int(time.time())
            candidate_id = f"{base_id}_{timestamp}"
            return candidate_id

