# Import helper functions
import json
import re
import random
from multilingual_evoschema.formatting_helpers import (
    normalize_sql_type,
    denormalize_sql_type,
    get_unique_table_name,
    get_unique_table_id
)


def format_sample_data(table_sample: str = None, max_rows: int = 2) -> str:
    """
    Format sample data (first N rows) from table for prompts.
    
    Args:
        table_sample: JSON string or dict containing table data with 'header' and 'rows'
        max_rows: Maximum number of rows to include (default: 2)
    
    Returns:
        Formatted string showing sample data, or empty string if no data
    """
    if not table_sample:
        return ""
    
    try:
        table = json.loads(table_sample) if isinstance(table_sample, str) else table_sample
        headers = table.get('header', [])
        rows = table.get('rows', [])
        
        if not headers or not rows:
            return ""
        
        # Take first max_rows rows
        sample_rows = rows[:max_rows]
        
        # Build formatted output
        lines = ["\nSAMPLE DATA (first 2 rows):"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---" for _ in headers]) + "|")
        
        for row in sample_rows:
            # Ensure row has same length as headers
            row_data = row[:len(headers)]
            row_data.extend([""] * (len(headers) - len(row_data)))
            # Don't truncate values - show full values for important verification
            formatted_row = [str(val) if val is not None else "" for val in row_data]
            lines.append("| " + " | ".join(formatted_row) + " |")
        
        return "\n".join(lines)
    except Exception as e:
        print(f"Error formatting sample data: {e}")
        return ""


def convert_table_to_schema(table: str) -> str:
    """
    Convert a table JSON string to a schema format showing table name and column names with types.
    
    Args:
        table: JSON string containing table data with 'id', 'header', and 'types' fields
        
    Returns:
        String representation of the schema in format:
        Table: <table_name>
        Columns:
        - <column_name> (<type>)
        - <column_name> (<type>)
        ...
    """
    import json
    
    table = json.loads(table) if isinstance(table, str) else table
    
    # Get table name (use 'id' or 'name' if available)
    table_name = table.get('name', table.get('id', 'unknown_table'))
    
    # Get headers and types
    headers = table.get('header', [])
    types = table.get('types', [])
    
    # Build schema string
    schema_lines = [f"Table: {table_name}", "Columns:"]
    
    # Match each header with its type
    for i, header in enumerate(headers):
        col_type = types[i] if i < len(types) else 'text'  # Default to 'text' if type missing
        # Normalize type names (e.g., 'text' -> 'TEXT', 'number' -> 'INTEGER' or 'REAL')
        sql_type = normalize_sql_type(col_type)
        schema_lines.append(f"  - {header} ({sql_type})")
    
    return "\n".join(schema_lines)
    
def rename_table(schema: str, new_table_name: str, table_sample: str = None) -> str:
    """
    Generate a prompt for renaming a table in the schema.
    
    Args:
        schema: Schema string
        new_table_name: Desired new table name
        table_sample: Optional table sample data
        
    Returns:
        Prompt string for LLM to rename the table
    """
    sample_data = format_sample_data(table_sample)
    
    prompt = f"""You are a database schema expert. Your task is to rename a table in the given schema.

Current Schema:
{schema}
{sample_data}

New Table Name: {new_table_name}

Please provide a JSON response with the following structure:
{{
    "old_name": "<current_table_name>",
    "new_name": "{new_table_name}"
}}

Return ONLY the JSON response, no additional text."""
    
    return prompt

def add_column(schema: str, table_sample: str = None) -> str:
    """
    Generate a prompt for adding a new column to the schema.
    
    Args:
        schema: Schema string
        table_sample: Optional table sample data
        
    Returns:
        Prompt string for LLM to add a column
    """
    sample_data = format_sample_data(table_sample)
    
    prompt = f"""You are a database schema expert. Your task is to add a new column to the given schema.

Current Schema:
{schema}
{sample_data}

Please add ONE new column that:
1. Is semantically meaningful and fits the domain
2. Has a descriptive name
3. Has an appropriate SQL type (TEXT, INTEGER, REAL, etc.)
4. Has 2 sample values that would appear in this column

Provide a JSON response with the following structure:
{{
    "columns": [
        {{
            "column_name": "<new_column_name>",
            "sql_type": "<SQL_TYPE>",
            "values": ["<value1>", "<value2>"]
        }}
    ]
}}

Return ONLY the JSON response, no additional text."""
    
    return prompt

def remove_column(schema: str, sql_query: str, nl: str, table_sample: str = None, column_to_remove: str = None) -> str:
    """
    Remove columns from the schema and datatable.
    If column_to_remove is provided, uses that column. Otherwise randomly chooses one.
    Only if column IS in SQL, asks LLM if NL can still be reasonably and fully answered.
    
    Args:
        schema: Schema string
        sql_query: SQL query string
        nl: Natural language question (REQUIRED - needed to determine if question can still be answered)
        table_sample: Optional table sample data
        column_to_remove: Optional column name to remove (if not provided, randomly selects one)
    """
    if not nl:
        raise ValueError("nl (natural language question) is required for remove_column - needed to determine if question can still be answered")
    import random
    import json
    import re
    
    sample_data = format_sample_data(table_sample)
    
    # If column_to_remove not provided, randomly select one
    if not column_to_remove:
        # Parse schema to extract column names
        column_names = []
        schema_lines = schema.split('\n')
        for line in schema_lines:
            # Match lines like "  - ColumnName (TYPE)" or "- ColumnName (TYPE)"
            match = re.match(r'\s*-\s*(.+?)\s*\(', line)
            if match:
                column_names.append(match.group(1).strip())
        
        if not column_names:
            # Fallback: try to extract from table_sample if available
            if table_sample:
                try:
                    table = json.loads(table_sample) if isinstance(table_sample, str) else table_sample
                    column_names = table.get('header', [])
                except:
                    pass
        
        if not column_names:
            # If we can't find columns, we can't proceed
            raise ValueError("Could not extract column names from schema or table_sample")
        
        column_to_remove = random.choice(column_names)
    
    prompt = f"""You are a database schema expert. A column is being removed from a table, and you need to determine if the natural language question can still be answered and rewrite the SQL query if needed.

Current Schema:
{schema}
{sample_data}

SQL Query:
{sql_query}

Natural Language Question:
{nl}

Column to Remove: {column_to_remove}

Your task:
1. Determine if the natural language question can still be reasonably and fully answered without the column "{column_to_remove}".
2. If YES: Rewrite the SQL query to work without this column, ensuring it still answers the NL question.
3. If NO: Return "unanswerable" as the new_query.

Provide a JSON response with the following structure:
{{
    "new_query": "<rewritten_sql_query_or_unanswerable>"
}}

Return ONLY the JSON response, no additional text."""
    
    return prompt

def rename_column(schema: str, table_sample: str = None) -> str:
    """
    Generate a prompt for renaming columns in the schema.
    
    Args:
        schema: Schema string
        table_sample: Optional table sample data
        
    Returns:
        Prompt string for LLM to rename columns
    """
    sample_data = format_sample_data(table_sample)
    
    prompt = f"""You are a database schema expert. Your task is to rename 2-3 columns in the given schema to make them more descriptive or follow a naming convention.

Current Schema:
{schema}
{sample_data}

Please rename 2-3 columns with new names that:
1. Are more descriptive or follow a consistent naming convention
2. Maintain semantic meaning (the new name should still make sense for the data)
3. Use snake_case or PascalCase consistently

Provide a JSON response with the following structure:
{{
    "renames": [
        {{
            "old_name": "<original_column_name>",
            "new_name": "<new_column_name>"
        }},
        {{
            "old_name": "<original_column_name>",
            "new_name": "<new_column_name>"
        }}
    ]
}}

Return ONLY the JSON response, no additional text."""
    
    return prompt

def split_column(schema: str, sql_query: str, table_sample: str = None) -> str:
    """
    Generate a prompt for splitting a column into multiple columns.
    
    Args:
        schema: Schema string
        sql_query: SQL query string
        table_sample: Optional table sample data
        
    Returns:
        Prompt string for LLM to split a column
    """
    sample_data = format_sample_data(table_sample)
    
    prompt = f"""You are a database schema expert. Your task is to split one column into multiple columns in the given schema.

Current Schema:
{schema}
{sample_data}

SQL Query:
{sql_query}

Please:
1. Identify ONE column that can be meaningfully split into 2-3 separate columns
2. The split should make semantic sense (e.g., "Name" -> "First_Name" and "Last_Name")
3. Provide sample values for the new columns

Provide a JSON response with the following structure:
{{
    "original_column": "<column_name_to_split>",
    "new_columns": [
        {{
            "column_name": "<new_column_name_1>",
            "sql_type": "<SQL_TYPE>",
            "values": ["<value1>", "<value2>"]
        }},
        {{
            "column_name": "<new_column_name_2>",
            "sql_type": "<SQL_TYPE>",
            "values": ["<value1>", "<value2>"]
        }}
    ]
}}

Return ONLY the JSON response, no additional text."""
    
    return prompt

def merge_column_in_query(schema: str, sql_query: str, table_sample: str = None) -> str:
    """
    Generate a prompt for merging columns that appear together in the SQL query.
    
    Args:
        schema: Schema string
        sql_query: SQL query string
        table_sample: Optional table sample data
        
    Returns:
        Prompt string for LLM to merge columns
    """
    sample_data = format_sample_data(table_sample)
    
    prompt = f"""You are a database schema expert. Your task is to merge 2-3 columns that appear together in the SQL query into a single column.

Current Schema:
{schema}
{sample_data}

SQL Query:
{sql_query}

Please:
1. Identify 2-3 columns that appear together in the SQL query
2. Merge them into a single column with a descriptive name
3. The merged column should combine the semantic meaning of the original columns

Provide a JSON response with the following structure:
{{
    "original_columns": ["<column1>", "<column2>"],
    "new_column": "<merged_column_name>"
}}

Return ONLY the JSON response, no additional text."""
    
    return prompt

def merge_column_outside_query(schema: str, table_sample: str = None) -> str:
    """
    Generate a prompt for merging columns that do NOT appear in the SQL query.
    
    Args:
        schema: Schema string
        table_sample: Optional table sample data
        
    Returns:
        Prompt string for LLM to merge columns
    """
    sample_data = format_sample_data(table_sample)
    
    prompt = f"""You are a database schema expert. Your task is to merge 2-3 columns that do NOT appear in the SQL query into a single column.

Current Schema:
{schema}
{sample_data}

Please:
1. Identify 2-3 columns that are NOT used in the SQL query
2. Merge them into a single column with a descriptive name
3. The merged column should combine the semantic meaning of the original columns

Provide a JSON response with the following structure:
{{
    "original_columns": ["<column1>", "<column2>"],
    "new_column": "<merged_column_name>"
}}

Return ONLY the JSON response, no additional text."""
    
    return prompt

def repair_query_column_split(schema: str, new_schema: str, sql_query: str, original_column: str, new_columns: list) -> str:
    """
    Generate a prompt for repairing a SQL query after a column split.
    
    Args:
        schema: Original schema string
        new_schema: New schema string after split
        sql_query: Original SQL query
        original_column: Column that was split
        new_columns: List of new column names
        
    Returns:
        Prompt string for LLM to repair the query
    """
    prompt = f"""You are an expert SQL developer. A column has been split into multiple columns, and the SQL query needs to be updated.

Original Schema:
{schema}

New Schema (after split):
{new_schema}

Original SQL Query:
{sql_query}

Original Column (split): {original_column}
New Columns: {', '.join(new_columns)}

Please rewrite the SQL query to work with the new schema. The query should maintain the same logical intent.

Return ONLY the repaired SQL query, no explanations."""
    
    return prompt

def repair_query_rename(query: str, change_info_json: str, schema_json: str) -> str:
    """
    Repair SQL query after column rename by replacing old column names with new ones.
    
    Args:
        query: Original SQL query
        change_info_json: JSON string with rename information
        schema_json: JSON string with schema information (unused but kept for compatibility)
        
    Returns:
        Repaired SQL query with new column names
    """
    import json
    
    try:
        change_info = json.loads(change_info_json) if isinstance(change_info_json, str) else change_info_json
    except:
        return query  # If parsing fails, return original query
    
    repaired_query = query
    
    if "renames" in change_info:
        renames = change_info["renames"] if isinstance(change_info["renames"], list) else [change_info["renames"]]
        
        for rename in renames:
            if isinstance(rename, dict):
                old_name = rename.get("old_name")
                new_name = rename.get("new_name")
                
                if old_name and new_name:
                    # Replace backticked column names: `old_name` -> `new_name`
                    escaped_old = re.escape(old_name)
                    pattern = rf'`{escaped_old}`'
                    repaired_query = re.sub(pattern, f'`{new_name}`', repaired_query, flags=re.IGNORECASE)
                    
                    # Replace plain column names (with word boundaries)
                    pattern2 = rf'\b{escaped_old}\b'
                    repaired_query = re.sub(pattern2, new_name, repaired_query, flags=re.IGNORECASE)
    
    return repaired_query

def parse_schema_change(response: str) -> dict:
    """
    Parse LLM response to extract schema change information.
    Handles common LLM JSON errors like trailing commas.
    
    Args:
        response: LLM response string (may contain JSON)
        
    Returns:
        Dict with schema change information, or None if parsing fails
    """
    import json
    
    # Try to extract JSON from code blocks first
    json_block = None
    if "```json" in response:
        json_block = response.split("```json", 1)[1].split("```")[0].strip()
    elif "```" in response:
        json_block = response.split("```", 1)[1].split("```")[0].strip()
    else:
        json_block = response.strip()
    
    # Try multiple parsing strategies
    strategies = [
        # Strategy 1: Direct JSON parse
        lambda x: json.loads(x),
        # Strategy 2: Remove trailing commas before closing braces/brackets
        lambda x: json.loads(re.sub(r',(\s*[}\]])', r'\1', x)),
        # Strategy 3: Extract JSON object with regex
        lambda x: json.loads(re.search(r'\{.*\}', x, re.DOTALL).group(0)) if re.search(r'\{.*\}', x, re.DOTALL) else None,
    ]
    
    for strategy in strategies:
        try:
            parsed = strategy(json_block)
            if parsed:
                return parsed
        except:
            continue
    
    # If all strategies fail, try manual parsing for specific patterns
    result = {}
    
    # Try to parse merge_column response (has original_columns array and new_column string)
    merge_match = re.search(r'"original_columns"\s*:\s*\[(.*?)\]', json_block, re.DOTALL)
    if merge_match:
        cols_str = merge_match.group(1)
        # Extract column names (handle quotes and commas)
        cols = [col.strip().strip('"\'') for col in re.findall(r'["\']([^"\']+)["\']', cols_str)]
        result["original_columns"] = cols
        
        new_col_match = re.search(r'"new_column"\s*:\s*["\']([^"\']+)["\']', json_block)
        if new_col_match:
            result["new_column"] = new_col_match.group(1)
            return result
    
    # Try to parse rename_column response (has renames array)
    renames_match = re.search(r'"renames"\s*:\s*\[(.*?)\]', json_block, re.DOTALL)
    if renames_match:
        renames_str = renames_match.group(1)
        renames = []
        # Extract each rename object
        for rename_match in re.finditer(r'\{\s*"old_name"\s*:\s*["\']([^"\']+)["\']\s*,\s*"new_name"\s*:\s*["\']([^"\']+)["\']\s*\}', renames_str):
            renames.append({
                "old_name": rename_match.group(1),
                "new_name": rename_match.group(2)
            })
        if renames:
            result["renames"] = renames
            return result
    
    # Try to parse remove_column response (has new_query)
    new_query_match = re.search(r'"new_query"\s*:\s*["\']([^"\']+)["\']', json_block, re.DOTALL)
    if new_query_match:
        result["new_query"] = new_query_match.group(1)
        return result
    
    # Try to parse split_column response
    split_match = re.search(r'"original_column"\s*:\s*["\']([^"\']+)["\']', json_block)
    if split_match:
        result["original_column"] = split_match.group(1)
        # Try to find new_columns array
        new_cols_match = re.search(r'"new_columns"\s*:\s*\[(.*?)\]', json_block, re.DOTALL)
        if new_cols_match:
            cols_str = new_cols_match.group(1)
            new_cols = []
            # Extract column objects
            for col_match in re.finditer(r'\{\s*"column_name"\s*:\s*["\']([^"\']+)["\']\s*,\s*"sql_type"\s*:\s*["\']([^"\']+)["\']', cols_str):
                new_cols.append({
                    "column_name": col_match.group(1),
                    "sql_type": col_match.group(2)
                })
            if new_cols:
                result["new_columns"] = new_cols
                return result
    
    # Try to parse add_column response
    add_match = re.search(r'"columns"\s*:\s*\[(.*?)\]', json_block, re.DOTALL)
    if add_match:
        cols_str = add_match.group(1)
        columns = []
        # Extract column objects
        for col_match in re.finditer(r'\{\s*"column_name"\s*:\s*["\']([^"\']+)["\']\s*,\s*"sql_type"\s*:\s*["\']([^"\']+)["\']', cols_str):
            columns.append({
                "column_name": col_match.group(1),
                "sql_type": col_match.group(2)
            })
        if columns:
            result["columns"] = columns
            return result
    
    print(f"⚠️  Failed to parse schema change response: {response[:200]}...")
    return None


def translate_schema(schema: str, target_language: str) -> str:
    """
    Generate a prompt for translating table and column names in a schema to a target language.
    
    Args:
        schema: Schema string with table and column names
        target_language: Target language code (e.g., "es", "fr", "hi", "hi_typed")
        
    Returns:
        Prompt string for LLM to translate the schema
    """
    prompt = f"""You are a multilingual database schema expert. Your task is to translate table and column names in the given schema to {target_language}.

Current Schema:
{schema}

Please translate:
1. The table name to {target_language}
2. All column names to {target_language}
3. Keep SQL types unchanged (TEXT, INTEGER, REAL, etc.)

Provide a JSON response with the following structure:
{{
    "translated_schema": [
        {{
            "original_table": "<original_table_name>",
            "translated_table": "<translated_table_name>",
            "column_translations": [
                {{
                    "original": "<original_column_name>",
                    "translated": "<translated_column_name>"
                }}
            ]
        }}
    ]
}}

Return ONLY the JSON response, no additional text."""
    
    return prompt


# ============================================================================
# Schema Change Application Functions
# ============================================================================

def apply_add_column_change(table: dict, change_info: dict) -> tuple[dict, str]:
    """
    Apply add_column change to table structure.
    
    Args:
        table: Table dict with 'header', 'types', 'rows'
        change_info: Dict with 'columns' list containing column info
        
    Returns:
        Tuple of (updated_table, updated_schema_string)
    """
    from multilingual_evoschema.schema_change import convert_table_to_schema
    
    # Store original column values for verification (first 2 rows of each original column)
    original_header_count = len(table["header"])
    original_column_values_before = {}
    for col_idx in range(original_header_count):
        col_name = table["header"][col_idx]
        first_two_values = []
        for row_idx in range(min(2, len(table["rows"]))):
            if col_idx < len(table["rows"][row_idx]):
                first_two_values.append(table["rows"][row_idx][col_idx])
        original_column_values_before[col_name] = first_two_values
    
    # Add new columns
    if "columns" in change_info:
        columns_to_add = change_info["columns"] if isinstance(change_info["columns"], list) else [change_info["columns"]]
        print(f"    Adding columns:")
        for col_info in columns_to_add:
            if isinstance(col_info, dict):
                col_name = col_info.get("column_name", "")
                col_type = col_info.get("sql_type", "TEXT")
                # Append new column name to header (preserves all original column names)
                table["header"].append(col_name)
                # Append new column type to types (preserves all original column types)
                table["types"].append(col_type.lower())
                
                # Get values from LLM response (should be a list of 2 values)
                values = col_info.get("values", [])
                
                # For each existing row, append a randomly sampled value from LLM values
                # This keeps all original column values intact and adds new values
                for row in table["rows"]:
                    if values and len(values) > 0:
                        # Randomly sample from the LLM-provided values (typically 2 values)
                        sampled_value = str(random.choice(values))
                        row.append(sampled_value)
                    else:
                        # Fallback: if no values provided, append empty string
                        row.append("")
                
                # Verify row length matches header length after appending
                expected_length = len(table["header"])
                for i, row in enumerate(table["rows"]):
                    if len(row) != expected_length:
                        print(f"      ⚠️  Warning: Row {i} length ({len(row)}) doesn't match header length ({expected_length})")
                
                print(f"      - {col_name} ({col_type}) with {len(values)} value(s) from LLM")
    
    # Verify original column values are unchanged
    print(f"    🔍 Verifying original column values are unchanged:")
    all_values_match = True
    for col_idx in range(original_header_count):
        col_name = table["header"][col_idx]
        original_values = original_column_values_before.get(col_name, [])
        current_first_two_values = []
        for row_idx in range(min(2, len(table["rows"]))):
            if col_idx < len(table["rows"][row_idx]):
                current_first_two_values.append(table["rows"][row_idx][col_idx])
        
        if original_values == current_first_two_values:
            print(f"      ✅ {col_name}: {original_values} (unchanged)")
        else:
            print(f"      ❌ {col_name}: BEFORE {original_values}, AFTER {current_first_two_values} (MISMATCH!)")
            all_values_match = False
    
    if all_values_match:
        print(f"    ✅ All original column values verified unchanged")
    else:
        print(f"    ⚠️  WARNING: Some original column values changed!")
    
    # Update schema string
    schema = convert_table_to_schema(json.dumps(table))
    print(f"    ✅ Query unchanged (added columns are not in the query)")
    
    return table, schema


def apply_remove_column_change(table: dict, change_info: dict, query: str) -> tuple[dict, str, str]:
    """
    Apply remove_column change to table structure.
    
    Args:
        table: Table dict with 'header', 'types', 'rows'
        change_info: Dict with 'removed_column' and optionally 'new_query'
        query: Original SQL query
        
    Returns:
        Tuple of (updated_table, updated_schema_string, updated_query)
    """
    from multilingual_evoschema.schema_change import convert_table_to_schema
    
    removed_col = change_info.get("removed_column")
    new_query = change_info.get("new_query")
    
    print(f"    Removing column: {removed_col}")
    
    if not removed_col:
        print(f"    ⚠️  No removed_column in change_info, skipping this change")
        return None, None, None
    
    # Manually remove column from table structure
    if removed_col in table["header"]:
        col_idx = table["header"].index(removed_col)
        table["header"].pop(col_idx)
        if col_idx < len(table["types"]):
            table["types"].pop(col_idx)
        for row in table["rows"]:
            if col_idx < len(row):
                row.pop(col_idx)
        print(f"    ✅ Removed column '{removed_col}' from table structure")
    else:
        print(f"    ⚠️  Column '{removed_col}' not found in table, skipping this change")
        return None, None, None
    
    # Regenerate schema from updated table
    schema = convert_table_to_schema(json.dumps(table))
    print(f"    ✅ Regenerated schema from updated table")
    
    # Update query - "unanswerable" is a valid result
    if new_query:
        updated_query = new_query
        if new_query.strip().lower() == "unanswerable":
            print(f"    ⚠️  Query marked as UNANSWERABLE - NL question cannot be answered without removed column")
        else:
            print(f"    ✅ Updated SQL query: {updated_query}")
    else:
        updated_query = query
        print(f"    ✅ Query unchanged (column was not in query)")
    
    return table, schema, updated_query


def apply_rename_column_change(table: dict, change_info: dict, query: str) -> tuple[dict, str, str]:
    """
    Apply rename_column change to table structure.
    
    Args:
        table: Table dict with 'header', 'types', 'rows'
        change_info: Dict with 'renames' list
        query: Original SQL query
        
    Returns:
        Tuple of (updated_table, updated_schema_string, updated_query)
    """
    from multilingual_evoschema.schema_change import convert_table_to_schema, repair_query_rename
    
    # Store original column values for verification (first 2 rows of each column)
    original_column_values_before = {}
    rename_mapping = {}  # Maps old_name -> new_name
    
    if "renames" in change_info:
        renames = change_info["renames"] if isinstance(change_info["renames"], list) else [change_info["renames"]]
        for rename in renames:
            if isinstance(rename, dict):
                old_name = rename.get("old_name")
                new_name = rename.get("new_name")
                rename_mapping[old_name] = new_name
        
        # Store values for all columns (both renamed and unchanged)
        for col_idx in range(len(table["header"])):
            col_name = table["header"][col_idx]
            first_two_values = []
            for row_idx in range(min(2, len(table["rows"]))):
                if col_idx < len(table["rows"][row_idx]):
                    first_two_values.append(table["rows"][row_idx][col_idx])
            original_column_values_before[col_name] = first_two_values
    
    # Rename columns
    if "renames" in change_info:
        renames = change_info["renames"] if isinstance(change_info["renames"], list) else [change_info["renames"]]
        print(f"    Renaming columns:")
        for rename in renames:
            if isinstance(rename, dict):
                old_name = rename.get("old_name")
                new_name = rename.get("new_name")
                if old_name in table["header"]:
                    idx = table["header"].index(old_name)
                    table["header"][idx] = new_name
                    print(f"      - {old_name} → {new_name}")
                else:
                    print(f"      ⚠️  Column '{old_name}' not found in table, skipping")
        
        # Verify column values are unchanged (only names changed)
        print(f"    🔍 Verifying column values are unchanged (only names changed):")
        all_values_match = True
        for col_idx in range(len(table["header"])):
            new_col_name = table["header"][col_idx]
            # Find the old name for this column
            old_col_name = None
            for old_name, new_name in rename_mapping.items():
                if new_name == new_col_name:
                    old_col_name = old_name
                    break
            
            # If this column was renamed, use old name to get original values
            # Otherwise, use current name (column wasn't renamed)
            lookup_name = old_col_name if old_col_name else new_col_name
            original_values = original_column_values_before.get(lookup_name, [])
            
            current_first_two_values = []
            for row_idx in range(min(2, len(table["rows"]))):
                if col_idx < len(table["rows"][row_idx]):
                    current_first_two_values.append(table["rows"][row_idx][col_idx])
            
            if original_values == current_first_two_values:
                display_name = f"{old_col_name} → {new_col_name}" if old_col_name else new_col_name
                print(f"      ✅ {display_name}: {original_values} (unchanged)")
            else:
                display_name = f"{old_col_name} → {new_col_name}" if old_col_name else new_col_name
                print(f"      ❌ {display_name}: BEFORE {original_values}, AFTER {current_first_two_values} (MISMATCH!)")
                all_values_match = False
        
        if all_values_match:
            print(f"    ✅ All column values verified unchanged (only names changed)")
        else:
            print(f"    ⚠️  WARNING: Some column values changed!")
        
        # Update schema
        schema = convert_table_to_schema(json.dumps(table))
        # Repair SQL query
        original_query_before_repair = query
        updated_query = repair_query_rename(query, json.dumps(change_info), json.dumps(change_info))
        print(f"    SQL query updated to use new column names")
        print(f"    Original query: {original_query_before_repair}")
        print(f"    Repaired query: {updated_query}")
        
        return table, schema, updated_query
    
    return table, convert_table_to_schema(json.dumps(table)), query


def apply_split_column_change(table: dict, change_info: dict, query: str, extract_table_name_fn) -> tuple[dict, str, str]:
    """
    Apply split_column change to table structure.
    
    Args:
        table: Table dict with 'header', 'types', 'rows'
        change_info: Dict with 'original_column' and 'new_columns'
        query: Original SQL query
        extract_table_name_fn: Function to extract table name from query
        
    Returns:
        Tuple of (updated_table, updated_schema_string, updated_query)
    """
    from multilingual_evoschema.schema_change import convert_table_to_schema, repair_query_column_split
    from multilingual_evoschema.llm import run_gpt
    from multilingual_evoschema.schema_change import parse_schema_change
    
    original_col = change_info.get("original_column")
    new_cols = change_info.get("new_columns", [])
    print(f"    Splitting column: {original_col}")
    print(f"    Into:")
    for new_col in new_cols:
        if isinstance(new_col, dict):
            col_name = new_col.get("column_name", "")
            col_type = new_col.get("sql_type", "TEXT")
            print(f"      - {col_name} ({col_type})")
    
    if original_col and original_col in table["header"]:
        col_idx = table["header"].index(original_col)
        # Remove original column
        table["header"].pop(col_idx)
        table["types"].pop(col_idx)
        for row in table["rows"]:
            if col_idx < len(row):
                row.pop(col_idx)
        # Add new columns
        for new_col in new_cols:
            if isinstance(new_col, dict):
                table["header"].append(new_col.get("column_name", ""))
                table["types"].append(new_col.get("sql_type", "TEXT").lower())
                values = new_col.get("values", [])
                for row in table["rows"]:
                    if values:
                        row.append(str(random.choice(values)))
                    else:
                        row.append("")
        # Update schema
        schema = convert_table_to_schema(json.dumps(table))
        # Repair SQL query using LLM
        new_col_names = [col.get("column_name", "") for col in new_cols if isinstance(col, dict)]
        
        # Extract original table name to preserve it
        original_table_name = extract_table_name_fn(query)
        
        repair_prompt = repair_query_column_split(schema, schema, query, original_col, new_col_names)
        print(f"    🤖 Using GPT (gpt-5-nano) for query repair after split")
        repair_response = run_gpt(repair_prompt, max_tokens=2048)
        # Extract SQL from response (handle code blocks and plain text)
        repaired_sql = repair_response.strip()
        if "```sql" in repaired_sql:
            repaired_sql = repaired_sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in repaired_sql:
            repaired_sql = repaired_sql.split("```")[1].split("```")[0].strip()
        
        # Try to parse as JSON first (in case LLM returns JSON)
        repair_info = parse_schema_change(repair_response)
        if repair_info and "repaired_sql" in repair_info:
            updated_query = repair_info["repaired_sql"]
        elif repaired_sql:
            # Use the extracted SQL directly
            updated_query = repaired_sql
        else:
            updated_query = query
        
        # Preserve the original table name (LLM might have changed it)
        if original_table_name:
            # Replace any table name in the repaired query with the original
            repaired_table_name = extract_table_name_fn(updated_query)
            if repaired_table_name and repaired_table_name != original_table_name:
                escaped_repaired = re.escape(repaired_table_name)
                pattern = rf'`{escaped_repaired}`'
                updated_query = re.sub(pattern, f'`{original_table_name}`', updated_query, flags=re.IGNORECASE)
                # Also handle FROM/JOIN clauses
                pattern2 = rf'\bFROM\s+`?{escaped_repaired}`?'
                updated_query = re.sub(pattern2, f'FROM `{original_table_name}`', updated_query, flags=re.IGNORECASE)
                pattern3 = rf'\bJOIN\s+`?{escaped_repaired}`?'
                updated_query = re.sub(pattern3, f'JOIN `{original_table_name}`', updated_query, flags=re.IGNORECASE)
                print(f"    Preserved original table name: {original_table_name}")
        
        return table, schema, updated_query
    
    return table, convert_table_to_schema(json.dumps(table)), query


def apply_merge_column_change(table: dict, change_info: dict, query: str, has_operation_on_multiple_columns_fn) -> tuple[dict, str, str]:
    """
    Apply merge_column change to table structure.
    
    Args:
        table: Table dict with 'header', 'types', 'rows'
        change_info: Dict with 'original_columns' and 'new_column'
        query: Original SQL query
        has_operation_on_multiple_columns_fn: Function to check if query has operations on multiple columns
        
    Returns:
        Tuple of (updated_table, updated_schema_string, updated_query)
    """
    from multilingual_evoschema.schema_change import convert_table_to_schema
    
    original_cols = change_info.get("original_columns", [])
    new_col = change_info.get("new_column")
    sql_type = change_info.get("sql_type", "TEXT")
    
    print(f"    Merging columns: {', '.join(original_cols)}")
    print(f"    Into: {new_col}")
    print(f"    SQL Type: {sql_type} (default, LLM no longer provides this)")
    
    if original_cols and new_col:
        # Get indices of columns being merged BEFORE removal (needed for generating merged values)
        merged_col_indices = []
        merged_col_headers = []
        for orig_col in original_cols:
            if orig_col in table["header"]:
                idx = table["header"].index(orig_col)
                merged_col_indices.append(idx)
                merged_col_headers.append(orig_col)
        
        # Generate merged values from original column data BEFORE removing columns
        merged_values = []
        for row in table["rows"]:
            # Get values from columns being merged (using original indices before removal)
            merged_parts = []
            for idx in merged_col_indices:
                if idx < len(row):
                    merged_parts.append(str(row[idx]) if row[idx] is not None else "")
            # Combine with a separator (comma or underscore)
            merged_value = ", ".join(merged_parts) if merged_parts else ""
            merged_values.append(merged_value)
        
        # Now remove original columns (sort indices in reverse to remove from end)
        indices_to_remove = sorted(merged_col_indices, reverse=True)
        for idx in indices_to_remove:
            table["header"].pop(idx)
            table["types"].pop(idx)
            for row in table["rows"]:
                if idx < len(row):
                    row.pop(idx)
        
        print(f"    ✅ merged values generated from data (first 3): {merged_values[:3]}")
        
        # Add merged column with actual merged values
        table["header"].append(new_col)
        table["types"].append(sql_type.lower())
        
        # Add merged values to rows
        for i, row in enumerate(table["rows"]):
            if i < len(merged_values):
                row.append(merged_values[i])
            else:
                row.append("")
        
        # Update schema
        schema = convert_table_to_schema(json.dumps(table))
        
        # Check if this is merge_column_in_query (columns appear together in query)
        is_merge_in_query = has_operation_on_multiple_columns_fn(query)
        
        if is_merge_in_query:
            # Check if any individual merged columns appear separately in the query
            # (not together, but individually in clauses like ORDER BY, WHERE, GROUP BY, HAVING, or separately in SELECT)
            print(f"    📋 Merge Column Details:")
            print(f"       New Column Name: {new_col}")
            print(f"       First 2 Merged Values in Table: {merged_values[:2] if len(merged_values) >= 2 else merged_values}")
            
            # Check if merged columns appear separately (individually) in any clause
            has_separate_columns = False
            separate_locations = []
            
            # Check ORDER BY clause
            order_by_match = re.search(r'ORDER\s+BY\s+([^;]+?)(?:;|$)', query, re.IGNORECASE | re.DOTALL)
            if order_by_match:
                order_by_clause = order_by_match.group(1)
                for old_col in original_cols:
                    escaped_old = re.escape(old_col)
                    if re.search(rf'`?{escaped_old}`?', order_by_clause, re.IGNORECASE):
                        has_separate_columns = True
                        separate_locations.append(f"ORDER BY ({old_col})")
                        break
            
            # Check GROUP BY clause
            if not has_separate_columns:
                group_by_match = re.search(r'GROUP\s+BY\s+([^;]+?)(?:\s+ORDER\s+BY|\s+HAVING|;|$)', query, re.IGNORECASE | re.DOTALL)
                if group_by_match:
                    group_by_clause = group_by_match.group(1)
                    for old_col in original_cols:
                        escaped_old = re.escape(old_col)
                        if re.search(rf'`?{escaped_old}`?', group_by_clause, re.IGNORECASE):
                            has_separate_columns = True
                            separate_locations.append(f"GROUP BY ({old_col})")
                            break
            
            # Check WHERE clause
            if not has_separate_columns:
                where_match = re.search(r'WHERE\s+([^;]+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|;|$)', query, re.IGNORECASE | re.DOTALL)
                if where_match:
                    where_clause = where_match.group(1)
                    for old_col in original_cols:
                        escaped_old = re.escape(old_col)
                        if re.search(rf'`?{escaped_old}`?', where_clause, re.IGNORECASE):
                            has_separate_columns = True
                            separate_locations.append(f"WHERE ({old_col})")
                            break
            
            # Check HAVING clause
            if not has_separate_columns:
                having_match = re.search(r'HAVING\s+([^;]+?)(?:\s+ORDER\s+BY|;|$)', query, re.IGNORECASE | re.DOTALL)
                if having_match:
                    having_clause = having_match.group(1)
                    for old_col in original_cols:
                        escaped_old = re.escape(old_col)
                        if re.search(rf'`?{escaped_old}`?', having_clause, re.IGNORECASE):
                            has_separate_columns = True
                            separate_locations.append(f"HAVING ({old_col})")
                            break
            
            # Check SELECT clause - if merged columns appear separately (not together)
            if not has_separate_columns:
                select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
                if select_match:
                    select_clause = select_match.group(1)
                    select_items = [item.strip() for item in select_clause.split(',')]
                    
                    # Count how many merged columns appear in SELECT
                    merged_cols_in_select = 0
                    for item in select_items:
                        item_clean = re.sub(r'\s+AS\s+[^,\s]+', '', item, flags=re.IGNORECASE).strip()
                        for old_col in original_cols:
                            if item_clean == f'`{old_col}`' or item_clean == old_col:
                                merged_cols_in_select += 1
                                break
                    
                    # If only one merged column appears (not all together), it's separate
                    if merged_cols_in_select > 0 and merged_cols_in_select < len(original_cols):
                        has_separate_columns = True
                        separate_locations.append(f"SELECT (only {merged_cols_in_select} of {len(original_cols)} merged columns)")
            
            if has_separate_columns:
                # Merged columns appear separately - mark as unanswerable
                print(f"    ⚠️  Merged columns appear separately in query: {', '.join(separate_locations)}")
                print(f"    ⚠️  Marking query as UNANSWERABLE - cannot answer question with merged columns used separately")
                updated_query = "unanswerable"
            else:
                # Columns appear together - can repair by replacing with merged column
                print(f"    ✅ Merged columns appear together in query - can repair")
                updated_query = query
                
                # Replace merged columns in SELECT with the merged column name
                select_match = re.search(r'SELECT\s+(.*?)\s+FROM', updated_query, re.IGNORECASE | re.DOTALL)
                if select_match:
                    select_clause = select_match.group(1)
                    select_items = [item.strip() for item in select_clause.split(',')]
                    
                    # Find items that are merged columns
                    merged_col_items = []
                    remaining_items = []
                    for item in select_items:
                        item_clean = re.sub(r'\s+AS\s+[^,\s]+', '', item, flags=re.IGNORECASE).strip()
                        # Check if this item is one of the old columns being merged
                        is_old_col = any(
                            item_clean == f'`{col}`' or item_clean == col 
                            for col in original_cols
                        )
                        if is_old_col:
                            merged_col_items.append(item)
                        else:
                            remaining_items.append(item)
                    
                    # Replace merged columns with single merged column reference
                    if merged_col_items:
                        new_select_items = remaining_items + [f'`{new_col}`']
                        new_select_clause = ', '.join(new_select_items)
                        updated_query = re.sub(
                            r'SELECT\s+' + re.escape(select_clause) + r'\s+FROM',
                            f'SELECT {new_select_clause} FROM',
                            updated_query,
                            flags=re.IGNORECASE | re.DOTALL
                        )
                        print(f"    ✅ Replaced merged columns in SELECT with `{new_col}`")
                
                # Replace CONCAT expressions that combine old columns
                def replace_concat_with_merged_col(match):
                    concat_expr = match.group(0)
                    # Check if this CONCAT contains all the old columns
                    contains_all = all(
                        f'`{old_col}`' in concat_expr or 
                        f'`{old_col.lower()}`' in concat_expr.lower() or
                        old_col in concat_expr
                        for old_col in original_cols
                    )
                    if contains_all:
                        return f'`{new_col}`'
                    return concat_expr
                
                # Match CONCAT expressions
                concat_pattern = r'CONCAT\s*\((?:[^()]+|\([^()]*\))*\)'
                updated_query = re.sub(concat_pattern, replace_concat_with_merged_col, updated_query, flags=re.IGNORECASE)
                
                # Replace any remaining references to old columns with the merged column
                # Use temporary placeholders to avoid double replacement when new_col contains old_col text
                temp_placeholder = "__MERGED_COL_PLACEHOLDER__"
                
                # First pass: replace all old columns with temporary placeholder
                for old_col in original_cols:
                    escaped_old = re.escape(old_col)
                    # Replace backticked: `old_col` -> placeholder
                    pattern = rf'`{escaped_old}`'
                    updated_query = re.sub(pattern, f'`{temp_placeholder}`', updated_query, flags=re.IGNORECASE)
                    # Replace plain (with word boundaries)
                    pattern2 = rf'\b{escaped_old}\b'
                    updated_query = re.sub(pattern2, temp_placeholder, updated_query, flags=re.IGNORECASE)
                
                # Second pass: replace placeholder with actual merged column name
                updated_query = updated_query.replace(f'`{temp_placeholder}`', f'`{new_col}`')
                updated_query = updated_query.replace(temp_placeholder, new_col)
                
                print(f"    ✅ Original SQL: {query}")
                print(f"    ✅ Updated SQL: {updated_query}")
        else:
            # merge_column_outside_query: No query repair needed
            updated_query = query
            print(f"    ✅ Query unchanged (merged columns were not in the query)")
        
        return table, schema, updated_query
    
    return table, convert_table_to_schema(json.dumps(table)), query


# Import database management function
from multilingual_evoschema.db_manager import add_new_table_to_db
