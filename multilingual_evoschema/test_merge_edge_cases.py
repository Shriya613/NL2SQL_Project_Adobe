"""
Test schema changes on the mock database to see what happens when tables get added.
Generates a CSV with nl, sql, schema (with first 2 rows), new_schema (with first 2 rows), new_sql.
"""
import os
import sys
import sqlite3
import json
import csv

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multilingual_evoschema import apply_schema_change
from multilingual_evoschema.schema_change import convert_table_to_schema, format_sample_data
import importlib.util
import sys

# Import internal functions directly
spec = importlib.util.spec_from_file_location("multilingual_evoschema_module", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                 "multilingual_evoschema", "__init__.py"))
evoschema_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evoschema_module)

# Get functions we need
apply_single_schema_change = evoschema_module.apply_single_schema_change
get_table_from_jsonl = evoschema_module.get_table_from_jsonl
extract_table_name_from_query = evoschema_module.extract_table_name_from_query

# Mock test queries matching test_schema_changes.py
'''

        "sql": """
            SELECT COUNT(`Player`) AS `Total Players`,
                   AVG(`Years in Toronto`) AS `Average Years in Toronto`,
                   MAX(`Years in Toronto`) - MIN(`Years in Toronto`) AS `Range of Years in Toronto`
            FROM `table_1_10015132_14`;
        """,
        "nl": "What is the total number of players, the average number of years they spent in Toronto, and the range of their tenure in the team?"
    },
    {
        "sql": """
            SELECT COUNT(`Player`) AS `Number_of_Players`,
                   `Position`,
                   AVG(`Pick #`) AS `Average_Pick_Number`
            FROM `table_1_10812938_4`
            GROUP BY `Position`;
        """,
        "nl": "What is the total number of credits for each hand when played with 1 to 5 credits, sorted from highest to lowest?"
    },
    {
        "sql": """
            SELECT `Country`,
                   COUNT(`Nomination`) AS `Number of Nominations`
            FROM `table_1_10236830_4`
            GROUP BY `Country`
            ORDER BY `Number of Nominations` DESC;
        """,
        "nl": "What is the total number of nominations each country received, and which countries received the highest number of nominations?"
    }
'''
test_queries = [
    {
        "sql": """
            SELECT `Player`, `No.` FROM `table_1_10015132_14`
            ORDER BY `No.`;
        """,
        "nl": "List the players and their numbers in order of their jersey numbers."
    }
]

test_merge_edge_cases = [
    {
        "sql": """
            SELECT `Player`, `No.` FROM `table_1_10015132_14`
            ORDER BY `No.`;
        """,
        "nl": "List the players and their numbers in order of their jersey numbers."
    },
    {
        "sql": """
            SELECT `Player`, `Points scored`, `No.`
FROM `table_1_10015132_14`
WHERE `Points scored` > (
    SELECT AVG(`Points scored`) FROM `table_1_10015132_14`
);
        """,
        "nl": "List the players and their numbers who scored more than the average points scored by all players."
    },
     {"sql": """
            SELECT `Player`, `No.` FROM `table_1_10015132_14`
            GROUP BY `No.`;
        """,
        "nl": "List the players and their numbers grouped by their jersey numbers."
    },
    {"sql": """
            SELECT `Player`, `No.` FROM `table_1_10015132_14`
            WHERE `No.` > 10;
        """,
        "nl": "List the players and their numbers who have a jersey number greater than 10."
    },
    {"sql": """
            SELECT `Player`, `No.` FROM `table_1_10015132_14`
            WHERE `No.` > (SELECT AVG(`No.`) FROM `table_1_10015132_14`);
        """,
        "nl": "List the players and their numbers who have a jersey number greater than 10."
    }
]

def get_schema_with_sample_data(table_dict):
    """Get schema string with first 2 rows of sample data."""
    schema = convert_table_to_schema(json.dumps(table_dict))
    sample_data = format_sample_data(table_dict, max_rows=2)
    return schema + sample_data

def analyze_merge_column_in_query(query: str, original_cols: list, new_col: str):
    """
    Detailed analysis of how merge_column_in_query would process a query.
    Shows step-by-step what happens to each column reference.
    """
    import re
    
    print("\n" + "="*80)
    print("DETAILED MERGE ANALYSIS")
    print("="*80)
    print(f"\nOriginal Query:")
    print(query.strip())
    print(f"\nMerging columns: {', '.join(original_cols)} → {new_col}")
    
    # Step 1: Analyze original query to find column usage
    print(f"\n{'─'*80}")
    print("STEP 1: Analyze column usage in original query")
    print(f"{'─'*80}")
    
    column_usage = {}
    for col in original_cols:
        escaped_col = re.escape(col)
        pattern = rf'`?{escaped_col}`?'
        matches = list(re.finditer(pattern, query, re.IGNORECASE))
        column_usage[col] = []
        
        for match in matches:
            start = max(0, match.start() - 50)
            end = min(len(query), match.end() + 50)
            context = query[start:end]
            
            # Determine which clause this appears in
            clause_type = "UNKNOWN"
            query_before = query[:match.start()]
            if re.search(r'ORDER\s+BY', query_before, re.IGNORECASE):
                clause_type = "ORDER BY"
            elif re.search(r'GROUP\s+BY', query_before, re.IGNORECASE):
                clause_type = "GROUP BY"
            elif re.search(r'WHERE', query_before, re.IGNORECASE):
                clause_type = "WHERE"
            elif re.search(r'HAVING', query_before, re.IGNORECASE):
                clause_type = "HAVING"
            elif re.search(r'SELECT', query_before, re.IGNORECASE):
                clause_type = "SELECT"
            
            column_usage[col].append({
                'position': match.start(),
                'context': context.strip(),
                'clause': clause_type
            })
    
    for col, usages in column_usage.items():
        print(f"\n  Column '{col}' appears {len(usages)} time(s):")
        for i, usage in enumerate(usages, 1):
            print(f"    {i}. In {usage['clause']}: ...{usage['context']}...")
    
    # Step 2: Check if columns appear alone or together
    # This matches the logic in __init__.py lines 690-707
    # Uses a 100-character context window around each occurrence
    print(f"\n{'─'*80}")
    print("STEP 2: Determine if columns appear alone or together")
    print(f"{'─'*80}")
    print("  (Using 100-character context window around each occurrence)")
    
    columns_appearing_alone = {}
    for idx, col in enumerate(original_cols):
        appears_alone = False
        
        # Check each occurrence of this column (matches actual clause-aware implementation)
        escaped_col = re.escape(col)
        col_pattern = rf'`?{escaped_col}`?'
        matches = list(re.finditer(col_pattern, query, re.IGNORECASE))
        
        for match in matches:
            match_pos = match.start()
            
            # Determine which SQL clause this column appears in (matches actual implementation)
            clause_start = 0
            clause_end = len(query)
            clause_type = None
            
            # Check ORDER BY
            order_by_pattern = r'ORDER\s+BY\s+([^;]+?)(?:;|$)'
            order_by_match = re.search(order_by_pattern, query, re.IGNORECASE | re.DOTALL)
            if order_by_match:
                order_by_start = order_by_match.start()
                semicolon_pos = query.find(';', order_by_match.end())
                if semicolon_pos != -1:
                    order_by_end = semicolon_pos + 1
                else:
                    order_by_end = len(query)
                if order_by_start <= match_pos < order_by_end:
                    clause_start = order_by_start
                    clause_end = order_by_end
                    clause_type = "ORDER BY"
            
            # Check GROUP BY
            if not clause_type:
                group_by_pattern = r'GROUP\s+BY\s+([^;]+?)(?:\s+ORDER\s+BY|\s+HAVING|;|$)'
                group_by_match = re.search(group_by_pattern, query, re.IGNORECASE | re.DOTALL)
                if group_by_match:
                    group_by_start = group_by_match.start()
                    next_clause = len(query)
                    for pattern in [r'\s+ORDER\s+BY', r'\s+HAVING', r';']:
                        next_match = re.search(pattern, query[group_by_match.end():], re.IGNORECASE)
                        if next_match:
                            next_clause = min(next_clause, group_by_match.end() + next_match.start())
                    group_by_end = next_clause if next_clause < len(query) else len(query)
                    if group_by_start <= match_pos < group_by_end:
                        clause_start = group_by_start
                        clause_end = group_by_end
                        clause_type = "GROUP BY"
            
            # Check WHERE
            if not clause_type:
                where_pattern = r'WHERE\s+([^;]+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|;|$)'
                where_match = re.search(where_pattern, query, re.IGNORECASE | re.DOTALL)
                if where_match:
                    where_start = where_match.start()
                    next_clause = len(query)
                    for pattern in [r'\s+GROUP\s+BY', r'\s+ORDER\s+BY', r'\s+HAVING', r';']:
                        next_match = re.search(pattern, query[where_match.end():], re.IGNORECASE)
                        if next_match:
                            next_clause = min(next_clause, where_match.end() + next_match.start())
                    where_end = next_clause if next_clause < len(query) else len(query)
                    if where_start <= match_pos < where_end:
                        clause_start = where_start
                        clause_end = where_end
                        clause_type = "WHERE"
            
            # Check HAVING
            if not clause_type:
                having_pattern = r'HAVING\s+([^;]+?)(?:\s+ORDER\s+BY|;|$)'
                having_match = re.search(having_pattern, query, re.IGNORECASE | re.DOTALL)
                if having_match:
                    having_start = having_match.start()
                    next_clause = len(query)
                    for pattern in [r'\s+ORDER\s+BY', r';']:
                        next_match = re.search(pattern, query[having_match.end():], re.IGNORECASE)
                        if next_match:
                            next_clause = min(next_clause, having_match.end() + next_match.start())
                    having_end = next_clause if next_clause < len(query) else len(query)
                    if having_start <= match_pos < having_end:
                        clause_start = having_start
                        clause_end = having_end
                        clause_type = "HAVING"
            
            # Check SELECT (default)
            if not clause_type:
                select_match = re.search(r'SELECT\s+([^;]+?)(?:\s+FROM)', query, re.IGNORECASE | re.DOTALL)
                if select_match:
                    clause_start = select_match.start()
                    from_match = re.search(r'\s+FROM\s+', query[select_match.end():], re.IGNORECASE)
                    if from_match:
                        clause_end = select_match.end() + from_match.start()
                    else:
                        clause_end = select_match.end()
                    clause_type = "SELECT"
            
            # Get the clause context (only within this specific clause)
            clause_context = query[clause_start:clause_end]
            
            # Check if other merged columns appear in THIS SAME CLAUSE
            other_cols = [c for c in original_cols if c != col]
            other_cols_in_same_clause = False
            
            for other_col in other_cols:
                other_pattern = rf'`?{re.escape(other_col)}`?'
                if re.search(other_pattern, clause_context, re.IGNORECASE):
                    other_cols_in_same_clause = True
                    print(f"  '{col}' in {clause_type}: Found '{other_col}' in same clause → TOGETHER")
                    break
            
            # If this column appears without other merged columns in the same clause, it's alone
            if not other_cols_in_same_clause:
                appears_alone = True
                print(f"  '{col}' in {clause_type}: No other merged columns in same clause → ALONE")
                break
        
        if not appears_alone:
            print(f"  '{col}': All occurrences have other merged columns in same clause → TOGETHER")
        
        columns_appearing_alone[col] = appears_alone
    
    # Step 3: Show what extraction SQL would be generated
    print(f"\n{'─'*80}")
    print("STEP 3: Extraction SQL for columns appearing alone")
    print(f"{'─'*80}")
    
    extraction_sqls = {}
    for idx, col in enumerate(original_cols):
        if columns_appearing_alone[col]:
            if idx == 0:
                extraction = f"substr(`{new_col}`, 1, instr(`{new_col}`, ',') - 1)"
            elif idx == 1:
                extraction = f"substr(`{new_col}`, instr(`{new_col}`, ',') + 2)"
            elif idx == 2:
                extraction = f"substr(`{new_col}`, instr(substr(`{new_col}`, instr(`{new_col}`, ',') + 1), ',') + instr(`{new_col}`, ',') + 2)"
            else:
                extraction = f"[Complex extraction for index {idx}]"
            extraction_sqls[col] = extraction
            print(f"  '{col}' (index {idx}): {extraction}")
            print(f"    Note: If original column was numeric, this will be wrapped in CAST(... AS TYPE)")
    
    # Step 4: Show the transformation step-by-step
    print(f"\n{'─'*80}")
    print("STEP 4: Query transformation step-by-step")
    print(f"{'─'*80}")
    
    current_query = query
    
    # Step 4a: Replace columns appearing alone with extraction
    print(f"\n  4a. Replace columns appearing alone with extraction SQL:")
    for col in original_cols:
        if columns_appearing_alone[col]:
            escaped_col = re.escape(col)
            extraction_sql = extraction_sqls[col]
            
            # Show what gets replaced
            pattern = rf'`{escaped_col}`'
            matches = list(re.finditer(pattern, current_query, re.IGNORECASE))
            for match in matches:
                before = current_query[max(0, match.start()-30):match.end()+30]
                print(f"      Replacing '{col}' in: ...{before}...")
                print(f"      With: {extraction_sql}")
            
            # Actually replace (for demonstration)
            current_query = re.sub(rf'`{escaped_col}`', extraction_sql, current_query, flags=re.IGNORECASE)
            current_query = re.sub(rf'\b{escaped_col}\b', extraction_sql.replace('`', ''), current_query, flags=re.IGNORECASE)
    
    # Step 4b: Handle SELECT clause - deduplicate if multiple merged columns appear together
    print(f"\n  4b. Handle SELECT clause (deduplicate merged columns appearing together):")
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', current_query, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        select_items = [item.strip() for item in select_clause.split(',')]
        
        # Check which items are merged columns (that appear together, not extracted)
        merged_together_items = []
        remaining_items = []
        for item in select_items:
            item_clean = re.sub(r'\s+AS\s+[^,\s]+', '', item, flags=re.IGNORECASE).strip()
            # Check if this is a merged column that appears together (not extracted)
            is_together_col = any(
                (item_clean == f'`{col}`' or item_clean == col) and not columns_appearing_alone.get(col, False)
                for col in original_cols
            )
            if is_together_col:
                merged_together_items.append(item)
            else:
                remaining_items.append(item)
        
        if merged_together_items:
            print(f"      Found {len(merged_together_items)} merged column(s) appearing together in SELECT")
            print(f"      Replacing with single merged column: `{new_col}`")
            # Build new SELECT clause
            new_select_items = remaining_items + [f'`{new_col}`']
            new_select_clause = ', '.join(new_select_items)
            current_query = re.sub(
                r'SELECT\s+' + re.escape(select_clause) + r'\s+FROM',
                f'SELECT {new_select_clause} FROM',
                current_query,
                flags=re.IGNORECASE | re.DOTALL
            )
    
    # Step 4c: Replace remaining columns appearing together with merged column
    print(f"\n  4c. Replace remaining columns appearing together with merged column:")
    for col in original_cols:
        if not columns_appearing_alone.get(col, False):
            escaped_col = re.escape(col)
            # Check if it still exists (not already replaced)
            if f'`{col}`' in current_query or re.search(rf'\b{col}\b', current_query, re.IGNORECASE):
                print(f"      Replacing '{col}' → `{new_col}`")
                current_query = re.sub(rf'`{escaped_col}`', f'`{new_col}`', current_query, flags=re.IGNORECASE)
                current_query = re.sub(rf'\b{escaped_col}\b', new_col, current_query, flags=re.IGNORECASE)
    
    # Final result
    print(f"\n{'─'*80}")
    print("FINAL TRANSFORMED QUERY:")
    print(f"{'─'*80}")
    print(current_query.strip())
    
    print(f"\n{'─'*80}")
    print("SUMMARY")
    print(f"{'─'*80}")
    print(f"Columns merged: {', '.join(original_cols)} → {new_col}")
    alone_cols = [col for col, alone in columns_appearing_alone.items() if alone]
    together_cols = [col for col, alone in columns_appearing_alone.items() if not alone]
    print(f"Columns extracted (appear alone): {alone_cols if alone_cols else 'None'}")
    print(f"Columns replaced with merged column (appear together): {together_cols if together_cols else 'None'}")
    
    # Check for potential issues
    print(f"\n{'─'*80}")
    print("ANALYSIS NOTES")
    print(f"{'─'*80}")
    
    # Check if a column appears in ORDER BY/GROUP BY alone but was detected as together
    order_by_match = re.search(r'ORDER\s+BY\s+([^;]+)', query, re.IGNORECASE)
    group_by_match = re.search(r'GROUP\s+BY\s+([^;]+)', query, re.IGNORECASE)
    
    potential_issues = []
    for col in original_cols:
        # Check if this column appears in ORDER BY or GROUP BY
        in_order_by = order_by_match and (f'`{col}`' in order_by_match.group(1) or col in order_by_match.group(1))
        in_group_by = group_by_match and (f'`{col}`' in group_by_match.group(1) or col in group_by_match.group(1))
        
        if (in_order_by or in_group_by) and not columns_appearing_alone.get(col, False):
            # This column is in ORDER BY/GROUP BY but was detected as appearing together
            # This might be because the 100-char context window includes the SELECT clause
            clause = "ORDER BY" if in_order_by else "GROUP BY"
            potential_issues.append(f"⚠️  '{col}' appears in {clause} but was detected as TOGETHER (context window may include SELECT clause)")
    
    if potential_issues:
        print("Potential issues detected:")
        for issue in potential_issues:
            print(f"  {issue}")
        print("\n  Note: The 100-character context window may include columns from different SQL clauses.")
        print("  Ideal behavior: Columns in ORDER BY/GROUP BY should be extracted if they appear alone in that clause.")
    else:
        print("✅ No obvious issues detected")
    
    print("="*80 + "\n")
    
    return current_query

def test_mock_schema_changes():
    """Test schema changes on mock database and generate CSV."""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mock_db_path = os.path.join(project_root, "data", "mock_test.db")
    mock_jsonl_path = os.path.join(project_root, "data", "mock_test.tables.jsonl")
    csv_output_path = os.path.join(project_root, "data", "schema_changes_results.csv")
    
    # Operation types to test (each once)
    operation_types = ["add_column", "remove_column", "rename_column", "rename_table", "split_column", "merge_column"]
    
    print("="*60)
    print("BEFORE SCHEMA CHANGES:")
    print("="*60)
    
    # Show original tables
    conn = sqlite3.connect(mock_db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    original_tables = [row[0] for row in cur.fetchall()]
    print(f"\nOriginal tables in database: {len(original_tables)}")
    for table in original_tables:
        cur.execute(f"SELECT COUNT(*) FROM `{table}`")
        count = cur.fetchone()[0]
        print(f"  - {table}: {count} rows")
    conn.close()
    
    # Count original JSONL entries
    with open(mock_jsonl_path, 'r', encoding='utf-8') as f:
        original_jsonl_count = sum(1 for line in f if line.strip())
    print(f"\nOriginal entries in JSONL: {original_jsonl_count}")
    
    print("\n" + "="*60)
    print("APPLYING SCHEMA CHANGES (one of each operation type):")
    print("="*60)
    
    # Prepare CSV data
    csv_rows = []
    
    # Use first query for all operations
    test_query = test_queries[0]
    table_name = extract_table_name_from_query(test_query['sql'])
    table_id = table_name.replace('table_', '').replace('_', '-')
    original_table = get_table_from_jsonl(table_id, mock_jsonl_path)
    
    if not original_table:
        print(f"❌ Could not find table {table_id} in JSONL")
        return
    
    original_schema_with_data = get_schema_with_sample_data(original_table)
    
    # Apply each operation type once
    for op_type in operation_types:
        print(f"\n{'='*60}")
        print(f"Testing {op_type}")
        print(f"{'='*60}")
        
        # Temporarily modify apply_schema_change to only do this one operation
        # We'll need to patch the function or create a wrapper
        # For now, let's use a modified version that forces one specific operation
        
        # Get original schema
        original_schema = convert_table_to_schema(json.dumps(original_table))
        
        # Apply the single schema change
        result = apply_single_schema_change(
            db_path=mock_db_path,
            original_schema=original_schema,
            original_table=original_table,
            nl=test_query['nl'],
            query=test_query['sql'],
            change_type=op_type,
            tables_jsonl_path=mock_jsonl_path
        )
        
        if result:
            # Get the new table from JSONL using the modified_table_id
            modified_table_id = result.get('db_id')
            if modified_table_id:
                # The modified table should be in JSONL with this ID
                modified_table = get_table_from_jsonl(modified_table_id, mock_jsonl_path)
                
                if modified_table:
                    new_schema_with_data = get_schema_with_sample_data(modified_table)
                    new_sql = result.get('SQL', '')
                    
                    csv_rows.append({
                        'operation_type': op_type,
                        'nl': test_query['nl'],
                        'sql': test_query['sql'].strip(),
                        'schema': original_schema_with_data,
                        'new_schema': new_schema_with_data,
                        'new_sql': new_sql.strip()
                    })
                    print(f"✅ {op_type} completed successfully")
                else:
                    print(f"⚠️  {op_type} completed but couldn't find modified table in JSONL")
            else:
                print(f"⚠️  {op_type} completed but no table ID returned")
        else:
            print(f"❌ {op_type} failed")
    
    # Write CSV
    if csv_rows:
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['operation_type', 'nl', 'sql', 'schema', 'new_schema', 'new_sql']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n✅ Generated CSV with {len(csv_rows)} rows: {csv_output_path}")
    else:
        print("\n❌ No results to write to CSV")
    
    print(f"\n✅ Total operations completed: {len(csv_rows)}/{len(operation_types)}")
    
    print("\n" + "="*60)
    print("AFTER SCHEMA CHANGES:")
    print("="*60)
    
    # Show all tables (including modified ones)
    conn = sqlite3.connect(mock_db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    all_tables = [row[0] for row in cur.fetchall()]
    print(f"\nAll tables in database: {len(all_tables)}")
    
    original_count = len([t for t in all_tables if not t.startswith('table_m_')])
    modified_count = len([t for t in all_tables if t.startswith('table_m_')])
    
    print(f"  - Original tables: {original_count}")
    print(f"  - Modified tables (with 'm_' prefix): {modified_count}")
    conn.close()
    
    # Count JSONL entries
    with open(mock_jsonl_path, 'r', encoding='utf-8') as f:
        all_jsonl_count = sum(1 for line in f if line.strip())
    print(f"\nAll entries in JSONL: {all_jsonl_count}")
    print(f"  - Original entries: {original_jsonl_count}")
    print(f"  - New modified entries: {all_jsonl_count - original_jsonl_count}")
    
    print(f"\n📊 CSV file saved to: {csv_output_path}")

def test_merge_analysis():
    """Test detailed analysis of merge_column_in_query on edge cases."""
    print("\n" + "="*80)
    print("TESTING MERGE COLUMN IN QUERY ANALYSIS")
    print("="*80)
    
    # Example 1: Player and No. merged, No. appears alone in ORDER BY
    print("\n" + "="*80)
    print("EXAMPLE 1: Player and No. merged, No. in ORDER BY")
    print("="*80)
    query1 = test_merge_edge_cases[0]['sql']
    analyze_merge_column_in_query(
        query=query1,
        original_cols=["Player", "No."],
        new_col="Player_ID"
    )
    
    # Example 2: Player and No. merged, Points scored separate
    print("\n" + "="*80)
    print("EXAMPLE 2: Player and No. merged, Points scored used separately")
    print("="*80)
    query2 = test_merge_edge_cases[1]['sql']
    analyze_merge_column_in_query(
        query=query2,
        original_cols=["Player", "No."],
        new_col="Player_ID"
    )

def test_all_merge_edge_cases():
    """Test all merge edge cases by actually running merge_column_in_query on them."""
    # apply_single_schema_change and convert_table_to_schema are already imported above
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mock_db_path = os.path.join(project_root, "data", "mock_test.db")
    mock_jsonl_path = os.path.join(project_root, "data", "mock_test.tables.jsonl")
    
    # Get the table that has Player and No. columns (table_1_10015132_14)
    # This should be the "Players-in-Toronto" table
    original_table = None
    with open(mock_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                table = json.loads(line)
                # Check if this table has Player and No. columns
                if 'Player' in table.get('header', []) and 'No.' in table.get('header', []):
                    original_table = table
                    break
    
    if not original_table:
        print("❌ Could not find table with Player and No. columns")
        return
    
    print("\n" + "="*80)
    print("TESTING ALL MERGE EDGE CASES")
    print("="*80)
    print(f"Using table: {original_table.get('id', 'unknown')}")
    print(f"Columns: {original_table.get('header', [])}")
    
    results = []
    
    for i, test_case in enumerate(test_merge_edge_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}/{len(test_merge_edge_cases)}")
        print(f"{'='*80}")
        print(f"NL: {test_case['nl']}")
        print(f"SQL: {test_case['sql'].strip()}")
        
        # Get original schema
        original_schema = convert_table_to_schema(json.dumps(original_table))
        
        # Apply merge_column operation
        result = apply_single_schema_change(
            db_path=mock_db_path,
            original_schema=original_schema,
            original_table=original_table.copy(),  # Deep copy to avoid modifying original
            nl=test_case['nl'],
            query=test_case['sql'],
            change_type="merge_column",
            tables_jsonl_path=mock_jsonl_path
        )
        
        if result:
            new_sql = result.get('SQL', '')
            modified_table_id = result.get('db_id')
            
            print(f"\n✅ Merge completed")
            print(f"   Modified table ID: {modified_table_id}")
            print(f"   Original SQL: {test_case['sql'].strip()}")
            print(f"   New SQL: {new_sql.strip()}")
            
            # Check if query became unanswerable or had clauses removed
            if "unanswerable" in new_sql.lower():
                print(f"   ⚠️  Query marked as unanswerable")
            elif "not supported" in new_sql.lower():
                print(f"   ⚠️  Some clauses were removed (not supported with merged column)")
            
            results.append({
                'test_num': i,
                'nl': test_case['nl'],
                'original_sql': test_case['sql'].strip(),
                'new_sql': new_sql.strip(),
                'success': True,
                'modified_table_id': modified_table_id
            })
        else:
            print(f"\n❌ Merge failed")
            results.append({
                'test_num': i,
                'nl': test_case['nl'],
                'original_sql': test_case['sql'].strip(),
                'new_sql': 'FAILED',
                'success': False,
                'modified_table_id': None
            })
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total test cases: {len(test_merge_edge_cases)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"\n{status} Test {result['test_num']}: {result['nl'][:60]}...")
        print(f"   Original: {result['original_sql'][:80]}...")
        print(f"   New:      {result['new_sql'][:80]}...")
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--analyze":
            test_merge_analysis()
        elif sys.argv[1] == "--test-merge":
            test_all_merge_edge_cases()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python test_mock_schema_changes.py [--analyze|--test-merge]")
    else:
        test_mock_schema_changes()

