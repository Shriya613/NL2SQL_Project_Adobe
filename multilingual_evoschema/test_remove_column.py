"""
Super verbose test script for remove_column behavior.
Tests remove_column on all mock queries, showing detailed information at each step.
"""
import os
import sys
import sqlite3
import json

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multilingual_evoschema import apply_single_schema_change
from multilingual_evoschema.schema_change import convert_table_to_schema, format_sample_data, remove_column
import importlib.util

# Import internal functions directly
spec = importlib.util.spec_from_file_location("multilingual_evoschema_module", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                 "multilingual_evoschema", "__init__.py"))
evoschema_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evoschema_module)

# Get functions we need
get_table_from_jsonl = evoschema_module.get_table_from_jsonl
extract_table_name_from_query = evoschema_module.extract_table_name_from_query

# Test queries from mock database
test_queries = [
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
    {
        "sql": """
            SELECT `Player`, `No.` FROM `table_1_10015132_14`
            GROUP BY `No.`;
        """,
        "nl": "List the players and their numbers grouped by their jersey numbers."
    },
    {
        "sql": """
            SELECT `Player`, `No.` FROM `table_1_10015132_14`
            WHERE `No.` > 10;
        """,
        "nl": "List the players and their numbers who have a jersey number greater than 10."
    },
    {
        "sql": """
            SELECT `Player`, `No.` FROM `table_1_10015132_14`
            WHERE `No.` > (SELECT AVG(`No.`) FROM `table_1_10015132_14`);
        """,
        "nl": "List the players and their numbers who have a jersey number greater than the average jersey number."
    },
    {
        "sql": """
            SELECT `Nationality` FROM `table_1_10015132_14`;
        """,
        "nl": "List all the nationalities of the players."
    },
    {
        "sql": """
            SELECT `Position` FROM `table_1_10015132_14`
            WHERE `Position` = 'Center';
        """,
        "nl": "List all players who play the Center position."
    }
]

def get_schema_with_sample_data(table_dict):
    """Get schema string with first 2 rows of sample data."""
    schema = convert_table_to_schema(json.dumps(table_dict))
    sample_data = format_sample_data(table_dict, max_rows=2)
    return schema + sample_data

def test_remove_column_verbose():
    """Super verbose test of remove_column behavior on all test queries."""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mock_db_path = os.path.join(project_root, "data", "mock_test.db")
    mock_jsonl_path = os.path.join(project_root, "data", "mock_test.tables.jsonl")
    
    # Get the table that has Player and No. columns (table_1_10015132_14)
    original_table = None
    with open(mock_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                table = json.loads(line)
                # Check if this table has Player and No. columns (or other columns from test queries)
                headers = table.get('header', [])
                if 'Player' in headers or 'Nationality' in headers or 'Position' in headers:
                    original_table = table
                    break
    
    if not original_table:
        print("❌ Could not find suitable table in mock database")
        return
    
    print("="*100)
    print("REMOVE COLUMN VERBOSE TEST")
    print("="*100)
    print(f"\n📊 Using table: {original_table.get('id', 'unknown')}")
    print(f"📋 Table columns: {original_table.get('header', [])}")
    print(f"📏 Table has {len(original_table.get('rows', []))} rows")
    
    # Show first 2 rows
    print(f"\n📄 First 2 rows of original table:")
    headers = original_table.get('header', [])
    rows = original_table.get('rows', [])
    print(f"   Headers: {headers}")
    for i, row in enumerate(rows[:2], 1):
        print(f"   Row {i}: {row}")
    
    results = []
    
    for test_num, test_case in enumerate(test_queries, 1):
        print(f"\n{'='*100}")
        print(f"TEST CASE {test_num}/{len(test_queries)}")
        print(f"{'='*100}")
        print(f"\n📝 Natural Language Question:")
        print(f"   {test_case['nl']}")
        print(f"\n🔍 Original SQL Query:")
        print(f"   {test_case['sql'].strip()}")
        
        # Get original schema
        original_schema = convert_table_to_schema(json.dumps(original_table))
        original_schema_with_data = get_schema_with_sample_data(original_table)
        
        print(f"\n📋 Original Schema (with sample data):")
        print(f"   {original_schema_with_data.replace(chr(10), chr(10) + '   ')}")
        
        # Extract columns used in the query
        import re
        query = test_case['sql']
        columns_in_query = []
        for col in original_table.get('header', []):
            escaped_col = re.escape(col)
            if re.search(rf'`?{escaped_col}`?', query, re.IGNORECASE):
                columns_in_query.append(col)
        
        print(f"\n🔍 Columns used in SQL query:")
        print(f"   {columns_in_query}")
        
        # Show all columns and explain the process
        all_columns = original_table.get('header', [])
        print(f"\n📋 All columns in table: {all_columns}")
        print(f"   → One column will be randomly selected for removal")
        print(f"   → If selected column is in query: LLM will be called to check if NL can still be answered")
        print(f"   → If selected column is NOT in query: Column removed manually, query unchanged")
        
        # Apply remove_column operation
        print(f"\n{'─'*100}")
        print("APPLYING REMOVE_COLUMN OPERATION")
        print(f"{'─'*100}")
        print(f"   (Full LLM prompt will be printed below if column is in query)")
        print(f"{'─'*100}")
        
        result = apply_single_schema_change(
            db_path=mock_db_path,
            original_schema=original_schema,
            original_table=original_table.copy(),  # Deep copy to avoid modifying original
            nl=test_case['nl'],
            query=test_case['sql'],
            change_type="remove_column",
            tables_jsonl_path=mock_jsonl_path
        )
        
        if result:
            new_sql = result.get('SQL', '')
            modified_table_id = result.get('db_id')
            
            print(f"\n✅ Remove column operation completed")
            print(f"\n📊 Results:")
            print(f"   Modified table ID: {modified_table_id}")
            
            # Get the modified table from JSONL
            if modified_table_id:
                modified_table = get_table_from_jsonl(modified_table_id, mock_jsonl_path)
                
                if modified_table:
                    new_schema_with_data = get_schema_with_sample_data(modified_table)
                    
                    print(f"\n📋 New Schema (with sample data):")
                    print(f"   {new_schema_with_data.replace(chr(10), chr(10) + '   ')}")
                    
                    print(f"\n🔍 Original SQL Query:")
                    print(f"   {test_case['sql'].strip()}")
                    
                    print(f"\n🔍 New SQL Query:")
                    print(f"   {new_sql.strip()}")
                    
                    # Check if query became unanswerable
                    if "unanswerable" in new_sql.lower():
                        print(f"\n⚠️  Query marked as UNANSWERABLE")
                        print(f"   Reason: The natural language question cannot be answered without the removed column")
                    elif new_sql.strip() == test_case['sql'].strip():
                        print(f"\n✅ Query unchanged")
                        print(f"   Reason: The removed column was not used in the SQL query")
                    else:
                        print(f"\n✅ Query was modified")
                        print(f"   Reason: The removed column was in the query, but the question can still be answered")
                    
                    # Show what changed
                    print(f"\n📊 Comparison:")
                    print(f"   Original columns: {original_table.get('header', [])}")
                    print(f"   New columns: {modified_table.get('header', [])}")
                    
                    removed_cols = set(original_table.get('header', [])) - set(modified_table.get('header', []))
                    if removed_cols:
                        print(f"   Removed column(s): {list(removed_cols)}")
                    
                    # Show first 2 rows of modified table
                    print(f"\n📄 First 2 rows of modified table:")
                    new_headers = modified_table.get('header', [])
                    new_rows = modified_table.get('rows', [])
                    print(f"   Headers: {new_headers}")
                    for i, row in enumerate(new_rows[:2], 1):
                        print(f"   Row {i}: {row}")
                    
                    results.append({
                        'test_num': test_num,
                        'nl': test_case['nl'],
                        'original_sql': test_case['sql'].strip(),
                        'new_sql': new_sql.strip(),
                        'removed_columns': list(removed_cols),
                        'success': True,
                        'unanswerable': "unanswerable" in new_sql.lower(),
                        'query_changed': new_sql.strip() != test_case['sql'].strip(),
                        'modified_table_id': modified_table_id
                    })
                else:
                    print(f"\n⚠️  Could not find modified table in JSONL")
                    results.append({
                        'test_num': test_num,
                        'nl': test_case['nl'],
                        'original_sql': test_case['sql'].strip(),
                        'new_sql': 'TABLE_NOT_FOUND',
                        'success': False
                    })
            else:
                print(f"\n⚠️  No table ID returned")
                results.append({
                    'test_num': test_num,
                    'nl': test_case['nl'],
                    'original_sql': test_case['sql'].strip(),
                    'new_sql': 'NO_TABLE_ID',
                    'success': False
                })
        else:
            print(f"\n❌ Remove column operation failed")
            results.append({
                'test_num': test_num,
                'nl': test_case['nl'],
                'original_sql': test_case['sql'].strip(),
                'new_sql': 'FAILED',
                'success': False
            })
    
    # Print summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"Total test cases: {len(test_queries)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"Queries marked unanswerable: {sum(1 for r in results if r.get('unanswerable', False))}")
    print(f"Queries that changed: {sum(1 for r in results if r.get('query_changed', False))}")
    print(f"Queries unchanged: {sum(1 for r in results if r.get('success') and not r.get('query_changed', False) and not r.get('unanswerable', False))}")
    
    print(f"\n{'='*100}")
    print("DETAILED RESULTS")
    print(f"{'='*100}")
    for result in results:
        status = "✅" if result['success'] else "❌"
        unanswerable = "⚠️ UNANSWERABLE" if result.get('unanswerable') else ""
        changed = "🔄 CHANGED" if result.get('query_changed') else "➡️ UNCHANGED"
        
        print(f"\n{status} Test {result['test_num']}: {result['nl'][:60]}...")
        if result.get('removed_columns'):
            print(f"   Removed column(s): {result['removed_columns']}")
        print(f"   Original SQL: {result['original_sql'][:80]}...")
        print(f"   New SQL: {result['new_sql'][:80]}...")
        print(f"   Status: {unanswerable} {changed}")
    
    return results

if __name__ == "__main__":
    test_remove_column_verbose()

