"""
Very verbose test script for schema changes.
Tests each change type and shows detailed information about:
- Original NL, SQL, and result
- The change applied
- New schema and new SQL
- New table name in JSONL and DB
- Result of running SQL on new DB
- Whether results are equal
"""
import os
import sys
import json
import sqlite3
import re

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multilingual_evoschema import apply_schema_change
from multilingual_evoschema.schema_change import convert_table_to_schema
from multilingual_evoschema.__init__ import extract_table_name_from_query, get_table_from_jsonl, execute_query_on_table

test_queries = [
    {
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
]


def execute_query_with_conversion(query: str, table: dict, db_path: str, table_name_in_db: str):
    """Execute SQL query with column name conversion."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Convert column names to placeholders
    conversion = {table["header"][i]: f"col{i}" for i in range(len(table["header"]))}
    test_query = query
    for original, replacement in sorted(conversion.items(), key=lambda x: len(x[0]), reverse=True):
        test_query = re.sub(rf'`{re.escape(original)}`', replacement, test_query, flags=re.IGNORECASE)
    
    # Replace table name in query
    test_query = re.sub(r'FROM\s+`?[^`\s,]+`?', f'FROM `{table_name_in_db}`', test_query, flags=re.IGNORECASE)
    test_query = re.sub(r'JOIN\s+`?[^`\s,]+`?', f'JOIN `{table_name_in_db}`', test_query, flags=re.IGNORECASE)
    
    try:
        cur.execute(test_query)
        result = cur.fetchall()
        conn.close()
        return result, None, test_query
    except Exception as e:
        conn.close()
        return None, str(e), test_query


def normalize_results(results):
    """Normalize query results for comparison (case-insensitive for strings)."""
    if results is None:
        return []
    
    def normalize_value(val):
        """Normalize a value for comparison (case-insensitive for strings)."""
        if val is None:
            return None
        # Handle bytes
        if isinstance(val, bytes):
            val = val.decode('utf-8', errors='ignore')
        # Convert strings to lowercase
        if isinstance(val, str):
            return val.lower()
        # For numbers, keep as is (don't convert to string)
        return val
    
    def normalize_row(row):
        """Normalize a row for comparison."""
        return tuple(normalize_value(x) for x in row)
    
    return sorted([normalize_row(row) for row in results])


def test_schema_changes_verbose():
    """Very verbose test of schema changes."""
    
    # Setup paths
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    db_path = os.path.join(project_root, "data/train.db")
    tables_jsonl_path = os.path.join(project_root, "data/train.tables.jsonl")
    
    print("=" * 100)
    print("VERBOSE SCHEMA CHANGE TEST")
    print("=" * 100)
    
    all_test_results = []
    
    for test_idx, test_case in enumerate(test_queries, 1):
        print(f"\n{'=' * 100}")
        print(f"TEST CASE {test_idx}/{len(test_queries)}")
        print(f"{'=' * 100}")
        
        nl = test_case['nl']
        sql = test_case['sql'].strip()
        
        print(f"\n📝 ORIGINAL INPUT:")
        print(f"   Natural Language: {nl}")
        print(f"   SQL Query: {sql}")
        
        # Extract table name
        table_name = extract_table_name_from_query(sql)
        if not table_name:
            print(f"❌ Could not extract table name from query")
            continue
        
        table_id = table_name.replace('table_', '').replace('_', '-')
        print(f"   Table Name: {table_name}")
        print(f"   Table ID: {table_id}")
        
        # Get original table
        original_table = get_table_from_jsonl(table_id, tables_jsonl_path)
        if not original_table:
            print(f"❌ Could not find table '{table_id}' in JSONL")
            continue
        
        print(f"\n📊 ORIGINAL TABLE INFO:")
        print(f"   Table ID in JSONL: {original_table.get('id')}")
        print(f"   Headers: {original_table.get('header', [])}")
        print(f"   Types: {original_table.get('types', [])}")
        print(f"   Number of Rows: {len(original_table.get('rows', []))}")
        
        # Convert to schema
        original_schema = convert_table_to_schema(json.dumps(original_table))
        print(f"\n📋 ORIGINAL SCHEMA:")
        print(original_schema)
        
        # Execute original query
        original_table_name_db = f"table_{table_id.replace('-', '_')}"
        print(f"\n🔍 EXECUTING ORIGINAL QUERY:")
        print(f"   Database Table Name: {original_table_name_db}")
        original_result, original_error, original_executed_query = execute_query_with_conversion(
            sql, original_table, db_path, original_table_name_db
        )
        
        if original_error:
            print(f"   ❌ Error: {original_error}")
            print(f"   Executed Query: {original_executed_query}")
            continue
        
        print(f"   ✅ Query executed successfully")
        print(f"   Executed Query: {original_executed_query}")
        print(f"   Result Rows: {len(original_result) if original_result else 0}")
        print(f"   Result: {original_result}")
        
        # Apply schema changes
        print(f"\n🔄 APPLYING SCHEMA CHANGES...")
        results = apply_schema_change(
            db_path=db_path,
            original_schema=original_schema,
            nl=nl,
            query=sql,
            tables_jsonl_path=tables_jsonl_path
        )
        
        if not results:
            print(f"\n⚠️  No results generated (all changes may have been skipped or rejected)")
            all_test_results.append({
                "test_case": test_idx,
                "table_id": table_id,
                "results_count": 0,
                "results": []
            })
            continue
        
        print(f"\n✅ Generated {len(results)} modified tables")
        
        test_case_results = []
        
        # Verify each result
        for result_idx, result in enumerate(results, 1):
            print(f"\n{'─' * 100}")
            print(f"RESULT {result_idx}/{len(results)}")
            print(f"{'─' * 100}")
            
            modified_table_id = result.get('db_id')
            modified_sql = result.get('SQL')
            modified_nl = result.get('question')
            change_type = result.get('change_type', 'unknown')
            change_info = result.get('change_info', {})
            original_query = result.get('original_query', sql)
            
            print(f"\n🔄 SCHEMA CHANGE DETAILS:")
            print(f"   Change Type: {change_type}")
            
            # Show detailed change information
            if change_type == "add_column":
                if "columns" in change_info:
                    columns_to_add = change_info["columns"] if isinstance(change_info["columns"], list) else [change_info["columns"]]
                    print(f"   Added Columns:")
                    for col_info in columns_to_add:
                        if isinstance(col_info, dict):
                            col_name = col_info.get("column_name", "")
                            col_type = col_info.get("sql_type", "")
                            values = col_info.get("values", [])
                            print(f"     - {col_name} ({col_type})")
                            if values:
                                print(f"       Sample values: {values[:3]}{'...' if len(values) > 3 else ''}")
            
            elif change_type == "remove_column":
                removed_col = change_info.get("removed_column", "")
                print(f"   Removed Column: {removed_col}")
            
            elif change_type == "rename_column":
                if "renames" in change_info:
                    renames = change_info["renames"] if isinstance(change_info["renames"], list) else [change_info["renames"]]
                    print(f"   Renamed Columns:")
                    for rename in renames:
                        if isinstance(rename, dict):
                            old_name = rename.get("old_name", "")
                            new_name = rename.get("new_name", "")
                            print(f"     - {old_name} → {new_name}")
            
            elif change_type == "rename_table":
                old_name = change_info.get("old_name", "")
                new_name = change_info.get("new_name", "")
                print(f"   Renamed Table: {old_name} → {new_name}")
            
            elif change_type == "split_column":
                original_col = change_info.get("original_column", "")
                new_cols = change_info.get("new_columns", [])
                print(f"   Split Column: {original_col}")
                print(f"   Into:")
                for new_col in new_cols:
                    if isinstance(new_col, dict):
                        col_name = new_col.get("column_name", "")
                        col_type = new_col.get("sql_type", "")
                        print(f"     - {col_name} ({col_type})")
            
            elif change_type == "merge_column":
                original_cols = change_info.get("original_columns", [])
                new_col = change_info.get("new_column", "")
                print(f"   Merged Columns: {', '.join(original_cols)}")
                print(f"   Into: {new_col}")
            
            print(f"\n📝 MODIFIED OUTPUT:")
            print(f"   Natural Language: {modified_nl}")
            print(f"   Modified Table ID: {modified_table_id}")
            
            # Show SQL query changes
            if original_query != modified_sql:
                print(f"\n🔀 SQL QUERY CHANGES:")
                print(f"   Original SQL:")
                print(f"   {original_query}")
                print(f"   Modified SQL:")
                print(f"   {modified_sql}")
            else:
                print(f"\n📝 SQL QUERY (unchanged):")
                print(f"   {modified_sql}")
            
            # Find the modified table in JSONL
            modified_table = None
            with open(tables_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            table = json.loads(line)
                            if table.get('id') == modified_table_id:
                                modified_table = table
                                break
                        except json.JSONDecodeError:
                            continue
            
            if not modified_table:
                print(f"   ❌ Could not find modified table in JSONL")
                continue
            
            print(f"\n📊 MODIFIED TABLE INFO:")
            print(f"   Table ID in JSONL: {modified_table.get('id')}")
            print(f"   Headers: {modified_table.get('header', [])}")
            print(f"   Types: {modified_table.get('types', [])}")
            print(f"   Number of Rows: {len(modified_table.get('rows', []))}")
            
            # Get modified schema
            modified_schema = convert_table_to_schema(json.dumps(modified_table))
            print(f"\n📋 MODIFIED SCHEMA:")
            print(modified_schema)
            
            # Find table name in database
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            
            if modified_table_id.startswith('m_'):
                id_parts = modified_table_id.split('_', 3)
                if len(id_parts) >= 4:
                    db_table_name = f"table_m_{id_parts[1]}_{id_parts[2]}_{id_parts[3].replace('-', '_')}"
                else:
                    db_table_name = f"table_{modified_table_id.replace('-', '_')}"
            else:
                db_table_name = f"table_{modified_table_id.replace('-', '_')}"
            
            # Verify table exists
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (db_table_name,))
            if not cur.fetchone():
                # Search for it
                if modified_table_id.startswith('m_'):
                    id_parts = modified_table_id.split('_', 3)
                    if len(id_parts) >= 3:
                        search_pattern = f"table_m_{id_parts[1]}_{id_parts[2]}_%"
                        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (search_pattern,))
                        result = cur.fetchone()
                        if result:
                            db_table_name = result[0]
            
            conn.close()
            
            print(f"\n💾 DATABASE INFO:")
            print(f"   Table Name in DB: {db_table_name}")
            
            # Execute modified query
            print(f"\n🔍 EXECUTING MODIFIED QUERY:")
            modified_result, modified_error, modified_executed_query = execute_query_with_conversion(
                modified_sql, modified_table, db_path, db_table_name
            )
            
            if modified_error:
                print(f"   ❌ Error: {modified_error}")
                print(f"   Executed Query: {modified_executed_query}")
                test_case_results.append({
                    "result_idx": result_idx,
                    "modified_table_id": modified_table_id,
                    "db_table_name": db_table_name,
                    "success": False,
                    "error": modified_error
                })
                continue
            
            print(f"   ✅ Query executed successfully")
            print(f"   Executed Query: {modified_executed_query}")
            print(f"   Result Rows: {len(modified_result) if modified_result else 0}")
            print(f"   Result: {modified_result}")
            
            # Compare results
            print(f"\n🔍 COMPARING RESULTS:")
            original_normalized = normalize_results(original_result)
            modified_normalized = normalize_results(modified_result)
            
            print(f"   Original Result (normalized): {original_normalized}")
            print(f"   Modified Result (normalized): {modified_normalized}")
            
            results_match = (original_normalized == modified_normalized)
            
            if results_match:
                print(f"   ✅ RESULTS MATCH!")
            else:
                print(f"   ❌ RESULTS DO NOT MATCH!")
                print(f"   Original: {original_result}")
                print(f"   Modified: {modified_result}")
            
            test_case_results.append({
                "result_idx": result_idx,
                "modified_table_id": modified_table_id,
                "db_table_name": db_table_name,
                "modified_sql": modified_sql,
                "original_result": original_result,
                "modified_result": modified_result,
                "results_match": results_match,
                "success": True
            })
        
        all_test_results.append({
            "test_case": test_idx,
            "table_id": table_id,
            "results_count": len(results),
            "results": test_case_results
        })
    
    # Summary
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    
    total_results = sum(r["results_count"] for r in all_test_results)
    total_matching = sum(
        sum(1 for res in r["results"] if res.get("results_match", False))
        for r in all_test_results
    )
    
    print(f"Total test cases: {len(test_queries)}")
    print(f"Total modified tables created: {total_results}")
    print(f"Total results matching original: {total_matching}/{total_results}")
    
    for test_result in all_test_results:
        print(f"\n  Test Case {test_result['test_case']} ({test_result['table_id']}):")
        print(f"    Results: {test_result['results_count']}")
        for res in test_result['results']:
            if res.get('success'):
                match_status = "✅ MATCH" if res.get('results_match') else "❌ NO MATCH"
                print(f"      Result {res['result_idx']}: {match_status} (Table: {res['modified_table_id']})")
            else:
                print(f"      Result {res['result_idx']}: ❌ FAILED ({res.get('error', 'Unknown error')})")
    
    print(f"\n✅ Test completed!")
    return all_test_results


if __name__ == "__main__":
    test_schema_changes_verbose()
