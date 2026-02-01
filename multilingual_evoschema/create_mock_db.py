"""
Create a mock database and JSONL file with 3 mock tables for testing schema changes.
"""
import json
import sqlite3
import os

# Mock tables based on test_schema_changes.py queries
mock_tables = [
    {
        "id": "1-10015132-14",
        "name": "table_10015132_14",
        "header": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team", "Points scored"],
        "types": ["text", "real", "text", "text", "text", "text", "real"],
        "rows": [
            ["Patrick O'Bryant", 13, "United States", "Center", "2009-10", "Bradley", 10],
            ["Jermaine O'Neal", 6, "United States", "Forward-Center", "2008-09", "Eau Claire High School", 6],
            ["Dan O'Sullivan", 45, "United States", "Center", "1995-96", "Fordham", 7],
            ["Charles Oakley", 34, "United States", "Forward", "1998-2001", "Virginia Union", 12],
            ["Hakeem Olajuwon", 34, "Nigeria / United States", "Center", "2001-02", "Houston", 8]
        ],
        "page_title": "Toronto Raptors all-time roster",
        "section_title": "O",
        "caption": "O"
    },
    {
        "id": "1-10812938-4",
        "name": "table_10812938_4",
        "header": ["Pick #", "CFL Team", "Player", "Position", "College"],
        "types": ["real", "text", "text", "text", "text"],
        "rows": [
            [26, "Edmonton Eskimos (via Hamilton)", "Andrew Brown", "LB", "Lafayette College"],
            [27, "Calgary Stampeders (via Winnipeg)", "Riley Clayton", "OL", "Manitoba"],
            [28, "Hamilton Tiger-Cats (via Ottawa)", "Chris Sutherland", "OL", "Saskatchewan"],
            [29, "Saskatchewan Roughriders", "Peter Hogarth", "OL", "McMaster"],
            [30, "Calgary Stampeders", "Gerald Commissiong", "RB", "Stanford"],
            [31, "Toronto Argonauts", "Obed Cétoute", "WR", "Central Michigan"],
            [32, "Montreal Alouettes (via BC)", "Ivan Birungi", "WR", "Acadia"],
            [33, "Montreal Alouettes", "Adrian Davis", "DL", "Marshall"]
        ],
        "page_title": "2006 CFL Draft",
        "section_title": "Round four",
        "caption": "Round four"
    },
    {
        "id": "1-10236830-4",
        "name": "table_10236830_4",
        "header": ["Nomination", "Actors Name", "Film Name", "Director", "Country"],
        "types": ["text", "text", "text", "text", "text"],
        "rows": [
            ["Best Actor in a Leading Role", "Yuriy Dubrovin", "Okraina", "Pyotr Lutsik", "Ukraine"],
            ["Best Actor in a Leading Role", "Zurab Begalishvili", "Zdes Rassvet", "Zaza Urushadze", "Georgia"],
            ["Best Actress in a Leading Role", "Galina Bokashevskaya", "Totalitarian Romance", "Vyacheslav Sorokin", "Russia"],
            ["Best Actor in a Supporting Role", "Vsevolod Shilovskiy", "Barhanov and his Bodyguard", "Valeriy Lanskoy", "Russia"],
            ["Best Actor in a Supporting Role", "Dragan Nikolić", "Barrel of Gunpowder", "Goran Paskaljevic", "Serbia"],
            ["Best Actress in a Supporting Role", "Zora Manojlovic", "Rane", "Srdjan Dragojevic", "Serbia"],
            ["Best Debut", "Agnieszka Włodarczyk", "Sara", "Maciej Ślesicki", "Poland"]
        ],
        "page_title": "Stozhary",
        "section_title": "Stozhary '99 Prize-Winners",
        "caption": "Stozhary '99 Prize-Winners"
    }
]

def create_mock_database_and_jsonl():
    """Create mock database and JSONL file with 3 tables."""
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mock_db_path = os.path.join(project_root, "data", "mock_test.db")
    mock_jsonl_path = os.path.join(project_root, "data", "mock_test.tables.jsonl")
    
    # Remove existing files if they exist
    if os.path.exists(mock_db_path):
        os.remove(mock_db_path)
    if os.path.exists(mock_jsonl_path):
        os.remove(mock_jsonl_path)
    
    # Create database connection
    conn = sqlite3.connect(mock_db_path)
    cur = conn.cursor()
    
    # Create JSONL file
    with open(mock_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for table in mock_tables:
            # Write to JSONL (with real column names)
            jsonl_file.write(json.dumps(table) + '\n')
            
            # Create table in database (with placeholder column names: col0, col1, etc.)
            table_name_in_db = f"table_{table['id'].replace('-', '_')}"
            
            # Build CREATE TABLE statement with placeholder columns
            col_defs = []
            for i, (col_name, col_type) in enumerate(zip(table['header'], table['types'])):
                # Convert type to SQLite type
                sql_type = col_type.upper()
                if sql_type == 'TEXT':
                    sql_type = 'TEXT'
                elif sql_type == 'REAL':
                    sql_type = 'REAL'
                elif sql_type == 'INTEGER':
                    sql_type = 'INTEGER'
                else:
                    sql_type = 'TEXT'
                
                col_defs.append(f"col{i} {sql_type}")
            
            create_sql = f"CREATE TABLE `{table_name_in_db}` ({', '.join(col_defs)})"
            cur.execute(create_sql)
            
            # Insert rows
            for row in table['rows']:
                placeholders = ','.join(['?' for _ in row])
                insert_sql = f"INSERT INTO `{table_name_in_db}` VALUES ({placeholders})"
                cur.execute(insert_sql, row)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Created mock database: {mock_db_path}")
    print(f"✅ Created mock JSONL: {mock_jsonl_path}")
    print(f"\n📊 Database contains {len(mock_tables)} tables:")
    for table in mock_tables:
        print(f"   - {table['name']} ({len(table['rows'])} rows)")
    
    print("\n" + "="*60)
    print("HOW TO VIEW DATABASE CONTENTS:")
    print("="*60)
    print("\n1. Using sqlite3 command line:")
    print(f"   sqlite3 {mock_db_path}")
    print("\n   Then run SQL commands like:")
    print("   .tables                    # List all tables")
    print("   .schema table_10015132_14  # Show table structure")
    print("   SELECT * FROM table_10015132_14 LIMIT 5;")
    print("   .quit                      # Exit")
    
    print("\n2. Using Python:")
    print("""
   import sqlite3
   conn = sqlite3.connect('data/mock_test.db')
   cur = conn.cursor()
   cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
   tables = cur.fetchall()
   print(tables)
   
   # Query a table
   cur.execute("SELECT * FROM table_10015132_14 LIMIT 5")
   rows = cur.fetchall()
   for row in rows:
       print(row)
   conn.close()
""")
    
    print("\n3. View all tables and their row counts:")
    print(f"   sqlite3 {mock_db_path} \"SELECT name FROM sqlite_master WHERE type='table';\"")
    print(f"   sqlite3 {mock_db_path} \"SELECT COUNT(*) FROM table_10015132_14;\"")
    
    print("\n4. View modified tables (after running schema changes):")
    print(f"   sqlite3 {mock_db_path} \"SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'table_m_%';\"")
    
    return mock_db_path, mock_jsonl_path

if __name__ == "__main__":
    create_mock_database_and_jsonl()

