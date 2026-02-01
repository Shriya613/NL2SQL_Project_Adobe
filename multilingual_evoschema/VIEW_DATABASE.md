# How to View Database Contents

## Mock Database Files
- **Database**: `data/mock_test.db`
- **JSONL**: `data/mock_test.tables.jsonl`

## Quick Commands

### 1. List All Tables
```bash
sqlite3 data/mock_test.db "SELECT name FROM sqlite_master WHERE type='table';"
```

### 2. List Only Modified Tables (with "m_" prefix)
```bash
sqlite3 data/mock_test.db "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'table_m_%';"
```

### 3. View Table Structure
```bash
sqlite3 data/mock_test.db ".schema table_1_10015132_14"
```

### 4. View Table Data
```bash
sqlite3 data/mock_test.db "SELECT * FROM table_1_10015132_14 LIMIT 5;"
```

### 5. Count Rows in a Table
```bash
sqlite3 data/mock_test.db "SELECT COUNT(*) FROM table_1_10015132_14;"
```

### 6. Interactive Mode
```bash
sqlite3 data/mock_test.db
```
Then run commands like:
- `.tables` - List all tables
- `.schema <table_name>` - Show table structure
- `.headers on` - Show column headers in SELECT results
- `.mode column` - Format output as columns
- `SELECT * FROM table_1_10015132_14;` - Query data
- `.quit` - Exit

## Python Script

```python
import sqlite3

# Connect to database
conn = sqlite3.connect('data/mock_test.db')
cur = conn.cursor()

# List all tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()
print("Tables:", tables)

# Query a table
cur.execute("SELECT * FROM table_1_10015132_14 LIMIT 5")
rows = cur.fetchall()
for row in rows:
    print(row)

# Get column names
cur.execute("PRAGMA table_info(table_1_10015132_14)")
columns = cur.fetchall()
print("Columns:", columns)

conn.close()
```

## After Schema Changes

When you run schema changes, new tables will be created with the "m_" prefix:
- Original: `table_1_10015132_14`
- Modified: `table_m_<timestamp>_<uuid>_1_10015132_14`

To see all modified tables:
```bash
sqlite3 data/mock_test.db "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'table_m_%' ORDER BY name;"
```

To see the structure of a modified table:
```bash
sqlite3 data/mock_test.db ".schema table_m_<timestamp>_<uuid>_1_10015132_14"
```

## View JSONL Contents

The JSONL file contains table schemas with real column names:
```bash
# View first table
head -1 data/mock_test.tables.jsonl | python -m json.tool

# Count total entries
wc -l data/mock_test.tables.jsonl

# View all table IDs
cat data/mock_test.tables.jsonl | python -c "import sys, json; [print(json.loads(line)['id']) for line in sys.stdin]"
```

