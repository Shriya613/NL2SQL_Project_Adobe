# Schema Change Pipeline: JSONL and Database Update Flow

This document explains how JSONL and database tables are updated when performing schema change operations (e.g., `add_column`).

## Step-by-Step Flow: `add_column` Operation

**Input:** Existing table ID (e.g., `10007452-3`), NL question, SQL query

### Step 1: Get Original Table from JSONL
- Extract table name from SQL query: `table_1_10007452_3`
- Convert to table ID: `10007452-3`
- Read original table from JSONL: `{"id": "10007452-3", "header": ["Order Year", "Manufacturer", ...], "rows": [[...], [...]]}`
- Store in memory as `original_table` dict

### Step 2: Execute Original Query (Baseline)
- Execute original query on original table in DB to get baseline result
- Table name in DB: `table_10007452_3` (uses generic `col0`, `col1`, etc.)
- Store result for later validation

### Step 3: Apply Schema Change (In Memory)
- Call LLM to generate new columns (e.g., `Order_Date`, `Fuel_Type`, `Total_Sold`)
- Modify `current_table` dict in memory:
  - Append new column names to `current_table["header"]`
  - Append new column types to `current_table["types"]`
  - For each existing row, append random values from the LLM's value list
- Query remains unchanged (new columns aren't used)

### Step 4: Create New Table in Database (`add_new_table_to_db`)
- Generate unique ID: `m_<timestamp>_<uuid>_10007452-3`
- Create table name: `table_m_<timestamp>_<uuid>_10007452_3`
- Create table in SQLite:
  ```sql
  CREATE TABLE `table_m_...` (
    `col0` TEXT,  -- "Order Year"
    `col1` TEXT,  -- "Manufacturer"
    ...
    `col6` DATE,  -- "Order_Date" (new)
    `col7` TEXT,  -- "Fuel_Type" (new)
    `col8` INTEGER -- "Total_Sold" (new)
  )
  ```
- Insert rows with data (original + new random values)
- **Database uses generic names:** `col0`, `col1`, ..., `col8`

### Step 5: Append to JSONL File
- Create new JSONL entry:
  ```json
  {
    "id": "m_<timestamp>_<uuid>_10007452-3",
    "header": ["Order Year", "Manufacturer", ..., "Order_Date", "Fuel_Type", "Total_Sold"],
    "types": ["text", "text", ..., "date", "text", "integer"],
    "rows": [[...], [...]]
  }
  ```
- Append to `train.tables.jsonl` (original entry remains)
- **JSONL uses real column names** (not `col0`, `col1`, etc.)

### Step 6: Convert Query for Execution
- Original query uses real names: `SELECT \`Manufacturer\`, \`Model\` ...`
- Convert to generic names for DB: `SELECT col1, col2 ...`
- Update table name: `FROM table_m_...` (instead of `table_1_10007452_3`)

### Step 7: Return Results
- Return both versions:
  - `sql`: Query with real column names (matches JSONL)
  - `db_sql`: Query with generic names (`col0`, `col1`, etc.) for DB execution

## Key Points

- **Database**: Always uses generic placeholder names (`col0`, `col1`, `col2`, ...)
- **JSONL**: Uses real column names in the `header` field
- **Query (`sql`)**: Uses real column names (matches JSONL)
- **Query (`db_sql`)**: Uses generic names for execution on the database

**Important:** The original table in the database is **never modified**. A new table is always created with a unique ID prefixed with `m_`.

## Other Schema Change Operations

The same pattern applies to other operations:
- **`rename_column`**: Column names change in JSONL header, but DB still uses `col0`, `col1`, etc.
- **`remove_column`**: Column removed from JSONL header, DB columns shift (but still use `col0`, `col1`, etc.)
- **`merge_column`**: Multiple columns become one in JSONL, DB columns shift
- **`split_column`**: One column becomes multiple in JSONL, DB columns shift
- **`add_column`**: New columns added to JSONL, new `colN` columns added to DB

