import sqlglot
import json
import pandas as pd
import re
from sqlglot import exp
import numpy as np

#node_types = []

def safe_parse_sql(sql: str):
    """
    Safely parse a SQL query, returning None if parsing fails.
    
    Args:
        sql: SQL query string to parse
        
    Returns:
        Parsed SQLGlot expression or None if parsing fails
    """
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None

def avg_words(df: pd.DataFrame, column: str) -> float:
    """Calculate the average number of words in the SQL queries"""
    return (
        df[column]
        .dropna()
        .replace("`", "")
        .replace("'", "")
        .apply(lambda x: len(re.sub(r"\s+", " ", x.strip()).split()))
        .mean()
    )

def avg_columns_used(df: pd.DataFrame, column: str) -> int:
    """Calculate the average number of columns in the SQL queries"""
    return (
        df[column]
        .dropna()
        .apply(lambda x: len(list(parsed.find_all(exp.Column))) if (parsed := safe_parse_sql(x)) else 0).mean()
    )

def avg_subqueries(df: pd.DataFrame, column: str) -> int:
    """Calculate the average number of subqueries in the SQL queries"""
    return (
        df[column]
        .dropna()
         .apply(lambda x: len(list(parsed.find_all(exp.Subquery))) if (parsed := safe_parse_sql(x)) else 0).mean()
    )
    
def count_nodes(expression: sqlglot.expressions.Expression) -> int:
    """
    Count the number of nodes in a SQLGlot expression tree.
    """
    if expression is None:
        return 0
    
    # Use walk() to traverse the tree and count each node exactly once
    return sum(1 for _ in expression.walk())

def print_tree_structure(expression, indent=0):
    """
    Print a tree representation of the SQLGlot expression.
    """
    if expression is None:
        return
    
    return expression.dump()

def count_operators(expression: sqlglot.expressions.Expression) -> dict:
    """
    Count different types of operators in the SQLGlot expression tree.
    """
    if expression is None:
        return {}
    
    operator_counts = {}
    
    # Define operator types to count
    all_operators = {
    # Basic comparison operators
    'EQ': "=",
    'NEQ': "!= <>",
    'LT': "<",
    'GT': ">",
    'LTE': "<=",
    'GTE': ">=",
    
    # Logical operators
    'And': "AND",
    'Or': "OR",
    'Not': "NOT",
    
    # String operations
    'Like': "LIKE",
    'In': "IN",
    'Between': "BETWEEN",
    'IsNull': "IS NULL",
    'IsNotNull': "IS NOT NULL",
    
    # Aggregation functions
    'Count': "COUNT",
    'Sum': "SUM",
    'Avg': "AVG",
    'Min': "MIN",
    'Max': "MAX",

    # Sorting and grouping
    'Order': "ORDER BY",
    'Group': "GROUP BY",
    'Having': "HAVING",
    
    # Joins (MySQL native)
    'Join': "JOIN",
    'LeftJoin': "LEFT JOIN",
    'RightJoin': "RIGHT JOIN",
    'InnerJoin': "INNER JOIN",
    'OuterJoin': "OUTER JOIN",
    
    # Set operations (MySQL native)
    'Union': "UNION",
    
    # Other MySQL-native operators
    'Distinct': "DISTINCT",
    'Limit': "LIMIT",
    'Case': "CASE",
    'Exists': "EXISTS",
    'As': "AS",
    'Where': "WHERE",
    'Select': "SELECT",
    'From': "FROM",
    'Insert': "INSERT",
    'Update': "UPDATE",
    'Delete': "DELETE",
    'Create': "CREATE",
    'Drop': "DROP",
    'Alter': "ALTER"
    }
    
    # Count each operator type
    for node in expression.walk():
        node_type = type(node).__name__
        #if node_type not in node_types:
        #    node_types.append(node_type)
        if node_type in all_operators:
            operator_name = all_operators[node_type]
            operator_counts[operator_name] = operator_counts.get(operator_name, 0) + 1
            operator_counts["ALL"] = operator_counts.get("ALL", 0) + 1
    
    return operator_counts

def count_splits(node: sqlglot.expressions.Expression) -> int:
    """
    Count the number of splits in a SQLGlot expression tree.
    """
    if not hasattr(node, "args") or not node.args:
        return 0
    # count 1 if this node has children
    count = 1 if any(isinstance(v, sqlglot.expressions.Expression) or isinstance(v, list) for v in node.args.values()) else 0
    # recurse through children
    for v in node.args.values():
        if isinstance(v, sqlglot.expressions.Expression):
            count += count_splits(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, sqlglot.expressions.Expression):
                    count += count_splits(item)
    return count

def avg_operators(df: pd.DataFrame, column: str) -> float:
    """Calculate the average number of operators in the SQL queries"""
    return (
        df[column]
        .dropna()
        .apply(lambda x: count_operators(parsed).get("ALL", 0) if (parsed := safe_parse_sql(x)) else 0).mean()
    )

def avg_splits(df: pd.DataFrame, column: str) -> float:
    """Calculate the average number of splits in the SQL queries"""
    return (
        df[column]
        .dropna()
        .apply(lambda x: count_splits(parsed) if (parsed := safe_parse_sql(x)) else 0).mean()
    )


def words(df: pd.DataFrame, column: str) -> float:
    """Calculate the words in the SQL queries"""
    return (
        df[column]
        .dropna()
        .replace("`", "")
        .replace("'", "")
        .apply(lambda x: len(re.sub(r"\s+", " ", x.strip()).split()))
    )

def columns_used(df: pd.DataFrame, column: str) -> int:
    """Calculate the columns in the SQL queries"""
    return (
        df[column]
        .dropna()
        .apply(lambda x: len(list(parsed.find_all(exp.Column))) if (parsed := safe_parse_sql(x)) else 0)
    )

def subqueries(df: pd.DataFrame, column: str) -> int:
    """Calculate the subqueries in the SQL queries"""
    return (
        df[column]
        .dropna()
         .apply(lambda x: len(list(parsed.find_all(exp.Subquery))) if (parsed := safe_parse_sql(x)) else 0)
    )

def operators(df: pd.DataFrame, column: str) -> float:
    """Calculate the operators in the SQL queries"""
    return (
        df[column]
        .dropna()
        .apply(lambda x: count_operators(parsed).get("ALL", 0) if (parsed := safe_parse_sql(x)) else 0)
    )

def splits(df: pd.DataFrame, column: str) -> float:
    """Calculate the splits in the SQL queries"""
    return (
        df[column]
        .dropna()
        .apply(lambda x: count_splits(parsed) if (parsed := safe_parse_sql(x)) else 0)
    )

def nodes(df: pd.DataFrame, column: str) -> float:
    """Calculate the nodes in the SQL queries"""
    return (
        df[column]
        .dropna()
        .apply(lambda x: count_nodes(parsed) if (parsed := safe_parse_sql(x)) else 0)
    )

if __name__ == "__main__":
    data = pd.read_json("data/train_bird.json")
    max_words = max(words(data, "SQL"))
    max_columns = max(columns_used(data, "SQL"))
    max_subqueries = max(subqueries(data, "SQL"))
    max_operators = max(operators(data, "SQL"))
    max_splits = max(splits(data, "SQL"))
    max_nodes = max(nodes(data, "SQL"))
    min_words = min(words(data, "SQL"))
    min_columns = min(columns_used(data, "SQL"))
    min_subqueries = min(subqueries(data, "SQL"))
    min_operators = min(operators(data, "SQL"))
    min_splits = min(splits(data, "SQL"))
    min_nodes = min(nodes(data, "SQL"))
    variance_words = np.var(words(data, "SQL"))
    variance_columns = np.var(columns_used(data, "SQL"))
    variance_subqueries = np.var(subqueries(data, "SQL"))
    variance_operators = np.var(operators(data, "SQL"))
    variance_splits = np.var(splits(data, "SQL"))
    variance_nodes = np.var(nodes(data, "SQL"))

    print("Max words:", max_words)
    print("Max columns:", max_columns)
    print("Max subqueries:", max_subqueries)
    print("Max operators:", max_operators)
    print("Max splits:", max_splits)
    print("Max nodes:", max_nodes)
    print("Min words:", min_words)
    print("Min columns:", min_columns)
    print("Min subqueries:", min_subqueries)
    print("Min operators:", min_operators)
    print("Min splits:", min_splits)
    print("Min nodes:", min_nodes)
    print("Variance of words:", variance_words)
    print("Variance of columns:", variance_columns)
    print("Variance of subqueries:", variance_subqueries)
    print("Variance of operators:", variance_operators)
    print("Variance of splits:", variance_splits)
    print("Variance of nodes:", variance_nodes)

