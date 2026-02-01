import json
import re
from collections import Counter


def extract_reserved_from_sql(sql_query: str, reserved_word_patterns: dict) -> list:
    """Extract SQL reserved words from SQL query."""
    
    # Convert to uppercase for consistent matching
    sql_upper = sql_query.upper()
    
    found_reserved = []
    # Check for each reserved word
    for reserved_word, pattern in reserved_word_patterns.items():
        matches = re.findall(pattern, sql_upper)
        found_reserved.extend([reserved_word] * len(matches))
    
    return found_reserved

def get_length(path: str, reserved_word_patterns: dict, SMOOTHING_FACTOR: int) -> dict:
    """function for seeing the composition of reserved words in a query"""
    length_counts = Counter()
    total_queries = 0
    
    try:
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different dataset formats
        if isinstance(data, list):
            queries = data
        else:
            queries = []
        
        for item in queries:
            # Extract SQL query based on dataset format
            sql_query = None
            
            if isinstance(item, dict):
                if 'SQL' in item:
                    sql_query = item['SQL']

            elif isinstance(item, str):
                sql_query = item
            
            if sql_query:
                reserved = extract_reserved_from_sql(sql_query, reserved_word_patterns)
                length = len(reserved)
                length_counts[length] += 1
                total_queries += 1
                
        # Apply smoothing to the counts
        # Find the maximum length to know the range
        max_length = max(length_counts.keys()) if length_counts else 0
        
        # Apply smoothing to all possible lengths from 0 to max_length
        adjusted_counts = {}
        for length in range(max_length + 1):
            adjusted_counts[length] = length_counts.get(length, 0) + SMOOTHING_FACTOR
        
        total_adjusted_count = sum(adjusted_counts.values())
        
        probabilities = {length: adjusted_counts[length] / total_adjusted_count for length in adjusted_counts.keys()}

        return probabilities
        
    except Exception as e:
        print(f"Error: {e}")
        return {}
