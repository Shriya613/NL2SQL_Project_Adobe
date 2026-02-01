import json
import re
import sys
import os
from collections import Counter, defaultdict
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reserved import reserved_word_patterns


def extract_reserved_from_sql(sql_query: str, reserved_word_patterns: dict) -> list:
    """Extract SQL reserved words from SQL query."""
    
    # Convert to uppercase for consistent matching
    sql_upper = sql_query.upper()
    
    found_reserved = []
    # Check for each reserved word
    for reserved_word, pattern in reserved_word_patterns.items():
        # Find all matches of this reserved word pattern
        matches = re.findall(pattern, sql_upper)
        # Add the reserved word once for each match found
        found_reserved.extend([reserved_word] * len(matches))
    
    return found_reserved

def get_length(path: str, reserved_word_patterns: dict, SMOOTHING_FACTOR: int) -> dict:
    """function for seeing the composition of reserved words in a query"""
    length_counts = Counter()
    length_examples = defaultdict(list)  # Store examples for each length
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
                
                # Store example for this length (limit to 1 per length)
                if len(length_examples[length]) < 1:
                    length_examples[length].append(sql_query)
        
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

def get_reserved_probabilities(path: str, reserved_word_patterns: dict, SMOOTHING_FACTOR: int) -> dict:
    """Get the reserved word probabilities for a given dataset."""
    reserved_counts = Counter()
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
                reserved_counts.update(reserved)
                total_queries += 1
        
        # Apply smoothing to the counts
        adjusted_counts = {reserved: reserved_counts.get(reserved, 0) + SMOOTHING_FACTOR for reserved in reserved_word_patterns.keys()}
        total_adjusted_count = sum(adjusted_counts.values())
        print("Total adjusted count: ", total_adjusted_count)
        print("Number of reserved words: ", len(reserved_word_patterns.keys()))
        probabilities = {reserved: adjusted_counts[reserved] / total_adjusted_count for reserved in reserved_word_patterns.keys()}

        return probabilities
        
    except Exception as e:
        print(f"Error: {e}")
        return {}

def main():
    SMOOTHING_FACTOR = 10
    path = "data/train_bird.json"
    
    # Get the probabilities of each reserved word for the dataset
    probabilities = get_reserved_probabilities(path, reserved_word_patterns, SMOOTHING_FACTOR)
    
    df = pd.DataFrame(list(probabilities.items()), columns=["keyword", "probability"])
    df.to_csv('data_gen/length_experiment/reserved_probabilities.csv', index=False)
    
    # Get the length distribution probabilities
    length_probabilities = get_length(path, reserved_word_patterns, SMOOTHING_FACTOR)
    
    length_df = pd.DataFrame(list(length_probabilities.items()), columns=["length", "probability"])
    length_df.to_csv('data_gen/length_experiment/reserved_length_probabilities.csv', index=False)
    print("Length probabilities saved to data_gen/length_experiment/reserved_length_probabilities.csv")


if __name__ == "__main__":
    main()
