"""
Generate a detailed table for reward_threshold_0.6 with all metrics.
"""
import pandas as pd
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from evaluation.eval_metrics import (
    avg_words, avg_columns_used, avg_subqueries, avg_operators, avg_splits
)
from evaluate import calculate_nl_difficulty_metrics, calculate_sql_difficulty_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the data for reward_threshold_0.6
file_path = os.path.join(project_root, 'filtering_experiments/source_2_synth_rep/finetuned_filtered_scores.csv')
df = pd.read_csv(file_path)

# Get predictions (reward > 0.6)
y_pred = (df['rewards'].fillna(0).astype(float) > 0.6).astype(int)
y_true = df['allignment'].fillna(0).astype(int)

# Calculate classification metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)  # This is sensitivity
f1 = f1_score(y_true, y_pred, zero_division=0)

# Calculate specificity
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = recall

# Split data
true_positives = df[(y_true == 1) & (y_pred == 1)].copy()  # Correctly kept
false_negatives = df[(y_true == 1) & (y_pred == 0)].copy()  # Should be kept but were rejected

# Calculate SQL complexity for true positives
sql_tp = calculate_sql_difficulty_metrics(true_positives, 'sql_queries')

# Calculate SQL complexity for false negatives
sql_fn = calculate_sql_difficulty_metrics(false_negatives, 'sql_queries') if len(false_negatives) > 0 else {
    'avg_words': 0, 'avg_columns': 0, 'avg_subqueries': 0, 'avg_operators': 0, 'avg_splits': 0
}

# Calculate NL complexity for true positives
nl_tp = calculate_nl_difficulty_metrics(true_positives, 'nl_questions')

# Calculate NL complexity for false negatives
nl_fn = calculate_nl_difficulty_metrics(false_negatives, 'nl_questions') if len(false_negatives) > 0 else {
    'avg_tokens_before_verb': 0, 'avg_constituents_per_word': 0, 'avg_subordinate_clauses': 0,
    'avg_coordinate_clauses': 0
}

# Load BIRD dataset metrics
bird_metrics_path = os.path.join(project_root, 'filtering_experiments/useful_eval_results_and_scripts/bird_complexity_metrics.csv')
bird_df = pd.read_csv(bird_metrics_path)

# Extract BIRD SQL metrics
bird_sql_words = float(bird_df[bird_df['Metric'] == '  Avg Words']['Value'].values[0])
bird_sql_columns = float(bird_df[bird_df['Metric'] == '  Avg Columns']['Value'].values[0])
bird_sql_subqueries = float(bird_df[bird_df['Metric'] == '  Avg Subqueries']['Value'].values[0])
bird_sql_operators = float(bird_df[bird_df['Metric'] == '  Avg Operators']['Value'].values[0])
bird_sql_splits = float(bird_df[bird_df['Metric'] == '  Avg Splits']['Value'].values[0])

# Extract BIRD NL metrics (only the important ones)
bird_nl_tokens_before_verb = float(bird_df[bird_df['Metric'] == '  Avg Tokens Before Verb']['Value'].values[0])
bird_nl_constituents_per_word = float(bird_df[bird_df['Metric'] == '  Avg Constituents Per Word']['Value'].values[0])
bird_nl_subordinate_clauses = float(bird_df[bird_df['Metric'] == '  Avg Subordinate Clauses']['Value'].values[0])
bird_nl_coordinate_clauses = float(bird_df[bird_df['Metric'] == '  Avg Coordinate Clauses']['Value'].values[0])
bird_total_examples = int(bird_df[bird_df['Metric'] == 'SQL Complexity']['Value'].values[0].split()[0])

# Create comprehensive table with BIRD metrics side by side
table_data = {
    'Metric': [
        # Classification metrics
        'Accuracy',
        'Specificity',
        'Sensitivity (Recall)',
        'Precision',
        'F1 Score',
        '',
        # Confusion Matrix
        'True Positives (TP)',
        'True Negatives (TN)',
        'False Positives (FP)',
        'False Negatives (FN)',
        '',
        # SQL Complexity - True Positives
        'SQL Complexity - True Positives',
        '  Avg Words',
        '  Avg Columns',
        '  Avg Subqueries',
        '  Avg Operators',
        '  Avg Splits',
        '',
        # SQL Complexity - False Negatives
        'SQL Complexity - False Negatives',
        '  Avg Words',
        '  Avg Columns',
        '  Avg Subqueries',
        '  Avg Operators',
        '  Avg Splits',
        '',
        # SQL Complexity - BIRD
        'SQL Complexity - BIRD Dataset',
        '  Avg Words',
        '  Avg Columns',
        '  Avg Subqueries',
        '  Avg Operators',
        '  Avg Splits',
        '',
        # NL Complexity - True Positives (only important metrics)
        'NL Complexity - True Positives',
        '  Avg Tokens Before Verb',
        '  Avg Constituents Per Word',
        '  Avg Subordinate Clauses',
        '  Avg Coordinate Clauses',
        '',
        # NL Complexity - False Negatives (only important metrics)
        'NL Complexity - False Negatives',
        '  Avg Tokens Before Verb',
        '  Avg Constituents Per Word',
        '  Avg Subordinate Clauses',
        '  Avg Coordinate Clauses',
        '',
        # NL Complexity - BIRD (only important metrics)
        'NL Complexity - BIRD Dataset',
        '  Avg Tokens Before Verb',
        '  Avg Constituents Per Word',
        '  Avg Subordinate Clauses',
        '  Avg Coordinate Clauses',
    ],
    'Reward t=0.6 (TP)': [
        # Classification metrics
        f'{accuracy:.4f}',
        f'{specificity:.4f}',
        f'{sensitivity:.4f}',
        f'{precision:.4f}',
        f'{f1:.4f}',
        '',
        # Confusion Matrix
        f'{tp}',
        f'{tn}',
        f'{fp}',
        f'{fn}',
        '',
        # SQL Complexity - True Positives
        f'{len(true_positives)} examples',
        f'{sql_tp.get("avg_words", 0):.3f}',
        f'{sql_tp.get("avg_columns", 0):.3f}',
        f'{sql_tp.get("avg_subqueries", 0):.3f}',
        f'{sql_tp.get("avg_operators", 0):.3f}',
        f'{sql_tp.get("avg_splits", 0):.3f}',
        '',
        # SQL Complexity - False Negatives
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        # SQL Complexity - BIRD
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        # NL Complexity - True Positives
        f'{len(true_positives)} examples',
        f'{nl_tp.get("avg_tokens_before_verb", 0):.3f}',
        f'{nl_tp.get("avg_constituents_per_word", 0):.3f}',
        f'{nl_tp.get("avg_subordinate_clauses", 0):.3f}',
        f'{nl_tp.get("avg_coordinate_clauses", 0):.3f}',
        '',
        # NL Complexity - False Negatives
        '',
        '',
        '',
        '',
        '',
        '',
        # NL Complexity - BIRD
        '',
        '',
        '',
        '',
        '',
    ],
    'Reward t=0.6 (FN)': [
        # Classification metrics
        '',
        '',
        '',
        '',
        '',
        '',
        # Confusion Matrix
        '',
        '',
        '',
        '',
        '',
        # SQL Complexity - True Positives
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        # SQL Complexity - False Negatives
        f'{len(false_negatives)} examples',
        f'{sql_fn.get("avg_words", 0):.3f}',
        f'{sql_fn.get("avg_columns", 0):.3f}',
        f'{sql_fn.get("avg_subqueries", 0):.3f}',
        f'{sql_fn.get("avg_operators", 0):.3f}',
        f'{sql_fn.get("avg_splits", 0):.3f}',
        '',
        # SQL Complexity - BIRD
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        # NL Complexity - True Positives
        '',
        '',
        '',
        '',
        '',
        '',
        # NL Complexity - False Negatives
        f'{len(false_negatives)} examples',
        f'{nl_fn.get("avg_tokens_before_verb", 0):.3f}',
        f'{nl_fn.get("avg_constituents_per_word", 0):.3f}',
        f'{nl_fn.get("avg_subordinate_clauses", 0):.3f}',
        f'{nl_fn.get("avg_coordinate_clauses", 0):.3f}',
        '',
        # NL Complexity - BIRD
        '',
        '',
        '',
        '',
        '',
    ],
    'BIRD Dataset': [
        # Classification metrics
        '',
        '',
        '',
        '',
        '',
        '',
        # Confusion Matrix
        '',
        '',
        '',
        '',
        '',
        # SQL Complexity - True Positives
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        # SQL Complexity - False Negatives
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        # SQL Complexity - BIRD
        f'{bird_total_examples} examples',
        f'{bird_sql_words:.3f}',
        f'{bird_sql_columns:.3f}',
        f'{bird_sql_subqueries:.3f}',
        f'{bird_sql_operators:.3f}',
        f'{bird_sql_splits:.3f}',
        '',
        # NL Complexity - True Positives
        '',
        '',
        '',
        '',
        '',
        '',
        # NL Complexity - False Negatives
        '',
        '',
        '',
        '',
        '',
        '',
        # NL Complexity - BIRD
        f'{bird_total_examples} examples',
        f'{bird_nl_tokens_before_verb:.3f}',
        f'{bird_nl_constituents_per_word:.3f}',
        f'{bird_nl_subordinate_clauses:.3f}',
        f'{bird_nl_coordinate_clauses:.3f}',
    ]
}

result_df = pd.DataFrame(table_data)

# Save to CSV
output_path = os.path.join(project_root, 'filtering_experiments/useful_eval_results_and_scripts/reward_threshold_0.6_detailed_metrics.csv')
result_df.to_csv(output_path, index=False)

print("=" * 70)
print("Reward Threshold 0.6 - Detailed Metrics Table")
print("=" * 70)
print()
print(result_df.to_string(index=False))
print()
print(f"✅ Saved to: {output_path}")

