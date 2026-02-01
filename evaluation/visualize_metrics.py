import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sqlglot
from sqlglot import exp
import re
from evaluation.eval_metrics import words, columns_used, subqueries, operators, splits, nodes

def create_distribution_plots(dataset: str, sql_column: str, dataset_name: str):
    """
    Create bar plots showing the distribution of SQL query metrics
    """
    
    # Load data
    if dataset.endswith(".json"):
        data = pd.read_json(dataset)
    elif dataset.endswith(".csv"):
        data = pd.read_csv(dataset)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset}")
    
    # Calculate metrics for all queries
    words_data = words(data, sql_column)
    columns_data = columns_used(data, sql_column)
    subqueries_data = subqueries(data, sql_column)
    operators_data = operators(data, sql_column)
    splits_data = splits(data, sql_column)
    nodes_data = nodes(data, sql_column)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Distribution of SQL Query Complexity Metrics in the {dataset_name} dataset', fontsize=16)
    
    # Plot 1: Words distribution
    ax1 = axes[0, 0]
    word_counts = words_data.value_counts().sort_index()
    ax1.bar(word_counts.index, word_counts.values, alpha=0.7, color='skyblue')
    ax1.set_title('Distribution of Word Count')
    ax1.set_xlabel('Number of Words')
    ax1.set_ylabel('Number of Queries')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Columns distribution
    ax2 = axes[0, 1]
    column_counts = columns_data.value_counts().sort_index()
    ax2.bar(column_counts.index, column_counts.values, alpha=0.7, color='lightgreen')
    ax2.set_title('Distribution of Column Count')
    ax2.set_xlabel('Number of Columns')
    ax2.set_ylabel('Number of Queries')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Subqueries distribution
    ax3 = axes[0, 2]
    subquery_counts = subqueries_data.value_counts().sort_index()
    ax3.bar(subquery_counts.index, subquery_counts.values, alpha=0.7, color='lightcoral')
    ax3.set_title('Distribution of Subquery Count')
    ax3.set_xlabel('Number of Subqueries')
    ax3.set_ylabel('Number of Queries')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Operators distribution
    ax4 = axes[1, 0]
    operator_counts = operators_data.value_counts().sort_index()
    ax4.bar(operator_counts.index, operator_counts.values, alpha=0.7, color='gold')
    ax4.set_title('Distribution of Operator Count')
    ax4.set_xlabel('Number of Operators')
    ax4.set_ylabel('Number of Queries')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Splits distribution
    ax5 = axes[1, 1]
    split_counts = splits_data.value_counts().sort_index()
    ax5.bar(split_counts.index, split_counts.values, alpha=0.7, color='plum')
    ax5.set_title('Distribution of Split Count')
    ax5.set_xlabel('Number of Splits')
    ax5.set_ylabel('Number of Queries')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Nodes distribution
    ax6 = axes[1, 2]
    node_counts = nodes_data.value_counts().sort_index()
    ax6.bar(node_counts.index, node_counts.values, alpha=0.7, color='lightblue')
    ax6.set_title('Distribution of Node Count')
    ax6.set_xlabel('Number of Nodes')
    ax6.set_ylabel('Number of Queries')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'sql_metrics_distribution_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("Summary Statistics:")
    print(f"Total queries: {len(data)}")
    print(f"Words - Min: {words_data.min()}, Max: {words_data.max()}, Mean: {words_data.mean():.2f}")
    print(f"Columns - Min: {columns_data.min()}, Max: {columns_data.max()}, Mean: {columns_data.mean():.2f}")
    print(f"Subqueries - Min: {subqueries_data.min()}, Max: {subqueries_data.max()}, Mean: {subqueries_data.mean():.2f}")
    print(f"Operators - Min: {operators_data.min()}, Max: {operators_data.max()}, Mean: {operators_data.mean():.2f}")
    print(f"Splits - Min: {splits_data.min()}, Max: {splits_data.max()}, Mean: {splits_data.mean():.2f}")
    print(f"Nodes - Min: {nodes_data.min()}, Max: {nodes_data.max()}, Mean: {nodes_data.mean():.2f}")

def create_binned_distribution(dataset: str, sql_column: str, dataset_name: str):
    """
    Create binned distribution plots (e.g., 5-10 words, 10-15 words, etc.)
    """
    
    # Load data
    if dataset.endswith(".json"):
        data = pd.read_json(dataset)
    elif dataset.endswith(".csv"):
        data = pd.read_csv(dataset)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset}")
    
    # Calculate metrics
    words_data = words(data, sql_column)
    
    # Create bins for words
    max_words = int(words_data.max())
    bin_edges = list(range(0, max_words + 10, 10))  # 10-word bins
    if bin_edges[-1] < max_words:
        bin_edges.append(max_words + 1)
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    counts, bins, patches = plt.hist(words_data, bins=bin_edges, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Calculate bin centers for proper x-axis positioning
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    
    # Customize x-axis labels to show ranges
    bin_labels = []
    for i in range(len(bins)-1):
        bin_labels.append(f'{int(bins[i])}-{int(bins[i+1])-1}')
    
    plt.xticks(bin_centers, bin_labels, rotation=45)
    plt.title(f'Distribution of SQL Query Word Count in the {dataset_name} dataset')
    plt.xlabel('Word Count Range')
    plt.ylabel('Number of Queries')
    plt.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        if count > 0:
            plt.text(bin_centers[i], count + 0.5, str(int(count)), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'words_binned_distribution_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print binned statistics
    print("\nBinned Word Count Distribution:")
    for i, count in enumerate(counts):
        if count > 0:
            print(f"{bin_labels[i]} words: {int(count)} queries")

if __name__ == "__main__":
    print("Creating distribution plots...")
    create_distribution_plots("../data_gen/order_experiment/generated_sql_nl.csv", "actual_sql", "generated_sql_nl")
    
    print("\nCreating binned distribution...")
    create_binned_distribution("../data_gen/order_experiment/generated_sql_nl.csv", "actual_sql", "generated_sql_nl")