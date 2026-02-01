"""
Create publication-ready visualizations for filtering experiment results.
Produces 1-2 comprehensive figures summarizing:
1. Filtering performance (accuracy vs specificity)
2. Difficulty of kept examples (SQL/NL complexity)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_data():
    """Load the metrics CSV."""
    metrics_path = os.path.join(project_root, 'filtering_experiments/useful_eval_results_and_scripts/all_experiments_metrics.csv')
    df = pd.read_csv(metrics_path)
    return df

def shorten_name(name):
    """Shorten experiment names for readability."""
    name = name.replace('prompt_filter_threshold_', 'Prompt t=')
    name = name.replace('reward_threshold_', 'Reward t=')
    name = name.replace('finetuned_qwen', 'Finetuned Qwen')
    name = name.replace('llama_teacher', 'Llama Teacher')
    return name
def get_threshold(exp):
    match = re.search(r'(\d\.\d+)', exp)
    return float(match.group(1)) if match else None
def get_color_map(experiments):
    color_map = {}

    # More contrastive palettes
    reward_palette = [
        "#C00000",  # darkest
        "#FF0040",
        "#FF6D00",
        "#FFB380"   # lightest
    ]
    prompt_palette = [
        "#8B6914",  # darkest
        "#D4AF37",
        "#FFD300",
        "#FFF5B1"   # lightest
    ]

    # ---------------- Reward models ----------------
    reward_exps = sorted(
        [e for e in experiments if 'reward_threshold' in e],
        key=lambda x: get_threshold(x)  # sort numerically
    )

    # Darkest for 0.6 → so reverse (lower threshold = darker)
    for i, exp in enumerate(reward_exps):
        # Assign based on position
        idx = int(i * (len(reward_palette)-1) / max(1, len(reward_exps)-1))
        color_map[exp] = reward_palette[idx]

    # ---------------- Prompt models ----------------
    prompt_exps = sorted(
        [e for e in experiments if 'prompt_filter' in e],
        key=lambda x: get_threshold(x),  # higher threshold = darker
        reverse=True
    )

    for i, exp in enumerate(prompt_exps):
        idx = int(i * (len(prompt_palette)-1) / max(1, len(prompt_exps)-1))
        color_map[exp] = prompt_palette[idx]

    # ---------------- Other models ----------------
    if 'finetuned_qwen' in experiments:
        color_map['finetuned_qwen'] = "#4B7BCC"
    if 'llama_teacher' in experiments:
        color_map['llama_teacher'] = "#2E8B57"

    return color_map


def create_comprehensive_summary(df):
    """
    Create a single comprehensive figure with 4 subplots:
    1. Accuracy vs Specificity scatter (with keep rate as size)
    2. SQL complexity of kept examples
    3. NL complexity of kept examples
    4. Summary metrics bar chart
    """
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, height_ratios=[1.2, 1])
    
    # Create consistent color mapping for all experiments
    experiments = sorted(df['experiment'].unique())
    color_map = get_color_map(experiments)
    
    # 1. Sensitivity vs Specificity scatter plot (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Plot each experiment with its consistent color
    for idx, row in df.iterrows():
        exp_name = row['experiment']
        color = color_map[exp_name]
        ax1.scatter(
            row['recall'],  # Sensitivity (recall) on x-axis
            row['specificity'],  # Specificity on y-axis
            s=row['accuracy'] * 2500,  # Size based on accuracy 
            c=[color],
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5,
            zorder=1,
            label=shorten_name(exp_name)
        )
        # Add labels for all points
        ax1.annotate(
            shorten_name(exp_name),
            (row['recall'], row['specificity']),
            fontsize=9,
            ha='center',
            va='center',
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
            zorder=5
        )
    
    # Add reference lines
    ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axvline(x=0.8, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('Sensitivity', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Specificity', fontweight='bold', fontsize=12)
    ax1.set_title('Specificity vs Sensitivity\n(Size = Accuracy, Color = Model)', 
                  fontweight='bold', fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Add quadrant labels (swapped for swapped axes: x=sensitivity, y=specificity)
    # Top left: Low Sensitivity, High Specificity (Underfilters - rejects too many good examples)
    ax1.text(0.15, 0.9, 'Low Sensitivity\nHigh Specificity\n(Underfilters)', 
            ha='center', va='center', fontsize=9, alpha=0.5, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    # Top right: High Sensitivity, High Specificity (Best)
    ax1.text(0.85, 0.9, 'High Sensitivity\nHigh Specificity\n(Best)', 
            ha='center', va='center', fontsize=9, alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    # Bottom left: Low Sensitivity, Low Specificity
    ax1.text(0.15, 0.1, 'Low Sensitivity\nLow Specificity', 
            ha='center', va='center', fontsize=9, alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    # Bottom right: High Sensitivity, Low Specificity (Overfilters - accepts too many bad examples)
    ax1.text(0.85, 0.1, 'High Sensitivity\nLow Specificity\n(Overfilters)', 
            ha='center', va='center', fontsize=9, alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 2. SQL Complexity of Kept Examples (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Calculate composite SQL difficulty score (normalized)
    sql_cols = ['sql_kept_avg_words', 'sql_kept_avg_columns', 'sql_kept_avg_subqueries', 
                'sql_kept_avg_operators', 'sql_kept_avg_splits']
    sql_composite = df[sql_cols].mean(axis=1)
    sql_composite_norm = (sql_composite - sql_composite.min()) / (sql_composite.max() - sql_composite.min())
    
    # Sort by composite score
    df_sorted_sql = df.copy()
    df_sorted_sql['sql_composite'] = sql_composite_norm
    df_sorted_sql = df_sorted_sql.sort_values('sql_composite', ascending=True)
    
    # Use consistent colors from color_map
    bar_colors = [color_map[name] for name in df_sorted_sql['experiment']]
    bars = ax2.barh(range(len(df_sorted_sql)), df_sorted_sql['sql_composite'], 
                    color=bar_colors, alpha=0.8)
    ax2.set_yticks(range(len(df_sorted_sql)))
    ax2.set_yticklabels([shorten_name(name) for name in df_sorted_sql['experiment']], fontsize=9)
    ax2.set_xlabel('Normalized SQL Complexity Score\n↑ Higher = Keeps More Difficult SQL', 
                   fontweight='bold', fontsize=11)
    ax2.set_title('SQL Complexity of Kept Examples', fontweight='bold', fontsize=13, pad=10)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. NL Complexity of Kept Examples (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate weighted NL difficulty score using specific metrics
    # Formula: 0.15 * tokens_before_main_verb + 0.30 * constituents_per_word + 
    #          0.35 * subordinate_clauses + 0.20 * coordinate_clauses
    nl_composite = (
        0.15 * df.get('nl_kept_avg_tokens_before_verb', 0) +
        0.30 * df.get('nl_kept_avg_constituents_per_word', 0) +
        0.35 * df.get('nl_kept_avg_subordinate_clauses', 0) +
        0.20 * df.get('nl_kept_avg_coordinate_clauses', 0)
    )
    nl_composite_norm = (nl_composite - nl_composite.min()) / (nl_composite.max() - nl_composite.min()) if nl_composite.max() > nl_composite.min() else nl_composite
    
    # Sort by composite score
    df_sorted_nl = df.copy()
    df_sorted_nl['nl_composite'] = nl_composite_norm
    df_sorted_nl = df_sorted_nl.sort_values('nl_composite', ascending=True)
    
    # Use consistent colors from color_map
    bar_colors_nl = [color_map[name] for name in df_sorted_nl['experiment']]
    bars = ax3.barh(range(len(df_sorted_nl)), df_sorted_nl['nl_composite'], 
                    color=bar_colors_nl, alpha=0.8)
    ax3.set_yticks(range(len(df_sorted_nl)))
    ax3.set_yticklabels([shorten_name(name) for name in df_sorted_nl['experiment']], fontsize=9)
    ax3.set_xlabel('Normalized NL Complexity Score\n↑ Higher = Keeps More Difficult NL', 
                   fontweight='bold', fontsize=11)
    ax3.set_title('NL Complexity of Kept Examples', fontweight='bold', fontsize=13, pad=10)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Key Metrics Comparison (bottom middle and right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Sort by combined quality score
    df['combined_quality'] = df['accuracy'] * df['specificity'] * df['f1_score']
    df_sorted = df.sort_values('combined_quality', ascending=True)
    
    # F1 Score (sorted by F1 score descending)
    df_sorted_f1 = df.sort_values('f1_score', ascending=True)
    # Use consistent colors from color_map
    f1_colors = [color_map[name] for name in df_sorted_f1['experiment']]
    bars4 = ax4.barh(range(len(df_sorted_f1)), df_sorted_f1['f1_score'], 
                     color=f1_colors, alpha=0.8)
    ax4.set_yticks(range(len(df_sorted_f1)))
    ax4.set_yticklabels([shorten_name(name) for name in df_sorted_f1['experiment']], fontsize=9)
    ax4.set_xlabel('F1', fontweight='bold', fontsize=11)
    ax4.set_title('F1', fontweight='bold', fontsize=13, pad=10)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim(0, 1)
    
    # Runtime per example (one per method, sorted from lowest to highest)
    # Map experiment names to method types
    method_mapping = {
        'finetuned_qwen': 'Finetuned Qwen',
        'llama_teacher': 'Llama Teacher',
        'prompt_filter_threshold_0.75': 'Prompt Filter',
        'prompt_filter_threshold_0.8': 'Prompt Filter',
        'prompt_filter_threshold_0.85': 'Prompt Filter',
        'prompt_filter_threshold_0.9': 'Prompt Filter',
        'reward_threshold_0.55': 'Reward Threshold',  # These use pre-computed rewards
        'reward_threshold_0.6': 'Reward Threshold',
        'reward_threshold_0.65': 'Reward Threshold',
        'reward_threshold_0.7': 'Reward Threshold',
    }
    
    # Get actual benchmarked runtimes only (no estimates)
    runtime_actual = {}
    try:
        import subprocess
        result = subprocess.run(
            ['python3', os.path.join(project_root, 'filtering_experiments/benchmark_runtime.py')],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=project_root
        )
        if result.returncode == 0:
            # Parse output to get actual runtimes from the "Runtime Results" section
            lines = result.stdout.split('\n')
            in_results_section = False
            for line in lines:
                if 'Runtime Results' in line:
                    in_results_section = True
                    continue
                if in_results_section and ':' in line and line.strip():
                    parts = line.split(':')
                    if len(parts) == 2:
                        method = parts[0].strip()
                        time_str = parts[1].strip().replace('s', '').strip()
                        if time_str != 'N/A' and time_str:
                            try:
                                runtime_val = float(time_str)
                                # Map benchmark method names to display names
                                if method == 'reward_model':
                                    runtime_actual['Reward Model'] = runtime_val
                                elif method == 'reward_threshold':
                                    runtime_actual['Reward Threshold'] = runtime_val
                                elif method == 'finetuned_qwen':
                                    runtime_actual['Finetuned Qwen'] = runtime_val
                                elif method == 'prompt_filter':
                                    runtime_actual['Prompt Filter'] = runtime_val
                                elif method == 'llama_teacher':
                                    runtime_actual['Llama Teacher'] = runtime_val
                            except (ValueError, TypeError):
                                pass
                # Stop parsing when we hit an empty line after results section
                if in_results_section and not line.strip() and runtime_actual:
                    break
    except Exception as e:
        print(f"  Warning: Could not benchmark runtime: {e}")
        runtime_actual = {}
    
    # Create unique methods list (one per method type)
    # Include Reward Model separately from Reward Threshold
    unique_methods = list(set(method_mapping.values()))
    # Add Reward Model if we have reward experiments (to show actual computation time)
    if any('reward' in exp for exp in df['experiment'].values):
        if 'Reward Model' not in unique_methods:
            unique_methods.append('Reward Model')
    
    # Only use actual benchmarked runtimes (no estimates)
    method_runtimes = [(method, runtime_actual.get(method)) for method in unique_methods]
    method_runtimes = [(m, r) for m, r in method_runtimes if r is not None and m != 'Reward Threshold']
    method_runtimes.sort(key=lambda x: x[1], reverse=True)  # Sort by runtime (highest to lowest - slowest to fastest)
    
    if len(method_runtimes) > 0:
        methods, runtimes = zip(*method_runtimes)
        # For runtime, we need to map methods back to experiments
        # Since runtime is per method type, we'll use a neutral color scheme
        # or try to match to the closest experiment color
        runtime_colors = []
        for method in methods:
            # Try to find a matching experiment for this method
            matching_exp = None
            if method == 'Reward Model':
                # Use color from reward_threshold_0.6 as it's the main reward method
                matching_exp = 'reward_threshold_0.6'
            elif method == 'Finetuned Qwen':
                matching_exp = 'finetuned_qwen'
            elif method == 'Llama Teacher':
                matching_exp = 'llama_teacher'
            elif method == 'Prompt Filter':
                # Use color from prompt_filter_threshold_0.8 as representative
                matching_exp = 'prompt_filter_threshold_0.8'
            
            if matching_exp and matching_exp in color_map:
                runtime_colors.append(color_map[matching_exp])
            else:
                # Fallback to a neutral gray
                runtime_colors.append('#7f7f7f')
        
        bars5 = ax5.barh(range(len(methods)), runtimes, 
                          color=runtime_colors, alpha=0.8)
        ax5.set_yticks(range(len(methods)))
        ax5.set_yticklabels(methods, fontsize=9)
        ax5.set_xlabel('Runtime (seconds)', fontweight='bold', fontsize=11)
        ax5.set_title('Runtime per Example', fontweight='bold', fontsize=13, pad=10)
        ax5.grid(True, alpha=0.3, axis='x')
    else:
        ax5.text(0.5, 0.5, 'Runtime data\nnot available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Runtime per Example', fontweight='bold', fontsize=13, pad=10)
    
    plt.suptitle('Filtering Methods Comparison: Performance & Difficulty Retention', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig

def create_summary_table(df):
    """
    Create a summary table with key metrics for each experiment.
    """
    # Select key columns
    summary_cols = ['experiment', 'accuracy', 'specificity', 'f1_score', 'recall',
                    'sql_kept_avg_words', 'sql_kept_avg_columns', 
                    'nl_kept_avg_words', 'nl_kept_avg_syllables']
    
    summary_df = df[summary_cols].copy()
    
    # Round numeric columns
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(3)
    
    # Rename columns for readability
    summary_df.columns = ['Experiment', 'Accuracy', 'Specificity', 'F1 Score', 'Sensitivity',
                         'SQL Words', 'SQL Columns', 'NL Words', 'NL Syllables']
    
    # Shorten experiment names
    summary_df['Experiment'] = summary_df['Experiment'].apply(shorten_name)
    
    # Sort by combined quality
    df['combined_quality'] = df['accuracy'] * df['specificity'] * df['f1_score']
    quality_map = dict(zip(df['experiment'], df['combined_quality']))
    summary_df['_quality'] = summary_df['Experiment'].map(
        {shorten_name(k): v for k, v in quality_map.items()}
    )
    summary_df = summary_df.sort_values('_quality', ascending=False)
    summary_df = summary_df.drop('_quality', axis=1)
    
    return summary_df

def load_false_negatives_data():
    """Load false negatives (incorrectly rejected examples) for each experiment."""
    from evaluation.eval_metrics import (
        avg_words, avg_columns_used, avg_subqueries, avg_operators, avg_splits
    )
    # Import calculate_nl_difficulty_metrics from evaluate.py (same directory)
    from evaluate import calculate_nl_difficulty_metrics
    
    experiments_config = [
        {
            'name': 'finetuned_qwen',
            'path': os.path.join(project_root, 'filtering_experiments/source_2_synth_rep/finetuned_filtered_scores.csv'),
            'score_column': 'finetuned_filtered_scores'
        },
        {
            'name': 'llama_teacher',
            'path': os.path.join(project_root, 'filtering_experiments/teacher_model_filter.py/llama_filtered_scores.csv'),
            'score_column': 'llama_filtered_scores'
        },
        {
            'name': 'prompt_filter_threshold_0.75',
            'path': os.path.join(project_root, 'filtering_experiments/prompt_filter.py/prompt_filtered_scores.csv'),
            'score_threshold_column': 'prompt_scores',
            'score_threshold_value': 0.75
        },
        {
            'name': 'prompt_filter_threshold_0.8',
            'path': os.path.join(project_root, 'filtering_experiments/prompt_filter.py/prompt_filtered_scores.csv'),
            'score_threshold_column': 'prompt_scores',
            'score_threshold_value': 0.8
        },
        {
            'name': 'prompt_filter_threshold_0.85',
            'path': os.path.join(project_root, 'filtering_experiments/prompt_filter.py/prompt_filtered_scores.csv'),
            'score_threshold_column': 'prompt_scores',
            'score_threshold_value': 0.85
        },
        {
            'name': 'prompt_filter_threshold_0.9',
            'path': os.path.join(project_root, 'filtering_experiments/prompt_filter.py/prompt_filtered_scores.csv'),
            'score_threshold_column': 'prompt_scores',
            'score_threshold_value': 0.9
        },
        {
            'name': 'reward_threshold_0.55',
            'path': os.path.join(project_root, 'filtering_experiments/source_2_synth_rep/finetuned_filtered_scores.csv'),
            'reward_column': 'rewards',
            'reward_threshold': 0.55
        },
        {
            'name': 'reward_threshold_0.6',
            'path': os.path.join(project_root, 'filtering_experiments/source_2_synth_rep/finetuned_filtered_scores.csv'),
            'reward_column': 'rewards',
            'reward_threshold': 0.6
        },
        {
            'name': 'reward_threshold_0.65',
            'path': os.path.join(project_root, 'filtering_experiments/source_2_synth_rep/finetuned_filtered_scores.csv'),
            'reward_column': 'rewards',
            'reward_threshold': 0.65
        },
        {
            'name': 'reward_threshold_0.7',
            'path': os.path.join(project_root, 'filtering_experiments/source_2_synth_rep/finetuned_filtered_scores.csv'),
            'reward_column': 'rewards',
            'reward_threshold': 0.7
        },
    ]
    
    results = []
    
    for exp_config in experiments_config:
        exp_name = exp_config['name']
        file_path = exp_config['path']
        
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping {exp_name}: file not found")
            continue
        
        try:
            df = pd.read_csv(file_path)
            
            # Determine predictions
            if 'reward_column' in exp_config:
                y_pred = (df[exp_config['reward_column']].fillna(0).astype(float) > exp_config['reward_threshold']).astype(int)
            elif 'score_threshold_column' in exp_config:
                y_pred = (df[exp_config['score_threshold_column']].fillna(0).astype(float) >= exp_config['score_threshold_value']).astype(int)
            else:
                y_pred = df[exp_config['score_column']].fillna(0).astype(int)
            
            # Ground truth
            y_true = df['allignment'].fillna(0).astype(int)
            
            # False negatives: should be kept (y_true=1) but were rejected (y_pred=0)
            false_negatives = df[(y_true == 1) & (y_pred == 0)].copy()
            
            if len(false_negatives) == 0:
                print(f"⚠️  {exp_name}: No false negatives")
                results.append({
                    'experiment': exp_name,
                    'num_false_negatives': 0,
                    'sql_avg_words': 0, 'sql_avg_columns': 0, 'sql_avg_subqueries': 0,
                    'sql_avg_operators': 0, 'sql_avg_splits': 0,
                    'nl_avg_words': 0, 'nl_avg_syllables': 0, 'nl_avg_sentences': 0,
                    'nl_avg_tokens_before_verb': 0
                })
                continue
            
            # Calculate SQL metrics for false negatives
            sql_metrics = {}
            try:
                sql_metrics = {
                    'sql_avg_words': avg_words(false_negatives, 'sql_queries'),
                    'sql_avg_columns': avg_columns_used(false_negatives, 'sql_queries'),
                    'sql_avg_subqueries': avg_subqueries(false_negatives, 'sql_queries'),
                    'sql_avg_operators': avg_operators(false_negatives, 'sql_queries'),
                    'sql_avg_splits': avg_splits(false_negatives, 'sql_queries'),
                }
            except Exception as e:
                print(f"⚠️  {exp_name}: Error calculating SQL metrics: {e}")
                sql_metrics = {'sql_avg_words': 0, 'sql_avg_columns': 0, 'sql_avg_subqueries': 0,
                              'sql_avg_operators': 0, 'sql_avg_splits': 0}
            
            # Calculate NL metrics for false negatives (using all metrics from nl_evaluation.py)
            nl_metrics = {}
            try:
                nl_metrics_dict = calculate_nl_difficulty_metrics(false_negatives, 'nl_questions')
                nl_metrics = {
                    'nl_avg_words': nl_metrics_dict.get('avg_words', 0),
                    'nl_avg_syllables': nl_metrics_dict.get('avg_syllables', 0),
                    'nl_avg_sentences': nl_metrics_dict.get('avg_sentences', 0),
                    'nl_avg_tokens': nl_metrics_dict.get('avg_tokens', 0),
                    'nl_avg_entities': nl_metrics_dict.get('avg_entities', 0),
                    'nl_avg_tokens_before_verb': nl_metrics_dict.get('avg_tokens_before_verb', 0),
                    'nl_avg_constituents_per_word': nl_metrics_dict.get('avg_constituents_per_word', 0),
                    'nl_avg_subordinate_clauses': nl_metrics_dict.get('avg_subordinate_clauses', 0),
                    'nl_avg_coordinate_clauses': nl_metrics_dict.get('avg_coordinate_clauses', 0),
                    'nl_avg_complexity_score': nl_metrics_dict.get('avg_complexity_score', 0),
                    'nl_avg_flesch_reading_ease': nl_metrics_dict.get('avg_flesch_reading_ease', 0),
                    'nl_avg_flesch_kincaid_grade_level': nl_metrics_dict.get('avg_flesch_kincaid_grade_level', 0),
                }
            except Exception as e:
                print(f"⚠️  {exp_name}: Error calculating NL metrics: {e}")
                nl_metrics = {
                    'nl_avg_words': 0, 'nl_avg_syllables': 0, 'nl_avg_sentences': 0,
                    'nl_avg_tokens': 0, 'nl_avg_entities': 0, 'nl_avg_tokens_before_verb': 0,
                    'nl_avg_constituents_per_word': 0, 'nl_avg_subordinate_clauses': 0,
                    'nl_avg_coordinate_clauses': 0, 'nl_avg_complexity_score': 0,
                    'nl_avg_flesch_reading_ease': 0, 'nl_avg_flesch_kincaid_grade_level': 0
                }
            
            results.append({
                'experiment': exp_name,
                'num_false_negatives': len(false_negatives),
                **sql_metrics,
                **nl_metrics
            })
            
        except Exception as e:
            print(f"⚠️  Error processing {exp_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return pd.DataFrame(results)


def create_false_negatives_summary(df):
    """
    Create a figure focused on SQL and NL complexity of false negatives (incorrectly rejected examples).
    Similar layout to the main summary but focused on what was incorrectly rejected.
    """
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, height_ratios=[1.2, 1])
    
    # Create consistent color mapping
    experiments = sorted(df['experiment'].unique())
    color_map = get_color_map(experiments)
    
    # Filter out experiments with no false negatives
    df_with_fn = df[df['num_false_negatives'] > 0].copy()
    
    if len(df_with_fn) == 0:
        fig.text(0.5, 0.5, 'No false negatives found in any experiment', 
                ha='center', va='center', fontsize=16, transform=fig.transFigure)
        return fig
    
    # 1. False Negatives Count (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    df_sorted_fn = df_with_fn.sort_values('num_false_negatives', ascending=True)
    bar_colors = [color_map[name] for name in df_sorted_fn['experiment']]
    bars = ax1.barh(range(len(df_sorted_fn)), df_sorted_fn['num_false_negatives'], 
                    color=bar_colors, alpha=0.8)
    ax1.set_yticks(range(len(df_sorted_fn)))
    ax1.set_yticklabels([shorten_name(name) for name in df_sorted_fn['experiment']], fontsize=9)
    ax1.set_xlabel('Number of False Negatives\n(Should be kept but were rejected)', 
                   fontweight='bold', fontsize=12)
    ax1.set_title('False Negatives Count', fontweight='bold', fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. SQL Complexity of False Negatives (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Calculate composite SQL difficulty score (normalized)
    sql_cols = ['sql_avg_words', 'sql_avg_columns', 'sql_avg_subqueries', 
                'sql_avg_operators', 'sql_avg_splits']
    sql_composite = df_with_fn[sql_cols].mean(axis=1)
    sql_composite_norm = (sql_composite - sql_composite.min()) / (sql_composite.max() - sql_composite.min()) if sql_composite.max() > sql_composite.min() else sql_composite
    
    df_sorted_sql = df_with_fn.copy()
    df_sorted_sql['sql_composite'] = sql_composite_norm
    df_sorted_sql = df_sorted_sql.sort_values('sql_composite', ascending=True)
    
    bar_colors = [color_map[name] for name in df_sorted_sql['experiment']]
    bars = ax2.barh(range(len(df_sorted_sql)), df_sorted_sql['sql_composite'], 
                    color=bar_colors, alpha=0.8)
    ax2.set_yticks(range(len(df_sorted_sql)))
    ax2.set_yticklabels([shorten_name(name) for name in df_sorted_sql['experiment']], fontsize=9)
    ax2.set_xlabel('Normalized SQL Complexity Score\n↑ Higher = More Complex SQL Rejected', 
                   fontweight='bold', fontsize=11)
    ax2.set_title('SQL Complexity of False Negatives', fontweight='bold', fontsize=13, pad=10)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. NL Complexity of False Negatives (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate weighted NL difficulty score using specific metrics
    # Formula: 0.15 * tokens_before_main_verb + 0.30 * constituents_per_word + 
    #          0.35 * subordinate_clauses + 0.20 * coordinate_clauses
    nl_composite = (
        0.15 * df_with_fn.get('nl_avg_tokens_before_verb', 0) +
        0.30 * df_with_fn.get('nl_avg_constituents_per_word', 0) +
        0.35 * df_with_fn.get('nl_avg_subordinate_clauses', 0) +
        0.20 * df_with_fn.get('nl_avg_coordinate_clauses', 0)
    )
    nl_composite_norm = (nl_composite - nl_composite.min()) / (nl_composite.max() - nl_composite.min()) if nl_composite.max() > nl_composite.min() else nl_composite
    
    df_sorted_nl = df_with_fn.copy()
    df_sorted_nl['nl_composite'] = nl_composite_norm
    df_sorted_nl = df_sorted_nl.sort_values('nl_composite', ascending=True)
    
    bar_colors = [color_map[name] for name in df_sorted_nl['experiment']]
    bars = ax3.barh(range(len(df_sorted_nl)), df_sorted_nl['nl_composite'], 
                    color=bar_colors, alpha=0.8)
    ax3.set_yticks(range(len(df_sorted_nl)))
    ax3.set_yticklabels([shorten_name(name) for name in df_sorted_nl['experiment']], fontsize=9)
    ax3.set_xlabel('Normalized NL Complexity Score\n↑ Higher = More Complex NL Rejected', 
                   fontsize=11)
    ax3.set_title('NL Complexity of False Negatives', fontweight='bold', fontsize=13, pad=10)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. SQL Words (bottom middle)
    ax4 = fig.add_subplot(gs[1, 1])
    
    df_sorted_words = df_with_fn.sort_values('sql_avg_words', ascending=True)
    bar_colors = [color_map[name] for name in df_sorted_words['experiment']]
    bars = ax4.barh(range(len(df_sorted_words)), df_sorted_words['sql_avg_words'], 
                    color=bar_colors, alpha=0.8)
    ax4.set_yticks(range(len(df_sorted_words)))
    ax4.set_yticklabels([shorten_name(name) for name in df_sorted_words['experiment']], fontsize=9)
    ax4.set_xlabel('Avg SQL Words', fontweight='bold', fontsize=11)
    ax4.set_title('SQL Words in False Negatives', fontweight='bold', fontsize=13, pad=10)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. NL Words (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    
    df_sorted_nl_words = df_with_fn.sort_values('nl_avg_words', ascending=True)
    bar_colors = [color_map[name] for name in df_sorted_nl_words['experiment']]
    bars = ax5.barh(range(len(df_sorted_nl_words)), df_sorted_nl_words['nl_avg_words'], 
                    color=bar_colors, alpha=0.8)
    ax5.set_yticks(range(len(df_sorted_nl_words)))
    ax5.set_yticklabels([shorten_name(name) for name in df_sorted_nl_words['experiment']], fontsize=9)
    ax5.set_xlabel('Avg NL Words', fontweight='bold', fontsize=11)
    ax5.set_title('NL Words in False Negatives', fontweight='bold', fontsize=13, pad=10)
    ax5.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('False Negatives Analysis: Complexity of Incorrectly Rejected Examples', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig


def main():
    """Generate visualizations and summary table."""
    print("Loading metrics data...")
    df = load_data()
    print(f"Loaded {len(df)} experiments")
    
    output_dir = os.path.join(project_root, 'filtering_experiments/useful_eval_results_and_scripts')
    
    print("\nGenerating comprehensive summary figure...")
    fig = create_comprehensive_summary(df)
    fig.savefig(os.path.join(output_dir, 'filtering_summary_comprehensive.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("  ✅ Saved: filtering_summary_comprehensive.png")
    
    print("\nGenerating false negatives analysis...")
    fn_df = load_false_negatives_data()
    print(f"  Loaded false negatives data for {len(fn_df)} experiments")
    fn_fig = create_false_negatives_summary(fn_df)
    fn_fig.savefig(os.path.join(output_dir, 'filtering_false_negatives_analysis.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
    print("  ✅ Saved: filtering_false_negatives_analysis.png")
    
    print("\nGenerating summary table...")
    summary_table = create_summary_table(df)
    summary_path = os.path.join(output_dir, 'filtering_summary_table.csv')
    summary_table.to_csv(summary_path, index=False)
    print(f"  ✅ Saved: filtering_summary_table.csv")
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(summary_table.to_string(index=False))
    print("\n" + "="*60)
    
    print(f"\n✅ All outputs saved to {output_dir}")
    print("\nGenerated files:")
    print("  1. filtering_summary_comprehensive.png - Single comprehensive figure with 6 subplots")
    print("  2. filtering_false_negatives_analysis.png - Analysis of incorrectly rejected examples")
    print("  3. filtering_summary_table.csv - Summary table with key metrics")

if __name__ == "__main__":
    main()
