import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import pi
import spacy
import seaborn as sns
import geopandas as gpd

# SQL Evaluation tools (for Radar Chart)
from sql_evaluation import (
    words,
    columns_used,
    subquery_counts,
    operator_counts,
    split_counts,
    normalize_metric,
)

from nl_evaluation import calculate_syntactic_complexity, get_nlp

def get_metrics_for_dataset(df, sql_column="sql") -> tuple[dict, dict]:
    """Compute raw average metrics for a DataFrame"""
    raw_sql_words = words(df, sql_column)
    raw_columns = columns_used(df, sql_column)
    raw_subqueries = subquery_counts(df, sql_column)
    raw_operators = operator_counts(df, sql_column)
    raw_splits = split_counts(df, sql_column)
    
    return {
        "SQL Words": raw_sql_words.mean(),
        "Columns": raw_columns.mean(),
        "Subqueries": raw_subqueries.mean(),
        "Operators": raw_operators.mean(),
        "Splits": raw_splits.mean()
    }, {
        "raw_sql_words": raw_sql_words,
        "raw_columns": raw_columns,
        "raw_subqueries": raw_subqueries,
        "raw_operators": raw_operators,
        "raw_splits": raw_splits
    }

def normalize_dataset_metrics(metrics_dict, raw_series, reference_stats=None) -> tuple[dict, dict]:
    """Normalize metrics using the same logic as evaluate_sql in sql_evaluation.py"""
    if reference_stats is None:
        reference_stats = {
            "sql_words": {
                "min": raw_series["raw_sql_words"].min(),
                "max": raw_series["raw_sql_words"].max(),
                "mean": raw_series["raw_sql_words"].mean(),
                "std": raw_series["raw_sql_words"].std(),
            },
            "columns": {
                "min": raw_series["raw_columns"].min(),
                "max": raw_series["raw_columns"].max(),
            },
            "subqueries": {
                "min": raw_series["raw_subqueries"].min(),
                "max": raw_series["raw_subqueries"].max(),
            },
            "operators": {
                "min": raw_series["raw_operators"].min(),
                "max": raw_series["raw_operators"].max(),
            },
            "splits": {
                "min": raw_series["raw_splits"].min(),
                "max": raw_series["raw_splits"].max(),
            },
        }

    norm_sql_words = normalize_metric(metrics_dict["SQL Words"], method="log_plus_one", stats=reference_stats["sql_words"])
    norm_columns = normalize_metric(metrics_dict["Columns"], method="minmax", stats=reference_stats["columns"])
    norm_subqueries = normalize_metric(metrics_dict["Subqueries"], method="log_plus_one", stats=reference_stats["subqueries"])
    norm_operators = normalize_metric(metrics_dict["Operators"], method="log_plus_one", stats=reference_stats["operators"])
    norm_splits = normalize_metric(metrics_dict["Splits"], method="log_plus_one", stats=reference_stats["splits"])

    return {
        "SQL Words": norm_sql_words,
        "Columns": norm_columns,
        "Subqueries": norm_subqueries,
        "Operators": norm_operators,
        "Splits": norm_splits
    }, reference_stats

def create_radar_chart():
    print("Generating Radar Chart...")
    print("Loading datasets...")
    
    try:
        df_pipeline = pd.read_csv("pipeline/final_translations/translated_en.csv")
        if "db_sql" in df_pipeline.columns:
            df_pipeline = df_pipeline.dropna(subset=["db_sql"])
    except FileNotFoundError:
        print("Error: pipeline/dataset/translated_en.csv not found.")
        return

    try:
        with open("data/train_bird.json", "r") as f:
            bird_data = json.load(f)
      
        df_bird = pd.DataFrame(bird_data)
        
        if "SQL" in df_bird.columns:
            df_bird.rename(columns={"SQL": "sql"}, inplace=True)
        elif "sql" not in df_bird.columns:
             if 'query' in df_bird.columns:
                 df_bird.rename(columns={'query': 'sql'}, inplace=True)
        
        df_bird = df_bird.dropna(subset=["sql"])
        
    except FileNotFoundError:
        print("Error: data/train_bird.json not found.")
        return

    print("Calculating metrics...")
    
    metrics_bird, raw_series_bird = get_metrics_for_dataset(df_bird, "sql")
    metrics_pipeline, raw_series_pipeline = get_metrics_for_dataset(df_pipeline, "db_sql")
    
    norm_metrics_bird, bird_stats = normalize_dataset_metrics(metrics_bird, raw_series_bird)
    norm_metrics_pipeline, _ = normalize_dataset_metrics(metrics_pipeline, raw_series_pipeline, reference_stats=bird_stats) # Normalize on BIRD's scale
    
    categories = list(norm_metrics_bird.keys())
    N = len(categories)
    
    values_bird = [float(v) if v is not None else 0.0 for v in list(norm_metrics_bird.values())]
    values_pipeline = [float(v) if v is not None else 0.0 for v in list(norm_metrics_pipeline.values())]
    
    values_bird += values_bird[:1]
    values_pipeline += values_pipeline[:1]
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], categories, color='black', size=18)
    ax.tick_params(axis='x', pad=30)
    
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=12)
    plt.ylim(0, max(max(values_bird), max(values_pipeline)) + 0.1)
    
    ax.plot(angles, values_bird, linewidth=2, linestyle='solid', color='#648FFF')
    ax.fill(angles, values_bird, '#648FFF', alpha=0.1)
    
    ax.plot(angles, values_pipeline, linewidth=2, linestyle='solid', color='#DC267F')
    ax.fill(angles, values_pipeline, '#DC267F', alpha=0.1)
    
    legend_handles = [
        patches.Patch(facecolor='#648FFF', edgecolor="#1A1A1A", linewidth=1.0, label='BIRD (Train)'),
        patches.Patch(facecolor='#DC267F', edgecolor="#1A1A1A", linewidth=1.0, label='Pipeline')
    ]
    plt.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1.0, 1),
        frameon=False,
        fontsize=24
    )
    
    output_path = "evaluation/visuals/complexity_radar_chart.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Radar chart saved to {output_path}")


def create_sunburst_chart():
    print("Generating NL Richness Sunburst Chart...")
    
    languages = ["en", "es", "tr", "ru", "zh", "ja", "hi", "de", "uk", "ro"]
    lang_names = {
        "en": "English", "es": "Spanish", "tr": "Turkish", "ru": "Russian",
        "zh": "Chinese", "ja": "Japanese", "hi": "Hindi", "de": "German",
        "uk": "Ukrainian", "ro": "Romanian"
    }
    dfs = {}
    
    for lang in languages:
        path = f"pipeline/final_translations/translated_{lang}.csv"
        try:
            df = pd.read_csv(path)
            if "nl" in df.columns:
                dfs[lang] = df.dropna(subset=["nl"])
            else:
                print(f"Skipping {lang}: no 'nl' column")
        except FileNotFoundError:
            print(f"Warning: {path} not found.")

    if not dfs:
        print("No data found.")
        return

    print("Calculating NL metrics...")
    
    lang_raw_means = {} 
    g_tokens = []
    g_const = []
    g_sub = []
    g_coord = []
    g_ent = []
    
    for lang, df in dfs.items():
        print(f"Processing {lang}...")
        try:
            nlp_model = get_nlp(lang)
        except Exception as e:
            print(f"Skipping {lang}: could not load model ({e})")
            continue
            
        # Use nlp.pipe for efficiency
        docs = list(nlp_model.pipe(df['nl'].astype(str), batch_size=50))
        
        l_tokens = []
        l_const = []
        l_sub = []
        l_coord = []
        
        for doc in docs:
            metrics = calculate_syntactic_complexity(doc, use_normalization=False)
            l_tokens.append(metrics['tokens_before_main_verb'])
            l_const.append(metrics['constituents_per_word'])
            l_sub.append(metrics['subordinate_clauses'])
            l_coord.append(metrics['coordinate_clauses'])
            
        l_tokens = np.array(l_tokens)
        l_const = np.array(l_const)
        l_sub = np.array(l_sub)
        l_coord = np.array(l_coord)
        
        lang_raw_means[lang] = {
            "Tokens Before Verb": l_tokens.mean(),
            "Constituents": l_const.mean(),
            "Subordinate": l_sub.mean(),
            "Coordinate": l_coord.mean(),
        }
        
        g_tokens.extend(l_tokens)
        g_const.extend(l_const)
        g_sub.extend(l_sub)
        g_coord.extend(l_coord)

    # If no languages processed successfully, return
    if not g_tokens:
        print("No languages processed successfully.")
        return

    g_tokens = np.array(g_tokens)
    g_const = np.array(g_const)
    g_sub = np.array(g_sub)
    g_coord = np.array(g_coord)
    g_ent = np.array(g_ent)
    
    global_stats = {
        "Tokens Before Verb": {"min": g_tokens.min(), "max": g_tokens.max()},
        "Constituents": {"min": g_const.min(), "max": g_const.max()},
        "Subordinate": {"min": g_sub.min(), "max": g_sub.max()},
        "Coordinate": {"min": g_coord.min(), "max": g_coord.max()},
    }

    metric_names = ["Tokens Before Verb", "Constituents", "Subordinate", "Coordinate"]
    
    custom_colors = [
        "#A2BDFF", # Lighter Blue
        "#FFA066", # Lighter Orange
        "#AE9EFF", # Lighter Purple
        "#EA7DB1", # Lighter Magenta
    ]
    metric_colors = {name: custom_colors[i] for i, name in enumerate(metric_names)}
    
    inner_fill_color = "#FF6B6B" # Lighter Adobe Red
    
    inner_values = []
    inner_labels = []
    inner_colors = []
    
    outer_values = []
    outer_labels = []
    outer_colors = []
    
    def norm(val, method, st):
        if method == "log_plus_one":
            base = np.log1p(st["max"]) if st["max"] else 1.0
            return np.log1p(val) / (base if base > 0 else 1.0)
        elif method == "minmax":
            denom = st["max"] - st["min"]
            return (val - st["min"]) / (denom if denom > 0 else 1.0)
        return val

    for lang in languages:
        if lang not in lang_raw_means: continue
        raw = lang_raw_means[lang]
        
        # Normalize
        n_tok = norm(raw["Tokens Before Verb"], "log_plus_one", global_stats["Tokens Before Verb"])
        n_con = norm(raw["Constituents"], "minmax", global_stats["Constituents"])
        n_sub = norm(raw["Subordinate"], "log_plus_one", global_stats["Subordinate"])
        n_coo = norm(raw["Coordinate"], "log_plus_one", global_stats["Coordinate"])
        
        # Weights
        vals = {
            "Tokens Before Verb": n_tok * 0.15,
            "Constituents": n_con * 0.30,
            "Subordinate": n_sub * 0.35,
            "Coordinate": n_coo * 0.20,
        }
        
        total = sum(vals.values())
        
        inner_values.append(total)
        inner_labels.append(lang_names[lang])
        inner_colors.append(inner_fill_color)
        
        for name in metric_names:
            v = vals[name]
            outer_values.append(v)
            outer_labels.append("") 
            outer_colors.append(metric_colors[name])

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.axis('equal')
    
    # Inner Pie
    wedges_in, texts_in = ax.pie(
        inner_values,
        radius=0.7,
        labels=inner_labels,
        labeldistance=0.78,
        colors=inner_colors,
        wedgeprops=dict(width=0.3, edgecolor='w'),
        rotatelabels=True
    )

    for text in texts_in:
        text.set_fontsize(18)
        text.set_horizontalalignment('center')
        text.set_verticalalignment('center')
    
    # Outer Pie
    wedges_out, texts_out = ax.pie(outer_values, radius=1.0, labels=None, 
           colors=outer_colors, wedgeprops=dict(width=0.3, edgecolor='w'))
           
    # Legend
    legend_elements = [
        patches.Patch(
            facecolor=metric_colors[n],
            label=n,
            edgecolor="#1A1A1A",
            linewidth=1.0
        )
        for n in metric_names
    ]
    plt.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(0.85, 1),
        title="NL Richness Metrics",
        frameon=False,
        fontsize=24,
        title_fontsize=24
    )
    
    output_path = "evaluation/visuals/complexity_sunburst.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Sunburst chart saved to {output_path}")

def create_bimodal_histogram_plot():
    df = pd.read_csv('data/cot_dataset_with_corruptions.csv')

    df['Data Source'] = df['is_corrupted'].map({False: 'Original BIRD Dataset', True: 'Corrupted Version'})

    colors = {'Original BIRD Dataset': '#648FFF', 'Corrupted Version': '#DC267F'} 

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        data=df,
        x='similarity_with_penalty',
        hue='Data Source',
        palette=colors,
        kde=True,
        bins=30,
        ax=ax,
        alpha=0.6
    )
    
    sns.despine()
    
    sns.move_legend(
        ax, "upper left",
        bbox_to_anchor=(1, 1),
        title=None,
        frameon=False,
    )

    ax.set_xlabel("Similarity Score", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    
    plt.tight_layout()
    output_path = "evaluation/visuals/bimodal_histogram_plot.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Bimodal histogram saved to {output_path}") 

def reward_model_performance_plot():
    metrics_path = "filtering_experiments/useful_eval_results_and_scripts/all_experiments_metrics.csv"
    df = pd.read_csv(metrics_path)

    wanted_ids = [
        'finetuned_qwen',
        'llama_teacher',
        'reward_threshold_0.6',
        'prompt_filter_threshold_0.8'
    ]
    
    def norm(val, method, mn, mx):
        if method == "log_plus_one":
            base = np.log1p(mx) if mx else 1.0
            denom = base if base > 0 else 1.0
            return np.log1p(val) / denom
        elif method == "minmax":
            denom = mx - mn
            return (val - mn) / (denom if denom > 0 else 1.0)
        return val

    # Calculate statistics across the entire dataset (all experiments)
    nl_stats = {
        "tokens": {"min": df["nl_kept_avg_tokens_before_verb"].min(), "max": df["nl_kept_avg_tokens_before_verb"].max()},
        "const": {"min": df["nl_kept_avg_constituents_per_word"].min(), "max": df["nl_kept_avg_constituents_per_word"].max()},
        "sub": {"min": df["nl_kept_avg_subordinate_clauses"].min(), "max": df["nl_kept_avg_subordinate_clauses"].max()},
        "coord": {"min": df["nl_kept_avg_coordinate_clauses"].min(), "max": df["nl_kept_avg_coordinate_clauses"].max()},
    }
    sql_stats = {
        "words": {"min": df["sql_kept_avg_words"].min(), "max": df["sql_kept_avg_words"].max()},
        "cols": {"min": df["sql_kept_avg_columns"].min(), "max": df["sql_kept_avg_columns"].max()},
        "subq": {"min": df["sql_kept_avg_subqueries"].min(), "max": df["sql_kept_avg_subqueries"].max()},
        "ops": {"min": df["sql_kept_avg_operators"].min(), "max": df["sql_kept_avg_operators"].max()},
        "splits": {"min": df["sql_kept_avg_splits"].min(), "max": df["sql_kept_avg_splits"].max()},
    }

    # Normalize NL metrics for all rows
    df['norm_nl_tokens'] = df['nl_kept_avg_tokens_before_verb'].apply(lambda x: norm(x, 'log_plus_one', nl_stats['tokens']['min'], nl_stats['tokens']['max']))
    df['norm_nl_const'] = df['nl_kept_avg_constituents_per_word'].apply(lambda x: norm(x, 'minmax', nl_stats['const']['min'], nl_stats['const']['max']))
    df['norm_nl_sub'] = df['nl_kept_avg_subordinate_clauses'].apply(lambda x: norm(x, 'log_plus_one', nl_stats['sub']['min'], nl_stats['sub']['max']))
    df['norm_nl_coord'] = df['nl_kept_avg_coordinate_clauses'].apply(lambda x: norm(x, 'log_plus_one', nl_stats['coord']['min'], nl_stats['coord']['max']))
    
    df['nl_richness_score'] = (
        df['norm_nl_tokens'] * 0.15 + 
        df['norm_nl_const'] * 0.30 + 
        df['norm_nl_sub'] * 0.35 + 
        df['norm_nl_coord'] * 0.20
    )

    # Normalize SQL metrics for all rows
    df['norm_sql_words'] = df['sql_kept_avg_words'].apply(lambda x: norm(x, 'log_plus_one', sql_stats['words']['min'], sql_stats['words']['max']))
    df['norm_sql_cols'] = df['sql_kept_avg_columns'].apply(lambda x: norm(x, 'minmax', sql_stats['cols']['min'], sql_stats['cols']['max']))
    df['norm_sql_subq'] = df['sql_kept_avg_subqueries'].apply(lambda x: norm(x, 'log_plus_one', sql_stats['subq']['min'], sql_stats['subq']['max']))
    df['norm_sql_ops'] = df['sql_kept_avg_operators'].apply(lambda x: norm(x, 'log_plus_one', sql_stats['ops']['min'], sql_stats['ops']['max']))
    df['norm_sql_splits'] = df['sql_kept_avg_splits'].apply(lambda x: norm(x, 'log_plus_one', sql_stats['splits']['min'], sql_stats['splits']['max']))
    
    df['sql_complexity_score'] = (
        df['norm_sql_words'] * 0.10 + 
        df['norm_sql_cols'] * 0.15 + 
        df['norm_sql_subq'] * 0.30 + 
        df['norm_sql_ops'] * 0.25 + 
        df['norm_sql_splits'] * 0.20
    )

    # Filter to selected experiments
    selected = (
        df.set_index('experiment')
        .loc[wanted_ids]
    )

    selected = selected.assign(
        recall=selected['recall'],
        precision=selected['precision'],
        f1=selected['f1_score'],
        keep=selected['keep_rate'],
        nl_richness=selected['nl_richness_score'],
        sql_complexity=selected['sql_complexity_score'],
        specificity=selected['specificity']
    )
    metric_columns = ['f1', 'precision', 'recall', 'specificity', 'keep', 'nl_richness', 'sql_complexity']

    experiments = [
        'Finetuned Qwen2.5-3B-Instruct',
        'Llama-3.3-70B-Versatile',
        'Reward',
        'Prompt'
    ]

    metrics = ["F1 Score", "Precision", "Recall", "Specificity", "Keep Rate", "NL Richness", "SQL Complexity"]

    data = selected[metric_columns].to_numpy()

    bar_width = 0.15
    x = np.arange(len(metrics))

    colors = ["#a2bbffff", "#9a87e0ff", "#ea7db2ff", "#fccab5ff"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, exp in enumerate(experiments):
        ax.bar(
            x + i * bar_width,
            data[i],
            width=bar_width,
            label=exp,
            color=colors[i],
            edgecolor="#1A1A1A",
            linewidth=1.0
        )

    metric_centers = x + bar_width * 1.5
    metric_mean = data.mean(axis=0)
    ax.plot(
        metric_centers,
        metric_mean,
        color="#FFD60A",
        linewidth=2.5,
        alpha=0.85,
        linestyle="-",
        marker="o",
        markerfacecolor="#FFD60A",
        markeredgecolor="#FFD60A",
        label="Metric Mean",
        markersize=8
    )

    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(metrics, fontsize=16, rotation=18)
    ax.set_ylabel("Score", fontsize=16)
    ax.set_ylim(0, 1.05)

    ax.set_axisbelow(True)

    sns.despine(ax=ax)
    legend = ax.legend(
        title=None,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=20
    )

    plt.tight_layout()
    output_path = "evaluation/visuals/reward_model_performance_plot.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Reward model performance plot saved to {output_path}")

def reserved_length_distribution_plot():
    csv_path = "data_gen/length_experiment/reserved_length_probabilities.csv"
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df,
        x="length",
        y="probability",
        color="#648FFF",
        edgecolor="#1A1A1A",
        linewidth=0.8
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.xlim(0, 25)

    output_path = "evaluation/visuals/reserved_length_distribution.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Reserved length distribution plot saved to {output_path}")


if __name__ == "__main__":
    # These are the charts on our poster
    create_radar_chart()
    # create_sunburst_chart()
    # reward_model_performance_plot()
    # reserved_length_distribution_plot()
