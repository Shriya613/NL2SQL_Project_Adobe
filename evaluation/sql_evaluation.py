from eval_metrics import (
    avg_operators,
    avg_splits,
    avg_words,
    avg_columns_used,
    avg_subqueries,
    words,
    columns_used,
    subqueries as subquery_counts,
    operators as operator_counts,
    splits as split_counts,
)
import pandas as pd
import numpy as np


def normalize_metric(value: float, method: str = "minmax", stats: dict | None = None) -> float:
    """Normalize a scalar using the same strategies as NL evaluation."""
    stats = stats or {}
    if method == "log_plus_one":
        baseline = np.log1p(stats.get("max", value)) if stats.get("max") else 1.0
        denom = baseline if baseline > 0 else 1.0
        return np.log1p(value) / denom
    if method == "zscore":
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)
        std = std if std not in (0, None) else 1.0
        return (value - mean) / std
    # default min-max
    min_val = stats.get("min", 0.0)
    max_val = stats.get("max", 1.0)
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def evaluate_sql(
    dataset,
    sql_column,
    use_normalization: bool = True,
    normalization_stats: dict | None = None,
):
    """Calculate SQL/NL dataset complexity with optional normalization."""

    df = pd.read_csv(dataset).dropna(subset=[sql_column])
    num_rows = len(df)

    # Raw per-query metrics
    raw_sql_words = words(df, sql_column)
    raw_columns = columns_used(df, sql_column)
    raw_subqueries = subquery_counts(df, sql_column)
    raw_operators = operator_counts(df, sql_column)
    raw_splits = split_counts(df, sql_column)

    sql_words = raw_sql_words.mean()
    columns = raw_columns.mean()
    subqueries = raw_subqueries.mean()
    operators = raw_operators.mean()
    splits = raw_splits.mean()

    if use_normalization:
        normalization_stats = normalization_stats or {
            "sql_words": {
                "min": raw_sql_words.min(),
                "max": raw_sql_words.max(),
                "mean": raw_sql_words.mean(),
                "std": raw_sql_words.std(),
            },
            "columns": {
                "min": raw_columns.min(),
                "max": raw_columns.max(),
            },
            "subqueries": {
                "min": raw_subqueries.min(),
                "max": raw_subqueries.max(),
            },
            "operators": {
                "min": raw_operators.min(),
                "max": raw_operators.max(),
            },
            "splits": {
                "min": raw_splits.min(),
                "max": raw_splits.max(),
            },
        }

        normalized_sql_words = normalize_metric(
            sql_words, method="log_plus_one", stats=normalization_stats["sql_words"]
        )
        normalized_columns = normalize_metric(
            columns, method="minmax", stats=normalization_stats["columns"]
        )
        normalized_subqueries = normalize_metric(
            subqueries, method="log_plus_one", stats=normalization_stats["subqueries"]
        )
        normalized_operators = normalize_metric(
            operators, method="log_plus_one", stats=normalization_stats["operators"]
        )
        normalized_splits = normalize_metric(
            splits, method="log_plus_one", stats=normalization_stats["splits"]
        )

        complexity_score = (
            normalized_sql_words * 0.10
            + normalized_columns * 0.15
            + normalized_subqueries * 0.30
            + normalized_operators * 0.25
            + normalized_splits * 0.20
        )
    else:
        complexity_score = (
            sql_words * 0.10
            + columns * 0.15
            + subqueries * 0.30
            + operators * 0.25
            + splits * 0.20
        )

    print(f"Number of rows: {num_rows}")
    print(f"Average number of words in SQL queries: {sql_words:.2f}")
    print(f"Average number of splits in SQL queries: {splits:.2f}")
    print(f"Average number of columns in SQL queries: {columns:.2f}")
    print(f"Average number of operators in SQL queries: {operators:.2f}")
    print(f"Average number of subqueries in SQL queries: {subqueries:.2f}")
    print(f"Composite complexity score ({'normalized' if use_normalization else 'raw'}): {complexity_score:.2f}")

if __name__ == "__main__":
    evaluate_sql("data_gen/persona_experiment/persona_seeding.csv", "actual_sql")