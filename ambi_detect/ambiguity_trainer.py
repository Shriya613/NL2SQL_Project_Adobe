
import argparse
import ast
import os
import random
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_first_sql(sql_field: str) -> str:
    """
    Many AMBROSIA rows store SQL as a stringified Python list with escaped CRLFs.
    This function tries to parse that and return the first query text.
    """
    if sql_field is None or (isinstance(sql_field, float) and np.isnan(sql_field)):
        return ""
    s = str(sql_field).strip()
    # Heuristic: if it looks like a list (starts with [ and ends with ]), parse it
    if s.startswith('[') and s.endswith(']'):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list) and len(parsed) > 0:
                return str(parsed[0])
        except Exception:
            pass
    return s


def pick_text_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Pick (nl_col, sql_col) from a DataFrame with best-effort fallbacks.
    """
    nl_candidates = ["question", "ambig_question", "nl", "utterance"]
    sql_candidates = ["gold_queries", "gold_query", "sql", "query"]

    nl_col = next((c for c in nl_candidates if c in df.columns), None)
    sql_col = next((c for c in sql_candidates if c in df.columns), None)

    if nl_col is None:
        raise ValueError(f"Could not find any NL column. Looked for: {nl_candidates}")

    if sql_col is None:
        raise ValueError(f"Could not find any SQL column. Looked for: {sql_candidates}")

    return nl_col, sql_col


def _safe_parse_listlike(value) -> Optional[List[str]]:
    """
    Try to parse a value into a list of non-empty strings.
    Handles:
      - actual Python lists
      - stringified Python lists like "['SELECT ...', 'SELECT ...']"
      - plain strings (return None here, handled elsewhere)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    # Already a list
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip() != ""]

    # Maybe a stringified list
    s = str(value).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip() != ""]
        except Exception:
            return None

    # Not list-like (scalar string)
    return None



def _count_gold_queries(df: pd.DataFrame) -> pd.Series:
    """
    Count how many gold queries a row has, preferring list-style columns if present.
    Fallback: scalar SQL string => count=1 if non-empty else 0.
    """
    # prefer list-like columns if they exist
    candidate_cols = [c for c in ["gold_queries", "gold_query", "sql", "query"] if c in df.columns]
    if not candidate_cols:
        # nothing to count; return ones (treat each row as single-query)
        return pd.Series([1] * len(df), index=df.index, dtype=int)

    def row_count(row) -> int:
        for c in candidate_cols:
            val = row[c]
            parsed_list = _safe_parse_listlike(val)
            if parsed_list is not None:
                return max(0, len(parsed_list))
            # If not list-like, treat as scalar string in this column and return 1 if non-empty
            s = "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val).strip()
            if s != "":
                return 1
        return 0

    return df.apply(row_count, axis=1).astype(int)


def build_label_series(df: pd.DataFrame) -> pd.Series:
    """
    Construct a 0–10 ambiguity score using BOTH:
      - 'gold_query' (canonical interpretation)
      - 'ambig_queries' (additional valid interpretations)
      - 'is_ambiguous' (boolean flag)
    Scheme:
      - If is_ambiguous == False  -> score = 0
      - If is_ambiguous == True   -> score in [5, 10] based on how many distinct SQLs there are
        among gold_query + ambig_queries, normalized over only the ambiguous rows.
    """

    # ---------- 1) Count distinct SQLs per row ----------
    def row_num_distinct_sqls(row) -> int:
        sql_set = set()

        # gold_query: scalar or list-like
        if "gold_query" in df.columns:
            val = row["gold_query"]
            parsed_list = _safe_parse_listlike(val)
            if parsed_list is not None:
                sql_set.update(parsed_list)
            else:
                s = "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val).strip()
                if s != "":
                    sql_set.add(s)

        # ambig_queries: typically a list / stringified list
        if "ambig_queries" in df.columns:
            val = row["ambig_queries"]
            parsed_list = _safe_parse_listlike(val)
            if parsed_list is not None:
                sql_set.update(parsed_list)
            else:
                s = "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val).strip()
                if s != "":
                    sql_set.add(s)

        # Fallback: if both are empty, but there is some 'sql'/'query' column, treat as 1
        if len(sql_set) == 0:
            for c in ["sql", "query"]:
                if c in df.columns:
                    base = row[c]
                    if base is not None and not (isinstance(base, float) and np.isnan(base)):
                        s = str(base).strip()
                        if s != "":
                            return 1
            return 0

        return len(sql_set)

    counts = df.apply(row_num_distinct_sqls, axis=1).astype(int)

    # ---------- 2) Normalize only among ambiguous rows ----------
    # Normalize counts over ambiguous subset, map to [5, 10]
    if "is_ambiguous" in df.columns:
        def norm_bool(x):
            if isinstance(x, str):
                return x.strip().lower() in ["true", "1", "yes", "y", "t"]
            return bool(x)

        is_amb = df["is_ambiguous"].apply(norm_bool).astype(bool)
    else:
        # If no flag exists, treat everything as ambiguous
        is_amb = pd.Series([True] * len(df), index=df.index)

    scores = pd.Series(0.0, index=df.index, dtype=float)

    # Only look at rows annotated as ambiguous
    amb_indices = is_amb[is_amb].index
    if len(amb_indices) > 0:
        amb_counts = counts.loc[amb_indices]
        max_amb_count = int(amb_counts.max())
        min_amb_count = int(amb_counts.min())

        if max_amb_count <= 1:
            # All ambiguous rows effectively have one interpretation; just set them to 10
            scores.loc[amb_indices] = 10.0
        else:
            # Map [min_amb_count, max_amb_count] -> [5, 10]
            # To be safe, ensure denominator > 0
            denom = max(max_amb_count - min_amb_count, 1)
            normalized = 5.0 + 5.0 * (amb_counts - min_amb_count) / float(denom)
            scores.loc[amb_indices] = normalized.astype(float)

    # Unambiguous rows remain at 0.0
    return scores.clip(lower=0.0, upper=10.0)





# ---------------------------
# Dataset
# ---------------------------

class NLSQLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256, with_labels: bool = True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.with_labels = with_labels

        nl_col, sql_col = pick_text_columns(df)

        # Clean text
        self.nls: List[str] = df[nl_col].fillna("").astype(str).tolist()
        # Special handling for SQL strings (list-like / CRLFs)
        self.sqls: List[str] = [safe_first_sql(s) for s in df[sql_col].tolist()]

        # Optional label
        self.labels = None
        if with_labels:
            labels = build_label_series(df)
            self.labels = labels.astype(float).tolist()

    def __len__(self):
        return len(self.nls)

    def __getitem__(self, idx):
        nl = self.nls[idx].strip()
        sql = self.sqls[idx].strip()

        text = f"NL: {nl} [SEP] SQL: {sql}"
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.with_labels and self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


# ---------------------------
# Model
# ---------------------------

class MeanPoolerRegressor(nn.Module):
    """
    A small regression head on top of a Transformer encoder.
    Uses simple mean pooling over the last_hidden_state to be compatible with
    models that do not expose a pooler_output (e.g., DistilBERT).
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(p=0.1)
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        # masked mean
        summed = torch.sum(last_hidden * mask, dim=1)  # (B, H)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)  # (B, 1)
        mean_pooled = summed / counts
        x = self.dropout(mean_pooled)
        score = self.regressor(x).squeeze(-1)  # (B,)
        return score


# ---------------------------
# Training / Evaluation
# ---------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        preds = model(input_ids, attention_mask)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * input_ids.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    running = 0.0
    for batch in tqdm(loader, desc="Val  ", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds = model(input_ids, attention_mask)
        loss = loss_fn(preds, labels)
        running += loss.item() * input_ids.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []
    for batch in tqdm(loader, desc="Predict", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        preds = model(input_ids, attention_mask)          # (B,)
        preds = torch.clamp(preds, 0.0, 10.0)             # clamp in torch
        # Convert to plain Python floats without using NumPy
        all_preds.extend([float(x) for x in preds.detach().cpu().flatten()])
    return all_preds



# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train an ambiguity regressor for NL–SQL pairs.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to ambrosia-like CSV (has labels).")
    parser.add_argument("--infer_csv", type=str, required=True, help="Path to generated_sql_nl.csv.")
    parser.add_argument("--output_csv", type=str, default="predicted_ambiguity.csv", help="Where to save predictions.")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="HF model name.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction of training for validation.")
    parser.add_argument("--save_dir", type=str, default="ambig_model", help="Directory to save model + tokenizer.")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training CSV
    train_df = pd.read_csv(args.train_csv)
    # Sanity check columns and build labels
    _ = pick_text_columns(train_df)  # raises if not found
    labels = build_label_series(train_df)
    train_df_with_scores = train_df.copy()
    train_df_with_scores["ambiguity_score_normalized"] = labels

    train_df_with_scores.to_csv("training_scores_debug.csv", index=False)
    print("\nSaved training scores to training_scores_debug.csv")

    plt.hist(labels, bins=20)
    plt.xlabel("Ambiguity Score (0–10)")
    plt.ylabel("Frequency")
    plt.title("Training Set Ambiguity Score Distribution")
    plt.savefig("score_histogram.png")
    print("\nSaved score histogram to score_histogram.png")
    plt.close()



    if labels.isna().any():
        raise ValueError("Unable to build labels for some rows in train_csv. Please check columns.")

    # Tokenizer / Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    full_ds = NLSQLDataset(train_df, tokenizer, max_len=args.max_len, with_labels=True)

    # Split into train/val
    n_total = len(full_ds)
    n_val = int(n_total * args.val_split)
    n_train = max(1, n_total - n_val)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if n_val > 0 else None

    # Model
    model = MeanPoolerRegressor(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Train
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        if val_loader is not None:
            val_loss = eval_one_epoch(model, val_loader, loss_fn, device)
            print(f"Train MSE: {tr_loss:.4f} | Val MSE: {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                os.makedirs(args.save_dir, exist_ok=True)
                model_state = {
                    "model_name": args.model_name,
                    "state_dict": model.state_dict(),
                }
                torch.save(model_state, os.path.join(args.save_dir, "pytorch_model.bin"))
                tokenizer.save_pretrained(args.save_dir)
        else:
            print(f"Train MSE : {tr_loss:.4f}")

    # If not saved during training (e.g., no val split), save now
    if not os.path.exists(os.path.join(args.save_dir, "pytorch_model.bin")):
        os.makedirs(args.save_dir, exist_ok=True)
        model_state = {"model_name": args.model_name, "state_dict": model.state_dict()}
        torch.save(model_state, os.path.join(args.save_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(args.save_dir)

    # ---------------- Inference ----------------
    infer_df = pd.read_csv(args.infer_csv)
    # Try to pick columns, but if this is missing ambrosia-style names, fall back to generated_sql_nl names
    try:
        _ = pick_text_columns(infer_df)
    except ValueError:
        # Common names in your provided generated file
        if "nl" in infer_df.columns and "sql" in infer_df.columns:
            pass
        else:
            raise

    infer_ds = NLSQLDataset(infer_df, tokenizer, max_len=args.max_len, with_labels=False)
    infer_loader = DataLoader(infer_ds, batch_size=args.batch_size, shuffle=False)

    # Load best model if available
    best_model_path = os.path.join(args.save_dir, "pytorch_model.bin")
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        if "model_name" in ckpt and ckpt["model_name"] != args.model_name:
            print(f"Warning: checkpoint model_name={ckpt['model_name']} != current {args.model_name}. Loading anyway.")
        model.load_state_dict(ckpt["state_dict"])

    preds = predict(model, infer_loader, device)

    # Save predictions
    # Reconstruct NL/SQL text for output
    nl_col, sql_col = pick_text_columns(infer_df)
    out_df = pd.DataFrame({
        "nl": infer_df[nl_col].astype(str).tolist(),
        "sql": [safe_first_sql(s) for s in infer_df[sql_col].tolist()],
        "predicted_ambiguity": preds
    })
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to: {args.output_csv}")
    print(f"Saved model to: {args.save_dir}")


if __name__ == "__main__":
    main()
