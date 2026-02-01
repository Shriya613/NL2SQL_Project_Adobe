"""Create a dataset by prompting an LLM to convert SQL into NL with CoT.

Outputs a CSV with columns: sql, reasoning, predicted_nl, true_nl, simcse_similarity.
Uses PyTorch on GPU with SimCSE for similarity computation.
"""

import json
import os
import torch
from typing import Tuple
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# Handle both relative and absolute imports
try:
    from .prompts import GEN_PROMPT_TEMPLATE, VAL_PROMPT_TEMPLATE
    from .db_corruption_values import DB_PLAUSIBLE_CORRUPTED_VALUES
except ImportError:
    from prompts import GEN_PROMPT_TEMPLATE, VAL_PROMPT_TEMPLATE
    from db_corruption_values import DB_PLAUSIBLE_CORRUPTED_VALUES

def build_generation_prompt(sql: str) -> str:
    return GEN_PROMPT_TEMPLATE.format(sql=sql)

def build_validation_prompt(sql: str, cot: str, nl: str) -> str:
    return VAL_PROMPT_TEMPLATE.format(sql=sql, cot=cot, nl=nl)





class SentenceBERTSimilarity:
    """Sentence-BERT-based similarity computation using PyTorch on GPU."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"🔄 Loading Sentence-BERT model: {model_name}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = "cuda"
            print("✅ Using GPU for Sentence-BERT")
        else:
            self.device = "cpu"
            print("⚠️  Using CPU for Sentence-BERT")
    
    def encode_texts(self, texts):
        """Encode texts using Sentence-BERT model."""
        # Use Sentence-BERT's built-in encoding
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings
    
    def compute_similarity(self, text1, text2):
        """Compute similarity between two texts."""
        if not text1.strip() or not text2.strip():
            return 0.0
            
        embeddings = self.encode_texts([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def extract_sql_entities(self, sql):
        """Extract and enrich SQL entities for semantic matching."""
        if not sql:
            return set()
        
        # SQL reserved words to filter out
        sql_reserved = {
            'select', 'from', 'where', 'order', 'by', 'group', 'having', 'limit', 'offset',
            'insert', 'update', 'delete', 'create', 'drop', 'alter', 'table', 'index',
            'and', 'or', 'not', 'in', 'like', 'between', 'is', 'null', 'exists',
            'count', 'sum', 'avg', 'min', 'max', 'distinct', 'as', 'asc', 'desc',
            'inner', 'left', 'right', 'outer', 'join', 'on', 'union', 'all',
            'case', 'when', 'then', 'else', 'end', 'if', 'while', 'for',
            '=', '!=', '<>', '<', '>', '<=', '>=', '(', ')', ',', ';', '+', '-', '*', '/', '%'
        }
        
        # Common table aliases to filter out
        import re
        table_aliases = set()
        # Match patterns like "table AS t1" or "table t1" or just "t1"
        alias_pattern = r'\b(t\d+|t\b|t1|t2|t3|t4|t5|t6|t7|t8|t9)\b'
        table_aliases = set(re.findall(alias_pattern, sql.lower()))
        
        # Extract quoted strings (values)
        quoted_strings = re.findall(r'[\"\']([^\"\']+)[\"\']', sql)
        
        # Remove quoted strings and extract column/table names
        sql_without_quotes = re.sub(r'[\"\']([^\"\']+)[\"\']', '', sql)
        sql_words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', sql_without_quotes.lower())
        
        # Filter out reserved words, table aliases, and keep only potential column/table names
        sql_entities = set()
        for word in sql_words:
            if word not in sql_reserved and word not in table_aliases and len(word) > 1:
                sql_entities.add(word)
        
        # Add quoted strings as entities
        for quoted in quoted_strings:
            sql_entities.add(quoted.lower())
        
        return sql_entities
    
    def enrich_sql_entity(self, entity):
        """Convert SQL entity to natural language phrase for better semantic matching."""
        # Handle quoted strings (values) - return as is
        if any(char in entity for char in ['"', "'"]):
            return entity.strip('"\'')
        
        # Convert snake_case to natural language
        if '_' in entity:
            # Split by underscore and join with spaces
            words = entity.split('_')
        else:
            # Handle camelCase (if any)
            if any(c.isupper() for c in entity[1:]):
                # Insert space before capital letters
                import re
                words = re.sub(r'([a-z])([A-Z])', r'\1 \2', entity).split()
            else:
                words = [entity]
        
        # Remove common suffixes like "name" or "title" for better matching
        # e.g., "director name" -> "director", "movie title" -> "movie"
        # this is to handle this like "who directed it?" "who produced it?" where name is implied but not stated directly
        if words:
            last_word = words[-1].lower()
            if last_word in ['name', 'title'] and len(words) > 1:
                words = words[:-1]
        
        return ' '.join(words)
    
    def compute_entity_similarity(self, text, entity, threshold=0.45):
        """Compute semantic similarity between text and SQL entity using word-level matching."""
        # Enrich the entity to natural language
        enriched_entity = self.enrich_sql_entity(entity)
        
        # Split enriched entity into words
        entity_words = enriched_entity.split()
        
        # If entity is a single word, use word-level similarity matching
        if len(entity_words) == 1:
            # Extract all words from the text (split by spaces, handle punctuation)
            import re
            text_words = re.findall(r'\b\w+\b', text.lower())
            
            if not text_words:
                return 0.0, False
            
            # Encode the entity word and all text words
            all_words = [enriched_entity] + text_words
            embeddings = self.encode_texts(all_words)
            
            # Get entity embedding
            entity_emb = embeddings[0]
            
            # Get text word embeddings
            text_embs = embeddings[1:]
            
            # Find the most similar word in the text
            max_similarity = 0.0
            best_match_idx = -1
            
            for i, text_emb in enumerate(text_embs):
                sim = cosine_similarity([entity_emb], [text_emb])[0][0]
                if sim > max_similarity:
                    max_similarity = sim
                    best_match_idx = i
            
            return float(max_similarity), max_similarity >= threshold
        else:
            # For multi-word entities, use sentence-level similarity
            # Encode both text and enriched entity
            embeddings = self.encode_texts([text, enriched_entity])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Return similarity and whether it exceeds threshold
            return float(similarity), similarity >= threshold
    
    def compute_enhanced_similarity(self, predicted_nl, true_nl, true_sql, corrupted_sql, corruption_count=0, debug=False):
        """Compute enhanced similarity with entity-based rewards and penalties."""
        if debug:
            print(f"\n🔍 COMPUTING ENHANCED SIMILARITY:")
            print(f"Predicted NL: '{predicted_nl}'")
            print(f"True NL: '{true_nl}'")
            print(f"True SQL: '{true_sql}'")
            print(f"Corrupted SQL: '{corrupted_sql}'")
            print(f"Corruption count: {corruption_count}")
        
        # 1. Base BERT similarity between predicted and true NL
        bert_similarity = self.compute_similarity(predicted_nl, true_nl)
        if debug:
            print(f"🤖 BERT similarity: {bert_similarity:.3f}")
        
        # 2. Extract entities from true and corrupted SQL
        true_entities = self.extract_sql_entities(true_sql)
        corrupted_entities = self.extract_sql_entities(corrupted_sql)
        
        # 3. Find entities that are in corrupted but not in true (corruption artifacts)
        corrupted_only_entities = corrupted_entities - true_entities
        
        if debug:
            print(f"🏷️ True entities: {true_entities}")
            print(f"🏷️ Corrupted entities: {corrupted_entities}")
            print(f"⚠️ Corrupted-only entities: {corrupted_only_entities}")
        
        # 4. Compute entity-based scores
        # Track which entities are mentioned in each text
        predicted_nl_mentioned_true_entities = set()
        predicted_nl_mentioned_corrupted_entities = set()
        true_nl_mentioned_true_entities = set()
        true_nl_mentioned_corrupted_entities = set()
        
        # Check each true entity for mention in predicted NL
        for entity in true_entities:
            enriched = self.enrich_sql_entity(entity)
            pred_similarity, pred_mentioned = self.compute_entity_similarity(predicted_nl, entity)
            true_similarity, true_mentioned = self.compute_entity_similarity(true_nl, entity)
            
            # Use relative similarity thresholds to override basic threshold
            # If true NL has high similarity (>0.7) but predicted has low (<0.5), consider it missing in predicted
            if true_similarity > 0.7 and pred_similarity < 0.5 or (true_similarity - pred_similarity > 0.3):
                # Override: don't add to predicted_mentioned even if it technically passed threshold
                if debug:
                    print(f"❌ True entity '{entity}' -> '{enriched}' effectively missing in predicted (true: {true_similarity:.3f}, pred: {pred_similarity:.3f})")
            elif pred_mentioned:
                predicted_nl_mentioned_true_entities.add(entity)
                if debug:
                    print(f"✅ True entity '{entity}' -> '{enriched}' mentioned in predicted (sim: {pred_similarity:.3f})")
            else:
                if debug:
                    print(f"❌ True entity '{entity}' -> '{enriched}' missing in predicted (sim: {pred_similarity:.3f})")
            
            # Add to true_mentioned if it meets the threshold
            if true_mentioned:
                true_nl_mentioned_true_entities.add(entity)
                if debug:
                    print(f"📝 True entity '{entity}' -> '{enriched}' mentioned in true NL (sim: {true_similarity:.3f})")
            else:
                if debug:
                    print(f"📝 True entity '{entity}' -> '{enriched}' missing in true NL (sim: {true_similarity:.3f})")
        
        # Check each corrupted-only entity for mention in predicted NL
        for entity in corrupted_only_entities:
            enriched = self.enrich_sql_entity(entity)
            pred_similarity, pred_mentioned = self.compute_entity_similarity(predicted_nl, entity)
            true_similarity, true_mentioned = self.compute_entity_similarity(true_nl, entity)
            
            # Use relative similarity thresholds for corrupted entities
            # If predicted has high similarity (>0.7) but true has low (<0.5), consider it only in predicted
            if pred_similarity > 0.7 and true_similarity < 0.5:
                predicted_nl_mentioned_corrupted_entities.add(entity)
                if debug:
                    print(f"⚠️ Corrupted entity '{entity}' -> '{enriched}' only mentioned in predicted (pred: {pred_similarity:.3f}, true: {true_similarity:.3f})")
            elif pred_mentioned:
                predicted_nl_mentioned_corrupted_entities.add(entity)
                if debug:
                    print(f"⚠️ Corrupted entity '{entity}' -> '{enriched}' mentioned in predicted (sim: {pred_similarity:.3f})")
            else:
                if debug:
                    print(f"✅ Corrupted entity '{entity}' -> '{enriched}' not mentioned in predicted (sim: {pred_similarity:.3f})")
            
            # Add to true_mentioned if it meets the threshold
            if true_mentioned:
                true_nl_mentioned_corrupted_entities.add(entity)
                if debug:
                    print(f"⚠️ Corrupted entity '{entity}' -> '{enriched}' mentioned in true NL (sim: {true_similarity:.3f})")
            else:
                if debug:
                    print(f"✅ Corrupted entity '{entity}' -> '{enriched}' not mentioned in true NL (sim: {true_similarity:.3f})")
        
        # 5. Compute entity-based score
        entity_score = 1.0
        
        # Penalty 1: Missing true entities that ARE mentioned in true NL
        # Only penalize if the true NL actually mentions the entity
        missing_true_entities = true_nl_mentioned_true_entities - predicted_nl_mentioned_true_entities
        entity_score = entity_score * .7 ** (len(missing_true_entities))
        
        # Penalty 2: Mentioning corrupted entities that are NOT mentioned in true NL
        # Only penalize if the true NL doesn't mention the corrupted entity
        bad_corrupted_entities = predicted_nl_mentioned_corrupted_entities - true_nl_mentioned_corrupted_entities
        entity_score = entity_score * .7 ** (len(bad_corrupted_entities))
        
        if debug:
            print(f"🔍 Missing true entities (mentioned in true NL but not predicted): {missing_true_entities}")
            print(f"🔍 Bad corrupted entities (mentioned in predicted but not true NL): {bad_corrupted_entities}")
            print(f"📊 Entity score: {entity_score:.3f}")
        
        # 6. Compute final enhanced similarity
        # 40% sentence similarity + 60% entity score
        final_similarity = bert_similarity * 0.4 + entity_score * 0.6
        
        # Ensure similarity is between 0 and 1
        final_similarity = max(0.0, min(1.0, final_similarity))
        
        if debug:
            print(f"🎯 Final enhanced similarity: {final_similarity:.3f}")
            print(f"   Sentence similarity (40%): {bert_similarity:.3f} × 0.4 = {bert_similarity * 0.4:.3f}")
            print(f"   Entity score (60%): {entity_score:.3f} × 0.6 = {entity_score * 0.6:.3f}")
        
        return final_similarity

    def compute_similarity_with_corruption_penalty(self, text1, text2, corruption_count=0, sql1=None, sql2=None, debug=False):
        """Compute similarity with systematic penalty for corruptions using enhanced entity-based scoring."""
        if not text1.strip() or not text2.strip():
            return 0.0
        
        # If we have both SQLs, use enhanced similarity system
        if sql1 and sql2:
            return self.compute_enhanced_similarity(text1, text2, sql1, sql2, corruption_count, debug)
        
        # Fallback to original system if SQLs not provided
        if debug:
            print(f"\n🔍 COMPUTING SIMILARITY (FALLBACK):")
            print(f"Text 1: '{text1}'")
            print(f"Text 2: '{text2}'")
            print(f"Corruption count: {corruption_count}")
            
        # Fallback to simple BERT similarity if SQLs not provided
        embeddings = self.encode_texts([text1, text2])
        bert_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        if debug:
            print(f"🤖 BERT similarity: {bert_similarity:.3f}")
        
        # Apply penalty for corruptions
        if corruption_count > 0:
            penalty = 0.1 * corruption_count  # 10% per corruption
            final_similarity = bert_similarity * (1 - penalty)
            if debug:
                print(f"⚠️ Corruption penalty: {penalty:.3f} (10% per corruption)")
                print(f"📉 Final similarity: {final_similarity:.3f}")
            return float(max(0.0, final_similarity))  # Don't go below 0
        
        if debug:
            print(f"✅ Final similarity: {bert_similarity:.3f}")
        
        return float(bert_similarity)

def run_llm_pytorch(model, tokenizer, prompt: str) -> str:
    """Run inference using PyTorch transformers pipeline."""
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,  # Much lower temperature for more deterministic output
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Stop at conversation markers to prevent dialogue loops
        for stop_marker in ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]:
            if stop_marker in generated_text:
                generated_text = generated_text.split(stop_marker)[0]
                break
        
        return generated_text.strip()
    except Exception as e:
        print(f"⚠️  PyTorch generation failed: {e}")
        return ""

import random
from sqlparse import parse, tokens as T
import sqlparse

import random
import re
import sqlparse

def corrupt_sql_query(sql: str, operation: str, db_id: str = None) -> str:
    """
    Apply a specific corruption operation to the SQL query.
    Produces realistic SQL-like output with wide lexical diversity.
    """
    sql = sql.strip().rstrip(";")
    corrupted = sql
    print(f"🔧 Applying operation: {operation}")

    op = operation

    # --- SQL operators ---
    operators = [
        "=", "!=", ">", "<", ">=", "<=", "LIKE", "NOT LIKE",
        "IN", "NOT IN", "BETWEEN", "IS NULL", "IS NOT NULL"
    ]


    # --- Helpers ---
    def get_columns(sql_text):
        match = re.search(r"SELECT\s+(.*?)\s+FROM", sql_text, re.IGNORECASE | re.DOTALL)
        if match:
            cols = [c.strip() for c in match.group(1).split(",")]
            return cols
        return []

    def replace_columns(sql_text, new_cols):
        return re.sub(
            r"SELECT\s+(.*?)\s+FROM",
            "SELECT " + ", ".join(new_cols) + " FROM",
            sql_text,
            flags=re.IGNORECASE | re.DOTALL
        )

    cols = get_columns(sql)

    # === Remove WHERE ===
    if op == "remove_where" and "WHERE" in sql.upper():
        before, _, after = sql.partition("WHERE")
        corrupted = before.strip()
        if "ORDER BY" in after.upper():
            _, _, after_order = after.partition("ORDER BY")
            corrupted += " ORDER BY " + after_order
        elif "GROUP BY" in after.upper():
            _, _, after_group = after.partition("GROUP BY")
            corrupted += " GROUP BY " + after_group
        corrupted += ";"


    # === Change aggregation ===
    elif op == "change_agg":
        aggs = ["COUNT", "SUM", "AVG", "MAX", "MIN"]
        found = [a for a in aggs if a in sql.upper()]
        if found:
            current = found[0]
            others = [a for a in aggs if a != current]
            corrupted = re.sub(current, random.choice(others), sql, flags=re.IGNORECASE)
        else:
            # If no aggregation found, this operation shouldn't have been chosen
            # But if it was, just return the original SQL
            corrupted = sql



    # === Add a new condition ===
    elif op == "add_condition":
        # Use database-specific columns and values if available, otherwise fall back to generic ones
        if db_id and db_id in DB_PLAUSIBLE_CORRUPTED_VALUES:
            db_columns = DB_PLAUSIBLE_CORRUPTED_VALUES[db_id]["columns"]
            db_values = DB_PLAUSIBLE_CORRUPTED_VALUES[db_id]["values"]
        else:
            # Generic fallback columns and values
            db_columns = ["item_id", "uuid", "name", "description", "status", "created_at", "updated_at"]
            db_values = ["item_001", "uuid_123", "Sample", "Active", "2023-01-01", "Test"]
        
        col = random.choice(db_columns)
        op_choice = random.choice(operators)
        val = random.choice(db_values)

        # Handle special operators like IS NULL, IN, BETWEEN
        if "IS" in op_choice:
            condition = f"{col} {op_choice}"
        elif op_choice in ["IN", "NOT IN"]:
            vals = ", ".join(random.sample(db_values, k=min(3, len(db_values))))
            condition = f"{col} {op_choice} ({vals})"
        elif op_choice == "BETWEEN":
            condition = f"{col} BETWEEN {random.randint(0, 50)} AND {random.randint(51, 100)}"
        else:
            condition = f"{col} {op_choice} {val}"

        if "WHERE" in sql.upper():
            before, _, after = sql.partition("WHERE")
            corrupted = before + "WHERE " + after.strip().rstrip(";")
            if not corrupted.strip().endswith(("AND", "OR")):
                corrupted += " AND " + condition + ";"
        else:
            corrupted = sql.rstrip(";") + f" WHERE {condition};"

    # === 6️⃣ Remove a column ===
    elif op == "remove_column" and len(cols) > 1:
        removed = random.choice(cols)
        new_cols = [c for c in cols if c != removed]
        corrupted = replace_columns(sql, new_cols)

    # === 7️⃣ Add a column ===
    elif op == "add_column":
        # Use database-specific columns if available, otherwise fall back to generic ones
        if db_id and db_id in DB_PLAUSIBLE_CORRUPTED_VALUES:
            db_columns = DB_PLAUSIBLE_CORRUPTED_VALUES[db_id]["columns"]
        else:
            # Generic fallback columns
            db_columns = ["item_id", "uuid", "name", "description", "status", "created_at", "updated_at"]
        
        # Ensure we add a column that doesn't already exist
        existing_cols_lower = [c.lower() for c in cols]
        available_cols = [col for col in db_columns if col not in existing_cols_lower]
        
        if available_cols:
            new_col = random.choice(available_cols)
            new_cols = cols + [new_col] if cols else [new_col]
            corrupted = replace_columns(sql, new_cols)
        else:
            # If all database-specific columns exist, add a numbered version
            base_col = random.choice(db_columns)
            counter = 1
            new_col = f"{base_col}_{counter}"
            while new_col in existing_cols_lower:
                counter += 1
                new_col = f"{base_col}_{counter}"
            
            new_cols = cols + [new_col] if cols else [new_col]
            corrupted = replace_columns(sql, new_cols)

    # === Swap two columns ===
    elif op == "swap_columns" and len(cols) > 1:
        i, j = random.sample(range(len(cols)), 2)
        cols[i], cols[j] = cols[j], cols[i]
        corrupted = replace_columns(sql, cols)

    # === Replace column with database-specific column ===
    elif op == "replace_column":
        # Use database-specific columns if available, otherwise fall back to generic ones
        if db_id and db_id in DB_PLAUSIBLE_CORRUPTED_VALUES:
            db_columns = DB_PLAUSIBLE_CORRUPTED_VALUES[db_id]["columns"]
        else:
            # Generic fallback columns
            db_columns = ["item_id", "uuid", "name", "description", "status", "created_at", "updated_at"]
        
        if cols and db_columns:
            # Replace a random existing column with a database-specific one
            col_to_replace = random.choice(cols)
            new_col = random.choice(db_columns)
            
            # Ensure the new column doesn't already exist
            existing_cols_lower = [c.lower() for c in cols]
            while new_col.lower() in existing_cols_lower:
                new_col = random.choice(db_columns)
            
            # Replace the column
            new_cols = [new_col if c == col_to_replace else c for c in cols]
            corrupted = replace_columns(sql, new_cols)

    # === Replace values with database-specific values ===
    elif op == "replace_values":
        # Use database-specific values if available, otherwise fall back to generic ones
        if db_id and db_id in DB_PLAUSIBLE_CORRUPTED_VALUES:
            db_values = DB_PLAUSIBLE_CORRUPTED_VALUES[db_id]["values"]
        else:
            # Generic fallback values
            db_values = ["item_001", "uuid_123", "Sample", "Active", "2023-01-01", "Test"]
        
        if db_values:
            # Find string literals in the SQL and replace them with database-specific values
            # This is a simple approach - replace quoted strings
            def replace_quoted_strings(match):
                return f"'{random.choice(db_values)}'"
            
            # Replace single-quoted strings
            corrupted = re.sub(r"'[^']*'", replace_quoted_strings, sql)
            
            # If no changes were made, add a WHERE condition with a database-specific value
            if corrupted == sql and "WHERE" not in sql.upper():
                col = random.choice(cols) if cols else "id"
                val = random.choice(db_values)
                corrupted = sql.rstrip(";") + f" WHERE {col} = '{val}';"

    # Normalize
    corrupted = sqlparse.format(corrupted, reindent=True, keyword_case="upper")
    return corrupted

def choose_corruption_type(sql: str, db_id: str = None) -> (str, int):
    """
    Choose one or two eligible corruption types based on SQL structure.
    Ensures that if corruption is sampled, it always chooses a corruption type that will work.
    """
    corrupted = random.random() < 0.55
    
    if not corrupted:
        print("NO CORRUPTION")
        print("😇😇😇 ORIGINAL SQL 😇😇😇")
        print(sql)
        return sql, 0
    else: 
        print("😇😇😇 ORIGINAL SQL 😇😇😇")
        print(sql)
        sql_upper = sql.upper()
        eligible = []

        # Detect which corruption types make sense for this SQL
        if "WHERE" in sql_upper:
            eligible.extend(["remove_where", "add_condition"])
        else:
            eligible.append("add_condition")


        if any(agg in sql_upper for agg in ["COUNT", "SUM", "AVG", "MAX", "MIN"]):
            eligible.append("change_agg")

        # Column-level corruptions - check if multiple columns exist for remove_column
        try:
            # Parse SQL to count columns
            parsed = sqlparse.parse(sql)[0]
            select_found = False
            column_count = 0
            for token in parsed.flatten():
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'SELECT':
                    select_found = True
                elif select_found and token.ttype is sqlparse.tokens.Keyword and token.value.upper() in ['FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT']:
                    break
                elif select_found and token.ttype is sqlparse.tokens.Name and token.value.strip():
                    column_count += 1
            
            # Add column corruptions based on what's possible
            if column_count > 1:
                eligible.append("remove_column")
                if column_count >= 2:
                    eligible.append("swap_columns")
            
            # add_column is always possible
            eligible.append("add_column")
            
            # Add database-specific column replacement if db_id is available
            if db_id and db_id in DB_PLAUSIBLE_CORRUPTED_VALUES:
                eligible.extend(["replace_column", "replace_values"])
            
        except:
            # Fallback: assume column corruptions are possible
            eligible.extend(["remove_column", "add_column", "swap_columns"])
            if db_id and db_id in DB_PLAUSIBLE_CORRUPTED_VALUES:
                eligible.extend(["replace_column", "replace_values"])

        # Ensure we have at least one eligible corruption type
        if not eligible:
            # If no specific corruptions are eligible, add_condition should always work
            eligible = ["add_condition"]

        # Randomly decide how many corruptions (but don't exceed available options)
        # Probabilities: 1 corruption (55%), 2 corruptions (30%), 3 corruptions (10%), 4 corruptions (5%)
        if len(eligible) >= 4 and 0 < random.random() < 0.05: # 5% chance to apply 4 corruptions
            n_corruptions = 4
        elif len(eligible) >= 3 and 0.05 < random.random() < 0.15: # 10% chance to apply three corruptions
            n_corruptions = 3
        elif len(eligible) >= 2 and 0.15 < random.random() < 0.45: # 30% chance to apply two corruptions
            n_corruptions = 2
        else: # 45% chance to apply one corruption
            n_corruptions = 1

        # Ensure we don't try to apply more corruptions than we have eligible types
        n_corruptions = min(n_corruptions, len(eligible))

    # Randomly pick unique corruptions from eligible ones
        chosen = random.sample(eligible, n_corruptions)
    
    # Apply all chosen corruptions sequentially
    corrupted_sql = sql
    for operation in chosen:
            corrupted_sql = corrupt_sql_query(corrupted_sql, operation, db_id)
    
    print("😈😈😈 FINAL CORRUPTED SQL 😈😈😈")
    print(corrupted_sql)
    return corrupted_sql, len(chosen)

def process_single_example(
    sql: str, 
    true_nl: str, 
    model, 
    tokenizer,
    sentence_bert_model: SentenceBERTSimilarity,
    db_id: str = None
) -> Tuple[str, str, str, float, bool]:
    """Process a single SQL-NL example to generate CoT reasoning and similarity score."""
    
    # Generate CoT reasoning
    corrupted_sql, count_of_corruptions = choose_corruption_type(sql, db_id)
    gen_prompt = build_generation_prompt(corrupted_sql)

    
    # Time the LLM generation
    start_time = time.time()
    reasoning = run_llm_pytorch(model, tokenizer, gen_prompt)
    llm_time = time.time() - start_time
    
    if not reasoning:
        return None  # Skip this example
    
    # Extract predicted NL from reasoning and clean reasoning
    predicted_nl = ""
    clean_reasoning = reasoning
    
    # Look for <question> tags first
    if '<question>' in reasoning and '</question>' in reasoning:
        start = reasoning.find('<question>') + len('<question>')
        end = reasoning.find('</question>')
        predicted_nl = reasoning[start:end].strip()
        # Remove the question tags and content from reasoning
        clean_reasoning = reasoning[:start-len('<question>')].strip() + reasoning[end+len('</question>'):].strip()
    else:
        # Look for "question:" pattern
        if 'question:' in reasoning:
            # Find the line with "question:" and extract the text after it
            lines = reasoning.strip().split('\n')
            for i, line in enumerate(lines):
                if 'question:' in line:
                    predicted_nl = line.split('question:', 1)[1].strip()
                    # Remove this line from reasoning
                    lines.pop(i)
                    clean_reasoning = '\n'.join(lines).strip()
                    break
        else:
            # Fallback: look for the final question - it's usually the last substantial line
            lines = reasoning.strip().split('\n')
            for i in reversed(range(len(lines))):
                line = lines[i].strip()
                if line and len(line) > 10:  # Skip very short lines
                    # Skip lines that are clearly not the final question
                    if not any(skip_word in line.lower() for skip_word in ['step', 'reasoning', 'think:', 'analysis:', 'conclusion:']):
                        predicted_nl = line
                        # Remove this line from reasoning
                        lines.pop(i)
                        clean_reasoning = '\n'.join(lines).strip()
                        break
            
            # If still no good extraction, take the last line
            if not predicted_nl:
                lines = reasoning.strip().split('\n')
                predicted_nl = lines[-1].strip()
                # Remove the last line from reasoning
                clean_reasoning = '\n'.join(lines[:-1]).strip()
    
    # Truncate at "Validate" if present
    if "Validate" in predicted_nl:
        validate_index = predicted_nl.find("Validate")
        predicted_nl = predicted_nl[:validate_index].strip()
    
    # Truncate at "think:" if present
    if "think:" in predicted_nl.lower():
        think_index = predicted_nl.lower().find("think:")
        predicted_nl = predicted_nl[:think_index].strip()
    
    # Also truncate true_nl at "think:" if present
    if "think:" in true_nl.lower():
        think_index = true_nl.lower().find("think:")
        true_nl = true_nl[:think_index].strip()
    
    # Update reasoning to be the clean version without the final question
    reasoning = clean_reasoning
    
    
    
    # Compute similarity with corruption penalty
    try:
        # Time the similarity computation
        start_time = time.time()
        print(f"\n=== DEBUG: SQL VALUES ===")
        print(f"Original SQL (sql): {sql}")
        print(f"Corrupted SQL (corrupted_sql): {corrupted_sql}")
        print(f"SQLs are equal: {sql == corrupted_sql}")
        print(f"========================\n")
        similarity = sentence_bert_model.compute_similarity_with_corruption_penalty(predicted_nl, true_nl, count_of_corruptions, sql, corrupted_sql, debug=True)
        simcse_time = time.time() - start_time
    except Exception as e:
        print(f"⚠️  Similarity computation failed: {e}")
        similarity = 0.0
        simcse_time = 0.0
    
    print("PREDICTED NL")
    print(predicted_nl)
    print("TRUE NL")
    print(true_nl)
    print("REASONING")
    print(clean_reasoning)
    print("SIMILARITY")
    print(similarity)
    print("COUNT OF CORRUPTIONS")
    print(count_of_corruptions)
    return reasoning, predicted_nl, true_nl, similarity, count_of_corruptions

def create_cot_dataset(
    input_csv: str,
    output_csv: str,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    max_examples: int = None
):
    """Create CoT dataset using PyTorch on GPU with SimCSE similarity."""
    
    print("🚀 Starting CoT dataset creation with PyTorch + SimCSE")
    print(f"📁 Input: {input_csv}")
    print(f"📁 Output: {output_csv}")
    
    # Load input data
    if not os.path.exists(input_csv):
        print(f"❌ Input file not found: {input_csv}")
        return
    
    df = pd.read_csv(input_csv)
    print(f"📊 Loaded {len(df)} examples")
    
    if max_examples:
        df = df.head(max_examples)
        print(f"🔢 Processing first {len(df)} examples")
    
    # Check if output file exists and load existing data
    existing_data = []
    if os.path.exists(output_csv):
        print(f"📁 Found existing dataset: {output_csv}")
        existing_df = pd.read_csv(output_csv)
        existing_data = existing_df.to_dict('records')
        print(f"📊 Found {len(existing_data)} existing examples")
        
        # Get already processed SQL queries to avoid duplicates
        existing_sqls = set(row['sql'] for row in existing_data)
        df = df[~df['sql'].isin(existing_sqls)]
        print(f"🔢 Processing {len(df)} new examples (skipping {len(existing_sqls)} already processed)")
    
    # Initialize models
    print("🔄 Loading LLM model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    print("🔄 Loading Sentence-BERT model...")
    sentence_bert_model = SentenceBERTSimilarity()
    
    # Process examples
    results = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"🔄 Processing example {i}/{len(df)}")
        
        sql = row['sql']
        true_nl = row['nl_question']  # Using 'nl_question' column for true NL
        db_id = row['db_id']  # Get db_id from the CSV
        
        result = process_single_example(
            sql, true_nl, model, tokenizer, sentence_bert_model, db_id
        )
        
        # Skip examples that failed to process
        if result is None:
            print(f"⚠️  Skipping example {i} (failed to generate reasoning)")
            continue
            
        reasoning, predicted_nl, _, similarity, count_of_corruptions = result
        
        # Generate validation prompt
        val_prompt = build_validation_prompt(sql, reasoning, predicted_nl)
        
        results.append({
            'sql': sql,
            'reasoning': reasoning,
            'predicted_nl': predicted_nl,
            'true_nl': true_nl,
            'similarity_with_penalty': similarity,
            'is_corrupted': count_of_corruptions > 0,
            'corruption_count': count_of_corruptions,
            'prompt': val_prompt
        })
        
        if i % 5 == 0:
            print(f"✅ Processed {i} examples")
            
            # Calculate average similarity for corrupted vs non-corrupted
            corrupted_results = [r for r in results if r['is_corrupted']]
            non_corrupted_results = [r for r in results if not r['is_corrupted']]
            
            if corrupted_results:
                avg_sim_corrupted = np.mean([r['similarity_with_penalty'] for r in corrupted_results])
                print(f"🔀 Corrupted examples: {len(corrupted_results)} - Avg similarity: {avg_sim_corrupted:.3f}")
                print(f"🔀 Corrupted example:")
                print(f"🔀 Predicted NL: {corrupted_results[0]['predicted_nl']}")
                print(f"🔀 True NL: {corrupted_results[0]['true_nl']}")
                print(f"🔀 Corruption count: {corrupted_results[0]['corruption_count']}")
            
            if non_corrupted_results:
                avg_sim_clean = np.mean([r['similarity_with_penalty'] for r in non_corrupted_results])
                print(f"✅ Clean examples: {len(non_corrupted_results)} - Avg similarity: {avg_sim_clean:.3f}")
                print(f"🔀 Clean example:")
                print(f"🔀 Predicted NL: {non_corrupted_results[0]['predicted_nl']}")
                print(f"🔀 True NL: {non_corrupted_results[0]['true_nl']}")
            
            # Overall average
            avg_sim_overall = np.mean([r['similarity_with_penalty'] for r in results])
            print(f"📊 Overall average similarity: {avg_sim_overall:.3f}")
            
            # Save progress every 5 examples
            all_results = existing_data + results
            result_df = pd.DataFrame(all_results)
            result_df.to_csv(output_csv, index=False)
            print(f"💾 Saved progress to {output_csv} ({len(result_df)} total examples)")
            
    
    # Combine existing data with new results
    all_results = existing_data + results
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(output_csv, index=False)
    
    print(f"✅ Dataset creation complete!")
    print(f"📁 Saved to: {output_csv}")
    print(f"📊 Total examples: {len(result_df)}")
    print(f"📈 Average similarity: {result_df['similarity_with_penalty'].mean():.3f}")
    print(f"📈 Similarity range: {result_df['similarity_with_penalty'].min():.3f} - {result_df['similarity_with_penalty'].max():.3f}")
    print(f"🔀 Corrupted examples: {result_df['is_corrupted'].sum()}/{len(result_df)} ({result_df['is_corrupted'].mean()*100:.1f}%)")
    
    return result_df

def main():
    """Main function to create the dataset."""
    input_csv = "../../data/bird_sqls_and_nl_questions.csv"
    output_csv = "../../data/cot_dataset_with_corruptions.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Create dataset
    create_cot_dataset(
        input_csv=input_csv,
        output_csv=output_csv,
        model_name="Qwen/Qwen2.5-3B-Instruct",  # Using Qwen model
        max_examples=None  # Process all examples
    )

if __name__ == "__main__":
    main()
