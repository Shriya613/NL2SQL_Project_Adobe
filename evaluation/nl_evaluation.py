import spacy
from spacy_syllables import SpacySyllables
import pandas as pd
import numpy as np
import sys
import json


SPACY_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "ru": "ru_core_news_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
    "ro": "ro_core_news_sm",
    "uk": "uk_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "pl": "pl_core_news_sm",
    "tr": "tr_core_news_md", 
    "hi": "en_core_web_sm", # No Hindi model
    "kk": "tr_core_news_md", # No Kazakh model
}

# Global cache for loaded models to avoid reloading
_loaded_models = {}

def get_nlp(lang="en"):
    """Load and return the appropriate spaCy model for the language."""
    if lang in _loaded_models:
        return _loaded_models[lang]
    
    model_name = SPACY_MODELS.get(lang, "en_ent_wiki_sm")
    
    try:
        print(f"Loading spaCy model '{model_name}' for language '{lang}'...")
        nlp = spacy.load(model_name)
        
        # Ensure sentence boundaries are set (required for doc.sents)
        if "sentencizer" not in nlp.pipe_names:
            try:
                nlp.add_pipe("sentencizer")
            except Exception:
                pass
        try:
            nlp.add_pipe("syllables", after="tagger")
        except Exception as e:
            pass
            
        _loaded_models[lang] = nlp
        return nlp
    except OSError:
        print(f"Warning: Model '{model_name}' not found. Falling back to 'en_core_web_sm' or failing.")
        # Fallback
        if lang != "en":
            try:
                fallback = spacy.load("en_core_web_sm")
                _loaded_models[lang] = fallback
                return fallback
            except:
                pass
        raise

def normalize_metric(value, method='log', stats=None):
    """
    Normalize a metric value using various methods.
    
    Args:
        value: The raw metric value
        method: 'log', 'minmax', 'zscore', 'robust', or 'log_plus_one'
        stats: Dict with normalization statistics (min, max, mean, std, p25, p75)
    
    Returns:
        Normalized value (typically 0-1 range)
    """
    if value == 0:
        return 0.0
    
    if method == 'log' or method == 'log_plus_one':
        # Log transformation: log(1 + x) to handle 0 values
        # For tokens_before_main_verb, this handles right-skewed distribution
        log_val = np.log1p(value)
        # Normalize to 0-1 range using reasonable bounds
        # log(1+10) ≈ 2.4, so we'll use 3 as max for normalization
        max_log = stats.get('max_log', 3.0) if stats else 3.0
        return min(log_val / max_log, 1.0)
    
    elif method == 'minmax':
        if stats and 'min' in stats and 'max' in stats:
            min_val = stats['min']
            max_val = stats['max']
            if max_val == min_val:
                return 0.0
            return (value - min_val) / (max_val - min_val)
        else:
            # Fallback to reasonable fixed bounds
            return min(value / 10.0, 1.0)  # Assume max ~10
    
    elif method == 'zscore':
        if stats and 'mean' in stats and 'std' in stats:
            mean_val = stats['mean']
            std_val = stats['std']
            if std_val == 0:
                return 0.0
            z = (value - mean_val) / std_val
            # Convert z-score to 0-1 range using sigmoid-like transformation
            return 1 / (1 + np.exp(-z * 0.5))
        else:
            return min(value / 10.0, 1.0)
    
    elif method == 'robust':
        if stats and 'p25' in stats and 'p75' in stats:
            p25 = stats['p25']
            p75 = stats['p75']
            iqr = p75 - p25
            if iqr == 0:
                return 0.0
            # Robust normalization using IQR
            normalized = (value - p25) / iqr
            return min(max(normalized, 0.0), 1.0)  # Clip to [0, 1]
        else:
            return min(value / 10.0, 1.0)
    
    return value

def calculate_syntactic_complexity(doc, normalization_stats=None, use_normalization=True):
    """
    Calculate syntactic complexity metrics:
    1. Mean number of words before the main verb
    2. Higher-level constituents per word in the sentence
    
    Args:
        doc: spaCy document
        normalization_stats: Dict with normalization statistics for each metric
        use_normalization: Whether to normalize metrics before combining
    """
    metrics = {
        'tokens_before_main_verb': 0,
        'constituents_per_word': 0,
        'subordinate_clauses': 0,
        'coordinate_clauses': 0,
        'complexity_score': 0
    }
    
    # Get all sentences in the document
    sentences = list(doc.sents)
    if not sentences:
        return metrics
    
    total_tokens_before_verb = 0
    total_constituents = 0
    total_words = 0
    total_subordinate = 0
    total_coordinate = 0
    
    for sent in sentences:
        # Count words (excluding punctuation)
        words = [token for token in sent if not token.is_punct and not token.is_space]
        total_words += len(words)
        
        # Find main verb (first verb that's not auxiliary)
        main_verb_found = False
        tokens_before_main_verb = 0
        
        for token in words:
            if token.pos_ == "VERB" and not main_verb_found:
                # Check if it's a main verb (not auxiliary)
                if token.dep_ not in ["aux", "auxpass"]:
                    main_verb_found = True
                    break
            if not main_verb_found:
                tokens_before_main_verb += 1
        
        total_tokens_before_verb += tokens_before_main_verb
        
        # Count higher-level constituents
        constituents = 0
        
        # Count noun phrases (NPs)
        try:
            for chunk in sent.noun_chunks:
                if len(chunk) > 1:  # Multi-word noun phrases
                    constituents += 1
        except (NotImplementedError, ValueError, IndexError, AttributeError):
            pass
        
        # Count verb phrases (VPs) - verbs with their dependents
        for token in sent:
            if token.pos_ == "VERB":
                # Count dependents of the verb
                dependents = [child for child in token.children]
                if len(dependents) > 0:
                    constituents += 1
        
        # Count prepositional phrases (PPs)
        for token in sent:
            if token.pos_ == "ADP":  # Preposition
                constituents += 1
        
        # Count subordinate clauses (marked by subordinating conjunctions)
        subordinate_clauses = 0
        for token in sent:
            if token.pos_ == "SCONJ":  # Subordinating conjunction
                subordinate_clauses += 1
        
        # Count coordinate clauses (marked by coordinating conjunctions)
        coordinate_clauses = 0
        for token in sent:
            if token.pos_ == "CCONJ":  # Coordinating conjunction
                coordinate_clauses += 1
        
        total_constituents += constituents
        total_subordinate += subordinate_clauses
        total_coordinate += coordinate_clauses
    
    # Calculate metrics
    if len(sentences) > 0:
        metrics['tokens_before_main_verb'] = total_tokens_before_verb / len(sentences)
        metrics['constituents_per_word'] = total_constituents / total_words if total_words > 0 else 0
        metrics['subordinate_clauses'] = total_subordinate / len(sentences)
        metrics['coordinate_clauses'] = total_coordinate / len(sentences)
        
        # Normalize metrics if requested
        if use_normalization:
            # Get normalization stats for each metric (if provided)
            tokens_stats = normalization_stats.get('tokens_before_main_verb', {}) if normalization_stats else {}
            constituents_stats = normalization_stats.get('constituents_per_word', {}) if normalization_stats else {}
            subordinate_stats = normalization_stats.get('subordinate_clauses', {}) if normalization_stats else {}
            coordinate_stats = normalization_stats.get('coordinate_clauses', {}) if normalization_stats else {}
            
            # Normalize each metric
            # Use log transformation for tokens_before_main_verb (right-skewed)
            normalized_tokens = normalize_metric(
                metrics['tokens_before_main_verb'], 
                method='log_plus_one',
                stats=tokens_stats
            )
            # Use minmax for constituents_per_word (already a ratio, typically 0-1)
            normalized_constituents = normalize_metric(
                metrics['constituents_per_word'],
                method='minmax',
                stats=constituents_stats
            )
            # Use log for subordinate_clauses (count-based, right-skewed)
            normalized_subordinate = normalize_metric(
                metrics['subordinate_clauses'],
                method='log_plus_one',
                stats=subordinate_stats
            )
            # Use log for coordinate_clauses (count-based, right-skewed)
            normalized_coordinate = normalize_metric(
                metrics['coordinate_clauses'],
                method='log_plus_one',
                stats=coordinate_stats
            )
            
            # Overall complexity score (weighted combination of normalized metrics)
            # Higher values indicate more complex syntax
            metrics['complexity_score'] = (
                normalized_tokens * 0.15 +
                normalized_constituents * 0.30 +
                normalized_subordinate * 0.35 +
                normalized_coordinate * 0.20
            )
        else:
            # Original unnormalized calculation
            metrics['complexity_score'] = (
                metrics['tokens_before_main_verb'] * 0.15 +
                metrics['constituents_per_word'] * 0.30 +
                metrics['subordinate_clauses'] * 0.35 +
                metrics['coordinate_clauses'] * 0.20
            )
    
    return metrics

def evaluate_nl(dataset, nl_column: str, lang: str = "en"):
    """Evaluate natural language questions and provide statistics.
    
    Args:
        dataset: Either a file path (str) or a pandas DataFrame
        nl_column: Column name containing the NL questions
        lang: Language code for spaCy model
    """
    if isinstance(dataset, str):
        data = pd.read_csv(dataset)
    else:
        data = dataset
    
    try:
        nlp_model = get_nlp(lang)
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    # Initialize lists to store metrics
    word_counts = []
    syllable_counts = []
    sentence_counts = []
    token_counts = []
    pos_counts = {}
    entity_counts = []
    all_sentences = []
    tokens_before_verb_counts = []
    
    print(f"Processing {len(data)} natural language questions for language '{lang}'...")
    
    for index, row in data.iterrows():
        nl = row[nl_column]
        # Skip rows with NaN or non-string values
        if pd.isna(nl) or not isinstance(nl, str):
            continue
        doc = nlp_model(nl)
        
        # Count words (excluding punctuation)
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        word_counts.append(len(words))
        
        # Count syllables
        total_syllables = 0
        # Only count if extension exists
        if doc[0].has_extension("syllables"):
            syllables = [token._.syllables for token in doc if not token.is_punct and not token.is_space]
            for syllable in syllables:
                if syllable:
                    total_syllables += len(syllable)
        syllable_counts.append(total_syllables)
        
        # Count sentences
        sentence_counts.append(len(list(doc.sents)))
        for sent in doc.sents:
            all_sentences.append(sent.text)
        
        # Count tokens
        token_counts.append(len(doc))
        
        # Calculate syntactic complexity metrics (first pass - collect raw metrics)
        syntactic_metrics = calculate_syntactic_complexity(doc, use_normalization=False)
        tokens_before_verb_counts.append(syntactic_metrics['tokens_before_main_verb'])
        
        # Count POS tags
        for token in doc:
            if not token.is_punct and not token.is_space:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1

        # Count entities
        entity_counts.append(len(doc.ents))

    
    # Convert to numpy arrays for statistics
    word_counts = np.array(word_counts)
    syllable_counts = np.array(syllable_counts)
    sentence_counts = np.array(sentence_counts)
    token_counts = np.array(token_counts)
    entity_counts = np.array(entity_counts)
    syntactic_complexity = np.array(tokens_before_verb_counts).mean()
    fkreadingease = 206.835 - (1.015 * (word_counts.mean() / sentence_counts.mean())) - (84.6 * (syllable_counts.mean() / word_counts.mean()))
    fkgl = 0.39 * (word_counts.mean() / sentence_counts.mean()) + 11.8 * (syllable_counts.mean() / word_counts.mean()) - 15.59
    
    # Print statistics
    print("\n" + "="*60)
    print(f"NATURAL LANGUAGE EVALUATION STATISTICS ({lang})")
    print("="*60)
    
    print(f"\nTotal questions processed: {len(data)}")
    
    print(f"\nWORD COUNT STATISTICS:")
    print(f"  Min words: {word_counts.min()}")
    print(f"  Max words: {word_counts.max()}")
    print(f"  Mean words: {word_counts.mean():.2f}")
    print(f"  Median words: {np.median(word_counts):.2f}")
    print(f"  Std deviation: {word_counts.std():.2f}")
    
    print(f"\nSYLLABLE COUNT STATISTICS:")
    print(f"  Min syllables: {syllable_counts.min()}")
    print(f"  Max syllables: {syllable_counts.max()}")
    print(f"  Mean syllables: {syllable_counts.mean():.2f}")
    print(f"  Median syllables: {np.median(syllable_counts):.2f}")
    print(f"  Std deviation: {syllable_counts.std():.2f}")
    
    print(f"\nSENTENCE COUNT STATISTICS:")
    print(f"  Min sentences: {sentence_counts.min()}")
    print(f"  Max sentences: {sentence_counts.max()}")
    print(f"  Mean sentences: {sentence_counts.mean():.2f}")
    print(f"  Median sentences: {np.median(sentence_counts):.2f}")
    
    print(f"\nTOKEN COUNT STATISTICS:")
    print(f"  Min tokens: {token_counts.min()}")
    print(f"  Max tokens: {token_counts.max()}")
    print(f"  Mean tokens: {token_counts.mean():.2f}")
    print(f"  Median tokens: {np.median(token_counts):.2f}")
    
    print(f"\nENTITY COUNT STATISTICS:")
    print(f"  Min entities: {entity_counts.min()}")
    print(f"  Max entities: {entity_counts.max()}")
    print(f"  Mean entities: {entity_counts.mean():.2f}")
    print(f"  Median entities: {np.median(entity_counts):.2f}")

    print(f"\nREADABILITY METRICS:")
    print(f"  Flesch-Kincaid Reading Ease: {fkreadingease:.2f}")
    print(f"  Flesch-Kincaid Grade Level: {fkgl:.2f}")
    
    print(f"\nSYNTACTIC COMPLEXITY METRICS:")
    print(f"  Mean tokens before main verb: {syntactic_complexity:.2f}")
    print(f"  Overall complexity score: {syntactic_complexity:.2f}")
    
    # Calculate and display detailed syntactic complexity statistics
    # First pass: collect raw metrics to compute normalization statistics
    all_raw_tokens_before_verb = []
    all_raw_constituents_per_word = []
    all_raw_subordinate_clauses = []
    all_raw_coordinate_clauses = []
    
    for index, row in data.iterrows():
        nl = row[nl_column]
        # Skip rows with NaN or non-string values
        if pd.isna(nl) or not isinstance(nl, str):
            continue
        doc = nlp_model(nl)
        metrics = calculate_syntactic_complexity(doc, use_normalization=False)
        all_raw_tokens_before_verb.append(metrics['tokens_before_main_verb'])
        all_raw_constituents_per_word.append(metrics['constituents_per_word'])
        all_raw_subordinate_clauses.append(metrics['subordinate_clauses'])
        all_raw_coordinate_clauses.append(metrics['coordinate_clauses'])
    
    # Compute normalization statistics
    raw_tokens = np.array(all_raw_tokens_before_verb)
    raw_constituents = np.array(all_raw_constituents_per_word)
    raw_subordinate = np.array(all_raw_subordinate_clauses)
    raw_coordinate = np.array(all_raw_coordinate_clauses)
    
    normalization_stats = {
        'tokens_before_main_verb': {
            'min': raw_tokens.min(),
            'max': raw_tokens.max(),
            'mean': raw_tokens.mean(),
            'std': raw_tokens.std(),
            'p25': np.percentile(raw_tokens, 25),
            'p75': np.percentile(raw_tokens, 75),
            'max_log': np.log1p(raw_tokens.max())  # For log normalization
        },
        'constituents_per_word': {
            'min': raw_constituents.min(),
            'max': raw_constituents.max(),
            'mean': raw_constituents.mean(),
            'std': raw_constituents.std(),
            'p25': np.percentile(raw_constituents, 25),
            'p75': np.percentile(raw_constituents, 75),
            'max_log': np.log1p(raw_constituents.max())
        },
        'subordinate_clauses': {
            'min': raw_subordinate.min(),
            'max': raw_subordinate.max(),
            'mean': raw_subordinate.mean(),
            'std': raw_subordinate.std(),
            'p25': np.percentile(raw_subordinate, 25),
            'p75': np.percentile(raw_subordinate, 75),
            'max_log': np.log1p(raw_subordinate.max())
        },
        'coordinate_clauses': {
            'min': raw_coordinate.min(),
            'max': raw_coordinate.max(),
            'mean': raw_coordinate.mean(),
            'std': raw_coordinate.std(),
            'p25': np.percentile(raw_coordinate, 25),
            'p75': np.percentile(raw_coordinate, 75),
            'max_log': np.log1p(raw_coordinate.max())
        }
    }
    
    # Second pass: recalculate with normalization
    all_complexity_scores = []
    all_constituents_per_word = []
    all_subordinate_clauses = []
    all_coordinate_clauses = []
    
    for index, row in data.iterrows():
        nl = row[nl_column]
        # Skip rows with NaN or non-string values
        if pd.isna(nl) or not isinstance(nl, str):
            continue
        doc = nlp_model(nl)
        metrics = calculate_syntactic_complexity(doc, normalization_stats=normalization_stats, use_normalization=True)
        all_complexity_scores.append(metrics['complexity_score'])
        all_constituents_per_word.append(metrics['constituents_per_word'])
        all_subordinate_clauses.append(metrics['subordinate_clauses'])
        all_coordinate_clauses.append(metrics['coordinate_clauses'])
    
    all_complexity_scores = np.array(all_complexity_scores)
    all_constituents_per_word = np.array(all_constituents_per_word)
    all_subordinate_clauses = np.array(all_subordinate_clauses)
    all_coordinate_clauses = np.array(all_coordinate_clauses)
    
    print(f"\nDETAILED SYNTACTIC COMPLEXITY STATISTICS:")
    print(f"  Complexity Score - Min: {all_complexity_scores.min():.2f}, Max: {all_complexity_scores.max():.2f}, Mean: {all_complexity_scores.mean():.2f}")
    print(f"  Constituents per Word - Min: {all_constituents_per_word.min():.2f}, Max: {all_constituents_per_word.max():.2f}, Mean: {all_constituents_per_word.mean():.2f}")
    print(f"  Subordinate Clauses per Sentence - Min: {all_subordinate_clauses.min():.2f}, Max: {all_subordinate_clauses.max():.2f}, Mean: {all_subordinate_clauses.mean():.2f}")
    print(f"  Coordinate Clauses per Sentence - Min: {all_coordinate_clauses.min():.2f}, Max: {all_coordinate_clauses.max():.2f}, Mean: {all_coordinate_clauses.mean():.2f}")
    
    print(f"\nTOP 10 PART-OF-SPEECH TAGS:")
    sorted_pos = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
    for pos, count in sorted_pos[:10]:
        print(f"  {pos}: {count}")

    print(f"Complexity score: {all_complexity_scores.mean():.2f}")


def evaluate_bird(json_path: str = "data/train_bird.json"):
    """Helper to evaluate the Bird training dataset."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} records from Bird dataset")
    return evaluate_nl(df, "question", "en")


if __name__ == "__main__":
    evaluate_bird("data/train_bird.json")
    evaluate_nl("pipeline/final_translations/translated_en.csv", "nl")
