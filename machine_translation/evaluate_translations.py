"""
Evaluate Machine Translation Quality
Uses metrics from evaluation folder to assess translation quality

IMPORTANT NOTES:
- Syntactic complexity: REMOVED - requires English spaCy model, won't work for Turkish/Russian/Spanish
- Fuzzy similarity: REMOVED - not useful for cross-language comparison (strings are different by design)

USEFUL METRICS FOR TRANSLATION EVALUATION:
1. Length ratios - detect truncated/verbose translations
2. Language detection - verify correct target language
3. BLEU scores - if reference translations available
4. Word/character counts - basic quality checks

BETTER ALTERNATIVES (not yet implemented):
- Semantic similarity using multilingual embeddings (sentence-transformers)
- Back-translation quality checks
- Language-specific syntactic analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional
import logging

# Add evaluation folder to path
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

try:
    from nl_evaluation import calculate_syntactic_complexity
    from sql_evaluation import evaluate_sql_query_fuzzy_match
    NL_EVAL_AVAILABLE = True
except ImportError as e:
    NL_EVAL_AVAILABLE = False
    logging.warning(f"Could not import evaluation modules: {e}")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    logging.warning("NLTK not available. Install with: pip install nltk")

try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("fuzzywuzzy not available. Install with: pip install fuzzywuzzy")

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

import spacy
try:
    nlp_en = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    nlp_en = None
    SPACY_AVAILABLE = False
    logging.warning("spaCy English model not available")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_bleu_score(reference: str, translation: str) -> float:
    """Calculate BLEU score between reference and translation"""
    if not BLEU_AVAILABLE:
        return None
    
    try:
        ref_tokens = word_tokenize(reference.lower())
        trans_tokens = word_tokenize(translation.lower())
        
        # Use smoothing function to handle cases where n-grams don't match
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], trans_tokens, smoothing_function=smoothing)
        return score
    except Exception as e:
        logger.warning(f"BLEU calculation failed: {e}")
        return None


def calculate_fuzzy_similarity(text1: str, text2: str) -> float:
    """Calculate fuzzy string similarity"""
    if not FUZZY_AVAILABLE:
        return None
    
    try:
        return fuzz.partial_ratio(text1, text2) / 100.0
    except Exception as e:
        logger.warning(f"Fuzzy similarity calculation failed: {e}")
        return None


def compare_text_metrics(original: str, translation: str) -> Dict:
    """Compare original and translated text using various metrics"""
    metrics = {}
    
    if not SPACY_AVAILABLE:
        return metrics
    
    try:
        doc_orig = nlp_en(original)
        doc_trans = nlp_en(translation) if translation else None
        
        # Word counts
        words_orig = [t for t in doc_orig if not t.is_punct and not t.is_space]
        metrics['original_word_count'] = len(words_orig)
        
        if doc_trans:
            words_trans = [t for t in doc_trans if not t.is_punct and not t.is_space]
            metrics['translation_word_count'] = len(words_trans)
            metrics['word_count_ratio'] = len(words_trans) / len(words_orig) if len(words_orig) > 0 else 0
        else:
            metrics['translation_word_count'] = 0
            metrics['word_count_ratio'] = 0
        
        # Character counts
        metrics['original_char_count'] = len(original)
        metrics['translation_char_count'] = len(translation) if translation else 0
        metrics['char_count_ratio'] = metrics['translation_char_count'] / metrics['original_char_count'] if metrics['original_char_count'] > 0 else 0
        
        # Sentence counts
        metrics['original_sentence_count'] = len(list(doc_orig.sents))
        if doc_trans:
            metrics['translation_sentence_count'] = len(list(doc_trans.sents))
        else:
            metrics['translation_sentence_count'] = 0
        
        # NOTE: Syntactic complexity removed - it requires English model and won't work
        # properly for cross-language evaluation (Turkish/Russian/Spanish)
        # The spaCy English model can't properly parse non-English text
        
    except Exception as e:
        logger.warning(f"Text metrics calculation failed: {e}")
    
    return metrics


def evaluate_translation_quality(
    original_csv: str,
    translation_csv: str,
    original_col: str = "original_text",
    translation_col: str = "translation",
    reference_csv: Optional[str] = None,
    reference_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluate translation quality by comparing original and translated texts
    
    Args:
        original_csv: Path to CSV with original English text
        translation_csv: Path to CSV with translations
        original_col: Column name for original text
        translation_col: Column name for translated text
        reference_csv: Optional path to reference translations (for BLEU)
        reference_col: Column name for reference translations
    """
    logger.info(f"Loading translation data from: {translation_csv}")
    df_trans = pd.read_csv(translation_csv)
    
    # Check if translation CSV already has original_text column
    if "original_text" in df_trans.columns:
        # Translation CSV already has original, use it directly
        df = df_trans.copy()
        actual_original_col = "original_text"  # Use this column
    else:
        # Need to load and merge with original CSV
        logger.info(f"Loading original data from: {original_csv}")
        df_orig = pd.read_csv(original_csv)
        # Merge on original text
        df = df_orig.merge(
            df_trans,
            on=original_col,
            how="inner",
            suffixes=("_orig", "_trans")
        )
        actual_original_col = original_col
    
    logger.info(f"Evaluating {len(df)} translation pairs...")
    
    # Load reference translations if provided
    df_ref = None
    if reference_csv and reference_col:
        logger.info(f"Loading reference translations from: {reference_csv}")
        df_ref = pd.read_csv(reference_csv)
        df = df.merge(
            df_ref,
            on=original_col,
            how="left",
            suffixes=("", "_ref")
        )
    
    results = []
    
    for idx, row in df.iterrows():
        original = row[actual_original_col]
        translation = row.get(translation_col, None)
        reference = row.get(reference_col, None) if df_ref is not None else None
        
        eval_result = {
            'index': idx,
            'original_text': original if pd.notna(original) else "",
            'translation': translation if pd.notna(translation) else "",
            'has_translation': pd.notna(translation) and translation != "",
        }
        
        # Basic metrics
        if translation and pd.notna(translation) and translation != "":
            # Length metrics
            original_str = str(original) if pd.notna(original) else ""
            translation_str = str(translation) if pd.notna(translation) else ""
            eval_result['original_length'] = len(original_str)
            eval_result['translation_length'] = len(translation_str)
            eval_result['length_ratio'] = len(translation_str) / len(original_str) if len(original_str) > 0 else 0
            
            # Text comparison metrics
            text_metrics = compare_text_metrics(original_str, translation_str)
            eval_result.update(text_metrics)
            
            # BLEU score (if reference available)
            if reference and BLEU_AVAILABLE and pd.notna(reference):
                bleu = calculate_bleu_score(str(reference), translation_str)
                eval_result['bleu_score'] = bleu
            
            # NOTE: Fuzzy similarity removed - not useful for cross-language comparison
            # Even perfect translations will have low scores because strings are different
            # This metric is only useful for same-language comparison
            
            # Language detection
            if LANGDETECT_AVAILABLE:
                try:
                    detected_lang = detect(translation_str)
                    eval_result['detected_language'] = detected_lang
                except:
                    eval_result['detected_language'] = None
        else:
            original_str = str(original) if pd.notna(original) else ""
            eval_result['original_length'] = len(original_str)
            eval_result['translation_length'] = 0
            eval_result['length_ratio'] = 0
        
        results.append(eval_result)
    
    results_df = pd.DataFrame(results)
    
    return results_df


def generate_evaluation_report(results_df: pd.DataFrame, output_file: str):
    """Generate a comprehensive evaluation report"""
    
    logger.info("Generating evaluation report...")
    
    # Filter to only successful translations
    successful = results_df[results_df['has_translation'] == True]
    
    if len(successful) == 0:
        logger.warning("No successful translations to evaluate!")
        return
    
    report = []
    report.append("=" * 70)
    report.append("MACHINE TRANSLATION EVALUATION REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Basic statistics
    report.append("BASIC STATISTICS:")
    report.append(f"  Total translations: {len(results_df)}")
    report.append(f"  Successful translations: {len(successful)} ({len(successful)/len(results_df)*100:.1f}%)")
    report.append(f"  Failed/Empty translations: {len(results_df) - len(successful)} ({(len(results_df) - len(successful))/len(results_df)*100:.1f}%)")
    report.append("")
    
    # Length metrics
    if 'length_ratio' in successful.columns:
        report.append("LENGTH METRICS:")
        report.append(f"  Average length ratio: {successful['length_ratio'].mean():.3f}")
        report.append(f"  Min length ratio: {successful['length_ratio'].min():.3f}")
        report.append(f"  Max length ratio: {successful['length_ratio'].max():.3f}")
        report.append(f"  Std deviation: {successful['length_ratio'].std():.3f}")
        report.append("")
    
    # Word count metrics
    if 'word_count_ratio' in successful.columns:
        report.append("WORD COUNT METRICS:")
        report.append(f"  Average word count ratio: {successful['word_count_ratio'].mean():.3f}")
        report.append(f"  Average original words: {successful['original_word_count'].mean():.2f}")
        report.append(f"  Average translation words: {successful['translation_word_count'].mean():.2f}")
        report.append("")
    
    # BLEU scores
    if 'bleu_score' in successful.columns and successful['bleu_score'].notna().any():
        bleu_scores = successful['bleu_score'].dropna()
        report.append("BLEU SCORES (vs reference):")
        report.append(f"  Average BLEU: {bleu_scores.mean():.4f}")
        report.append(f"  Min BLEU: {bleu_scores.min():.4f}")
        report.append(f"  Max BLEU: {bleu_scores.max():.4f}")
        report.append(f"  Translations with BLEU > 0.5: {(bleu_scores > 0.5).sum()} ({(bleu_scores > 0.5).sum()/len(bleu_scores)*100:.1f}%)")
        report.append("")
    
    # NOTE: Fuzzy similarity removed - not meaningful for cross-language evaluation
    
    # Language detection
    if 'detected_language' in successful.columns:
        lang_counts = successful['detected_language'].value_counts()
        report.append("LANGUAGE DETECTION:")
        for lang, count in lang_counts.head(5).items():
            report.append(f"  {lang}: {count} ({count/len(successful)*100:.1f}%)")
        report.append("")
    
    # NOTE: Syntactic complexity removed - requires English model, won't work for Turkish/Russian/Spanish
    
    report.append("=" * 70)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"Evaluation report saved to: {output_file}")
    print(report_text)
    
    return report_text


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate machine translation quality")
    parser.add_argument("--translation_csv", required=True, help="Path to translation CSV")
    parser.add_argument("--original_csv", default="data_gen/judge_experiment/compare_judge.csv", 
                       help="Path to original CSV (default: judge experiment data)")
    parser.add_argument("--original_col", default="nl", help="Column name for original text")
    parser.add_argument("--translation_col", default="translation", help="Column name for translation")
    parser.add_argument("--reference_csv", default=None, help="Optional: Path to reference translations")
    parser.add_argument("--reference_col", default=None, help="Column name for reference translations")
    parser.add_argument("--output_csv", default="machine_translation/translation_evaluation_results.csv",
                       help="Output CSV for detailed results")
    parser.add_argument("--output_report", default="machine_translation/translation_evaluation_report.txt",
                       help="Output text report")
    
    args = parser.parse_args()
    
    # Evaluate translations
    results_df = evaluate_translation_quality(
        original_csv=args.original_csv,
        translation_csv=args.translation_csv,
        original_col=args.original_col,
        translation_col=args.translation_col,
        reference_csv=args.reference_csv,
        reference_col=args.reference_col
    )
    
    # Save detailed results
    results_df.to_csv(args.output_csv, index=False)
    logger.info(f"Detailed evaluation results saved to: {args.output_csv}")
    
    # Generate and save report
    generate_evaluation_report(results_df, args.output_report)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

