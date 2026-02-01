# Machine Translation Evaluation Guide

## Overview

The `evaluate_translations.py` script uses code from the `evaluation/` folder to assess the quality of your machine translation pipeline. It provides comprehensive metrics to understand how well translations preserve meaning, structure, and quality.

## Available Metrics

### 1. **Basic Statistics**
- Total translations processed
- Success rate (translations that are not empty/null)
- Failure rate

### 2. **Length Metrics** (from `nl_evaluation.py`)
- **Length Ratio**: Translation length / Original length
  - Helps identify if translations are too short or too long
  - Ideal range: 0.7 - 1.3 (varies by language)
- **Word Count Ratio**: Translation words / Original words
  - Shows if word count is preserved
- **Character Count Ratio**: Translation chars / Original chars

### 3. **Syntactic Complexity** (from `nl_evaluation.py`)
- **Complexity Score**: Measures syntactic complexity
  - Tokens before main verb
  - Constituents per word
  - Subordinate/coordinate clauses
- Compares original vs translation complexity to see if structure is preserved

### 4. **Fuzzy Similarity** (from `sql_evaluation.py`)
- Uses `fuzzywuzzy` to calculate string similarity
- Lower scores are expected for different languages (English vs Turkish)
- Useful for detecting if translation is completely wrong (very low similarity might indicate issues)

### 5. **BLEU Score** (if reference translations available)
- Standard machine translation metric
- Requires reference translations for comparison
- Range: 0.0 - 1.0 (higher is better)

### 6. **Language Detection**
- Verifies translations are in the correct target language
- Uses `langdetect` library
- Flags translations that are in the wrong language

## Usage

### Basic Evaluation

```bash
python machine_translation/evaluate_translations.py \
    --translation_csv final_data/tr_translations_nllb_improved_ner.csv \
    --translation_col translation
```

### With Original Dataset (if translation CSV doesn't have original_text)

```bash
python machine_translation/evaluate_translations.py \
    --translation_csv final_data/tr_translations_nllb_improved_ner.csv \
    --original_csv data_gen/judge_experiment/compare_judge.csv \
    --original_col nl \
    --translation_col translation
```

### With Reference Translations (for BLEU score)

```bash
python machine_translation/evaluate_translations.py \
    --translation_csv final_data/tr_translations_nllb_improved_ner.csv \
    --translation_col translation \
    --reference_csv path/to/reference_translations.csv \
    --reference_col reference_translation
```

## Output Files

1. **Detailed Results CSV** (`translation_evaluation_results.csv`)
   - Contains all metrics for each translation pair
   - Can be used for further analysis or visualization

2. **Evaluation Report** (`translation_evaluation_report.txt`)
   - Summary statistics
   - Easy-to-read metrics overview

## Interpreting Results

### Good Translation Indicators:
- ✅ Length ratio between 0.7 - 1.3
- ✅ Language detection matches target language (>95%)
- ✅ Complexity scores similar between original and translation
- ✅ High BLEU score (>0.5) if reference available
- ✅ Low failure rate (<5%)

### Warning Signs:
- ⚠️ Very low length ratio (<0.5) - translations might be truncated
- ⚠️ Very high length ratio (>2.0) - translations might be verbose
- ⚠️ Wrong language detected - translation quality issue
- ⚠️ High failure rate - pipeline issues

## Example Output

```
======================================================================
MACHINE TRANSLATION EVALUATION REPORT
======================================================================

BASIC STATISTICS:
  Total translations: 498
  Successful translations: 497 (99.8%)
  Failed/Empty translations: 1 (0.2%)

LENGTH METRICS:
  Average length ratio: 0.906
  Min length ratio: 0.385
  Max length ratio: 1.542
  Std deviation: 0.143

WORD COUNT METRICS:
  Average word count ratio: 0.762
  Average original words: 21.91
  Average translation words: 16.65

LANGUAGE DETECTION:
  tr: 496 (99.8%)
  ro: 1 (0.2%)
```

## Dependencies

The script uses:
- `evaluation/nl_evaluation.py` - For syntactic complexity and text metrics
- `evaluation/sql_evaluation.py` - For fuzzy matching
- `fuzzywuzzy` - String similarity
- `langdetect` - Language detection
- `spacy` - NLP processing (optional, for complexity metrics)
- `nltk` - BLEU scores (optional)

## Limitations

1. **Syntactic Complexity**: Only works for English (spaCy English model)
   - Can't compare complexity between English and Turkish directly
   - Use length ratios instead for cross-language comparison

2. **BLEU Scores**: Require reference translations
   - Not available if you don't have gold-standard translations

3. **Fuzzy Similarity**: Low scores are expected for different languages
   - Use for detecting completely wrong translations, not quality assessment

## Next Steps

1. **Visualization**: Use `evaluation/visualize_metrics.py` as inspiration to create translation-specific visualizations
2. **Custom Metrics**: Add domain-specific metrics (e.g., SQL keyword preservation)
3. **Comparison**: Compare different translation models/pipelines using the same evaluation script

