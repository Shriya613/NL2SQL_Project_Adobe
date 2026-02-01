# Translation Filtering Mechanisms

## Overview

The improved machine translation pipeline includes multiple filtering mechanisms to remove wrong or low-quality translations.

## Current Filtering Mechanisms

### 1. **Entity-Level Filtering** (Pre-Translation)

#### Entity Type Filtering
- **Excludes**: DATE, TIME, ORDINAL, PERCENT, QUANTITY, CARDINAL, LANGUAGE, MONEY
- **Reason**: These entity types don't need Wikipedia translation (dates/numbers are language-independent)
- **Location**: `ImprovedNER` class, `dont_include` list

#### Entity Priority Filtering
- **Prioritizes**: PERSON, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART
- **Reason**: These are most likely to have Wikipedia pages and need translation
- **Location**: `ImprovedNER` class, `priority_types` list

#### Entity Deduplication
- Removes duplicate entities from multiple NER models
- Case-insensitive deduplication
- **Location**: `ImprovedNER.merge_entities()`

### 2. **Wikipedia API Filtering** (Entity Translation)

#### Disambiguation Page Filtering
- **Filters out**: Wikipedia disambiguation pages
- **Patterns**:
  - Turkish: `(anlam ayrımı)`
  - Russian: `(значения)`
  - Spanish: `(desambiguación)`
- **Location**: `EntityTranslator.get_lang_title()`, line 316

#### Entity Validation
- Only translates entities that have Wikipedia langlinks
- Returns `None` if no langlink found (entity not translated)
- **Location**: `EntityTranslator.get_lang_title()`

#### Suspicious Pattern Detection
- **Filters out**:
  - Empty or whitespace-only translations
  - Translations with only punctuation/symbols
  - Translations with only numbers
  - Translations shorter than 2 characters
- **Location**: `TranslationQualityFilter.check_suspicious_patterns()`

#### Basic Quality Checks
- Validates translation is not None/empty
- Checks all quality metrics
- **Location**: `TranslationQualityFilter.check_basic_quality()`

## Filtering Flow

```
Input Text
    ↓
1. Entity Extraction (NER)
    ├─ Filter by entity type (exclude DATE, TIME, etc.)
    ├─ Prioritize important types (PERSON, ORG, etc.)
    └─ Deduplicate entities
    ↓
2. Entity Translation (Wikipedia API)
    ├─ Check cache first
    ├─ Fetch from Wikipedia if not cached
    ├─ Filter disambiguation pages
    └─ Return None if no langlink found
    ↓
3. Full Text Translation (NLLB)
    ├─ Translate with NLLB model
    └─ Handle translation errors
    ↓
4. Quality Filtering
    ├─ Check length ratio (0.3 - 3.0)
    ├─ Detect language (verify target language)
    ├─ Check suspicious patterns
    └─ Reject if any check fails
    ↓
Output (Accepted) or Filtered Out
```

## Filtering Statistics

The pipeline now tracks:
- **Successful**: Translations that passed all filters
- **Failed**: Translations that failed (model error)
- **Filtered Out**: Translations rejected by quality filter

## Configuration

### Quality Filter Parameters

```python
TranslationQualityFilter(
    target_lang="tr",
    min_length_ratio=0.3,  # Minimum acceptable length
    max_length_ratio=3.0   # Maximum acceptable length
)
```

### Adjustable Thresholds

You can adjust filtering strictness:
- **Stricter**: Lower `max_length_ratio` (e.g., 2.0), require language match
- **More lenient**: Higher `max_length_ratio` (e.g., 4.0), allow language mismatch

## Installation Requirements

For quality filtering to work:
```bash
pip install langdetect
```

If not installed, the pipeline will:
- Still work (accepts all translations)
- Log a warning
- Skip quality filtering

## Example Filtering Scenarios

### Scenario 1: Empty Translation
```
Original: "What is the total number?"
Translation: ""
Result: ❌ FILTERED - "Translation too short"
```

### Scenario 2: Wrong Language
```
Original: "What is the total number?"
Translation: "What is the total number?"  (English, not Turkish)
Result: ❌ FILTERED - Language mismatch
```

### Scenario 3: Too Short
```
Original: "What is the total number of players in each team?"
Translation: "?"  (Only 1 character)
Result: ❌ FILTERED - Length ratio too low
```

### Scenario 4: Valid Translation
```
Original: "What is the total number?"
Translation: "Toplam sayı nedir?"  (Turkish, correct length)
Result: ✅ ACCEPTED
```

## Output Format

When quality filter is enabled, output CSV includes:
- `filtered`: Boolean (True if filtered out)
- `filter_reason`: Reason for filtering (if filtered)
- `length_ratio`: Length ratio metric
- `detected_language`: Detected language code

## Future Improvements

Potential additional filters:
1. **BLEU Score**: Compare with reference translations
2. **Semantic Similarity**: Use embeddings to check meaning preservation
3. **SQL Term Preservation**: Check if SQL-related terms are preserved
4. **Repetition Detection**: Filter translations with excessive repetition
5. **Confidence Scores**: Use model confidence scores if available

## Summary

**Current Filtering:**
- ✅ Entity type filtering
- ✅ Disambiguation page filtering
- ✅ Length ratio validation
- ✅ Language detection
- ✅ Suspicious pattern detection

**Missing (Could Add):**
- ❌ BLEU score validation
- ❌ Semantic similarity checks
- ❌ Model confidence thresholds
- ❌ Manual review queue for edge cases

