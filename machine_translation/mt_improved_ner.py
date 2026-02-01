"""
Improved Machine Translation Pipeline with Enhanced NER
- Uses better NER models and techniques
- Caches entity translations
- Smart entity replacement
- Better error handling
"""

import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm
import torch
import requests
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import quality filter
try:
    import sys
    # Add machine_translation directory to path for imports
    mt_dir = Path(__file__).parent
    if str(mt_dir) not in sys.path:
        sys.path.insert(0, str(mt_dir))
    from translation_quality_filter import TranslationQualityFilter
    QUALITY_FILTER_AVAILABLE = True
except ImportError as e:
    QUALITY_FILTER_AVAILABLE = False
    logger.warning(f"Translation quality filter not available: {e}. Install langdetect: pip install langdetect")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache directory for entity translations
CACHE_DIR = Path("machine_translation/.cache")
CACHE_DIR.mkdir(exist_ok=True)

# Load dataset (same as previous version)
df = pd.read_csv("data_gen/judge_experiment/compare_judge.csv")
df.dropna(subset=['nl'], inplace=True)

# Initialize NER models - using multiple approaches for better coverage
try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    logger.info("spaCy NER model loaded")
except Exception as e:
    nlp_spacy = None
    SPACY_AVAILABLE = False
    logger.warning(f"spaCy not available: {e}")

# Flair NER - excellent accuracy, often better than BERT for NER
ner_flair = None
FLAIR_NER_AVAILABLE = False
try:
    from flair.models import SequenceTagger
    from flair.data import Sentence
    ner_flair = SequenceTagger.load("flair/ner-english-fast")
    FLAIR_NER_AVAILABLE = True
    logger.info("Flair NER model loaded")
except Exception as e:
    ner_flair = None
    FLAIR_NER_AVAILABLE = False
    logger.warning(f"Flair NER not available: {e}. Install with: pip install flair")

# BERT-based NER for better accuracy (gracefully disable if numpy is missing or incompatible)
ner_bert = None
BERT_NER_AVAILABLE = False
try:
    import numpy as _np  # noqa: F401
    try:
        ner_bert = pipeline(
            "token-classification",
            model="dslim/bert-large-NER",
            device=device,
            aggregation_strategy="simple",
        )
        BERT_NER_AVAILABLE = True
        logger.info("BERT NER model loaded")
    except Exception as e:
        ner_bert = None
        BERT_NER_AVAILABLE = False
        logger.warning(
            "BERT NER disabled (transformers/torch setup issue). Details: %s",
            e,
        )
except Exception as e:
    ner_bert = None
    BERT_NER_AVAILABLE = False
    logger.warning(
        "NumPy not available or incompatible; disabling BERT NER. Details: %s",
        e,
    )

# Translation pipelines
tr_translation = pipeline("translation", 
                        model="facebook/nllb-200-distilled-600M", 
                        src_lang="eng_Latn", 
                        tgt_lang="tur_Latn", 
                        device=device)
ru_translation = pipeline("translation", 
                         model="facebook/nllb-200-distilled-600M", 
                         src_lang="eng_Latn", 
                         tgt_lang="rus_Cyrl", 
                         device=device)
es_translation = pipeline("translation", 
                         model="facebook/nllb-200-distilled-600M", 
                         src_lang="eng_Latn", 
                         tgt_lang="spa_Latn", 
                         device=device)


class ImprovedNER:
    """Enhanced NER with multiple models and better filtering"""
    
    def __init__(self):
        self.spacy_model = nlp_spacy
        self.bert_model = ner_bert
        self.flair_model = ner_flair
        
        # Entity types to exclude (common in SQL queries but not needed for translation)
        self.dont_include = [
            "DATE", "TIME", "ORDINAL", "PERCENT", "QUANTITY", 
            "CARDINAL", "LANGUAGE", "MONEY"
        ]
        
        # Entity types to prioritize (likely to have Wikipedia pages)
        self.priority_types = [
            "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"
        ]
    
    def extract_entities_spacy(self, text: str) -> List[Dict[str, str]]:
        """Extract entities using spaCy"""
        entities = []
        if not self.spacy_model:
            return entities
        
        try:
            doc = self.spacy_model(text)
            for ent in doc.ents:
                if ent.label_ not in self.dont_include:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "source": "spacy"
                    })
        except Exception as e:
            logger.warning(f"spaCy NER failed: {e}")
        
        return entities
    
    def extract_entities_flair(self, text: str) -> List[Dict[str, str]]:
        """Extract entities using Flair (often more accurate than BERT)"""
        entities = []
        if not self.flair_model:
            return entities
        
        try:
            sentence = Sentence(text)
            self.flair_model.predict(sentence)
            
            for entity in sentence.get_spans("ner"):
                entity_text = entity.text
                entity_label = entity.tag
                
                # Filter out unwanted types and short entities
                if (entity_label not in self.dont_include and 
                    len(entity_text) > 2):
                    entities.append({
                        "text": entity_text,
                        "label": entity_label,
                        "start": entity.start_position,
                        "end": entity.end_position,
                        "source": "flair"
                    })
        except Exception as e:
            logger.warning(f"Flair NER failed: {e}")
        
        return entities
    
    def extract_entities_bert(self, text: str) -> List[Dict[str, str]]:
        """Extract entities using BERT"""
        entities = []
        if not self.bert_model:
            return entities
        
        try:
            results = self.bert_model(text)
            for ent in results:
                # BERT returns different format
                if isinstance(ent, dict):
                    entity_text = ent.get("word", "").strip()
                    entity_label = ent.get("entity_group", "")
                    
                    # Filter out unwanted types and short entities
                    if (entity_label not in self.dont_include and 
                        len(entity_text) > 2 and 
                        entity_text.isalnum() or " " in entity_text):
                        entities.append({
                            "text": entity_text,
                            "label": entity_label,
                            "start": ent.get("start", 0),
                            "end": ent.get("end", 0),
                            "source": "bert"
                        })
        except Exception as e:
            logger.warning(f"BERT NER failed: {e}")
        
        return entities
    
    def merge_entities(self, entities: List[Dict[str, str]]) -> List[str]:
        """Merge entities from multiple sources and deduplicate"""
        # Combine all entities
        all_entities = []
        seen_texts = set()
        
        # Sort by priority (priority types first, then others)
        priority_entities = []
        other_entities = []
        
        for ent in entities:
            text = ent["text"].strip()
            if text and text.lower() not in seen_texts:
                seen_texts.add(text.lower())
                
                if ent["label"] in self.priority_types:
                    priority_entities.append(text)
                else:
                    other_entities.append(text)
        
        # Return priority entities first
        return priority_entities + other_entities
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract entities using multiple models and merge results"""
        all_entities = []
        
        # Extract from spaCy
        spacy_entities = self.extract_entities_spacy(text)
        all_entities.extend(spacy_entities)
        
        # Extract from Flair (preferred over BERT if available)
        if FLAIR_NER_AVAILABLE:
            flair_entities = self.extract_entities_flair(text)
            all_entities.extend(flair_entities)
        
        # Extract from BERT (fallback if Flair not available)
        if BERT_NER_AVAILABLE and not FLAIR_NER_AVAILABLE:
            bert_entities = self.extract_entities_bert(text)
            all_entities.extend(bert_entities)
        
        # Merge and deduplicate
        merged = self.merge_entities(all_entities)
        
        return merged


class EntityTranslator:
    """Handle entity translation with caching"""
    
    def __init__(self, target_lang: str):
        self.target_lang = target_lang
        self.cache_file = CACHE_DIR / f"entity_cache_{target_lang}.json"
        self.cache = self._load_cache()
        
        self.disambiguation = {
            "tr": "(anlam ayrımı)",
            "ru": "(значения)",
            "es": "(desambiguación)"
        }
    
    def _load_cache(self) -> Dict[str, Optional[str]]:
        """Load cached entity translations"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save entity translations to cache"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_lang_title(self, en_title: str, max_retries: int = 2) -> Optional[str]:
        """Get translated Wikipedia title with caching and retry logic"""
        # Check cache first
        cache_key = f"{en_title}_{self.target_lang}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = "https://en.wikipedia.org/w/api.php"
        headers = {"User-Agent": "nl2sql-translation-improved/1.0"}
        
        for attempt in range(max_retries):
            try:
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": en_title,
                    "prop": "langlinks",
                    "lllang": self.target_lang
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    if "langlinks" in page:
                        translated = page["langlinks"][0]["*"]
                        # Filter out disambiguation pages
                        if self.disambiguation.get(self.target_lang) not in translated:
                            self.cache[cache_key] = translated
                            self._save_cache()
                            return translated
                
                # No langlink found
                self.cache[cache_key] = None
                self._save_cache()
                return None
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching Wikipedia for '{en_title}' (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.warning(f"Error fetching Wikipedia for '{en_title}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        self.cache[cache_key] = None
        self._save_cache()
        return None
    
    def replace_entities_smart(self, text: str, entity_translations: Dict[str, str]) -> str:
        """Replace entities using word boundaries to avoid partial matches"""
        result = text
        
        # Sort by length (longest first) to handle overlapping entities
        sorted_entities = sorted(
            entity_translations.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        for entity, translation in sorted_entities:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(entity) + r'\b'
            result = re.sub(pattern, translation, result, flags=re.IGNORECASE)
        
        return result
    
    def process_entities(self, text: str, entities: List[str]) -> str:
        """Process entities: translate and replace in text"""
        if not entities:
            return text
        
        entity_translations = {}
        for entity in entities:
            translated = self.get_lang_title(entity)
            if translated:
                entity_translations[entity] = translated
                logger.debug(f"Translated entity: {entity} -> {translated}")
        
        if entity_translations:
            text = self.replace_entities_smart(text, entity_translations)
        
        return text


def run_translation(text: str, translation_pipeline) -> Optional[str]:
    """Translate text with error handling"""
    try:
        results = translation_pipeline(text)
        if results and len(results) > 0:
            return results[0]["translation_text"]
    except Exception as e:
        logger.error(f"Translation failed: {e}")
    
    return None


def main(lang: str = "tr"):
    """Main translation function"""
    logger.info(f"Starting translation to {lang}")
    logger.info(f"Dataset size: {len(df)} rows")
    
    # Initialize components
    ner_extractor = ImprovedNER()
    entity_translator = EntityTranslator(target_lang=lang)
    
    # Initialize quality filter if available
    quality_filter = None
    if QUALITY_FILTER_AVAILABLE:
        quality_filter = TranslationQualityFilter(target_lang=lang)
        logger.info("Translation quality filter enabled")
    else:
        logger.warning("Translation quality filter disabled - all translations will be accepted")
    
    # Get translation pipeline
    pipelines = {
        "tr": tr_translation,
        "ru": ru_translation,
        "es": es_translation
    }
    
    if lang not in pipelines:
        logger.error(f"Unsupported language: {lang}")
        return
    
    translation_pipeline = pipelines[lang]
    
    # Prepare results (using list for better performance)
    results = []
    filtered_results = []  # Separate list for filtered translations
    stats = {
        "total": 0,
        "successful": 0,
        "failed": 0,
        "filtered_out": 0,  # New: translations filtered by quality
        "entities_found": 0,
        "entities_translated": 0
    }
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Translating to {lang}"):
        stats["total"] += 1
        original_text = row["nl"]
        
        # Extract entities with improved NER
        entities = ner_extractor.extract_entities(original_text)
        if entities:
            stats["entities_found"] += len(entities)
            logger.debug(f"Found {len(entities)} entities: {entities}")
        
        # Process entities: translate and replace
        text_to_translate = entity_translator.process_entities(original_text, entities)
        if text_to_translate != original_text:
            stats["entities_translated"] += len(entity_translator.cache)
        
        # Translate the text
        translation = run_translation(text_to_translate, translation_pipeline)
        
        if translation:
            # Apply quality filter if available
            if quality_filter:
                should_keep, filtered_translation, metrics = quality_filter.filter_translation(
                    original_text, translation
                )
                
                if not should_keep:
                    stats["filtered_out"] += 1
                    logger.warning(
                        f"Translation filtered out: {metrics.get('reason', 'Unknown reason')}. "
                        f"Original: {original_text[:50]}..."
                    )
                    # Save to filtered results (separate file)
                    filtered_results.append({
                        "original_text": original_text,
                        "rejected_translation": translation,  # Save the bad translation for review
                        "entities_found": len(entities),
                        "entities": ", ".join(entities) if entities else "",
                        "filter_reason": metrics.get('reason', 'Quality check failed'),
                        "length_ratio": metrics.get('length_ratio', 0.0),
                        "detected_language": metrics.get('detected_language', 'unknown'),
                        "original_length": len(original_text),
                        "translation_length": len(translation) if translation else 0
                    })
                    # Also add to main results with None translation
                    results.append({
                        "original_text": original_text,
                        "translation": None,
                        "entities_found": len(entities),
                        "entities": ", ".join(entities) if entities else "",
                        "filtered": True,
                        "filter_reason": metrics.get('reason', 'Quality check failed')
                    })
                else:
                    stats["successful"] += 1
                    results.append({
                        "original_text": original_text,
                        "translation": filtered_translation,
                        "entities_found": len(entities),
                        "entities": ", ".join(entities) if entities else "",
                        "filtered": False,
                        "length_ratio": metrics.get('length_ratio', 0.0),
                        "detected_language": metrics.get('detected_language', 'unknown')
                    })
            else:
                # No quality filter - accept all translations
                stats["successful"] += 1
                results.append({
                    "original_text": original_text,
                    "translation": translation,
                    "entities_found": len(entities),
                    "entities": ", ".join(entities) if entities else ""
                })
        else:
            stats["failed"] += 1
            logger.warning(f"Translation failed for: {original_text[:50]}...")
            results.append({
                "original_text": original_text,
                "translation": None,
                "entities_found": len(entities),
                "entities": ", ".join(entities) if entities else ""
            })
        
        # Save periodically
        if len(results) % 50 == 0:
            temp_df = pd.DataFrame(results)
            output_file = f"final_data/{lang}_translations_nllb_improved_ner.csv"
            temp_df.to_csv(output_file, index=False)
            
            # Also save filtered translations periodically
            if filtered_results:
                filtered_df = pd.DataFrame(filtered_results)
                filtered_file = f"final_data/{lang}_translations_nllb_improved_ner_FILTERED.csv"
                filtered_df.to_csv(filtered_file, index=False)
            
            logger.info(f"Saved progress: {len(results)} translations ({len(filtered_results)} filtered)")
    
    # Create final DataFrame and save
    results_df = pd.DataFrame(results)
    output_file = f"final_data/{lang}_translations_nllb_improved_ner.csv"
    results_df.to_csv(output_file, index=False)
    
    # Save filtered translations to separate file
    if filtered_results:
        filtered_df = pd.DataFrame(filtered_results)
        filtered_file = f"final_data/{lang}_translations_nllb_improved_ner_FILTERED.csv"
        filtered_df.to_csv(filtered_file, index=False)
        logger.info(f"Filtered translations saved to: {filtered_file}")
    
    # Save entity cache
    entity_translator._save_cache()
    
    # Print statistics
    logger.info("=" * 70)
    logger.info("Translation Statistics")
    logger.info("=" * 70)
    logger.info(f"Total translations: {stats['total']}")
    logger.info(f"Successful: {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)")
    logger.info(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    if stats['filtered_out'] > 0:
        logger.info(f"Filtered out (low quality): {stats['filtered_out']} ({stats['filtered_out']/stats['total']*100:.1f}%)")
    logger.info(f"Entities found: {stats['entities_found']}")
    logger.info(f"Unique entities translated: {len(entity_translator.cache)}")
    logger.info(f"Output saved to: {output_file}")
    if filtered_results:
        logger.info(f"Filtered translations saved to: final_data/{lang}_translations_nllb_improved_ner_FILTERED.csv")
    logger.info("=" * 70)


if __name__ == "__main__":
    import sys
    
    # Default to Turkish, allow command line argument
    lang = sys.argv[1] if len(sys.argv) > 1 else "tr"
    
    if lang not in ["tr", "ru", "es"]:
        logger.error("Language must be one of: tr, ru, es")
        sys.exit(1)
    
    main(lang=lang)

