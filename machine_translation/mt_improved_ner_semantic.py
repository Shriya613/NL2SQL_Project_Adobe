"""
Improved Machine Translation Pipeline with Enhanced NER + Semantic QC
- Enhanced NER coverage/caching (same as mt_improved_ner.py)
- Adds semantic similarity scoring using multilingual sentence transformers
- Adds back-translation validation for stronger quality checks
- Keeps legacy heuristics (length/language) as fast first-pass filters
"""

import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm
import torch
import requests
import re
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from machine_translation.llm import run_gpt, parse_json, TRANSLATION_PROMPT
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np
    SEMANTIC_AVAILABLE = True
except Exception as semantic_exc:  # noqa: F841
    SentenceTransformer = None
    np = None  # type: ignore
    SEMANTIC_AVAILABLE = False

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose debug logs from third-party libraries
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

if not SEMANTIC_AVAILABLE:
    logger.warning("Semantic similarity disabled (sentence-transformers not installed). Install with: pip install sentence-transformers")

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

# NLLB codes mapping (for back-translation)
NLLB_CODES = {
    "tr": "tur_Latn",
    "ru": "rus_Cyrl",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "hi": "hin_Deva",
    "de": "deu_Latn",
    "uk": "ukr_Cyrl",
    "kk": "kaz_Cyrl",
    "ro": "ron_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "en": "eng_Latn"
}

# Language name mapping for OpenAI prompts
LANGUAGE_NAMES = {
    "tr": "Turkish",
    "ru": "Russian",
    "es": "Spanish",
    "zh": "Chinese",
    "ja": "Japanese",
    "hi": "Hindi",
    "de": "German",
    "uk": "Ukrainian",
    "kk": "Kazakh",
    "ro": "Romanian",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "en": "English"
}

# Alias for backward compatibility
LANG_NAMES = LANGUAGE_NAMES

# Shared model cache
_shared_model = None
_shared_tokenizer = None

def get_shared_model():
    global _shared_model, _shared_tokenizer
    if _shared_model is None:
        model_name = "facebook/nllb-200-distilled-600M"
        logger.info(f"Loading shared NLLB model: {model_name}")
        _shared_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _shared_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return _shared_model, _shared_tokenizer

def get_translation_pipeline(src_lang, tgt_lang):
    model, tokenizer = get_shared_model()
    return pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=device
    )

class SemanticSimilarityScorer:
    """Lazy-loading semantic similarity scorer using sentence-transformers"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self.available = SEMANTIC_AVAILABLE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_model(self):
        if not self.available:
            return None
        if self.model is None:
            try:
                logger.info("Loading semantic similarity model: %s", self.model_name)
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info("Semantic similarity model ready")
            except Exception as e:
                logger.error("Failed to load semantic similarity model: %s", e)
                self.available = False
                self.model = None
        return self.model

    def score(self, text_a: str, text_b: str) -> Optional[float]:
        """Return cosine similarity between two texts"""
        if not self.available or not text_a or not text_b:
            return None

        model = self._ensure_model()
        if not model:
            return None

        try:
            # Use torch tensors instead of numpy to avoid numpy dependency issues
            embeddings = model.encode(
                [text_a, text_b],
                convert_to_tensor=True,  # Use torch tensors
                show_progress_bar=False,
            )
            vec_a, vec_b = embeddings[0], embeddings[1]
            
            # Calculate cosine similarity using torch
            # Cosine similarity = dot(a, b) / (norm(a) * norm(b))
            dot_product = torch.dot(vec_a, vec_b)
            norm_a = torch.norm(vec_a)
            norm_b = torch.norm(vec_b)
            denom = (norm_a * norm_b) + 1e-8
            
            if denom == 0:
                return None
            
            similarity = float((dot_product / denom).item())
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.warning("Semantic similarity computation failed: %s", e)
            return None


class TranslationQualityGate:
    """
    Multi-stage quality gate that combines:
    1) Heuristic filter (length/language/symbol checks)
    2) Semantic similarity between English and target text
    3) Back-translation similarity (target -> English)
    """

    def __init__(self, lang: str, quality_filter: Optional["TranslationQualityFilter"] = None):
        self.lang = lang
        self.quality_filter = quality_filter
        self.semantic_scorer = SemanticSimilarityScorer() if SEMANTIC_AVAILABLE else None
        
        # Use Gemini for back-translation (no pipeline needed)
        self.back_translator = "gemini"  # Marker to use Gemini instead of NLLB

        self.semantic_threshold = 0.65
        self.back_translation_threshold = 0.70
        self.combined_threshold = 0.72
        self.manual_review_band = (0.60, 0.72)

    def _combine_scores(self, semantic_score: Optional[float], back_score: Optional[float]) -> Optional[float]:
        scores = [score for score in [semantic_score, back_score] if score is not None]
        if not scores:
            return None
        if len(scores) == 1:
            return scores[0]
        # Weight back-translation slightly higher (more sensitive to information loss)
        print("SEMANTIC SCORE: ", semantic_score)
        print("BACK SCORE: ", back_score)
        print("COMBINED SCORE: ", (0.4 * semantic_score) + (0.6 * back_score))
        return (0.4 * semantic_score) + (0.6 * back_score)  # type: ignore

    def _decision(self, semantic_score: Optional[float], back_score: Optional[float], combined_score: Optional[float]) -> Tuple[bool, str]:
        """
        Return (should_keep, reason)
        """
        if combined_score is not None:
            print("COMBINED SCORE: ", combined_score)
            print("THRESHOLD: ", self.combined_threshold)
            print("SHOULD KEEP: ", combined_score >= self.combined_threshold)
            if combined_score >= self.combined_threshold:
                print("RETURNING TRUE, SEMANTIC_BACK_PASS")
                return True, "semantic_back_pass"
            if self.manual_review_band[0] <= combined_score < self.manual_review_band[1]:
                return False, "semantic_back_review"
            return False, "semantic_back_fail"

        if back_score is not None:
            print("BACK SCORE: ", back_score)
            print("BACK THRESHOLD: ", self.back_translation_threshold)
            print("SHOULD KEEP: ", back_score >= self.back_translation_threshold)
            if back_score >= self.back_translation_threshold:
                return True, "back_only_pass"
            return False, "back_only_fail"

        if semantic_score is not None:
            print("SEMANTIC SCORE: ", semantic_score)
            print("SEMANTIC THRESHOLD: ", self.semantic_threshold)
            print("SHOULD KEEP: ", semantic_score >= self.semantic_threshold)
            if semantic_score >= self.semantic_threshold:
                return True, "semantic_only_pass"
            return False, "semantic_only_fail"

        # If no advanced metrics are available, fallback to accepting (heuristics already ran)
        return True, "semantic_metrics_unavailable"

    def evaluate(self, original: str, translation: str) -> Tuple[bool, str, str, Dict[str, Optional[float]]]:
        """
        Run the full quality gate. Returns:
        (should_keep, stage_reason, cleaned_translation, metrics_dict)
        """
        metrics: Dict[str, Optional[float]] = {}

        # Stage 1: heuristic filter
        if self.quality_filter:
            heuristics_pass, normalized_translation, heuristic_metrics = self.quality_filter.filter_translation(
                original, translation
            )
            metrics.update(heuristic_metrics)
            if not heuristics_pass:
                stage_reason = heuristic_metrics.get("reason", "heuristic_filter_fail")
                metrics["quality_stage"] = stage_reason
                return False, stage_reason, translation, metrics
            translation = normalized_translation

        # Stage 2: semantic + back translation
        semantic_score = self.semantic_scorer.score(original, translation) if self.semantic_scorer else None  # type: ignore
        metrics["semantic_similarity"] = semantic_score

        back_translation_text = None
        back_translation_score = None
        if self.back_translator and translation:
            # Use Gemini for back-translation
            back_translation_text = run_back_translation_gemini(translation, self.lang)
            metrics["back_translation_text"] = back_translation_text
            if back_translation_text and self.semantic_scorer:
                back_translation_score = self.semantic_scorer.score(original, back_translation_text)
        metrics["back_translation_similarity"] = back_translation_score

        combined = self._combine_scores(semantic_score, back_translation_score)
        metrics["combined_similarity"] = combined

        decision, reason = self._decision(semantic_score, back_translation_score, combined)
        metrics["quality_stage"] = reason
        return decision, reason, translation, metrics


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
            "es": "(desambiguación)",
            "zh": "(消歧义)",
            "ja": "(曖昧さ回避)",
            "hi": "(बहुविकल्पी)",
            "de": "(Begriffsklärung)",
            "uk": "(значення)",
            "kk": "(айрық)",
            "ro": "(dezambiguizare)"
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


def run_translation_nllb(text: str, translation_pipeline) -> Optional[str]:
    """Translate text using NLLB with error handling (for back-translation)"""
    try:
        results = translation_pipeline(text)
        if results and len(results) > 0:
            return results[0]["translation_text"]
    except Exception as e:
        logger.error(f"NLLB translation failed: {e}")
    
    return None

def run_back_translation_gemini(text: str, source_lang: str) -> Optional[str]:
    """Back-translate text from target language to English using Gemini 2.5 Pro"""
    try:
        source_language_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        
        # Create a prompt for back-translation to English
        back_translation_prompt = f"""You are an expert translator. Translate the following text from {source_language_name} to English.
Make sure the translation is accurate and preserves the original meaning.

Text to translate:
{text}

Return ONLY the English translation, no explanations or additional text."""
        
        logger.debug(f"Back-translating from {source_language_name} to English")
        logger.debug(f"Text: {text[:100]}...")
        
        # Call Gemini using run_gpt
        response = run_gpt(back_translation_prompt)
        
        if not response:
            logger.error("Gemini returned empty response for back-translation")
            return None
        
        # Clean up the response (remove any extra formatting)
        translation = response.strip()
        
        # Remove quotes if the response is wrapped in them
        if translation.startswith('"') and translation.endswith('"'):
            translation = translation[1:-1]
        elif translation.startswith("'") and translation.endswith("'"):
            translation = translation[1:-1]
        
        logger.debug(f"Back-translation result: {translation[:100]}...")
        return translation
       
    except Exception as e:
        logger.error(f"Gemini back-translation failed: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def run_translation_gemini(text: str, target_lang: str) -> Optional[str]:
    """Translate a single text using Gemini 2.5 Pro with error handling"""
    try:
        language_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        
        logger.debug(f"Translating question to {language_name}: {text[:100]}...")
        
        prompt = TRANSLATION_PROMPT.format(
            language=language_name,
            original_question=text
        )
        
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Call Gemini using run_gpt (which uses Gemini)
        response = run_gpt(prompt)
        
        if not response:
            logger.error("Gemini returned empty response")
            return None
        
        logger.debug(f"Gemini response length: {len(response)} characters")
        logger.debug(f"Gemini response preview (first 500 chars): {response[:500]}")
        
        # Parse the JSON response - look for "translation" key
        translation = None
        try:
            # Try standard JSON parsing first
            parsed = json.loads(response.strip())
            if isinstance(parsed, dict) and "translation" in parsed:
                translation = parsed["translation"]
        except json.JSONDecodeError:
            # Try extracting from code blocks
            try:
                json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', response, re.DOTALL | re.IGNORECASE)
                if json_match:
                    parsed = json.loads(json_match.group(1).strip())
                    if isinstance(parsed, dict) and "translation" in parsed:
                        translation = parsed["translation"]
            except (json.JSONDecodeError, AttributeError):
                # Try regex to find "translation": "..."
                pattern = r'["\']?translation["\']?\s*:\s*["\']((?:[^"\'\\]|\\.)*)["\']'
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    translation = match.group(1).replace('\\"', '"').replace("\\'", "'")
        
        if translation:
            logger.debug(f"Successfully parsed translation: {translation[:100]}...")
            return translation
        else:
            logger.warning(f"Failed to parse translation from response: {response[:200]}...")
            return None
       
    except Exception as e:
        logger.error(f"Gemini translation failed: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
    
    # Use Gemini for forward translation, NLLB for back-translation
    logger.info(f"Using Gemini 2.5 Pro for forward translation to {lang}")
    logger.info("Using NLLB for back-translation")
    
    quality_gate = TranslationQualityGate(lang=lang, quality_filter=quality_filter)
    
    # Prepare results (using list for better performance)
    results = []
    filtered_results = []  # Separate list for filtered translations
    stats = {
        "total": 0,
        "successful": 0,
        "failed": 0,
        "filtered_out": 0,  # New: translations filtered by quality
        "entities_found": 0,
        "entities_translated": 0,
        "heuristic_rejected": 0,
        "semantic_rejected": 0
    }
    
    output_stub = f"final_data/{lang}_translations_nllb_improved_ner_semantic"
    rejected_stub = f"{output_stub}_REJECTED"
    
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
        
        # Translate the text using OpenAI
        translation = run_translation_openai(text_to_translate, lang)
        
        if translation:
            should_keep, reason, cleaned_translation, gate_metrics = quality_gate.evaluate(
                original_text, translation
            )
            
            if not should_keep:
                stats["filtered_out"] += 1
                if reason.startswith("semantic") or reason.startswith("back"):
                    stats["semantic_rejected"] += 1
                else:
                    stats["heuristic_rejected"] += 1
                
                logger.warning(
                    "Translation rejected (%s): %s...",
                    reason,
                    original_text[:80],
                )
                
                filtered_payload = {
                    "original_text": original_text,
                    "rejected_translation": translation,
                    "entities_found": len(entities),
                    "entities": ", ".join(entities) if entities else "",
                    "filter_reason": reason,
                    "quality_stage": gate_metrics.get("quality_stage"),
                    "length_ratio": gate_metrics.get("length_ratio"),
                    "detected_language": gate_metrics.get("detected_language"),
                    "semantic_similarity": gate_metrics.get("semantic_similarity"),
                    "back_translation_similarity": gate_metrics.get("back_translation_similarity"),
                    "combined_similarity": gate_metrics.get("combined_similarity"),
                    "back_translation_text": gate_metrics.get("back_translation_text"),
                    "original_length": len(original_text),
                    "translation_length": len(translation) if translation else 0,
                }
                filtered_results.append(filtered_payload)
                
                results.append({
                    "original_text": original_text,
                    "translation": None,
                    "entities_found": len(entities),
                    "entities": ", ".join(entities) if entities else "",
                    "filtered": True,
                    "filter_reason": reason,
                    "semantic_similarity": gate_metrics.get("semantic_similarity"),
                    "back_translation_similarity": gate_metrics.get("back_translation_similarity"),
                    "combined_similarity": gate_metrics.get("combined_similarity"),
                })
            else:
                stats["successful"] += 1
                results.append({
                    "original_text": original_text,
                    "translation": cleaned_translation,
                    "entities_found": len(entities),
                    "entities": ", ".join(entities) if entities else "",
                    "filtered": False,
                    "length_ratio": gate_metrics.get("length_ratio"),
                    "detected_language": gate_metrics.get("detected_language"),
                    "semantic_similarity": gate_metrics.get("semantic_similarity"),
                    "back_translation_similarity": gate_metrics.get("back_translation_similarity"),
                    "combined_similarity": gate_metrics.get("combined_similarity"),
                    "quality_stage": gate_metrics.get("quality_stage"),
                })
        else:
            stats["failed"] += 1
            logger.warning(f"Translation failed for: {original_text[:50]}...")
            results.append({
                "original_text": original_text,
                "translation": None,
                "entities_found": len(entities),
                "entities": ", ".join(entities) if entities else "",
                "filtered": True,
                "filter_reason": "translation_failed",
                "quality_stage": "translation_failed"
            })
        
        # Save periodically
        if len(results) % 50 == 0:
            temp_df = pd.DataFrame(results)
            output_file = f"{output_stub}.csv"
            temp_df.to_csv(output_file, index=False)
            
            # Also save filtered translations periodically
            if filtered_results:
                filtered_df = pd.DataFrame(filtered_results)
                filtered_df.to_csv(f"{rejected_stub}.csv", index=False)
            
            logger.info(f"Saved progress: {len(results)} translations ({len(filtered_results)} rejected)")
    
    # Create final DataFrame and save
    results_df = pd.DataFrame(results)
    output_file = f"{output_stub}.csv"
    results_df.to_csv(output_file, index=False)
    
    # Save filtered translations to separate file
    if filtered_results:
        filtered_df = pd.DataFrame(filtered_results)
        filtered_df.to_csv(f"{rejected_stub}.csv", index=False)
        logger.info(f"Rejected translations saved to: {rejected_stub}.csv")
    
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
        logger.info(f"  Heuristic rejections: {stats['heuristic_rejected']}")
        logger.info(f"  Semantic/back-trans rejections: {stats['semantic_rejected']}")
    logger.info(f"Entities found: {stats['entities_found']}")
    logger.info(f"Unique entities translated: {len(entity_translator.cache)}")
    logger.info(f"Output saved to: {output_file}")
    if filtered_results:
        logger.info(f"Filtered translations saved to: {output_stub}_FILTERED.csv")
    logger.info("=" * 70)


if __name__ == "__main__":
    import sys
    
    # Default to Turkish, allow command line argument
    lang = sys.argv[1] if len(sys.argv) > 1 else "tr"
    
    if lang not in ["tr", "ru", "es", "zh", "ja", "hi", "de", "uk", "kk", "ro"]:
        logger.error("Language must be one of: tr, ru, es, zh, ja, hi, de, uk, kk, ro")
        sys.exit(1)
    
    main(lang=lang)

