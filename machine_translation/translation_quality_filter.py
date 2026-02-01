"""
Translation Quality Filter
Filters out low-quality or incorrect translations
"""

import re
import logging
from typing import Optional, Dict, Tuple
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

# Language code mapping
LANG_CODE_MAP = {
    "tr": "tr",  # Turkish
    "ru": "ru",  # Russian
    "es": "es",  # Spanish
}


class TranslationQualityFilter:
    """Filter translations based on quality metrics"""
    
    def __init__(self, target_lang: str, min_length_ratio: float = 0.3, max_length_ratio: float = 3.0):
        """
        Initialize quality filter
        
        Args:
            target_lang: Target language code (tr, ru, es)
            min_length_ratio: Minimum acceptable length ratio (translation/original)
            max_length_ratio: Maximum acceptable length ratio
        """
        self.target_lang = target_lang
        self.expected_lang_code = LANG_CODE_MAP.get(target_lang, target_lang)
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
        
        # Common issues to detect
        self.suspicious_patterns = [
            r'^[^a-zA-Zа-яА-ЯçğıöşüÇĞIİÖŞÜñáéíóúüÑÁÉÍÓÚÜ]*$',  # Only punctuation/symbols
            r'^[\s]*$',  # Only whitespace
            r'^[0-9\s]*$',  # Only numbers
        ]
    
    def check_length_ratio(self, original: str, translation: str) -> Tuple[bool, float]:
        """
        Check if translation length is reasonable
        
        Returns:
            (is_valid, ratio)
        """
        if not original or not translation:
            return False, 0.0
        
        ratio = len(translation) / len(original) if len(original) > 0 else 0.0
        
        is_valid = self.min_length_ratio <= ratio <= self.max_length_ratio
        
        return is_valid, ratio
    
    def check_language(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if translation is in the correct language
        
        Returns:
            (is_correct_language, detected_language)
        """
        if not text or len(text.strip()) < 3:
            return False, None
        
        try:
            detected = detect(text)
            is_correct = detected == self.expected_lang_code
            return is_correct, detected
        except LangDetectException:
            # If detection fails, we can't verify but don't reject
            logger.warning(f"Language detection failed for: {text[:50]}...")
            return True, None  # Give benefit of doubt
        except Exception as e:
            logger.warning(f"Language detection error: {e}")
            return True, None
    
    def check_suspicious_patterns(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check for suspicious patterns that indicate bad translation
        
        Returns:
            (is_valid, reason_if_invalid)
        """
        if not text:
            return False, "Empty translation"
        
        text_stripped = text.strip()
        
        if len(text_stripped) < 2:
            return False, "Translation too short"
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.match(pattern, text_stripped):
                return False, f"Matches suspicious pattern: {pattern}"
        
        # Additional checks could be added here
        # e.g., check if translation is identical to original (no translation happened)
        
        return True, None
    
    def check_basic_quality(self, original: str, translation: str) -> Tuple[bool, Optional[str]]:
        """
        Basic quality checks
        
        Returns:
            (is_valid, reason_if_invalid)
        """
        if not translation:
            return False, "Translation is None or empty"
        
        # Check length ratio
        length_valid, ratio = self.check_length_ratio(original, translation)
        if not length_valid:
            return False, f"Length ratio {ratio:.2f} outside acceptable range [{self.min_length_ratio}, {self.max_length_ratio}]"
        
        # Check suspicious patterns
        pattern_valid, reason = self.check_suspicious_patterns(translation)
        if not pattern_valid:
            return False, reason
        
        return True, None
    
    def validate_translation(self, original: str, translation: Optional[str]) -> Dict[str, any]:
        """
        Comprehensive translation validation
        
        Returns:
            Dictionary with validation results:
            {
                'is_valid': bool,
                'reason': str (if invalid),
                'metrics': {
                    'length_ratio': float,
                    'detected_language': str,
                    'language_match': bool,
                    ...
                }
            }
        """
        result = {
            'is_valid': False,
            'reason': None,
            'metrics': {}
        }
        
        # Check if translation exists
        if not translation:
            result['reason'] = "Translation is None or empty"
            return result
        
        # Basic quality checks
        basic_valid, basic_reason = self.check_basic_quality(original, translation)
        if not basic_valid:
            result['reason'] = basic_reason
            return result
        
        # Calculate metrics
        length_valid, length_ratio = self.check_length_ratio(original, translation)
        lang_valid, detected_lang = self.check_language(translation)
        
        result['metrics'] = {
            'length_ratio': length_ratio,
            'detected_language': detected_lang,
            'language_match': lang_valid,
            'original_length': len(original),
            'translation_length': len(translation)
        }
        
        # Final validation: must pass basic checks
        # Language detection is informative but not required (can fail on short texts)
        result['is_valid'] = True
        
        # Log warnings for potential issues
        if not lang_valid and detected_lang:
            logger.warning(
                f"Language mismatch: expected {self.expected_lang_code}, got {detected_lang}. "
                f"Original: {original[:50]}... Translation: {translation[:50]}..."
            )
        
        return result
    
    def filter_translation(self, original: str, translation: Optional[str]) -> Tuple[bool, Optional[str], Dict]:
        """
        Filter translation - returns whether to keep it
        
        Returns:
            (should_keep, filtered_translation, metrics)
        """
        validation = self.validate_translation(original, translation)
        
        if validation['is_valid']:
            return True, translation, validation['metrics']
        else:
            logger.warning(f"Translation filtered out: {validation['reason']}. Original: {original[:50]}...")
            return False, None, validation['metrics']

