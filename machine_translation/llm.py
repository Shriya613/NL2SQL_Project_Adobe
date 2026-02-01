from openai import OpenAI
import json
import re
from typing import Dict, Any, Optional

# note we should add something like:
# ote: This is to model questions that a person is asking an LLM. If the target language has a formal "you", consider whether the person would be more likely to use a formal or informal "you" when talking to an LLM. In Russian, for example, the informal "ты" is more likely to be used when talking to an LLM.
TRANSLATION_PROMPT = """
You are an expert translator tasked with translating a natural language question into {language}.
Make sure the translation is accurate, natural, and follows the same style and meaning as the original natural language question, while remaining grammatical in {language}.

Translate the following natural language question into {language}:

{original_question}

Output your response as a JSON object with a single key "translation" containing the translated text.

Example response:
```json
{{
    "translation": "..."
}}
```

Do not include any other text in your response besides the JSON object.
"""


def run_gpt(prompt: str) -> str:
    """Use Gemini 2.5 Flash to translate"""
    import google.generativeai as genai
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.5-pro")
        
        logger.debug(f"Calling Gemini with prompt length: {len(prompt)}")
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            logger.error("Gemini returned empty or None response")
            logger.error(f"Response object: {response}")
            if hasattr(response, 'prompt_feedback'):
                logger.error(f"Prompt feedback: {response.prompt_feedback}")
            raise ValueError("Empty response from Gemini")
        
        result = response.text.strip()
        logger.debug(f"Gemini returned {len(result)} characters")
        print("RAW GEMINI RESPONSE: \n", result)
        return result
    except Exception as e:
        logger.error(f"Error in run_gpt: {type(e).__name__}: {e}")
        if hasattr(e, 'response'):
            logger.error(f"Error response: {e.response}")
        raise


def parse_json(text: str) -> list:
    """
    Robust JSON parser that tries multiple strategies and returns a list of NL questions.
    Extracts values from JSON objects (especially question1, question2, etc.) and returns them as a list.
    """
    if not text or not text.strip():
        return []
    
    parsed_dict = {}
    
    # Strategy 1: Try standard JSON parsing
    try:
        parsed_dict = json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Try to extract JSON from code blocks
    if not parsed_dict:
        try:
            # Look for ```json ... ``` or ``` ... ```
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_text = json_match.group(1).strip()
                parsed_dict = json.loads(json_text)
        except (json.JSONDecodeError, AttributeError):
            pass
    
    # Strategy 3: Try to fix common JSON issues and parse
    if not parsed_dict:
        try:
            cleaned = text.strip()
            
            # Remove JSON comments (// and /* */)
            cleaned = re.sub(r'//.*?$', '', cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
            
            # Remove trailing commas before closing braces/brackets
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            
            # Try parsing the cleaned version
            parsed_dict = json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Keyword search for question pattern (question1, question2, etc.)
    if not parsed_dict:
        try:
            # Look for pattern like "question1": "...", "question2": "...", etc.
            # Match keys that look like question followed by numbers
            # Handle both single and double quotes, and allow escaped quotes in values
            pattern = r'["\']?(question\d+)["\']?\s*:\s*["\']((?:[^"\'\\]|\\.)*)["\']'
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            
            for match in matches:
                key = match.group(1).lower()  # Normalize to lowercase
                value = match.group(2)
                # Unescape escaped quotes
                value = value.replace('\\"', '"').replace("\\'", "'")
                parsed_dict[key] = value
        except Exception:
            pass
    
    # Strategy 5: Regex-based fallback parsing for general JSON objects
    if not parsed_dict:
        try:
            # Try to find JSON object pattern { ... }
            obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if obj_match:
                obj_text = obj_match.group(0)
                
                # Extract key-value pairs using regex
                # Match: "key": value or 'key': value
                pattern = r'["\']([^"\']+)["\']\s*:\s*["\']([^"\']+)["\']'
                matches = re.finditer(pattern, obj_text)
                
                for match in matches:
                    key = match.group(1)
                    value = match.group(2)
                    parsed_dict[key] = value
        except Exception:
            pass
    
    # Convert dictionary to list of values, maintaining order
    if parsed_dict:
        # If keys are question1, question2, etc., sort by number
        if any(key.lower().startswith('question') for key in parsed_dict.keys()):
            # Extract and sort by question number
            items = []
            for key, value in parsed_dict.items():
                if isinstance(key, str) and key.lower().startswith('question'):
                    # Extract number from question1, question2, etc.
                    num_match = re.search(r'(\d+)', key)
                    if num_match:
                        items.append((int(num_match.group(1)), value))
                    else:
                        items.append((999, value))  # Put non-numbered questions at end
                else:
                    items.append((999, value))  # Put non-question keys at end
            
            # Sort by number and return just the values
            items.sort(key=lambda x: x[0])
            return [value for _, value in items]
        else:
            # For other dictionaries, just return values in order
            return list(parsed_dict.values())
    
    # If all strategies fail, return empty list
    return []