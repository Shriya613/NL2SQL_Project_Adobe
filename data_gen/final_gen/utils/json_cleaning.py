import json

def strip_markdown(json_str: str) -> str:
    """Clean the output of the LLM"""
    text = json_str.strip()
    if text.startswith("```json"):
        text = text[7:] 
    elif text.startswith("```"):
        text = text[3:]
    return text.rstrip("`").strip()

def extract_json(text: str) -> str:
    """Extract JSON from the response by matching braces"""
    first_brace = text.find('{')
    if first_brace == -1:
        return None

    brace_count = 0
    for i, ch in enumerate(text[first_brace:], start=first_brace):
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[first_brace:i + 1]

    return None

def load_json_loose(text: str) -> dict:
    """Try multiple extraction strategies"""
    text = strip_markdown(text)
    json_candidate = extract_json(text)

    if not json_candidate:
        return None

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        return None
    

    