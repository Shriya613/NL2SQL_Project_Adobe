import google.genai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import time

# Define a predicate to retry only on potential transient errors
def is_retryable_error(exception):
    error_str = str(exception).lower()
    # Retry on 429 (Resource Exhausted), 500 (Internal), 503 (Unavailable)
    if "429" in error_str or "resource" in error_str and "exhausted" in error_str:
        return True
    if "500" in error_str or "503" in error_str or "internal" in error_str or "unavailable" in error_str:
        return True
    if "quota" in error_str:
        return True
    return False

def log_retry_attempt(retry_state):
    print(f"⚠️  Gemini API transient error: {retry_state.outcome.exception()}. Retrying attempt {retry_state.attempt_number}...")

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception(is_retryable_error),
    before_sleep=log_retry_attempt
)
def run_llm_gemini(pipe: genai.Client, prompt: str) -> str:
    response = pipe.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )
    gen_text = response.text
    return gen_text

