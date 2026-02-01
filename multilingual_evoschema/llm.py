# Using Qwen model for schema changes

import os
import torch
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
# Get project root (parent of multilingual_evoschema directory)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)
groq_token = os.getenv("GROQ_TOKEN")

def _select_best_device():
    """Select the CUDA device with the most free memory (fallback to CPU)."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(idx)
        except RuntimeError:
            free_mem, total_mem = 0, 0
        if free_mem > best_free:
            best_idx = idx
            best_free = free_mem
            best_total = total_mem

    gb = 1024 ** 3
    print(
        f"📈 Selected cuda:{best_idx} "
        f"({best_free / gb:.2f} / {best_total / gb:.2f} GiB free) for Qwen model"
    )
    return torch.device(f"cuda:{best_idx}")


# Initialize device and model
device = _select_best_device()
model_name = "Qwen/Qwen2.5-3B-Instruct"
pipe = None


def _get_pipeline():
    """Lazy load the pipeline to avoid loading at import time."""
    global pipe
    if pipe is None:
        print(f"🔄 Loading Qwen model: {model_name}...")
        pipe = pipeline(model=model_name, device=device)
        print("✅ Qwen model loaded successfully!")
    return pipe


def run_gpt(prompt: str, temperature: float = None, max_tokens: int = 512) -> str:
    """Uses OpenAI GPT-5.1 for schema changes."""
    try:
        from openai import OpenAI
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        client = OpenAI(api_key=openai_key)
        
        # Combine system message and user prompt
        system_message = "You are an expert SQL developer."
        full_prompt = prompt
        
        # Prepare messages for OpenAI API
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": full_prompt}
        ]
        
        # Prepare parameters
        params = {
            "model": "gpt-5.1",
            "messages": messages,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            # GPT-5.1 uses max_completion_tokens instead of max_tokens
            params["max_completion_tokens"] = max_tokens
        
        # Call OpenAI API
        response = client.chat.completions.create(**params)
        
        # Extract text from response
        if not response.choices or not response.choices[0].message:
            raise ValueError("OpenAI returned empty response")
        
        text = response.choices[0].message.content
        
        if not text:
            raise ValueError("OpenAI returned empty text")
        
        return text.strip()
    except Exception as e:
        print(f"Error calling OpenAI GPT-5.1: {e}")
        raise
def run_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    """Run LLM with the given prompt and return the response."""
    try:
        pipeline_model = _get_pipeline()
        outputs = pipeline_model(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=10,
            return_full_text=False
        )
        # Extract generated text from pipeline output
        gen_text = ""
        for output in outputs:
            gen_text = output.get("generated_text", "")
            if gen_text:
                break
        return gen_text
    except Exception as e:
        print(f"Error calling Qwen model: {e}")
        raise
