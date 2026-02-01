import difflib
from data_gen.final_gen.utils.llm import run_llm_gemini
from data_gen.final_gen.utils.json_cleaning import load_json_loose
from data_gen.final_gen.prompts import (
    NL_GEN_RESERVED_PROMPT,
    LLM_AS_A_JUDGE_PROMPT,
    NL_GEN_PERSONA_PROMPT,
    AMBIGUOUS_INDUCING_PROMPT,
    LLM_AS_A_JUDGE_PROMPT_PERSONA,
)
import google.genai as genai
gemini_client = genai.Client()
def similarity_score(a, b):
    """Calculate the similarity between two strings"""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_nl_ambiguous_questions(pipe, table: str, sql_query: str) -> tuple[str, str, str]:
    """Get NL questions and reasoning from the LLM"""
    prompt_formatted = AMBIGUOUS_INDUCING_PROMPT.format(table=table, sql_query=sql_query)
    nl_response = run_llm_gemini(pipe, prompt_formatted)
    data = load_json_loose(nl_response)
    if not data:
        return "", "", ""
    question = data.get("question", "")
    reasoning = data.get("reasoning", "")
    ambiguity_type = data.get("ambiguity_type", "")
    ambiguity_explanation = data.get("ambiguity_explanation", "")
    details_parts = []
    if ambiguity_type:
        details_parts.append(f"type: {ambiguity_type}")
    if ambiguity_explanation:
        details_parts.append(f"explanation: {ambiguity_explanation}")
    ambiguity_details = " | ".join(details_parts)
    if not ambiguity_details and (ambiguity_type or ambiguity_explanation):
        ambiguity_details = ambiguity_type or ambiguity_explanation
    return question, reasoning, ambiguity_details

def get_nl_reserved_questions(pipe, table: str, sql_query: str) -> tuple[list[str], list[str]]:
    """Get NL questions and reasoning from the LLM"""
    prompt_formatted = NL_GEN_RESERVED_PROMPT.format(table=table, sql_query=sql_query)
    nl_response = run_llm_gemini(pipe, prompt_formatted)
    data = load_json_loose(nl_response)
    if not data:
        return [], []
    
    questions = []
    reasoning = []
    for i in range(1, 5):
        # Try different key variations
        q = data.get(f"question{i}") or data.get(f"Question{i}") or data.get(f"question {i}")
        r = data.get(f"reasoning{i}") or data.get(f"Reasoning{i}") or data.get(f"reasoning {i}")
        if q:
            questions.append(str(q).strip())
            reasoning.append(str(r).strip() if r else "")
            
    return questions, reasoning

def get_nl_persona_questions(pipe, table: str, sql_query: str, persona: str, goals: str) -> tuple[list[str], list[str]]:
    """Get NL questions and reasoning from the LLM"""
    prompt_formatted = NL_GEN_PERSONA_PROMPT.format(table=table, sql_query=sql_query, persona=persona, goals=goals)
    nl_response = run_llm_gemini(pipe, prompt_formatted)
    data = load_json_loose(nl_response)
    if not data:
        return [], []
    questions = [data.get(f"question{i}", "") for i in range(1, 5)]
    reasoning = [data.get(f"reasoning{i}", "") for i in range(1, 5)]
    return questions, reasoning

def choose_best_question_persona(pipe, nl_questions: list[str], reasoning: list[str], sql_query: str, table: str, persona: str) -> tuple[str, str]:
    """Choose the best natural language question and reasoning from the LLM"""
    if not nl_questions:
        return None, None
    judge_prompt = LLM_AS_A_JUDGE_PROMPT_PERSONA.format(nl_question=nl_questions, sql_query=sql_query, table=table, persona=persona)
    raw = run_llm_gemini(pipe, judge_prompt)
    judge_data = load_json_loose(raw)
    if not judge_data:
        return None, None

    best_nl_index = judge_data.get("index", 0)
    if best_nl_index < 0 or best_nl_index >= len(nl_questions):
        return None, None

    best = nl_questions[best_nl_index]
    returned = judge_data.get("question", best)

    if similarity_score(best, returned) >= 0.8:
        return best, reasoning[best_nl_index]
    return None, None
    

def choose_best_question(pipe, nl_questions: list[str], reasoning: list[str], sql_query: str, table: str) -> tuple[str, str]:
    """Choose the best natural language question and reasoning from the LLM"""
    if not nl_questions:
        return None, None
    
    judge_prompt = LLM_AS_A_JUDGE_PROMPT.format(nl_question=nl_questions, sql_query=sql_query, table=table)
    raw = run_llm_gemini(pipe, judge_prompt)
    judge_data = load_json_loose(raw)

    if not judge_data:
        return None, None

    best_nl_index = judge_data.get("index", 0)
    if best_nl_index < 0 or best_nl_index >= len(nl_questions):
        return None, None

    best = nl_questions[best_nl_index]
    returned = judge_data.get("question", best)

    if similarity_score(best, returned) >= 0.8:
        return best, reasoning[best_nl_index]
    return None, None

def nl_gen_ambiguous_pipeline(pipe, table: str, sql_query: str) -> tuple[str, str, str]:
    """Generate the natural language question and reasoning from the LLM"""
    nl_question, reasoning, ambiguity_details = get_nl_ambiguous_questions(pipe, table, sql_query)
    if not nl_question:
        return None, None, ""
    # We won't do any filtering here, because ambiguous queries might get filtered out
    return nl_question, reasoning, ambiguity_details

def nl_gen_reserved_pipeline(pipe, table: str, sql_query: str) -> tuple[str, str]:
    """Generate the natural language question and reasoning from the LLM"""
    nl_questions, reasoning = get_nl_reserved_questions(pipe, table, sql_query)
    if not nl_questions:
        return None, None
    return choose_best_question(pipe, nl_questions, reasoning, sql_query, table)

def nl_gen_persona_pipeline(pipe, table: str, sql_query: str, persona: str, goals: str) -> tuple[str, str]:
    """Generate the natural language question and reasoning from the LLM"""
    nl_questions, reasoning = get_nl_persona_questions(pipe, table, sql_query, persona, goals)
    if not nl_questions:
        return None, None
    return choose_best_question_persona(pipe, nl_questions, reasoning, sql_query, table, persona)