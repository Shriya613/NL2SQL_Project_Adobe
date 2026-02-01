SQL_GEN_RESERVED_PROMPT = """
You are an expert SQL developer tasked with generating a creative and non-trivial SQLite query.

<database_schema>
{table}
</database_schema>

<task_requirements>
1. **Source Material:** Construct the query using ONLY information from the <database_schema> provided above.
2. **Logic:** The query must be executable, logically sound, and produce actionable insights.
3. **Identifier Formatting:** You must encase ALL column names and row values in backticks (`).
   - Example Column: `State/territory` (Do NOT replace spaces with underscores).
   - Example Value: `New South Wales`.
4. **Formatting:** Output a single SQLite query inside ```sql code blocks.
5. **Syntax:** End with a semicolon (;).
6. **Efficiency:** Keep the query efficient with minimal steps.
</task_requirements>

<constraint_checklist>
- The query must contain EXACTLY {length_seed} reserved words.
- Reserved words to choose from: {reserved_word_patterns}
- Do not use fewer or more than {length_seed}.
</constraint_checklist>

Step-by-step instructions:
1. Analyze the table schema.
2. Use exactly {length_seed} reserved words from the list.
3. Construct a query that uses exactly those reserved words and answers a non-trivial question.
4. Verify backtick formatting.

Output ONLY the SQL query inside code blocks. No explanations.
"""

PERSONA_PROMPT_OUT_OF_THE_BOX = """
You are tasked with inventing a creative, uncommon, yet plausible persona who interprets the provided database.

<database_schema>
{table}
</database_schema>

<instructions>
Create a single persona. They should be "outside the box"—unique or unusual, but their motivation must be grounded in the data provided.

Output your response as a JSON object following this exact schema:
{{
    "persona_description": "Who they are and what makes them unique (Max 1 sentence).",
    "goals": "What they are trying to achieve (Imaginative but plausible).",
    "example_queries": [
        "Query idea 1 (Natural language or SQL description)",
        "Query idea 2",
        "Query idea 3"
    ]
}}
</instructions>
"""

PERSONA_PROMPT_COMMON = """
You are tasked with generating a realistic and likely persona who would query this database in a professional capacity.

<database_schema>
{table}
</database_schema>

<instructions>
Create a single, common persona (e.g., Analyst, Manager, Auditor). Avoid generic or previously used personas.

Output your response as a JSON object following this exact schema:
{{
    "persona_description": "Role and background (Max 2 sentences).",
    "goals": "What they are trying to analyze.",
    "example_queries": [
        "Practical query idea 1",
        "Practical query idea 2",
        "Practical query idea 3"
    ]
}}
</instructions>
"""

SQL_GEN_PERSONA_PROMPT = """
You are an expert SQL developer. Generate a SQLite query that matches the specific goals of the user persona described below.

<database_schema>
{table}
</database_schema>

<persona_profile>
**Persona:** {persona}
**Goals:** {goals}
**Inspiration Queries:** {example_queries}
</persona_profile>

<instructions>
1. **Relevance:** The query must answer a question this specific persona would genuinely ask. Use the inspiration queries as a guide for *complexity*, but do not copy them exactly.
2. **Strict Formatting:**
   - Use valid SQLite syntax.
   - Encase ALL column names and string literals in backticks (e.g., `Column Name`, `Row Value`).
   - Keep names verbatim (preserve spaces).
3. **Output:** A single executable SQL query inside ```sql tags, ending with a semicolon.
4. **Efficiency:** Minimal steps to achieve the insight.
</instructions>

Output ONLY the SQL query. No natural language.
"""

SQL_CORRECT_RESERVED_PROMPT = """
You are an SQL debugging expert. Correct the following SQLite query based on the error provided.

<context>
<database_schema>
{table}
</database_schema>

<broken_query>
{sql_query}
</broken_query>

<error_message>
{error}
</error_message>

<constraints>
Reserved Word Count Requirement: {length_seed}
</constraints>
</context>

<correction_rules>
1. **Fix the Error:** Address the specific error message (e.g., close parentheses, fix syntax).
2. **Handle No Results:** If the error is "no results found", rewrite the logic to find valid data.
3. **Handle Cut-offs:** If the query is incomplete, finish it efficiently.
4. **Reserved Words:** Ensure the final query uses exactly {length_seed} reserved words.
5. **Formatting:**
   - Enclose all column names in backticks (`Column Name`).
   - Do NOT split backticked names across lines.
   - End with a semicolon (;).
</correction_rules>

Return ONLY the corrected SQLite query.
"""

SQL_CORRECT_PERSONA_PROMPT = """
You are an SQL debugging expert. Correct the query below while maintaining the intent of the specified persona.

<context>
<database_schema>
{table}
</database_schema>

<persona_context>
**Persona:** {persona}
**Goals:** {goals}
</persona_context>

<broken_query>
{sql_query}
</broken_query>

<error_message>
{error}
</error_message>
</context>

<correction_rules>
1. **Fix the Error:** Address syntax errors, missing brackets, or logic failures.
2. **Relevance:** Ensure the corrected query still aligns with the Persona's goals defined above.
3. **Formatting:**
   - Enclose all column names in backticks (`Column Name`).
   - Do NOT split backticked names across lines.
   - End with a semicolon (;).
</correction_rules>

Return ONLY the corrected SQLite query.
"""

NL_GEN_RESERVED_PROMPT = """
You are a natural language expert. Translate the SQLite query below into 4 distinct natural language questions.

<context>
<database_schema>
{table}
</database_schema>

<sql_query>
{sql_query}
</sql_query>
</context>

<instructions>
Analyze the SQL intent and generate 4 questions that logically lead to this query.
1. **No SQL Leakage:** Do not use technical terms (SELECT, GROUP BY) or column names with underscores. Use natural English.
2. **Accuracy:** Do not infer details not present in the query.
3. **Variety:** Phrase the questions differently (e.g., different levels of formality or sentence structure).

Structure your thinking for EACH question:
1. Parse SQL Intent -> 2. Map to Schema -> 3. Draft Question -> 4. Validate.

Output must be a single JSON object with this exact structure:
{{
    "reasoning1": "Analysis for Q1 (Max 4 sentences)",
    "question1": "Natural language question 1",
    "reasoning2": "Analysis for Q2",
    "question2": "Natural language question 2",
    "reasoning3": "Analysis for Q3",
    "question3": "Natural language question 3",
    "reasoning4": "Analysis for Q4",
    "question4": "Natural language question 4"
}}
</instructions>
"""

NL_GEN_PERSONA_PROMPT = """
You are a natural language expert. Translate the SQLite query into 4 questions using the voice and style of a specific persona.

<context>
<database_schema>
{table}
</database_schema>

<sql_query>
{sql_query}
</sql_query>

<persona_profile>
**Persona:** {persona}
**Goals:** {goals}
</persona_profile>
</context>

<instructions>
Generate 4 natural language questions that this specific persona would ask to get this data.
1. **Voice & Tone:** Mimic their likely jargon, casualness, or strictness.
   - Does this persona know SQL terms? Or do they use vague business terms?
   - Are they polite or demanding?
2. **Accuracy:** The question must be answerable by the provided SQL.

Output must be a single JSON object:
{{
    "reasoning1": "How the persona views this data (Max 2 sentences)",
    "question1": "The question in the persona's voice",
    "reasoning2": "...",
    "question2": "...",
    "reasoning3": "...",
    "question3": "...",
    "reasoning4": "...",
    "question4": "..."
}}
</instructions>
"""


AMBIGUOUS_INDUCING_PROMPT = """
    You are a SQL to NL expert tasked with translating a SQLite query into a natural language question that is ambiguous.

    Here are some examples of types of ambigiuty in NL to SQL:
    - **Scope ambiguity**  
    *Definition:* Uncertainty about whether a phrase like "each", "all", or "every" applies to the entire set collectively or individually.  
    *Example:* “Across all teams, which player scored the most?”  
    → (a) the top scorer overall • (b) the top scorer per team.

    - **Attachment ambiguity**  
    *Definition:* Uncertainty about which word or phrase a modifier refers to.  
    *Example:* “Show screenwriters and editors on contract”  
    → (a) both screenwriters and editors are on contract • (b) only editors are.

    - **Entity Vagueness**  
    *Definition:* The meaning of an entity is unclear because multiple valid granularities or types could be implied.  
    *Example:* “Which bank made the most profit?”  
    → (a) the entire bank company • (b) an individual branch • (c) both.

    Your task is to generate a natural language question that is a valid interpretation of the SQLite query, but is ambiguous.
    Start by producing a breif reasoning chain with the following steps:
    1.) Parse the SQL intent
    2.) Determine a type of ambiguity that would fit well for this type of query 
    3.) Draft the question
    4.) Explain the type of ambiguity and how it is interpreted in the original query, as well as how it could be interpreted differently in a new query.

    From the reasoning, it should be clear what the ambiguity is.

    Keep the reasoning chains concise and to the point. 

    Output your reasoning chain, the natural language question, and the type of ambiguity in the following format:
    ```json
    {{
        "reasoning": "STEP 1: ..., (parse the SQL intent), STEP 2: ..., (make it super clear what the ambiguity type is and why it is fitting for the query), STEP 3: ..., (draft the question), STEP 4:  ..., (explain the type of ambiguity and how it is interpreted in the original query, as well as how it could be interpreted differently in a new query)"
        "question": ...,
        "ambiguity_type": ...,
        "ambiguity_explanation": ..., (another sentence explaining the ambiguity, how it is interpreted in the original query, as well as how it could be interpreted differently in a new query)
    }}
    ```

    Do not include any other text in your response besides the JSON object. Make sure to include all the steps in the reasoning chain.

    SQLite Query: {sql_query}

    Table Information: {table}

    Your response:

    """
    
LLM_AS_A_JUDGE_PROMPT = """
You are an expert evaluator of Natural Language to SQL datasets.

<task>
Select the best Natural Language Question from the options provided.
</task>

<inputs>
<database_schema>
{table}
</database_schema>

<sql_query>
{sql_query}
</sql_query>

<candidate_questions>
{nl_question}
</candidate_questions>
</inputs>

<evaluation_criteria>
1. **Completeness:** The question implies all filters/logic present in the SQL.
2. **No Hallucination:** It does not ask for data absent from the SQL.
3. **Naturalness:** It sounds like a human wrote it (no underscores, no "SELECT *").
4. **Independence:** The question must make sense without seeing the table schema.
</evaluation_criteria>

Output as JSON:
{{
    "index": <integer_index_of_best_question>,
    "question": "<text_of_best_question>"
}}
"""

LLM_AS_A_JUDGE_PROMPT_PERSONA = """
You are an expert evaluator. Select the Natural Language Question that best fits the specific **Persona**.

<inputs>
<database_schema>
{table}
</database_schema>

<persona>
{persona}
</persona>

<sql_query>
{sql_query}
</sql_query>

<candidate_questions>
{nl_question}
</candidate_questions>
</inputs>

<evaluation_criteria>
1. **Persona Alignment:** Does the vocabulary, tone, and technical depth match the persona?
   - Non-technical personas should NOT use SQL jargon.
   - Experts might use specific acronyms or precise terminology.
2. **Accuracy:** The SQL must be a valid answer to the question.
</evaluation_criteria>

Output as JSON:
{{
    "index": <integer_index_of_best_question>,
    "question": "<text_of_best_question>"
}}
"""

PROMPT_FILTER_PROMPT = """
You are an expert in SQL interpretation and ambiguity detection.

<task>
Determine if the provided SQL query is a valid interpretation of the potentially ambiguous Natural Language (NL) question.
</task>

<inputs>
<nl_question>
{nl_question}
</nl_question>

<sql_query>
{sql_query}
</sql_query>

<database_schema>
{table}
</database_schema>

<ambiguity_reasoning>
{reasoning}
</ambiguity_reasoning>
</inputs>

<ambiguity_types>
Only the following ambiguity types are considered valid for ambiguous queries:
1. **Scope**: Where a modifier or constraint applies (e.g., global vs filtered scope, which set of data a condition applies to)
2. **Attachment**: Which part of the sentence a modifier attaches to (e.g., prepositional phrase attachment)
3. **Entity**: Which entity or concept is being referred to (e.g., entity resolution, referential ambiguity)
4. **Broadness**: How broadly or narrowly to interpret a term or concept (e.g., inclusive vs exclusive interpretation)

**REJECT** the following as NOT valid ambiguity types:
- Pattern matching variations (e.g., LIKE '%X%' vs LIKE 'X%' vs exact match)
- Conditional priority differences (e.g., order of CASE statement evaluation)
- Implementation style differences (e.g., subquery vs JOIN, different SQL syntax for same logic)
- Grouping semantics differences (e.g., ordered tuples vs unordered sets) - unless it represents a true Scope or Entity ambiguity
- Table/schema confusion (querying wrong table)
- Minor formatting or syntax variations that don't change semantic interpretation
</ambiguity_types>

<scoring_criteria>
Assign a score (0.0 to 1.0) based on these rules:
- **1.0 (Valid):** The SQL represents a valid interpretation of the NL question that differs from other interpretations due to one of the four valid ambiguity types (Scope, Attachment, Entity, or Broadness). The difference must be semantic, not just implementation style.
- **0.0 (Invalid):** The SQL either:
  - Contradicts the NL question or fetches irrelevant data
  - Differs only in implementation style, pattern matching approach, conditional priority, or other non-semantic differences
  - Does not represent a valid ambiguity type (Scope, Attachment, Entity, or Broadness)

**Key requirement**: The ambiguity must be a genuine semantic ambiguity of type Scope, Attachment, Entity, or Broadness. Differences in how to implement the same semantic interpretation (e.g., different LIKE patterns, different CASE statement order) should be scored 0.0.
If the ambiguity is a valid ambiguity, then it should be scored 1.0. if it is a possibly way to interpret the question
</scoring_criteria>

Output as JSON:
{{
    "score": <float>
}}
"""

MULTIPLE_SQL_GEN_PROMPT = """
You are an expert SQL developer.

<task>
The provided NL question is ambiguous. The first SQL query below provides ONE interpretation. Your job is to generate a **DIFFERENT** SQL query that provides a SECOND valid interpretation.
</task>

<inputs>
<nl_question>
{nl_question}
</nl_question>

<first_interpretation_sql>
{first_sql_query}
</first_interpretation_sql>

<database_schema>
{table}
</database_schema>
</inputs>

<instructions>
1. Analyze the ambiguity (Scope, Attachment, Entity, or Broadness).
2. Construct a new SQL query that interprets the question differently than the first query.
3. Adhere to strict formatting:
   - Use information from the <database_schema> ONLY.
   - Enclose columns in backticks (`Column Name`).
   - End with a semicolon (;).
</instructions>

Output as JSON:
{{
    "reasoning": "Identify the ambiguity and how this new query interprets it differently (Max 1 sentence).",
    "sql_query": "The new SQLite query"
}}
"""

MULTIPLE_SQL_CORRECT_PROMPT = """
You are an SQL debugging expert. Correct the query below.

<context>
<nl_question>
{nl_question}
</nl_question>

<broken_query>
{sql_query}
</broken_query>

<error_message>
{error}
</error_message>

<database_schema>
{table}
</database_schema>
</context>

<instructions>
1. Fix the syntax or logic error specified in the error message.
2. Ensure the query still answers the NL question provided.
3. **Formatting:**
   - Enclose column names in backticks (`Column Name`).
   - Keep backticked names on a single line.
   - End with a semicolon (;).
</instructions>

Output ONLY the corrected SQLite query.
"""