SQL_GEN_PROMPT = """
    You are an expert SQL developer tasked with generating a complex, 
    creative, and non-trivial SQLite query. 

    Your query must: 
    1. Be constructed using only the information from the table provided: {table}, {table_description}
    2. Be complex and challenging.
    3. Have sound logic and be executable (it is okay if the variables names have spaces)
    4. The intended result from the query must produce actionable insights.
    5. Use the column and row values verbatim to the table information but encapsulate them in backticks. 
    This means that underscores are not used to replace spaces. For example, 
    if the column name is "State/territory", then the SQLite query should use "`State/territory`".
    If the row value is "New South Wales", then the SQLite query should use "`New South Wales`".
    6. Produce one SQLite query, and format it inside of ```sql tags.
    7. End with a semicolon.
    
    Do not include any other text in your response besides the SQLite query. 
    Absolutely NO explanations or natural language. 
    """

SQL_CORRECT_PROMPT = """
    Correct the following SQLite query based on the error message,
    and table information. ONLY return the corrected SQLite query.
    Make sure to enclose the column names in backticks.
    
    SQLite Query: {sql_query}

    Error: {error}

    Table Information: {table}, {table_description}
    """

NL_GEN_PROMPT = """
    You are a natural language expert tasked with translating a SQLite 
    query into a question that it answers.

    Given the table information and the SQLite query, generate a natural 
    language question that would logically lead to this query. 

    Your natural language question should:
    1. Only produce a question. Do not include the table or SQLite query in 
    your response.
    2. Be specific enough to be answered by the SQLite query.
    3. Not include details that the SQLite query does not contain.
    4. Not include any information besides the question.
    5. Produce one natural language question, and format it inside of ```question tags.

    Table information: {table}, {table_description}

    SQLite Query: {sql_query}
    """

LLM_AS_A_JUDGE_PROMPT = """
    You are an expert natural language expert tasked with correcting a natural language question.
    Check whether or not the natural language question is valid based on the SQLite query and table information and provide a 
    score between 0 and 100.

    Your score should be based on the following criteria:
    1. The natural language question should produce a question. It does not include the SQLite query or table information.
    2. The natural language question should be specific enough to be answered by the SQLite query.
    3. The natural language question should not include details that the SQLite query does not contain.
    4. The natural language question should not include any information besides the question.
    5. The natural language question should only be one singular natural language question.

    Natural Language Question: {nl_question}

    SQLite Query: {sql_query}

    Table Information: {table}, {table_description}

    Format your response as a JSON object with the following keys:
    - "score": the score between 0 and 100
    - "reasoning": the reasoning for the score

    Example response:
    ```json
    {{
        "score": 80,
        "reasoning": "The natural language question is valid based on the SQLite query and table information but isn't specific enough to be answered by the SQLite query."
    }}
    ```

    Do not include any other text in your response besides the JSON object.
    """

NL_CORRECT_PROMPT = """
    Correct the following natural language question based on the judge's score and reasoning.
    ONLY return the corrected natural language question. Do not include any other information besides the natural language question.
    Make sure that the question is enclosed in ```question tags.

    Table Information: {table}
    SQLite Query: {sql_query}

    Judge's Score: {score}
    Judge's Reasoning: {reasoning}

    Natural Language Question: {nl_question}

    """

MULTIPLE_SQL_GEN_PROMPT = """
    You are an expert SQL developer tasked with generating a complex, 
    creative, and non-trivial SQLite query that answers a natural language question in a different way from the first query.

    Your query must: 
    1. Interpret the natural language question in a different way from the first query: {first_sql_query}
    2. Be constructed using only the information from the table provided: {table} 
    3. Be complex and challenging.
    4. Have sound logic and be executable (it is okay if the variables names have spaces)
    5. The intended result from the query must produce actionable insights.
    6. Use the column and row values verbatim to the table information but encapsulate them in backticks. 
    This means that underscores are not used to replace spaces. For example, 
    if the column name is "State/territory", then the SQLite query should use "`State/territory`".
    If the row value is "New South Wales", then the SQLite query should use "`New South Wales`".
    7. Produce one SQLite query, and format it inside of ```sql tags.
    8. End with a semicolon.
    9. Answer the natural language question: {nl_question}
    
    Do not include any other text in your response besides the SQLite query. 
    Absolutely NO explanations or natural language. 
    """

MULTIPLE_SQL_CORRECT_PROMPT = """
    Correct the following SQLite query based on the error message,
    natural language question, and table information.
    ONLY return the corrected SQLite query.
    Make sure to enclose the column names in backticks.
    
    First SQLite Query: {first_sql_query}
    SQLite Query: {sql_query}

    Natural Language Question: {nl_question}

    Error: {error}

    Table Information: {table}
    """