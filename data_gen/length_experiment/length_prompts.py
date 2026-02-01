SQL_GEN_PROMPT = """
    You are an expert SQL developer tasked with generating a complex, 
    creative, and non-trivial SQLite query. 

    Your query must: 
    1. Be constructed using only the information from the table provided: {table} 
    2. Be complex and challenging.
    3. Have sound logic and be executable (it is okay if the variables names have spaces)
    4. The intended result from the query must produce actionable insights.
    5. Use the column and row values verbatim to the table information but encapsulate them in backticks. 
    This means that underscores are not used to replace spaces. For example, 
    if the column name is "State/territory", then the SQLite query should use "`State/territory`".
    If the row value is "New South Wales", then the SQLite query should use "`New South Wales`".
    6. Produce one SQLite query, and format it inside of ```sql tags.
    7. End with a semicolon.
    8. Must contain exactly {length_seed} reserved words. Do not include less that or more than {length_seed} reserved words.

    Reserved words are: {reserved_word_patterns}
    
    Think step-by-step and produce a final SQL query that meets the requirements.
    Do not include any other text in your response besides the SQLite query. 
    Absolutely NO explanations or natural language. 
  
    """

SQL_CORRECT_PROMPT = """
    Correct the following SQLite query based on the error message,
    and table information. ONLY return the corrected SQLite query.
    Make sure to enclose the column names in backticks. You must also 
    ensure that the query contains {length_seed} reserved words.
    
    SQLite Query: {sql_query}

    Error: {error}

    Table Information: {table}

    Reserved Word Requirement: {length_seed}
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

    Table information: {table}

    SQLite Query: {sql_query}
    """

