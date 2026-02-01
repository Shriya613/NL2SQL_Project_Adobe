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
    8. Use the persona {persona} to generate the query. Here are the goals and example queries: {goals} {example_queries}
    
    Do not include any other text in your response besides the SQLite query. 
    Absolutely NO explanations or natural language. 
    """

SQL_CORRECT_PROMPT = """
    Correct the following SQLite query based on the error message,
    and table information. ONLY return the corrected SQLite query.
    Make sure to enclose the column names in backticks.
    
    SQLite Query: {sql_query}

    Error: {error}

    Table Information: {table}

    Persona: {persona}
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
    6. Use the persona {persona} to generate the question.

    Table information: {table}

    SQLite Query: {sql_query}
    """


PERSONA_PROMPT_OUT_OF_THE_BOX = """
    You are tasked with inventing a highly creative and uncommon persona who might want to query this database.

    Your output should include the following included in ```json tags:
    1. **SHORT Persona description** – who they are, what makes them unique or unusual in the context of this database. (max 1 sentence)  
    2. **Goals** – what they are trying to achieve by querying the data (be imaginative but plausible).  
    3. **Example queries** – 3 realistic SQL-style queries this persona might generate to meet those goals.  

    Guidelines:
    - Only generate ONE persona.
    - Think outside the box — the persona should be rare or eccentric, not typical!  The more unique, the better.
    - Use the **context of the database and its schema** to ground your ideas. The persona should be specifically related to this database.
    - Avoid repeating generic personas or personas you have already created.

    Database schema:
    {table}
    """


PERSONA_PROMPT_COMMON = """
    You are tasked with generating realistic and likely personas who might want to query this database for persona.

    Your output should include the following included in ```json tags:
    1. **SHORT Persona description** – who they are, what their role or background is, and why they would use this database. (max 2 sentences)  
    2. **Goals** – what they are trying to achieve or analyze by querying the data.  
    3. **Example queries** – 3 practical SQL-style queries this persona might write to accomplish their goals.  

    Guidelines:
    - Only generate ONE persona.
    - Focus on **plausible and common roles**   
    - Use the **context of the database and its schema** to keep your personas grounded in reality.  
    - Make each persona distinct and purposeful — avoid overlap in motivation or job type.  
    - Keep the tone clear, professional, and realistic.
    - Avoid repeating generic personas or personas you have already created.

    Database schema:
    {table}
    """
