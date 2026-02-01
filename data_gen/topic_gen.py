def generate_seed_topic(table: dict) -> str:
    """Create seed topic to introduce diveristy and give the model guidance"""

    example_table = str({
        "id": "1-1000181-1", 
        "header": ["State/territory", "Text/background colour", "Format", "Current slogan", "Current series", "Notes"], 
        "types": ["text", "text", "text", "text", "text", "text"],
        "rows": [
            ["Australian Capital Territory", "blue/white", "Yaa\u00b7nna", "ACT \u00b7 CELEBRATION OF A CENTURY 2013", "YIL\u00b700A", "Slogan screenprinted on plate"],
            ["New South Wales", "black/yellow", "aa\u00b7nn\u00b7aa", "NEW SOUTH WALES", "BX\u00b799\u00b7HI", "No slogan on current series"], 
            ["New South Wales", "black/white", "aaa\u00b7nna", "NSW", "CPX\u00b712A", "Optional white slimline series"], 
            ["Northern Territory", "ochre/white", "Ca\u00b7nn\u00b7aa", "NT \u00b7 OUTBACK AUSTRALIA", "CB\u00b706\u00b7ZZ", "New series began in June 2011"], 
            ["Queensland", "maroon/white", "nnn\u00b7aaa", "QUEENSLAND \u00b7 SUNSHINE STATE", "999\u00b7TLG", "Slogan embossed on plate"], 
            ["South Australia", "black/white", "Snnn\u00b7aaa", "SOUTH AUSTRALIA", "S000\u00b7AZD", "No slogan on current series"], 
            ["Victoria", "blue/white", "aaa\u00b7nnn", "VICTORIA - THE PLACE TO BE", "ZZZ\u00b7562", "Current series will be exhausted this year"]
            ], 
            "name": "table_1000181_1"
    })
    
    prompt = """
    # Instructions:
    You are an expert database engineer. Your task is to generate one interesting 
    statement or fact based on provided table information.
    
    # Example:
    Table:
    {example_table}
    Statement/Fact:
    The territory with the word "celebration" in the slogan is the Australian Capital Territory and has blue text and a white background.

    # Your Response:

    Table:
    {table}

    Statement/Fact:
    """
    prompt_formatted = prompt.format(example_table=example_table, table=table)
    return prompt_formatted

