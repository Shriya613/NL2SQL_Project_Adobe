# Multilingual NL to SQL

## Data Generation
* We are improving upon the data generation done in the source2synth-experiment repo.
* Our planned improvements
    * Add in SQL operator diversity statistics to mimic natural occurrence
    * Beef up prompts the LLM prompts
    * Ensure that we aren't getting empty results or errors from the SQL query filter

### `reserved_probs.py`
The purpose of this script is to get the probabilities of each reserved word in the MySQL language. This is to help use generate random seed opperators that reflect real usage in order to encourage SQL diversity. The choosen dataset is BIRD.


### `nl_sql_gen.py`
This file synthetically generates the seed topic, sql query, and natural language question

## Filtering
* Improvements from Source2Synth
    * Use CoT to filter both the NL questions and SQL queries (in addition to running the queries against the database)

## Experiments

### First Experiment: Finetuning + CoT

Open-source models:
1. gpt-oss-20b
2. llama 3
3. Gemma
4. Mistral
5. Qwen 3

All five models will be fintuned to take input as NL + CoT pair and output SQL

3 sub-experiments:
1. Keep everything in English
2. Translate the NL, but leave the CoT in English
3. Translate both the NL and CoT








