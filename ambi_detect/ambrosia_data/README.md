#  Data
The dataset structure is as follows:
```
data
|-- scope
|   |-- Agriculture
|   |   |-- scope_agricultural_machinery_stores_brands
|   |   |   |-- concepts.json
|   |   |   |-- concepts_db.json
|   |   |   |-- examples.json
|   |   |   |-- scope_agricultural_machinery_stores_brands.sqlite
|   |   |   |-- scope_agricultural_machinery_stores_brands_dump.sql
|   |   |-- ...
|   |-- Airport
|   |   |-- ...
|-- attachment
|   |-- ...
|-- vague
|   |-- ...
```

Ambiguous questions for each database are available in `examples.json` along with their interpretations and SQL queries. We also provide key concepts and relations in `concepts.json`, and these concepts grounded to the database elements in `concepts_db.json`.

A CSV file with all instances is available here: (ambrosia.csv) [ambrosia.csv](ambrosia.csv)

The [Croissant](https://github.com/mlcommons/croissant) format is available in [ambrosia_croissant.json](ambrosia_croissant.json). 

AMBROSIA is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en).