"""
Database-specific corruption values for SQL query corruption.

This module contains realistic column names and values for each database in the BIRD dataset,
used to generate contextually appropriate corruptions during SQL query processing.
"""

DB_PLAUSIBLE_CORRUPTED_VALUES = {
    "address": {
        "columns": [
            "zip_code", "country_code", "county", "timezone", 
            "population_density", "median_income", "crime_rate", "elevation"
        ],
        "values": [
            "90210", "USA", "Los Angeles County", "PST", 
            "3200", "75000", "Low", "100 meters"
        ]
    },
    "airline": {
        "columns": [
            "baggage_allowance", "seats_available", "fuel_consumption", "carbon_emissions",
            "premium_seats", "in_flight_entertainment", "wifi_available", "meal_included"
        ],
        "values": [
            "23 kg", "125", "4500 gallons", "12 tons CO2",
            "Business Class", "Yes", "Free WiFi", "Standard Meal"
        ]
    },
    "app_store": {
        "columns": [
            "data_usage_mb", "battery_consumption", "privacy_score", "content_rating",
            "supported_languages", "file_size_mb", "update_frequency", "support_email"
        ],
        "values": [
            "45 MB", "Low", "8.5/10", "4+",
            "EN, ES, FR, DE", "125 MB", "Monthly", "support@app.com"
        ]
    },
    "authors": {
        "columns": [
            "books_sold_millions", "awards_won", "net_worth_usd", "agent_name",
            "social_media_followers", "best_seller_list", "translation_count", "movie_adaptations"
        ],
        "values": [
            "25", "3", "5 million", "Jane Doe",
            "2.5M", "5 weeks", "15 languages", "2 films"
        ]
    },
    "beer_factory": {
        "columns": [
            "company_id", "trading_as", "established_year", "brewery_location",
            "beer_style", "output_volume", "abv_percentage", "quality_score",
            "export_country", "staff_size"
        ],
        "values": [
            "Thunderbolt Brewing", "Misty Peak Brewery", "Coastal Craft", "Stout", "Pilsner"
        ]
    },
    "bike_share_1": {
        "columns": [
            "dock_id", "location_designation", "locality", "bike_inventory",
            "bikes_available", "docking_stations", "last_sync", "operational_state",
            "usage_tally", "maintenance_status"
        ],
        "values": [
            "Lakeside Station", "Metro Central Hub", "Under Maintenance", "Active", "Needs Service"
        ]
    },
    "book_publishing_company": {
        "columns": [
            "company_id", "imprint", "launch_year", "based_in",
            "chief_executive", "publishing_focus", "divisional_brands", "yearly_revenue",
            "contracted_writers", "hq_city"
        ],
        "values": [
            "Crimson Press", "Vintage Books", "Sapphire Publishing", "Boston", "San Francisco"
        ]
    },
    "books": {
        "columns": [
            "publication_id", "title", "writer", "year_published",
            "category", "isbn_number", "language_code", "page_total",
            "star_rating", "publishing_house"
        ],
        "values": [
            "The Catcher in the Rye", "Brave New World", "Wuthering Heights", "French", "German"
        ]
    },
    "car_retails": {
        "columns": [
            "showroom_id", "doing_business_as", "manufacturer", "locality",
            "sales_total", "customer_rating", "opened_year", "sales_territory",
            "staff_count", "vehicle_count"
        ],
        "values": [
            "City Auto Mart", "Express Motors Ltd", "Lexus", "Audi", "Mercedes"
        ]
    },
    "cars": {
        "columns": [
            "car_id", "model", "brand", "year",
            "fuel_type", "price_usd", "horsepower", "transmission",
            "color", "body_type"
        ],
        "values": [
            "Model S", "Civic", "Camry", "Gasoline", "Electric"
        ]
    },
    "chicago_crime": {
        "columns": [
            "case_number", "crime_type", "date", "district",
            "latitude", "longitude", "arrest_made", "domestic",
            "weapon_used", "year"
        ],
        "values": [
            "Assault", "Burglary", "Homicide", "Yes", "No"
        ]
    },
    "citeseer": {
        "columns": [
            "paper_id", "title", "author", "year",
            "journal", "citation_count", "keywords", "abstract",
            "institution", "field"
        ],
        "values": [
            "Deep Learning", "Transformers", "Knowledge Graphs", "NLP", "Computer Vision"
        ]
    },
    "codebase_comments": {
        "columns": [
            "file_path", "function_signature", "author", "comment_text",
            "date", "commit_id", "lines_changed", "review_status",
            "language", "repository"
        ],
        "values": [
            "main.py", "process_data()", "JohnDoe", "Approved", "Refactor Needed"
        ]
    },
    "coinmarketcap": {
        "columns": [
            "symbol", "ticker", "market_cap_usd", "price_usd",
            "volume_24h", "circulating_supply", "max_supply", "percent_change_24h",
            "rank", "launch_year"
        ],
        "values": [
            "BTC", "ETH", "DOGE", "Cardano", "Solana"
        ]
    },
    "college_completion": {
        "columns": [
            "institution", "state", "degree_type", "enrollment",
            "graduation_rate", "dropout_rate", "year", "tuition_usd",
            "public_private", "student_faculty_ratio"
        ],
        "values": [
            "Stanford", "Harvard", "UCLA", "Public", "Private"
        ]
    },
    "computer_student": {
        "columns": ["student_id", "full_name", "major", "gpa", "year"],
        "values": ["CS101", "John", "Computer Science", "3.5", "Sophomore"]
    },
    "cookbook": {
        "columns": ["recipe_code", "dish", "cuisine", "difficulty", "cook_time"],
        "values": ["Pasta", "Italian", "Easy", "30 min", "Chicken"]
    },
    "craftbeer": {
        "columns": ["brewery_code", "brand", "style", "abv", "rating"],
        "values": ["IPA", "Stout", "Pale Ale", "5.5%", "4.2"]
    },
    "cs_semester": {
        "columns": ["course_code", "subject", "credits", "instructor", "semester"],
        "values": ["CS101", "Data Structures", "3", "Dr. Smith", "Fall 2023"]
    },
    "disney": {
        "columns": ["character_code", "character", "movie", "year", "voice_actor"],
        "values": ["Mickey", "Frozen", "2013", "Elsa", "Anna"]
    },
    "donor": {
        "columns": ["donor_code", "contributor", "amount", "date", "organization"],
        "values": ["$1000", "Red Cross", "2023-01-15", "Charity", "Foundation"]
    },
    "european_football_1": {
        "columns": ["player_code", "squad_member", "team", "position", "goals"],
        "values": ["Messi", "Barcelona", "Forward", "25", "Ronaldo"]
    },
    "food_inspection": {
        "columns": ["inspection_number", "restaurant", "score", "date", "violations"],
        "values": ["A", "B", "C", "Pass", "Fail"]
    },
    "food_inspection_2": {
        "columns": ["inspection_number", "restaurant", "score", "date", "violations"],
        "values": ["A", "B", "C", "Pass", "Fail"]
    },
    "genes": {
        "columns": ["gene_code", "gene_symbol", "chromosome", "function", "expression"],
        "values": ["BRCA1", "TP53", "Chr17", "Tumor Suppressor", "High"]
    },
    "hockey": {
        "columns": ["player_code", "player", "team", "position", "goals"],
        "values": ["Crosby", "Penguins", "Center", "25", "Ovechkin"]
    },
    "human_resources": {
        "columns": ["employee_id", "full_name", "department", "salary", "hire_date"],
        "values": ["John", "Engineering", "$75000", "2022-01-15", "Manager"]
    },
    "ice_hockey_draft": {
        "columns": ["player_code", "prospect", "team", "draft_year", "position"],
        "values": ["McDavid", "Oilers", "2015", "Center", "Bedard"]
    },
    "image_and_language": {
        "columns": ["image_code", "caption", "language", "tags", "confidence"],
        "values": ["cat", "dog", "English", "nature", "0.95"]
    },
    "language_corpus": {
        "columns": ["document_code", "language", "length", "genre", "year"],
        "values": ["English", "Spanish", "News", "2020", "Literature"]
    },
    "law_episode": {
        "columns": ["episode_number", "episode_title", "season", "episode", "air_date"],
        "values": ["Pilot", "Season 1", "Episode 1", "2020-01-15", "Drama"]
    },
    "legislator": {
        "columns": ["legislator_code", "representative", "party", "state", "term_start"],
        "values": ["Democrat", "Republican", "California", "2021-01-01", "Senator"]
    },
    "mental_health_survey": {
        "columns": ["response_number", "age", "gender", "score", "date"],
        "values": ["25", "Female", "7.5", "2023-03-15", "Moderate"]
    },
    "menu": {
        "columns": ["menu_item_code", "dish", "price", "category", "description"],
        "values": ["Pizza", "$12.99", "Main Course", "Delicious", "Pasta"]
    },
    "mondial_geo": {
        "columns": ["country_code", "country", "capital", "population", "area"],
        "values": ["France", "Paris", "67000000", "643801", "Germany"]
    },
    "movie": {
        "columns": ["film_code", "film_title", "year", "genre", "rating"],
        "values": ["Inception", "2010", "Sci-Fi", "8.8", "The Matrix"]
    },
    "movie_3": {
        "columns": ["film_code", "film_title", "year", "genre", "rating"],
        "values": ["Inception", "2010", "Sci-Fi", "8.8", "The Matrix"]
    },
    "movie_platform": {
        "columns": ["film_code", "film_title", "year", "genre", "rating"],
        "values": ["Inception", "2010", "Sci-Fi", "8.8", "The Matrix"]
    },
    "movielens": {
        "columns": ["film_code", "film_title", "year", "genre", "rating"],
        "values": ["Inception", "2010", "Sci-Fi", "8.8", "The Matrix"]
    },
    "movies_4": {
        "columns": ["film_code", "film_title", "year", "genre", "rating"],
        "values": ["Inception", "2010", "Sci-Fi", "8.8", "The Matrix"]
    },
    "music_platform_2": {
        "columns": ["track_code", "track_title", "artist", "genre", "duration"],
        "values": ["Bohemian Rhapsody", "Queen", "Rock", "5:55", "Imagine"]
    },
    "music_tracker": {
        "columns": ["track_code", "track_title", "artist", "genre", "duration"],
        "values": ["Bohemian Rhapsody", "Queen", "Rock", "5:55", "Imagine"]
    },
    "olympics": {
        "columns": ["athlete_code", "competitor", "sport", "country", "medal"],
        "values": ["Phelps", "Swimming", "USA", "Gold", "Bolt"]
    },
    "professional_basketball": {
        "columns": ["player_code", "player", "team", "position", "points"],
        "values": ["LeBron", "Lakers", "Forward", "25.5", "Curry"]
    },
    "public_review_platform": {
        "columns": ["review_number", "business", "rating", "text", "date"],
        "values": ["5 stars", "Great service", "2023-01-15", "Excellent", "Amazing"]
    },
    "regional_sales": {
        "columns": ["transaction_code", "region", "product", "amount", "date"],
        "values": ["North", "Laptop", "$1200", "2023-01-15", "Electronics"]
    },
    "restaurant": {
        "columns": ["restaurant_code", "establishment", "cuisine", "rating", "price_range"],
        "values": ["Italian", "4.5", "$$", "Pizza Palace", "Fine Dining"]
    },
    "retail_complains": {
        "columns": ["complaint_number", "customer", "product", "issue", "status"],
        "values": ["Defective", "Resolved", "Refund", "Quality Issue", "Pending"]
    },
    "retail_world": {
        "columns": ["product_code", "product", "category", "price", "stock"],
        "values": ["Electronics", "$299", "50", "Laptop", "Smartphone"]
    },
    "retails": {
        "columns": ["product_code", "product", "category", "price", "stock"],
        "values": ["Electronics", "$299", "50", "Laptop", "Smartphone"]
    },
    "sales": {
        "columns": ["transaction_code", "product", "amount", "date", "customer"],
        "values": ["$150", "2023-01-15", "John", "Laptop", "Electronics"]
    },
    "sales_in_weather": {
        "columns": ["transaction_code", "product", "weather", "amount", "date"],
        "values": ["Sunny", "Rainy", "$200", "2023-01-15", "Umbrella"]
    },
    "shakespeare": {
        "columns": ["work_code", "play_title", "type", "year", "character"],
        "values": ["Hamlet", "Tragedy", "1603", "Romeo", "Juliet"]
    },
    "shipping": {
        "columns": ["shipment_number", "tracking", "destination", "status", "date"],
        "values": ["Delivered", "In Transit", "New York", "ABC123", "2023-01-15"]
    },
    "shooting": {
        "columns": ["incident_number", "location", "date", "casualties", "weapon"],
        "values": ["School", "2023-01-15", "0", "Handgun", "Mall"]
    },
    "simpson_episodes": {
        "columns": ["episode_number", "episode_title", "season", "episode", "air_date"],
        "values": ["Homer", "Season 1", "Episode 1", "1989-12-17", "Bart"]
    },
    "soccer_2016": {
        "columns": ["player_code", "player", "team", "position", "goals"],
        "values": ["Messi", "Barcelona", "Forward", "25", "Ronaldo"]
    },
    "social_media": {
        "columns": ["post_number", "user", "content", "likes", "date"],
        "values": ["@john", "Great post!", "150", "2023-01-15", "Tweet"]
    },
    "software_company": {
        "columns": ["employee_id", "full_name", "department", "salary", "project"],
        "values": ["Engineering", "$90000", "AI Project", "Sarah", "Manager"]
    },
    "student_loan": {
        "columns": ["loan_number", "student", "amount", "interest_rate", "status"],
        "values": ["$25000", "4.5%", "Active", "John", "Graduate"]
    },
    "superstore": {
        "columns": ["product_id", "product", "category", "price", "region"],
        "values": ["Electronics", "$299", "West", "Laptop", "Office Supplies"]
    },
    "synthea": {
        "columns": ["patient_code", "age", "gender", "condition", "medication"],
        "values": ["45", "Female", "Diabetes", "Metformin", "Hypertension"]
    },
    "talkingdata": {
        "columns": ["user_code", "device", "app", "duration", "timestamp"],
        "values": ["Android", "Facebook", "120", "2023-01-15", "iPhone"]
    },
    "trains": {
        "columns": ["train_number", "route", "departure", "arrival", "status"],
        "values": ["Express", "On Time", "8:00 AM", "10:30 AM", "Delayed"]
    },
    "university": {
        "columns": ["student_id", "full_name", "major", "gpa", "year"],
        "values": ["CS101", "John", "Computer Science", "3.5", "Sophomore"]
    },
    "video_games": {
        "columns": ["game_code", "game_title", "genre", "platform", "rating"],
        "values": ["Action", "PlayStation", "9.5", "Cyberpunk", "RPG"]
    },
    "works_cycles": {
        "columns": ["work_cycle_code", "employee", "start_time", "end_time", "productivity"],
        "values": ["John", "9:00 AM", "5:00 PM", "High", "Work Day"]
    },
    "world": {
        "columns": ["country_code", "country", "capital", "population", "area"],
        "values": ["France", "Paris", "67000000", "643801", "Germany"]
    },
    "world_development_indicators": {
        "columns": ["indicator_code", "country", "year", "value", "category"],
        "values": ["GDP", "USA", "2023", "25.5", "Economic"]
    }
}