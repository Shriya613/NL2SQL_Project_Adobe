BEGIN TRANSACTION;
CREATE TABLE Bars(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Year_of_establishment INT,
    Type_of_bar TEXT,
    Capacity INT);
INSERT INTO "Bars" VALUES(1,'The Roof Garden','New York',1980,'Roof Bar',200);
INSERT INTO "Bars" VALUES(2,'Speakeasy Spirits','New York',2005,'Cocktail Bar',100);
INSERT INTO "Bars" VALUES(3,'Windy City Pub','Chicago',2010,'Irish Pub',250);
INSERT INTO "Bars" VALUES(4,'Beachside Brewery','Miami',1970,'Brewery',150);
INSERT INTO "Bars" VALUES(5,'Whiskey Way Station','Dallas',1990,'Whiskey Bar',100);
CREATE TABLE EventVenues(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Year_of_establishment INT,
    Maximum_capacity INT,
    Types_of_events TEXT);
INSERT INTO "EventVenues" VALUES(1,'Metropolitan Ballroom','New York',1980,500,'Conferences, Weddings, Parties');
INSERT INTO "EventVenues" VALUES(2,'Union Hall','New York',2000,300,'Corporate Events, Exhibitions, Concerts');
INSERT INTO "EventVenues" VALUES(3,'Lakeview Pavilion','Chicago',2010,600,'Outdoor Weddings, Festivals, Sporting Events');
INSERT INTO "EventVenues" VALUES(4,'Palm Beach Convention Center','Miami',1970,800,'Trade Shows, Banquets, Awards Ceremonies');
INSERT INTO "EventVenues" VALUES(5,'Texas State Fairgrounds','Dallas',1990,1000,'Agricultural Fairs, Music Festivals, Car Shows');
CREATE TABLE Hotels(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Year_of_establishment INT,
    Number_of_rooms INT,
    Star_rating INT);
INSERT INTO "Hotels" VALUES(1,'Grand Central Hotel','New York',1980,300,5);
INSERT INTO "Hotels" VALUES(2,'Times Square Inn','New York',2000,200,3);
INSERT INTO "Hotels" VALUES(3,'Chicago Hilton','Chicago',2010,400,4);
INSERT INTO "Hotels" VALUES(4,'South Beach Resort','Miami',1970,250,4);
INSERT INTO "Hotels" VALUES(5,'Northstar Lodge','Dallas',1990,150,3);
CREATE TABLE SeaFoodRestaurants(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Year_of_establishment INT,
    Number_of_employees INT,
    Specialty_dish TEXT);
INSERT INTO "SeaFoodRestaurants" VALUES(1,'Ocean Treasures','Boston',1980,40,'Lobster Roll');
INSERT INTO "SeaFoodRestaurants" VALUES(2,'Bay View Cafe','San Francisco',2005,50,'Crab Cakes');
INSERT INTO "SeaFoodRestaurants" VALUES(3,'Harbor Grill','Atlanta',1990,60,'Shrimp Scampi');
INSERT INTO "SeaFoodRestaurants" VALUES(4,'Seaside Shack','Washington DC',2015,30,'Clam Chowder');
INSERT INTO "SeaFoodRestaurants" VALUES(5,'Coastal Kitchen','Philadelphia',1975,20,'Salmon');
CREATE TABLE SteakHouses(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Year_of_establishment INT,
    Number_of_employees INT,
    Average_meal_price REAL);
INSERT INTO "SteakHouses" VALUES(1,'Old Steak','New York',1980,30,20.0);
INSERT INTO "SteakHouses" VALUES(2,'Tenderloin Palace','Los Angeles',2000,50,25.0);
INSERT INTO "SteakHouses" VALUES(3,'Ribeye Ranch','Chicago',2010,60,30.0);
INSERT INTO "SteakHouses" VALUES(4,'Filet Mignon Manor','Miami',1970,40,22.0);
INSERT INTO "SteakHouses" VALUES(5,'Sirloin Shack','Dallas',1990,20,15.0);
COMMIT;
