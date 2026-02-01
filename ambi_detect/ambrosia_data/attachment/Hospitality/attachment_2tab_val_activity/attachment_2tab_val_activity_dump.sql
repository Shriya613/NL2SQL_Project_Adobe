BEGIN TRANSACTION;
CREATE TABLE Accommodations(
    AccomodationID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    StarRating INTEGER,
    PricePerNight REAL
);
INSERT INTO "Accommodations" VALUES(1,'Hotel Royal','New York City',3,180.0);
INSERT INTO "Accommodations" VALUES(2,'Grand Hyatt','New York City',5,360.0);
CREATE TABLE Events(
    EventID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Date TEXT,
    Time TEXT,
    Venue TEXT,
    Description TEXT
);
INSERT INTO "Events" VALUES(1,'Concert at Madison Square Garden','2022-11-10','8PM','Madison Square Garden','A live concert by popular music artist.');
INSERT INTO "Events" VALUES(2,'Exhibition at MOMA','2022-11-11','10AM - 6PM','MOMA New York','An art exhibition featuring famous artists.');
CREATE TABLE Local_Markets(
    MarketID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    OperatingDays TEXT,
    ProductsSold TEXT
);
INSERT INTO "Local_Markets" VALUES(1,'Borough Market','London','Mon - Sat','fruits, vegetables, meats, cheese, baked goods');
INSERT INTO "Local_Markets" VALUES(2,'Les Halles de Lyon','Lyon','Mon - Sun','seafood, meats, cheeses, produce');
CREATE TABLE Restaurants(
    RestaurantID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    TypeOfCuisine TEXT,
    Location TEXT,
    AveragePriceForMeal REAL
);
INSERT INTO "Restaurants" VALUES(1,'Le Jules Verne','French','Paris',100.0);
INSERT INTO "Restaurants" VALUES(2,'The Capital Grille','Steakhouse','New York City',75.0);
CREATE TABLE Tourist_Attractions(
    AttractionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    EntranceFee REAL,
    OpeningHours TEXT
);
INSERT INTO "Tourist_Attractions" VALUES(1,'Big Ben','London',10.0,'9am - 7pm');
INSERT INTO "Tourist_Attractions" VALUES(2,'Eiffel Tower','Paris',20.0,'9am - 11pm');
CREATE TABLE Tours(
    TourID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Duration TEXT,
    StartingLocation TEXT,
    Cost REAL
);
INSERT INTO "Tours" VALUES(1,'Statue of Liberty Cruise','2 hours','New York City',25.0);
INSERT INTO "Tours" VALUES(2,'Louvre Museum Guided Tour','3 hours','Paris',40.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Tourist_Attractions',2);
INSERT INTO "sqlite_sequence" VALUES('Local_Markets',2);
INSERT INTO "sqlite_sequence" VALUES('Accommodations',2);
INSERT INTO "sqlite_sequence" VALUES('Restaurants',2);
INSERT INTO "sqlite_sequence" VALUES('Tours',2);
INSERT INTO "sqlite_sequence" VALUES('Events',2);
COMMIT;
