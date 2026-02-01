BEGIN TRANSACTION;
CREATE TABLE Aircrafts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT,
    max_passengers INTEGER,
    max_range INTEGER,
    airline_id INTEGER REFERENCES Airlines(id));
INSERT INTO "Aircrafts" VALUES(1,'Boeing 787 Dreamliner',300,7639,1);
INSERT INTO "Aircrafts" VALUES(2,'Airbus A380',525,8000,2);
CREATE TABLE Airlines(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    headquartered_at INTEGER REFERENCES Airports(id));
INSERT INTO "Airlines" VALUES(1,'BA',1);
INSERT INTO "Airlines" VALUES(2,'KLM',2);
CREATE TABLE Airports(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    country TEXT,
    city TEXT);
INSERT INTO "Airports" VALUES(1,'Heathrow','UK','London');
INSERT INTO "Airports" VALUES(2,'Schiphol','Netherlands','Amsterdam');
CREATE TABLE InflightEntertainments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entertainment_name TEXT,
    description TEXT,
    aircraft_id INTEGER REFERENCES Aircrafts(id));
INSERT INTO "InflightEntertainments" VALUES(1,'Movie Selection','Wide range of movies available for viewing.',1);
INSERT INTO "InflightEntertainments" VALUES(2,'Games Library','Variety of games available for play.',2);
CREATE TABLE OnboardServices(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT,
    description TEXT,
    aircraft_id INTEGER REFERENCES Aircrafts(id));
INSERT INTO "OnboardServices" VALUES(1,'Meal Service','A variety of meals served on board.',1);
INSERT INTO "OnboardServices" VALUES(2,'Snack Service','Light snacks available for purchase.',2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airports',2);
INSERT INTO "sqlite_sequence" VALUES('Airlines',2);
INSERT INTO "sqlite_sequence" VALUES('Aircrafts',2);
INSERT INTO "sqlite_sequence" VALUES('OnboardServices',2);
INSERT INTO "sqlite_sequence" VALUES('InflightEntertainments',2);
COMMIT;
