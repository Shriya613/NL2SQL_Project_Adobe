BEGIN TRANSACTION;
CREATE TABLE Locations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
INSERT INTO "Locations" VALUES(1,'Orlando, Florida');
INSERT INTO "Locations" VALUES(2,'Los Angeles, California');
INSERT INTO "Locations" VALUES(3,'New York City, New York');
INSERT INTO "Locations" VALUES(4,'Paris, France');
INSERT INTO "Locations" VALUES(5,'Tokyo, Japan');
CREATE TABLE ThemeParks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parkName TEXT,
    locationId INTEGER,
    familyEntertainment TEXT ,
    FOREIGN KEY (locationId) REFERENCES Locations(id));
INSERT INTO "ThemeParks" VALUES(1,'Disney World',1,'Amusement Park');
INSERT INTO "ThemeParks" VALUES(2,'Universal Studios Hollywood',2,'Amusement Park');
INSERT INTO "ThemeParks" VALUES(3,'Europa-Park',3,'Amusement Park');
INSERT INTO "ThemeParks" VALUES(4,'Walt Disney Studios Park',4,'Amusement Park');
INSERT INTO "ThemeParks" VALUES(5,'Aquatica Orlando',1,'Water Park');
INSERT INTO "ThemeParks" VALUES(6,'Typhoon Lagoon',1,'Water Park');
INSERT INTO "ThemeParks" VALUES(7,'Legoland Billund Resort',5,'Amusement Park');
INSERT INTO "ThemeParks" VALUES(8,'Port Aventura',6,'Amusement Park');
INSERT INTO "ThemeParks" VALUES(9,'Atlantis, The Palm',7,'Water Park');
INSERT INTO "ThemeParks" VALUES(10,'Discovery Cove',8,'Water Park');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Locations',5);
INSERT INTO "sqlite_sequence" VALUES('ThemeParks',10);
COMMIT;
