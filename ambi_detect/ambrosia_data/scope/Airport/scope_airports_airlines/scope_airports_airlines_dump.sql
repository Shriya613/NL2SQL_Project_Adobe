BEGIN TRANSACTION;
CREATE TABLE Airlines(
    AirlineID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT
);
INSERT INTO "Airlines" VALUES(1,'American_Airlines');
INSERT INTO "Airlines" VALUES(2,'United_Airlines');
INSERT INTO "Airlines" VALUES(3,'Delta_Airlines');
INSERT INTO "Airlines" VALUES(4,'Southwest_Airlines');
INSERT INTO "Airlines" VALUES(5,'JetBlue_Airlines');
CREATE TABLE Airports(
    AirportID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT
);
INSERT INTO "Airports" VALUES(1,'New_York_John_F._Kennedy_International_Airport','New York, USA');
INSERT INTO "Airports" VALUES(2,'Chicago_O''Hare_International_Airport','Chicago, USA');
INSERT INTO "Airports" VALUES(3,'Dallas_Love_Field','Dallas, USA');
INSERT INTO "Airports" VALUES(4,'Los_Angeles_International_Airport','Los Angeles, USA');
INSERT INTO "Airports" VALUES(5,'Miami_International_Airport','Miami, USA');
CREATE TABLE Airports_Airlines(
    AirportID INTEGER,
    AirlineID INTEGER,
    FOREIGN KEY(AirportID) REFERENCES Airports(AirportID),
    FOREIGN KEY(AirlineID) REFERENCES Airlines(AirlineID),
    PRIMARY KEY(AirportID, AirlineID)
);
INSERT INTO "Airports_Airlines" VALUES(1,1);
INSERT INTO "Airports_Airlines" VALUES(2,1);
INSERT INTO "Airports_Airlines" VALUES(3,1);
INSERT INTO "Airports_Airlines" VALUES(4,1);
INSERT INTO "Airports_Airlines" VALUES(5,1);
INSERT INTO "Airports_Airlines" VALUES(5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airlines',5);
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
COMMIT;
