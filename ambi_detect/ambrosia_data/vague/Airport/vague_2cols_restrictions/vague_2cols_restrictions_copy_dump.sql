BEGIN TRANSACTION;
CREATE TABLE Airports (
    AirportCode CHAR(3) PRIMARY KEY,
    AirportName TEXT,
    City TEXT,
    Country TEXT,
    Latitude REAL,
    Longitude REAL
);
INSERT INTO "Airports" VALUES('LAX','Los Angeles International Airport','Los Angeles','USA',-118.405594,33.942934);
INSERT INTO "Airports" VALUES('JFK','John F Kennedy International Airport','New York','USA',-73.778929,40.639742);
CREATE TABLE Flights (
    FlightID INTEGER PRIMARY KEY,
    AirlineName TEXT,
    DepartureTime TIMESTAMP,
    ArrivalTime TIMESTAMP,
    DestinationCity TEXT,
    NumberOfPassengers INTEGER DEFAULT 0 ,
    BaggageAllowance INTEGER DEFAULT 25 
);
INSERT INTO "Flights" VALUES(1,'United','2021-08-26 10:00:00','2021-08-26 17:00:00','Los Angeles',200,25);
INSERT INTO "Flights" VALUES(2,'Delta','2021-08-27 12:00:00','2021-08-27 19:00:00','New York',150,20);
CREATE TABLE "Security" (
    RestrictionId INTEGER PRIMARY KEY AUTOINCREMENT,
    Restrictions TEXT,
    Airlines TEXT,
    FOREIGN KEY (Airlines) REFERENCES Flights(AirlineName)
);
INSERT INTO "Security" VALUES(1,'Over 12oz','United');
INSERT INTO "Security" VALUES(2,'More than 100ml','Delta');
CREATE TABLE Terminals (
    TerminalId INTEGER PRIMARY KEY AUTOINCREMENT,
    AirportCode CHAR(3),
    FloorSpace INTEGER ,
    NumberOfGates INTEGER ,
    FOREIGN KEY (AirportCode) REFERENCES Airports(AirportCode)
);
INSERT INTO "Terminals" VALUES(1,'LAX',3000000,100);
INSERT INTO "Terminals" VALUES(2,'JFK',500000,80);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Terminals',2);
INSERT INTO "sqlite_sequence" VALUES('Security',2);
COMMIT;
