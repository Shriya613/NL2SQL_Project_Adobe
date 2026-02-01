BEGIN TRANSACTION;
CREATE TABLE Airports(
    AirportID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT);
INSERT INTO "Airports" VALUES(1,'JFK International','New York');
INSERT INTO "Airports" VALUES(2,'LAX International','Los Angeles');
INSERT INTO "Airports" VALUES(3,'ORD International','Chicago');
INSERT INTO "Airports" VALUES(4,'IAD International','Washington D.C.');
INSERT INTO "Airports" VALUES(5,'ATL Hartsfield-Jackson','Atlanta');
CREATE TABLE Airports_Facilities(
    AirportID INTEGER,
    FacilityID INTEGER,
    FOREIGN KEY(AirportID) REFERENCES Airports(AirportID),
    FOREIGN KEY(FacilityID) REFERENCES Facilities(FacilityID),
    PRIMARY KEY(AirportID, FacilityID));
INSERT INTO "Airports_Facilities" VALUES(1,1);
INSERT INTO "Airports_Facilities" VALUES(2,1);
INSERT INTO "Airports_Facilities" VALUES(3,1);
INSERT INTO "Airports_Facilities" VALUES(4,1);
INSERT INTO "Airports_Facilities" VALUES(5,1);
INSERT INTO "Airports_Facilities" VALUES(5,2);
CREATE TABLE Facilities(
    FacilityID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT);
INSERT INTO "Facilities" VALUES(1,'Duty Free Shopping');
INSERT INTO "Facilities" VALUES(2,'Restaurant');
INSERT INTO "Facilities" VALUES(3,'Car Rental');
INSERT INTO "Facilities" VALUES(4,'Hotel Booking Desk');
INSERT INTO "Facilities" VALUES(5,'Baggage Storage');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Facilities',5);
COMMIT;
