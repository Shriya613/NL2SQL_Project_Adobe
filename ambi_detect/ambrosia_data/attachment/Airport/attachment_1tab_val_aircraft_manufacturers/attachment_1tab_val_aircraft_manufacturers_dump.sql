BEGIN TRANSACTION;
CREATE TABLE Airlines(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    airlineName TEXT,
    headquarteredIn TEXT,
    homeAirportId INTEGER,
    FOREIGN KEY (homeAirportId) REFERENCES Airports(id));
INSERT INTO "Airlines" VALUES(1,'Alaska Airlines','USA',NULL);
INSERT INTO "Airlines" VALUES(2,'Delta Airlines','USA',NULL);
INSERT INTO "Airlines" VALUES(3,'United Airlines','USA',NULL);
INSERT INTO "Airlines" VALUES(4,'American Airlines','USA',NULL);
INSERT INTO "Airlines" VALUES(5,'JetBlue Airways','USA',NULL);
CREATE TABLE Airports(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    city TEXT,
    country TEXT);
INSERT INTO "Airports" VALUES(1,'A1','City A','Country A');
INSERT INTO "Airports" VALUES(2,'A2','City B','Country B');
INSERT INTO "Airports" VALUES(3,'A3','City C','Country C');
INSERT INTO "Airports" VALUES(4,'A4','City D','Country D');
INSERT INTO "Airports" VALUES(5,'A5','City E','Country E');
CREATE TABLE Flights(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flightNo INTEGER,
    departureTime TIMESTAMP,
    arrivalTime TIMESTAMP,
    originAirportId INTEGER,
    destinationAirportId INTEGER,
    aircraftId INTEGER,
    FOREIGN KEY (originAirportId) REFERENCES Airports(id),
    FOREIGN KEY (destinationAirportId) REFERENCES Airports(id),
    FOREIGN KEY (aircraftId) REFERENCES Manufactures(id));
INSERT INTO "Flights" VALUES(1,'AA1','2022-01-01 10:00:00','2022-01-01 15:00:00',1,2,1);
INSERT INTO "Flights" VALUES(2,'DL1','2022-01-01 11:00:00','2022-01-01 16:00:00',2,3,2);
INSERT INTO "Flights" VALUES(3,'UA1','2022-01-01 12:00:00','2022-01-01 17:00:00',3,4,3);
INSERT INTO "Flights" VALUES(4,'AS1','2022-01-01 13:00:00','2022-01-01 18:00:00',4,5,4);
INSERT INTO "Flights" VALUES(5,'B61','2022-01-01 14:00:00','2022-01-01 19:00:00',5,1,5);
CREATE TABLE "Manufactures"(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    aircraftModel TEXT,
    manufacturers TEXT,
    numberOfSeats INT);
INSERT INTO "Manufactures" VALUES(1,'737 MAX','Boeing',180);
INSERT INTO "Manufactures" VALUES(2,'A320neo','Airbus',180);
INSERT INTO "Manufactures" VALUES(3,'Bombardier CRJ900','Canada',65);
INSERT INTO "Manufactures" VALUES(4,'Embraer ERJ-170','Brazil',68);
INSERT INTO "Manufactures" VALUES(5,'Airbus A220','Airbus',130);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Airlines',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('Manufactures',5);
COMMIT;
