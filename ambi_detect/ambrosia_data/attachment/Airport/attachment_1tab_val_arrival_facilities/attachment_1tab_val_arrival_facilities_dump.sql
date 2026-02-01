BEGIN TRANSACTION;
CREATE TABLE Airlines(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Headquarters TEXT,
    IATA_Code CHAR(2) UNIQUE);
INSERT INTO "Airlines" VALUES(1,'Emirates','Dubai, United Arab Emirates','EK');
INSERT INTO "Airlines" VALUES(2,'American Airlines','Fort Worth, Texas, U.S.','AA');
CREATE TABLE Airports(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    NumberOfTerminals INTEGER,
    FOREIGN KEY(NumberOfTerminals) REFERENCES TerminalFacility(id));
INSERT INTO "Airports" VALUES(1,'Los Angeles International Airport','California, USA',9);
INSERT INTO "Airports" VALUES(2,'Dubai International Airport','United Arab Emirates',3);
CREATE TABLE Flights(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FlightNo INTEGER,
    DepartureAirportId INTEGER,
    DestinationAirportId INTEGER,
    AirlineId INTEGER,
    Departs TIMESTAMP,
    Arrives TIMESTAMP,
    FOREIGN KEY(DepartureAirportId) REFERENCES Airports(id),
    FOREIGN KEY(DestinationAirportId) REFERENCES Airports(id),
    FOREIGN KEY(AirlineId) REFERENCES Airlines(id));
INSERT INTO "Flights" VALUES(1,1,1,2,1,'2023-04-01 10:00:00','2023-04-01 14:00:00');
INSERT INTO "Flights" VALUES(2,2,1,3,2,'2023-04-01 11:00:00','2023-04-01 15:00:00');
CREATE TABLE Passengers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    FlightId INTEGER,
    FOREIGN KEY(FlightId) REFERENCES Flights(id));
INSERT INTO "Passengers" VALUES(1,'John','Smith','1990-01-01',1);
INSERT INTO "Passengers" VALUES(2,'Jane','Doe','1992-02-02',2);
CREATE TABLE TerminalFacility(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Arrival_Facilities TEXT,
    Maximum_Capacity TEXT,
    OpeningHours TEXT);
INSERT INTO "TerminalFacility" VALUES(1,'Arrival Hall','500 passengers','06:00 - 23:00');
INSERT INTO "TerminalFacility" VALUES(2,'Arrival Hall','1000 passengers','07:00 - 22:00');
INSERT INTO "TerminalFacility" VALUES(3,'Baggage Reclaim','500 passengers','08:00 - 21:00');
INSERT INTO "TerminalFacility" VALUES(4,'Baggage Reclaim','800 passengers','09:00 - 20:00');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('TerminalFacility',4);
INSERT INTO "sqlite_sequence" VALUES('Airports',2);
INSERT INTO "sqlite_sequence" VALUES('Airlines',2);
INSERT INTO "sqlite_sequence" VALUES('Flights',2);
INSERT INTO "sqlite_sequence" VALUES('Passengers',2);
COMMIT;
