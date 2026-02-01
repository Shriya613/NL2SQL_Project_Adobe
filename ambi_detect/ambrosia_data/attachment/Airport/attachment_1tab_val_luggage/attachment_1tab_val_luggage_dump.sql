BEGIN TRANSACTION;
CREATE TABLE Aircrafts(
    AircraftID INTEGER PRIMARY KEY,
    Model TEXT,
    Capacity INTEGER,
    RegistrationNumber VARCHAR(10) UNIQUE
);
INSERT INTO "Aircrafts" VALUES(1,'Boeing 737',200,'N12345');
INSERT INTO "Aircrafts" VALUES(2,'Airbus A320',180,'N67890');
INSERT INTO "Aircrafts" VALUES(3,'Embraer E190',100,'N23456');
CREATE TABLE Assignments(
    AssignmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FlightID INTEGER,
    AircraftID INTEGER,
    AssignedDate DATE,
    ReturnedDate DATE,
    FOREIGN KEY(FlightID) REFERENCES Flights(FlightID),
    FOREIGN KEY(AircraftID) REFERENCES Aircrafts(AircraftID)
);
INSERT INTO "Assignments" VALUES(1,1,1,'2023-01-01','2023-01-02');
INSERT INTO "Assignments" VALUES(2,2,2,'2023-01-02','2023-01-03');
INSERT INTO "Assignments" VALUES(3,3,3,'2023-01-03','2023-01-04');
CREATE TABLE CheckIns(
    CheckInID INTEGER PRIMARY KEY AUTOINCREMENT,
    FlightID INTEGER,
    PassengerID INTEGER,
    CheckInDateTime DATETIME,
    BoardingGate TEXT,
    Seat ASSIGNMENT,
    FOREIGN KEY(FlightID) REFERENCES Flights(FlightID),
    FOREIGN KEY(PassengerID) REFERENCES Passengers(PassengerID)
);
INSERT INTO "CheckIns" VALUES(1,1,1,'2023-01-01 09:00:00','Gates C1-C10','12A');
INSERT INTO "CheckIns" VALUES(2,2,2,'2023-01-02 10:00:00','Gates D1-D20','23B');
INSERT INTO "CheckIns" VALUES(3,3,3,'2023-01-03 11:00:00','Gates E1-E10','34A');
INSERT INTO "CheckIns" VALUES(4,1,4,'2023-01-01 09:30:00','Gates C1-C10',NULL);
INSERT INTO "CheckIns" VALUES(5,2,5,'2023-01-02 10:30:00','Gates D1-D20',NULL);
CREATE TABLE Flights(
    FlightID INTEGER PRIMARY KEY,
    Origin TEXT,
    Destination TEXT,
    DepartureTime DATETIME,
    ArrivalTime DATETIME
);
INSERT INTO "Flights" VALUES(1,'JFK','LAX','2023-01-01 10:00:00','2023-01-01 17:00:00');
INSERT INTO "Flights" VALUES(2,'ORD','DFW','2023-01-02 11:00:00','2023-01-02 18:00:00');
INSERT INTO "Flights" VALUES(3,'SEA','SFO','2023-01-03 12:00:00','2023-01-03 19:00:00');
CREATE TABLE Luggage(
    ItemID INTEGER PRIMARY KEY AUTOINCREMENT,
    PassengerID INTEGER,
    Description TEXT,
    Fees TEXT,
    FOREIGN KEY(PassengerID) REFERENCES Passengers(PassengerID)
);
INSERT INTO "Luggage" VALUES(1,1,'Cabin Baggage','$25');
INSERT INTO "Luggage" VALUES(2,2,'Checked Baggage','$30');
INSERT INTO "Luggage" VALUES(3,3,'Checked Baggage','$25');
INSERT INTO "Luggage" VALUES(4,4,'Cabin Baggage','€35');
INSERT INTO "Luggage" VALUES(5,5,'Checked Baggage','$30');
CREATE TABLE Passengers(
    PassengerID INTEGER PRIMARY KEY,
    DateOfBirth DATE,
    Nationality TEXT
, FullName TEXT);
INSERT INTO "Passengers" VALUES(1,'1990-01-01','US','John Doe');
INSERT INTO "Passengers" VALUES(2,'1985-06-15','UK','Jane Smith');
INSERT INTO "Passengers" VALUES(3,'1992-12-20','CA','Alice Johnson');
INSERT INTO "Passengers" VALUES(4,'1988-03-03','AU','Michael Williams');
INSERT INTO "Passengers" VALUES(5,'1995-07-12','FR','Emily Brown');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Luggage',5);
INSERT INTO "sqlite_sequence" VALUES('Assignments',3);
INSERT INTO "sqlite_sequence" VALUES('CheckIns',5);
COMMIT;
