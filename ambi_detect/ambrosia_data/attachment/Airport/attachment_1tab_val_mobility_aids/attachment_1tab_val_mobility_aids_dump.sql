BEGIN TRANSACTION;
CREATE TABLE Flight (
    flightID INTEGER PRIMARY KEY AUTOINCREMENT,
    departureTime DATETIME,
    destination VARCHAR(255),
    duration INT
);
INSERT INTO "Flight" VALUES(1,'2023-03-17 09:00','Paris',8);
INSERT INTO "Flight" VALUES(2,'2023-03-17 11:00','London',6);
CREATE TABLE Gate (
    gateID INTEGER PRIMARY KEY AUTOINCREMENT,
    gateNumber INT,
    terminal INT,
    capacity INT
);
INSERT INTO "Gate" VALUES(1,101,1,20);
INSERT INTO "Gate" VALUES(2,102,1,25);
CREATE TABLE Passenger (
    passengerID INTEGER PRIMARY KEY AUTOINCREMENT,
    mobilityAids TEXT ,
    pickUpDropOffLocation VARCHAR(100)
, FullName TEXT);
INSERT INTO "Passenger" VALUES(1,'Wheelchair Assistance','Entrance A','John Doe');
INSERT INTO "Passenger" VALUES(2,'Wheelchair Assistance','Exit C','Jane Smith');
INSERT INTO "Passenger" VALUES(3,'Mobility Scooter','Entrance A','Alice Brown');
INSERT INTO "Passenger" VALUES(4,'Mobility Scooter','Gate F','Bob White');
CREATE TABLE ServiceRequest (
    requestID INTEGER PRIMARY KEY AUTOINCREMENT,
    flightID INTEGER,
    passengerID INTEGER,
    serviceType VARCHAR(100),
    FOREIGN KEY (flightID) REFERENCES Flight(flightID),
    FOREIGN KEY (passengerID) REFERENCES Passenger(passengerID)
);
INSERT INTO "ServiceRequest" VALUES(1,1,1,'Assisted Boarding');
INSERT INTO "ServiceRequest" VALUES(2,1,2,'Meal Request');
INSERT INTO "ServiceRequest" VALUES(3,2,3,'Extra Luggage Space');
INSERT INTO "ServiceRequest" VALUES(4,2,4,'Special Meal');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Flight',2);
INSERT INTO "sqlite_sequence" VALUES('Gate',2);
INSERT INTO "sqlite_sequence" VALUES('Passenger',4);
INSERT INTO "sqlite_sequence" VALUES('ServiceRequest',4);
COMMIT;
