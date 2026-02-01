BEGIN TRANSACTION;
CREATE TABLE Aircraft(
    aircraftID INT PRIMARY KEY,
    model TEXT,
    capacity INT);
INSERT INTO "Aircraft" VALUES(1,'Airbus A320',200);
INSERT INTO "Aircraft" VALUES(2,'Boeing 737',180);
INSERT INTO "Aircraft" VALUES(3,'Embraer E190',100);
INSERT INTO "Aircraft" VALUES(4,'Airbus A380',500);
INSERT INTO "Aircraft" VALUES(5,'Boeing 777',350);
CREATE TABLE Amenity(
    amenityID INT PRIMARY KEY,
    description TEXT);
INSERT INTO "Amenity" VALUES(1,'Free high-speed internet access');
INSERT INTO "Amenity" VALUES(2,'Power outlets at every seat');
INSERT INTO "Amenity" VALUES(3,'Quiet zones for relaxation');
INSERT INTO "Amenity" VALUES(4,'Children''s play area');
INSERT INTO "Amenity" VALUES(5,'Showers facilities');
CREATE TABLE Flight(
    flightID INT PRIMARY KEY,
    departureTime TIMESTAMP,
    arrivalTime TIMESTAMP,
    aircraftID INT,
    terminalID INT,
    FOREIGN KEY(aircraftID) REFERENCES Aircraft(aircraftID),
    FOREIGN KEY(terminalID) REFERENCES Terminal(terminalID));
INSERT INTO "Flight" VALUES(1,'2023-01-01 07:00:00','2023-01-01 10:00:00',1,1);
INSERT INTO "Flight" VALUES(2,'2023-01-01 09:00:00','2023-01-01 11:00:00',2,1);
INSERT INTO "Flight" VALUES(3,'2023-01-01 11:00:00','2023-01-01 13:00:00',3,2);
INSERT INTO "Flight" VALUES(4,'2023-01-01 14:00:00','2023-01-01 16:00:00',4,1);
INSERT INTO "Flight" VALUES(5,'2023-01-01 16:00:00','2023-01-01 18:00:00',5,2);
CREATE TABLE Lounge(
    loungeID INT PRIMARY KEY,
    name TEXT,
    exclusiveArea TEXT,
    amenities TEXT,
    capacity INT,
    terminalID INT,
    FOREIGN KEY(terminalID) REFERENCES Terminal(terminalID));
INSERT INTO "Lounge" VALUES(1,'Business Class Lounge','Frequent Flyer Lounge','Complimentary Wi-Fi',100,1);
INSERT INTO "Lounge" VALUES(2,'Premium Economy Lounge','Frequent Flyer Lounge','Not Complimentary Wi-Fi',80,1);
INSERT INTO "Lounge" VALUES(3,'First Class Lounge','VIP Lounge','Complimentary Wi-Fi',50,2);
INSERT INTO "Lounge" VALUES(4,'Golden Wings Lounge','VIP Lounge','Not Complimentary Wi-Fi',60,2);
INSERT INTO "Lounge" VALUES(5,'Silver Stars Lounge','','',40,1);
CREATE TABLE Passenger(
    passengerID INT PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    flightID INT,
    loungeID INT,
    FOREIGN KEY(flightID) REFERENCES Flight(flightID),
    FOREIGN KEY(loungeID) REFERENCES Lounge(loungeID));
INSERT INTO "Passenger" VALUES(1,'John','Doe',1,1);
INSERT INTO "Passenger" VALUES(2,'Jane','Smith',2,1);
INSERT INTO "Passenger" VALUES(3,'Alice','Brown',3,2);
INSERT INTO "Passenger" VALUES(4,'Michael','Johnson',4,1);
INSERT INTO "Passenger" VALUES(5,'Emily','White',5,2);
CREATE TABLE Terminal(
    terminalID INT PRIMARY KEY,
    location TEXT);
INSERT INTO "Terminal" VALUES(1,'Terminal 1');
INSERT INTO "Terminal" VALUES(2,'Terminal 2');
COMMIT;
