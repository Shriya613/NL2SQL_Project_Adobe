BEGIN TRANSACTION;
CREATE TABLE Airports(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    city TEXT,
    country TEXT);
INSERT INTO "Airports" VALUES(1,'New York JFK','New York','USA');
INSERT INTO "Airports" VALUES(2,'London Heathrow','London','UK');
INSERT INTO "Airports" VALUES(3,'Tokyo Haneda','Tokyo','Japan');
INSERT INTO "Airports" VALUES(4,'Dubai International','Dubai','UAE');
INSERT INTO "Airports" VALUES(5,'Frankfurt Airport','Frankfurt','Germany');
CREATE TABLE Flights(
    flightNumber INTEGER PRIMARY KEY AUTOINCREMENT,
    departureAirportId INTEGER REFERENCES Airports(id) ON DELETE CASCADE,
    arrivalAirportId INTEGER REFERENCES Airports(id) ON DELETE CASCADE,
    departureTime TIMESTAMP,
    arrivalTime TIMESTAMP);
INSERT INTO "Flights" VALUES(1,1,2,'2023-06-01 10:00:00','2023-06-01 18:00:00');
INSERT INTO "Flights" VALUES(2,3,4,'2023-07-01 12:00:00','2023-07-01 20:00:00');
INSERT INTO "Flights" VALUES(3,5,2,'2023-08-01 14:00:00','2023-08-01 22:00:00');
INSERT INTO "Flights" VALUES(4,3,5,'2023-09-01 16:00:00','2023-09-01 23:00:00');
INSERT INTO "Flights" VALUES(5,1,4,'2023-10-01 08:00:00','2023-10-01 16:00:00');
CREATE TABLE Passengers(
    passengerId INTEGER PRIMARY KEY AUTOINCREMENT,
    dateOfBirth DATE,
    isMinor BOOLEAN DEFAULT FALSE, FullName TEXT);
INSERT INTO "Passengers" VALUES(1,'1990-01-01',0,'John Doe');
INSERT INTO "Passengers" VALUES(2,'1985-02-02',0,'Jane Smith');
INSERT INTO "Passengers" VALUES(3,'1991-03-03',0,'Emily Johnson');
INSERT INTO "Passengers" VALUES(4,'1988-04-04',0,'Michael Williams');
INSERT INTO "Passengers" VALUES(5,'1992-05-05',0,'David Brown');
INSERT INTO "Passengers" VALUES(6,'2005-07-15',1,'Alice Morgan');
INSERT INTO "Passengers" VALUES(7,'2002-11-20',1,'Brian Case');
INSERT INTO "Passengers" VALUES(8,'1995-05-22',0,'Cathy Lee');
INSERT INTO "Passengers" VALUES(9,'1998-03-30',0,'Derek Woods');
CREATE TABLE "Requirements"(
    requirementId INTEGER PRIMARY KEY AUTOINCREMENT, passengerId INTEGER REFERENCES Passengers(passengerId),
    description TEXT,
    assistant TEXT);
INSERT INTO "Requirements" VALUES(1,1,'Wheelchair assistance required','Alice Brown');
INSERT INTO "Requirements" VALUES(2,2,'Oxygen tank needed on board','Bob Johnson');
INSERT INTO "Requirements" VALUES(3,3,'Diabetic meal request','Charlie White');
INSERT INTO "Requirements" VALUES(4,4,'Special dietary needs','Daniel Green');
INSERT INTO "Requirements" VALUES(5,5,'Hearing aid support','Eva Black');
INSERT INTO "Requirements" VALUES(6,6,'Unaccompanied minor','Clare Allen');
INSERT INTO "Requirements" VALUES(7,7,'Unaccompanied minor','Mark Taylor');
INSERT INTO "Requirements" VALUES(8,8,'Tailored assistance request','Clare Allen');
INSERT INTO "Requirements" VALUES(9,9,'Tailored assistance request','Nancy Davis');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('Passengers',9);
INSERT INTO "sqlite_sequence" VALUES('Requirements',9);
COMMIT;
