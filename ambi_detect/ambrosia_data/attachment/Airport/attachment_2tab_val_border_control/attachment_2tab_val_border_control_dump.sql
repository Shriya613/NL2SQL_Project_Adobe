BEGIN TRANSACTION;
CREATE TABLE Aircraft(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Model TEXT,
    Capacity INTEGER,
    RegistrationNumber TEXT,
    Manufacturer TEXT);
INSERT INTO "Aircraft" VALUES(1,'Boeing 747-400',400,'G-BNAC','Boeing');
INSERT INTO "Aircraft" VALUES(2,'Airbus A380',800,'G-KBJS','Airbus');
INSERT INTO "Aircraft" VALUES(3,'Bombardier CRJ-1000',100,'EI-EEE','Bombardier');
INSERT INTO "Aircraft" VALUES(4,'Embraer E190',100,'PS-AMZ','Embraer');
INSERT INTO "Aircraft" VALUES(5,'AIRBUS A320',180,'VQ-BEA','Airbus');
CREATE TABLE Customs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Airport TEXT,
    CustomOfficerName TEXT,
    CustomOfficePhoneNumber TEXT,
    OpeningHours TEXT);
INSERT INTO "Customs" VALUES(1,'Gatwick','John Doe','1234567890','06:00 - 22:00');
INSERT INTO "Customs" VALUES(2,'Heathrow','Jane Smith','0987654321','06:30 - 21:30');
INSERT INTO "Customs" VALUES(3,'Stansted','Alice Brown','1122334455','07:00 - 21:00');
INSERT INTO "Customs" VALUES(4,'Luton','Bob Johnson','2233445566','06:00 - 22:00');
INSERT INTO "Customs" VALUES(5,'Manchester','Charlie White','3344556677','06:30 - 21:30');
CREATE TABLE Flight(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    DepartureTime TIMESTAMP,
    ArrivalTime TIMESTAMP,
    DestinationAirport TEXT,
    PlaneId INTEGER,
    FOREIGN KEY (PlaneId) REFERENCES Aircraft(id));
INSERT INTO "Flight" VALUES(1,'2023-04-01 08:00:00','2023-04-01 11:00:00','New York',1);
INSERT INTO "Flight" VALUES(2,'2023-04-01 10:00:00','2023-04-01 13:00:00','Los Angeles',2);
INSERT INTO "Flight" VALUES(3,'2023-04-01 12:00:00','2023-04-01 15:00:00','Paris',3);
INSERT INTO "Flight" VALUES(4,'2023-04-01 14:00:00','2023-04-01 17:00:00','Tokyo',4);
INSERT INTO "Flight" VALUES(5,'2023-04-01 16:00:00','2023-04-01 19:00:00','Rome',5);
CREATE TABLE Immigration(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Airport TEXT,
    ImmigrationOfficerName TEXT,
    ImmigrationOfficePhoneNumber TEXT,
    OpeningHours TEXT);
INSERT INTO "Immigration" VALUES(1,'Gatwick','Michael Black','1357924680','06:00 - 22:00');
INSERT INTO "Immigration" VALUES(2,'Heathrow','Emma Green','0987654321','06:30 - 21:30');
INSERT INTO "Immigration" VALUES(3,'Stansted','David Blue','1122334455','07:00 - 21:00');
INSERT INTO "Immigration" VALUES(4,'Luton','Nathan Red','2233445566','06:00 - 22:00');
INSERT INTO "Immigration" VALUES(5,'Manchester','Simon Yellow','3344556677','06:30 - 21:30');
CREATE TABLE Runway(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Airport TEXT,
    Length REAL,
    Width REAL,
    SurfaceType TEXT);
INSERT INTO "Runway" VALUES(1,'Gatwick',3000.0,45.0,'Asphalt');
INSERT INTO "Runway" VALUES(2,'Heathrow',3200.0,60.0,'Concrete');
INSERT INTO "Runway" VALUES(3,'Stansted',2800.0,40.0,'Asphalt');
INSERT INTO "Runway" VALUES(4,'Luton',2500.0,35.0,'Asphalt');
INSERT INTO "Runway" VALUES(5,'Manchester',3100.0,50.0,'Concrete');
CREATE TABLE Terminal(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Airport TEXT,
    Name TEXT,
    NumberOfDoors INTEGER,
    Facilities TEXT);
INSERT INTO "Terminal" VALUES(1,'Gatwick','Terminal A',10,'Shops, Restaurants, Baggage Claim');
INSERT INTO "Terminal" VALUES(2,'Heathrow','Terminal 2',15,'Duty Free Shops, Lounge, Currency Exchange');
INSERT INTO "Terminal" VALUES(3,'Stansted','Terminal 1',8,'Cafes, Information Desk, Baby Changing Rooms');
INSERT INTO "Terminal" VALUES(4,'Luton','Main Terminal',6,'ATMs, Lost & Found, Car Rental');
INSERT INTO "Terminal" VALUES(5,'Manchester','Terminal 1',12,'WiFi, Parking, Disabled Access');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customs',5);
INSERT INTO "sqlite_sequence" VALUES('Immigration',5);
INSERT INTO "sqlite_sequence" VALUES('Runway',5);
INSERT INTO "sqlite_sequence" VALUES('Terminal',5);
INSERT INTO "sqlite_sequence" VALUES('Flight',5);
INSERT INTO "sqlite_sequence" VALUES('Aircraft',5);
COMMIT;
