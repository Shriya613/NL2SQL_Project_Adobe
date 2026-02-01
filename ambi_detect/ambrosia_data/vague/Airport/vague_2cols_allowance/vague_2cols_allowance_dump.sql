BEGIN TRANSACTION;
CREATE TABLE Airlines (
    AirlineID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Headquarters TEXT,
    Founded DATE,
    Website TEXT UNIQUE);
INSERT INTO "Airlines" VALUES(1,'American Airlines','Fort Worth, Texas','1930-04-15','https://www.aa.com');
INSERT INTO "Airlines" VALUES(2,'United Airlines','Chicago, Illinois','1926-07-01','https://www.united.com');
INSERT INTO "Airlines" VALUES(3,'Delta Air Lines','Atlanta, Georgia','1928-04-17','https://www.delta.com');
INSERT INTO "Airlines" VALUES(4,'Southwest Airlines','Dallas, Texas','1967-03-15','https://www.southwest.com');
INSERT INTO "Airlines" VALUES(5,'JetBlue Airways','New York City, New York','2000-02-11','https://www.jetblue.com');
CREATE TABLE AirportFacilities (
    FacilityID INTEGER PRIMARY KEY AUTOINCREMENT,
    AirportName TEXT,
    TerminalCount INTEGER,
    RestaurantCount INTEGER,
    ShoppingMallCount INTEGER,
    ParkingCapacity INTEGER,
    WiFiAvailability BOOLEAN DEFAULT true);
INSERT INTO "AirportFacilities" VALUES(1,'DFW International Airport',5,100,20,10000,1);
INSERT INTO "AirportFacilities" VALUES(2,'Los Angeles International Airport',9,150,30,12000,1);
INSERT INTO "AirportFacilities" VALUES(3,'Hartsfield–Jackson Atlanta International Airport',6,120,40,15000,1);
INSERT INTO "AirportFacilities" VALUES(4,'Seattle–Tacoma International Airport',4,80,25,10000,1);
INSERT INTO "AirportFacilities" VALUES(5,'Newark Liberty International Airport',3,60,15,12000,1);
CREATE TABLE BaggageLimits (
    LimitID INTEGER PRIMARY KEY AUTOINCREMENT,
    BagWeightLimit INTEGER,
    BagPieceLimit INTEGER,
    Airlines TEXT,
    CheckInDeadline TIMESTAMP);
INSERT INTO "BaggageLimits" VALUES(1,30,2,'All','2023-03-01 02:00:00');
INSERT INTO "BaggageLimits" VALUES(2,40,3,'Delta','2023-03-01 03:00:00');
INSERT INTO "BaggageLimits" VALUES(3,35,2,'Southwest','2023-03-01 04:00:00');
INSERT INTO "BaggageLimits" VALUES(4,25,1,'JetBlue','2023-03-01 05:00:00');
INSERT INTO "BaggageLimits" VALUES(5,20,1,'Other','2023-03-01 06:00:00');
CREATE TABLE FlightRoutes (
    RouteID INTEGER PRIMARY KEY AUTOINCREMENT,
    OriginAirport TEXT,
    DestinationAirport TEXT,
    DepartureTime TIMESTAMP,
    ArrivalTime TIMESTAMP,
    AircraftType TEXT,
    SeatsAvailable INTEGER );
INSERT INTO "FlightRoutes" VALUES(1,'DFW','LAX','2023-03-01 07:00:00','2023-03-01 11:00:00','Boeing 737',150);
INSERT INTO "FlightRoutes" VALUES(2,'ATL','ORD','2023-03-01 08:00:00','2023-03-01 12:00:00','Airbus A320',200);
INSERT INTO "FlightRoutes" VALUES(3,'SEA','SFO','2023-03-01 09:00:00','2023-03-01 13:00:00','Boeing 787',250);
INSERT INTO "FlightRoutes" VALUES(4,'EWR','MIA','2023-03-01 10:00:00','2023-03-01 14:00:00','Airbus A321',225);
INSERT INTO "FlightRoutes" VALUES(5,'IAH','LAX','2023-03-01 11:00:00','2023-03-01 15:00:00','Boeing 777',300);
CREATE TABLE Passengers (
    PassengerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Email TEXT UNIQUE);
INSERT INTO "Passengers" VALUES(1,'John','Doe','1990-01-01','john.doe@example.com');
INSERT INTO "Passengers" VALUES(2,'Jane','Smith','1992-02-15','jane.smith@example.com');
INSERT INTO "Passengers" VALUES(3,'Alice','Johnson','1988-03-20','alice.johnson@example.com');
INSERT INTO "Passengers" VALUES(4,'Bob','Williams','1991-04-05','bob.williams@example.com');
INSERT INTO "Passengers" VALUES(5,'Emma','Martinez','1993-05-10','emma.martinez@example.com');
CREATE TABLE SecurityCheckpoints (
    CheckpointID INTEGER PRIMARY KEY AUTOINCREMENT,
    AirportName TEXT,
    Location TEXT,
    OpeningHours TEXT,
    StaffCount INTEGER,
    WaitTimes INTEGER);
INSERT INTO "SecurityCheckpoints" VALUES(1,'DFW International Airport','Terminal A','04:00 - 22:00',100,30);
INSERT INTO "SecurityCheckpoints" VALUES(2,'Los Angeles International Airport','Tom Bradley International Terminal','04:00 - 22:00',120,45);
INSERT INTO "SecurityCheckpoints" VALUES(3,'Hartsfield–Jackson Atlanta International Airport','Domestic Terminal North','04:00 - 22:00',150,40);
INSERT INTO "SecurityCheckpoints" VALUES(4,'Seattle–Tacoma International Airport','North Satellite','04:00 - 22:00',80,35);
INSERT INTO "SecurityCheckpoints" VALUES(5,'Newark Liberty International Airport','Terminal A','04:00 - 22:00',110,50);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airlines',5);
INSERT INTO "sqlite_sequence" VALUES('FlightRoutes',5);
INSERT INTO "sqlite_sequence" VALUES('Passengers',5);
INSERT INTO "sqlite_sequence" VALUES('BaggageLimits',5);
INSERT INTO "sqlite_sequence" VALUES('AirportFacilities',5);
INSERT INTO "sqlite_sequence" VALUES('SecurityCheckpoints',5);
COMMIT;
