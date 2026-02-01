BEGIN TRANSACTION;
CREATE TABLE Airlines (
    AirlineID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Headquarters TEXT);
INSERT INTO "Airlines" VALUES(1,'American Airlines','Fort Worth');
INSERT INTO "Airlines" VALUES(2,'Delta Air Lines','Atlanta');
INSERT INTO "Airlines" VALUES(3,'United Airlines','Chicago');
INSERT INTO "Airlines" VALUES(4,'Southwest Airlines','Dallas');
INSERT INTO "Airlines" VALUES(5,'JetBlue Airways','New York');
CREATE TABLE Bookings (
    BookingID INTEGER PRIMARY KEY AUTOINCREMENT,
    FlightID INTEGER REFERENCES Flights(FlightID),
    PassengerID INTEGER REFERENCES Passengers(PassengerID),
    ClassOfService TEXT ,
    BookedDate DATETIME);
INSERT INTO "Bookings" VALUES(1,1,1,'Economy','2022-12-31 10:00');
INSERT INTO "Bookings" VALUES(2,2,2,'Business','2023-01-01 11:00');
INSERT INTO "Bookings" VALUES(3,3,3,'Premium Economy','2023-01-02 12:00');
INSERT INTO "Bookings" VALUES(4,4,4,'Economy','2023-01-03 13:00');
INSERT INTO "Bookings" VALUES(5,5,5,'Business','2023-01-04 14:00');
CREATE TABLE Flights (
    FlightID INTEGER PRIMARY KEY AUTOINCREMENT,
    OriginAirport TEXT,
    DestinationAirport TEXT,
    DepartureTime DATETIME,
    ArrivalTime DATETIME,
    AircraftType TEXT,
    OperatedBy INTEGER REFERENCES Airlines(AirlineID));
INSERT INTO "Flights" VALUES(1,'DFW','LAX','2023-01-01 08:00','2023-01-01 11:00','Boing 737',1);
INSERT INTO "Flights" VALUES(2,'ORD','MIA','2023-01-02 09:00','2023-01-02 12:00','Airbus A320',2);
INSERT INTO "Flights" VALUES(3,'IAH','PHX','2023-01-03 10:00','2023-01-03 13:00','Embraer E175',3);
INSERT INTO "Flights" VALUES(4,'SFO','SEA','2023-01-04 11:00','2023-01-04 14:00','Boeing 757',4);
INSERT INTO "Flights" VALUES(5,'ATL','MCO','2023-01-05 12:00','2023-01-05 15:00','Bombardier CRJ900',5);
CREATE TABLE Frequent_Flyer_Lounges (
    LoungeID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    Amenities TEXT,
    Capacity INTEGER);
INSERT INTO "Frequent_Flyer_Lounges" VALUES(1,'The Club','DFW Terminal C','Complimentary Wi-Fi',100);
INSERT INTO "Frequent_Flyer_Lounges" VALUES(2,'Club ATLAS','ATL International','No Complimentary Wi-Fi',150);
CREATE TABLE Passengers (
    PassengerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth TEXT,
    Nationality TEXT);
INSERT INTO "Passengers" VALUES(1,'John','Doe','1990-06-15','USA');
INSERT INTO "Passengers" VALUES(2,'Jane','Smith','1992-08-20','UK');
INSERT INTO "Passengers" VALUES(3,'Emma','Nelson','1988-10-05','Canada');
INSERT INTO "Passengers" VALUES(4,'Michael','Jackson','1995-12-25','Australia');
INSERT INTO "Passengers" VALUES(5,'David','Brown','1985-02-12','Germany');
CREATE TABLE VIP_Lounges (
    LoungeID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    Amenities TEXT,
    AccessLevel TEXT );
INSERT INTO "VIP_Lounges" VALUES(1,'First Class Lounge','SFO Terminal G','Complimentary Wi-Fi','Invitation Only');
INSERT INTO "VIP_Lounges" VALUES(2,'Platinum Lounge','LAX Tom Bradley Intl','No Complimentary Wi-Fi','By Invitation/Membership');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airlines',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('Passengers',5);
INSERT INTO "sqlite_sequence" VALUES('Bookings',5);
INSERT INTO "sqlite_sequence" VALUES('Frequent_Flyer_Lounges',2);
INSERT INTO "sqlite_sequence" VALUES('VIP_Lounges',2);
COMMIT;
