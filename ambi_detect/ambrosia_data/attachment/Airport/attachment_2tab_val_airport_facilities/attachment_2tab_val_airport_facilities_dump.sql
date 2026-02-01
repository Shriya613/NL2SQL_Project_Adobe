BEGIN TRANSACTION;
CREATE TABLE "BaggageClaims"(
    ClaimID INTEGER PRIMARY KEY AUTOINCREMENT, HandlerName TEXT,
    FlightID INTEGER,
    Location TEXT,
    FOREIGN KEY(FlightID) REFERENCES Flights(FlightID));
INSERT INTO "BaggageClaims" VALUES(1,'Sarah Miller',1,'Terminal 2');
INSERT INTO "BaggageClaims" VALUES(2,'Michael Brown',2,'Terminal 3');
INSERT INTO "BaggageClaims" VALUES(3,'Lisa White',3,'Terminal 1');
INSERT INTO "BaggageClaims" VALUES(4,'James Wilson',4,'Terminal 2');
INSERT INTO "BaggageClaims" VALUES(5,'Patricia Taylor',5,'Terminal 4');
CREATE TABLE Bookings(
    BookingID INTEGER PRIMARY KEY AUTOINCREMENT,
    PassengerID INTEGER,
    FlightID INTEGER,
    Class TEXT ,
    Price DECIMAL(8,2),
    FOREIGN KEY(PassengerID) REFERENCES Passengers(PassengerID),
    FOREIGN KEY(FlightID) REFERENCES Flights(FlightID));
INSERT INTO "Bookings" VALUES(1,1,1,'Economy',250);
INSERT INTO "Bookings" VALUES(2,1,2,'Business',1000);
INSERT INTO "Bookings" VALUES(3,2,3,'Premium Economy',500);
INSERT INTO "Bookings" VALUES(4,3,4,'Economy',250);
INSERT INTO "Bookings" VALUES(5,4,5,'First Class',3000);
CREATE TABLE Flights(
    FlightID INTEGER PRIMARY KEY AUTOINCREMENT,
    Airline TEXT,
    Destination TEXT,
    DepartureTime DATETIME,
    ArrivalTime DATETIME);
INSERT INTO "Flights" VALUES(1,'United Airlines','New York','10:30 AM','12:00 PM');
INSERT INTO "Flights" VALUES(2,'Delta Airlines','Los Angeles','9:00 AM','11:00 AM');
INSERT INTO "Flights" VALUES(3,'American Airlines','Chicago','7:00 AM','9:00 AM');
INSERT INTO "Flights" VALUES(4,'Southwest Airlines','Dallas','6:30 AM','8:00 AM');
INSERT INTO "Flights" VALUES(5,'JetBlue Airlines','Miami','5:00 AM','7:00 AM');
CREATE TABLE Passengers(
    PassengerID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INTEGER,
    Nationality TEXT);
INSERT INTO "Passengers" VALUES(1,'John Doe',30,'USA');
INSERT INTO "Passengers" VALUES(2,'Jane Smith',25,'UK');
INSERT INTO "Passengers" VALUES(3,'Alice Johnson',28,'Canada');
INSERT INTO "Passengers" VALUES(4,'Tom Brown',45,'Australia');
INSERT INTO "Passengers" VALUES(5,'Emily Davis',35,'Germany');
CREATE TABLE "SecurityChecks"(
    CheckpointID INTEGER PRIMARY KEY AUTOINCREMENT, OfficerName TEXT,
    FlightID INTEGER,
    Location TEXT,
    Time DATETIME,
    FOREIGN KEY(FlightID) REFERENCES Flights(FlightID));
INSERT INTO "SecurityChecks" VALUES(1,'David Smith',1,'Terminal 2','8:00 AM');
INSERT INTO "SecurityChecks" VALUES(2,'Maria Garcia',2,'Terminal 3','7:00 AM');
INSERT INTO "SecurityChecks" VALUES(3,'Robert Johnson',3,'Terminal 1','6:00 AM');
INSERT INTO "SecurityChecks" VALUES(4,'Jennifer Lee',4,'Terminal 2','4:00 AM');
INSERT INTO "SecurityChecks" VALUES(5,'William Jones',5,'Terminal 4','3:00 AM');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('Passengers',5);
INSERT INTO "sqlite_sequence" VALUES('Bookings',5);
INSERT INTO "sqlite_sequence" VALUES('BaggageClaims',5);
INSERT INTO "sqlite_sequence" VALUES('SecurityChecks',5);
COMMIT;
