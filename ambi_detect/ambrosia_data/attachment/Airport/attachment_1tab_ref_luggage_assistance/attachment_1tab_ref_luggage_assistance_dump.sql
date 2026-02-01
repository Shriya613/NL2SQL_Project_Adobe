BEGIN TRANSACTION;
CREATE TABLE Bookings(
    BookingID INTEGER PRIMARY KEY AUTOINCREMENT,
    LuggageRequestID INTEGER,
    LocationID INTEGER,
    SlotStart TIME STAMP,
    SlotEnd TIME STAMP,
    BookingStatus TEXT  ,
    FOREIGN KEY (LuggageRequestID) REFERENCES LuggageRequests(RequestID),
    FOREIGN KEY (LocationID) REFERENCES Locations(LocationID)
);
INSERT INTO "Bookings" VALUES(1,1,1,'2023-01-01 08:30:00','2023-01-01 09:00:00','Confirmed');
INSERT INTO "Bookings" VALUES(2,2,2,'2023-01-01 08:30:00','2023-01-01 09:00:00','Confirmed');
INSERT INTO "Bookings" VALUES(3,3,1,'2023-01-02 08:30:00','2023-01-02 09:00:00','Pending');
INSERT INTO "Bookings" VALUES(4,4,2,'2023-01-02 08:30:00','2023-01-02 09:00:00','Pending');
CREATE TABLE Flights(
    FlightID INTEGER PRIMARY KEY AUTOINCREMENT,
    DepartureTime DATETIME,
    Destination TEXT,
    AvailableSeats INTEGER
);
INSERT INTO "Flights" VALUES(1,'2023-01-01 10:00:00','New York',100);
INSERT INTO "Flights" VALUES(2,'2023-01-02 10:00:00','Los Angeles',80);
INSERT INTO "Flights" VALUES(3,'2023-01-03 10:00:00','Chicago',70);
INSERT INTO "Flights" VALUES(4,'2023-01-04 10:00:00','Dallas',60);
INSERT INTO "Flights" VALUES(5,'2023-01-05 10:00:00','Houston',50);
CREATE TABLE Locations(
    LocationID INTEGER PRIMARY KEY AUTOINCREMENT,
    LocationName TEXT UNIQUE
);
INSERT INTO "Locations" VALUES(1,'Entrance B');
INSERT INTO "Locations" VALUES(2,'Exit A');
CREATE TABLE LuggageRequests(
    RequestID INTEGER PRIMARY KEY AUTOINCREMENT,
    PassengerID INTEGER,
    AssistanceType TEXT ,
    Location VARCHAR(10),
    RequestDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (PassengerID) REFERENCES Passengers(PassengerID)
);
INSERT INTO "LuggageRequests" VALUES(1,1,'Luggage Trolley','Entrance B','2023-01-01 09:00:00');
INSERT INTO "LuggageRequests" VALUES(2,1,'Luggage Trolley','Exit A','2023-01-01 09:00:00');
INSERT INTO "LuggageRequests" VALUES(3,2,'Porter Service','Entrance B','2023-01-02 09:00:00');
INSERT INTO "LuggageRequests" VALUES(4,2,'Porter Service','Exit A','2023-01-02 09:00:00');
CREATE TABLE Passengers(
    PassengerID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    FlightID INTEGER
);
INSERT INTO "Passengers" VALUES(1,'John Doe',1);
INSERT INTO "Passengers" VALUES(2,'Jane Smith',2);
INSERT INTO "Passengers" VALUES(3,'Mike Johnson',3);
INSERT INTO "Passengers" VALUES(4,'Emily Davis',4);
INSERT INTO "Passengers" VALUES(5,'David Brown',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Passengers',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('LuggageRequests',4);
INSERT INTO "sqlite_sequence" VALUES('Locations',2);
INSERT INTO "sqlite_sequence" VALUES('Bookings',4);
COMMIT;
