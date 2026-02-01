BEGIN TRANSACTION;
CREATE TABLE Airports(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    CityId INTEGER REFERENCES Cities(Id),
    Code CHAR(2),
    RunwayCount INTEGER DEFAULT 1,
    FOREIGN KEY(CityId) REFERENCES Cities(Id));
INSERT INTO "Airports" VALUES(1,1,'JFK',3);
INSERT INTO "Airports" VALUES(2,2,'LHR',2);
INSERT INTO "Airports" VALUES(3,3,'CDG',2);
INSERT INTO "Airports" VALUES(4,4,'HND',4);
INSERT INTO "Airports" VALUES(5,5,'DXB',6);
CREATE TABLE Cities(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Country TEXT);
INSERT INTO "Cities" VALUES(1,'New York','USA');
INSERT INTO "Cities" VALUES(2,'London','UK');
INSERT INTO "Cities" VALUES(3,'Paris','France');
INSERT INTO "Cities" VALUES(4,'Tokyo','Japan');
INSERT INTO "Cities" VALUES(5,'Dubai','UAE');
CREATE TABLE Flights(
    FlightId INTEGER PRIMARY KEY AUTOINCREMENT,
    DepartureAirportId INTEGER REFERENCES Airports(Id),
    ArrivalAirportId INTEGER REFERENCES Airports(Id),
    ScheduledDeparture TIMESTAMP,
    ActualDeparture TIMESTAMP,
    EstimatedArrival TIMESTAMP,
    Status TEXT ,
    AircraftModel TEXT);
INSERT INTO "Flights" VALUES(1,1,2,'2023-03-25 12:00:00','2023-03-25 12:15:00','2023-03-25 13:45:00','On Time','Boeing 747');
INSERT INTO "Flights" VALUES(2,2,3,'2023-03-25 15:00:00','2023-03-25 15:30:00','2023-03-25 18:30:00','Delayed','Airbus A380');
INSERT INTO "Flights" VALUES(3,3,4,'2023-03-25 18:00:00','2023-03-25 18:30:00','2023-03-25 22:15:00','On Time','Boeing 787');
INSERT INTO "Flights" VALUES(4,4,5,'2023-03-25 21:00:00','2023-03-25 21:30:00','2023-03-25 23:15:00','Delayed','Boeing 777');
INSERT INTO "Flights" VALUES(5,5,1,'2023-03-26 02:00:00','2023-03-26 02:30:00','2023-03-26 04:15:00','On Time','Emirates A380');
CREATE TABLE Runways(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Length REAL,
    Width REAL,
    SurfaceType TEXT,
    AirportId INTEGER REFERENCES Airports(Id),
    FOREIGN KEY(AirportId) REFERENCES Airports(Id));
INSERT INTO "Runways" VALUES(1,9000.0,150.0,'Asphalt',1);
INSERT INTO "Runways" VALUES(2,8000.0,170.0,'Concrete',2);
INSERT INTO "Runways" VALUES(3,9500.0,200.0,'Grass',3);
INSERT INTO "Runways" VALUES(4,8800.0,180.0,'Asphalt',4);
INSERT INTO "Runways" VALUES(5,12000.0,220.0,'Tarmac',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Cities',5);
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Runways',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
COMMIT;
