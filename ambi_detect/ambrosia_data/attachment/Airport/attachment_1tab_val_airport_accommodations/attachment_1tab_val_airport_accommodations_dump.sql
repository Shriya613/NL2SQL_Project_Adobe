BEGIN TRANSACTION;
CREATE TABLE Airlines (
    AirlineID INTEGER PRIMARY KEY,
    Name VARCHAR(50),
    Headquarters VARCHAR(50)
);
INSERT INTO "Airlines" VALUES(1,'United Airlines','Chicago');
INSERT INTO "Airlines" VALUES(2,'Delta Airlines','Atlanta');
INSERT INTO "Airlines" VALUES(3,'Southwest Airlines','Dallas');
CREATE TABLE AssignPilotToFlight (
    FlightID INTEGER,
    PilotID INTEGER,
    FOREIGN KEY (FlightID) REFERENCES Flights(FlightID),
    FOREIGN KEY (PilotID) REFERENCES Pilots(PilotID),
    PRIMARY KEY (FlightID, PilotID)
);
INSERT INTO "AssignPilotToFlight" VALUES(1,1);
INSERT INTO "AssignPilotToFlight" VALUES(2,2);
INSERT INTO "AssignPilotToFlight" VALUES(3,3);
CREATE TABLE Flights (
    FlightID INTEGER PRIMARY KEY,
    DepartureTime DATETIME,
    ArrivalTime DATETIME,
    Origin VARCHAR(2),
    Destination VARCHAR(2),
    AircraftType VARCHAR(10),
    SeatsAvailable INT
);
INSERT INTO "Flights" VALUES(1,'2023-07-01 10:00','2023-07-01 14:00','LA','SF','Boeing 737',100);
INSERT INTO "Flights" VALUES(2,'2023-07-01 11:00','2023-07-01 15:00','SF','LA','Airbus A320',90);
INSERT INTO "Flights" VALUES(3,'2023-07-02 08:00','2023-07-02 12:00','NY','MI','Embraer E175',100);
CREATE TABLE Hotels (
    HotelID INTEGER PRIMARY KEY,
    Name VARCHAR(50),
    Location VARCHAR(50),
    Price DECIMAL(8, 2),
    AccomodationType VARCHAR(20) 
);
INSERT INTO "Hotels" VALUES(1,'Airport Hotel','Inside the airport premises',150,'Hotel');
INSERT INTO "Hotels" VALUES(2,'Sleep Pod','Outside the airport near terminal',150,'Pod');
INSERT INTO "Hotels" VALUES(3,'Another Airport Hotel','Inside the airport premises',200,'Hotel');
INSERT INTO "Hotels" VALUES(4,'Another Sleep Pod','Outside the airport near terminal',200,'Pod');
INSERT INTO "Hotels" VALUES(5,'Grand Plaza','Close to the city center',250,'Resort');
INSERT INTO "Hotels" VALUES(6,'Economy Inn','Far from the city center',50,'Motel');
CREATE TABLE Pilots (
    PilotID INTEGER PRIMARY KEY,
    FirstName VARCHAR(20),
    LastName VARCHAR(20),
    LicenseNumber VARCHAR(15),
    DateOfBirth DATE
);
INSERT INTO "Pilots" VALUES(1,'John','Doe','ABCD1234','1980-06-01');
INSERT INTO "Pilots" VALUES(2,'Jane','Smith','EFGH5678','1990-05-01');
INSERT INTO "Pilots" VALUES(3,'Alice','Wonderland','XYZL9876','1985-08-01');
COMMIT;
