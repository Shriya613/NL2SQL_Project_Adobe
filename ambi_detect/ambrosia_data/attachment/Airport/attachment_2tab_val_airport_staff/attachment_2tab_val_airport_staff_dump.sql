BEGIN TRANSACTION;
CREATE TABLE Airlines(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Code TEXT,
    Headquarters TEXT
);
INSERT INTO "Airlines" VALUES(1,'American Airlines','AA','Fort Worth, Texas');
INSERT INTO "Airlines" VALUES(2,'United Airlines','UA','Chicago, Illinois');
INSERT INTO "Airlines" VALUES(3,'Delta Airlines','DL','Atlanta, Georgia');
INSERT INTO "Airlines" VALUES(4,'Southwest Airlines','WN','Dallas, Texas');
INSERT INTO "Airlines" VALUES(5,'Spirit Airlines','NK','Miramar, Florida');
CREATE TABLE Baggage(
    ID INTEGER PRIMARY KEY,
    FlightID INTEGER REFERENCES Flights(ID),
    OwnerID INTEGER REFERENCES Passengers(ID),
    Weight INT,
    Description TEXT
);
INSERT INTO "Baggage" VALUES(1,1,1,10,'Laptop bag');
INSERT INTO "Baggage" VALUES(2,1,2,15,'Skis case');
INSERT INTO "Baggage" VALUES(3,2,3,20,'Golf clubs');
INSERT INTO "Baggage" VALUES(4,2,4,25,'Musical instruments');
INSERT INTO "Baggage" VALUES(5,3,5,30,'Stroller');
CREATE TABLE Flights(
    ID INTEGER PRIMARY KEY,
    AirlineID INTEGER REFERENCES Airlines(ID),
    DepartureTime TIMESTAMP,
    ArrivalTime TIMESTAMP,
    DestinationAirport TEXT
);
INSERT INTO "Flights" VALUES(1,1,'2023-04-01 10:00:00','2023-04-01 12:00:00','New York JFK');
INSERT INTO "Flights" VALUES(2,2,'2023-04-01 11:00:00','2023-04-01 13:00:00','Los Angeles LAX');
INSERT INTO "Flights" VALUES(3,3,'2023-04-01 12:00:00','2023-04-01 14:00:00','Atlanta Hartsfield-Jackson');
INSERT INTO "Flights" VALUES(4,4,'2023-04-01 13:00:00','2023-04-01 15:00:00','Dallas Love Field');
INSERT INTO "Flights" VALUES(5,5,'2023-04-01 14:00:00','2023-04-01 16:00:00','Fort Lauderdale Hollywood');
CREATE TABLE Gate_Agents(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Language TEXT,
    YearsOfExperience INT,
    LicenseNumber TEXT UNIQUE
);
INSERT INTO "Gate_Agents" VALUES(1,'Juan','Spanish',8,'A0923');
INSERT INTO "Gate_Agents" VALUES(2,'Emily','English',7,'B1263');
INSERT INTO "Gate_Agents" VALUES(3,'Michael','French',6,'C3456');
INSERT INTO "Gate_Agents" VALUES(4,'Olivia','German',5,'D5678');
INSERT INTO "Gate_Agents" VALUES(5,'Liam','Italian',4,'E7890');
CREATE TABLE Passengers(
    ID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Age INT,
    Nationality TEXT
);
INSERT INTO "Passengers" VALUES(1,'John','Smith',30,'USA');
INSERT INTO "Passengers" VALUES(2,'Jane','Doe',25,'UK');
INSERT INTO "Passengers" VALUES(3,'Tom','Brown',40,'Canada');
INSERT INTO "Passengers" VALUES(4,'Anna','White',35,'Australia');
INSERT INTO "Passengers" VALUES(5,'Paul','Green',45,'France');
CREATE TABLE Pilots(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Language TEXT,
    FlightHours INT,
    LicenseNumber TEXT UNIQUE
);
INSERT INTO "Pilots" VALUES(1,'Alberto','Spanish',2000,'F0011');
INSERT INTO "Pilots" VALUES(2,'David','English',1800,'G0022');
INSERT INTO "Pilots" VALUES(3,'Isabelle','French',1600,'H0033');
INSERT INTO "Pilots" VALUES(4,'Leo','German',1400,'I0044');
INSERT INTO "Pilots" VALUES(5,'Milan','Italian',1200,'J0055');
COMMIT;
