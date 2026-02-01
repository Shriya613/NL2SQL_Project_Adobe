BEGIN TRANSACTION;
CREATE TABLE Airlines(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Code TEXT
);
INSERT INTO "Airlines" VALUES(1,'United Airlines','UA');
INSERT INTO "Airlines" VALUES(2,'Delta Airlines','DL');
INSERT INTO "Airlines" VALUES(3,'Southwest Airlines','WN');
INSERT INTO "Airlines" VALUES(4,'JetBlue Airways','B6');
INSERT INTO "Airlines" VALUES(5,'Spirit Airlines','NK');
CREATE TABLE Airports(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    City TEXT,
    Country TEXT
);
INSERT INTO "Airports" VALUES(1,'Chicago O''Hare International Airport','Chicago','USA');
INSERT INTO "Airports" VALUES(2,'Dallas/Fort Worth International Airport','DFW','USA');
INSERT INTO "Airports" VALUES(3,'Los Angeles International Airport','LAX','USA');
INSERT INTO "Airports" VALUES(4,'New York John F. Kennedy International Airport','NYC','USA');
INSERT INTO "Airports" VALUES(5,'London Heathrow Airport','London','UK');
CREATE TABLE Benefits(
    BenefitId INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT
);
INSERT INTO "Benefits" VALUES(1,'Elite Status');
INSERT INTO "Benefits" VALUES(2,'Priority Boarding');
INSERT INTO "Benefits" VALUES(3,'Free Baggage Check');
INSERT INTO "Benefits" VALUES(4,'Lounge Access');
INSERT INTO "Benefits" VALUES(5,'Preferred Seating');
CREATE TABLE Flights(
    FlightID INTEGER PRIMARY KEY AUTOINCREMENT,
    DepartureTime DATETIME,
    ArrivalTime DATETIME,
    AirlineId INTEGER REFERENCES Airlines(ID) ON DELETE CASCADE,
    FromAirportId INTEGER REFERENCES Airports(ID) ON DELETE CASCADE,
    ToAirportId INTEGER REFERENCES Airports(ID) ON DELETE CASCADE
);
CREATE TABLE Frequent_Flyer_Program(
    ProgramId INTEGER PRIMARY KEY AUTOINCREMENT,
    AirlineId INTEGER REFERENCES Airlines(ID) ON DELETE SET NULL,
    MemberName TEXT,
    JoinDate DATE
);
INSERT INTO "Frequent_Flyer_Program" VALUES(1,1,'John Doe','2022-01-01');
INSERT INTO "Frequent_Flyer_Program" VALUES(2,2,'Jane Smith','2022-07-01');
INSERT INTO "Frequent_Flyer_Program" VALUES(3,3,'Mike Johnson','2022-10-01');
INSERT INTO "Frequent_Flyer_Program" VALUES(4,4,'Emily Brown','2022-11-01');
INSERT INTO "Frequent_Flyer_Program" VALUES(5,5,'David Wilson','2022-12-01');
CREATE TABLE "Frequent_Flyer_Program_Benefits"(
    ProgramId INTEGER REFERENCES Frequent_Flyer_Program(ProgramId) ON DELETE CASCADE,
    BenefitId INTEGER REFERENCES Benefits(BenefitId) ON DELETE CASCADE,
    PRIMARY KEY (ProgramId, BenefitId)
);
INSERT INTO "Frequent_Flyer_Program_Benefits" VALUES(1,1);
INSERT INTO "Frequent_Flyer_Program_Benefits" VALUES(2,1);
INSERT INTO "Frequent_Flyer_Program_Benefits" VALUES(3,1);
INSERT INTO "Frequent_Flyer_Program_Benefits" VALUES(4,1);
INSERT INTO "Frequent_Flyer_Program_Benefits" VALUES(5,1);
INSERT INTO "Frequent_Flyer_Program_Benefits" VALUES(5,2);
INSERT INTO "Frequent_Flyer_Program_Benefits" VALUES(3,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Frequent_Flyer_Program',5);
INSERT INTO "sqlite_sequence" VALUES('Benefits',5);
COMMIT;
