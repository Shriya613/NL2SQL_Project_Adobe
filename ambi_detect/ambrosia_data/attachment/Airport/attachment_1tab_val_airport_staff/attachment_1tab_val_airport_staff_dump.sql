BEGIN TRANSACTION;
CREATE TABLE Airports(
    AirportID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    City TEXT,
    Country TEXT);
INSERT INTO "Airports" VALUES(1,'JFK','New York','USA');
INSERT INTO "Airports" VALUES(2,'LAX','Los Angeles','USA');
INSERT INTO "Airports" VALUES(3,'MAD','Madrid','Spain');
INSERT INTO "Airports" VALUES(4,'CDG','Paris','France');
INSERT INTO "Airports" VALUES(5,'FRA','Frankfurt','Germany');
CREATE TABLE Flights(
    FlightID INTEGER PRIMARY KEY AUTOINCREMENT,
    DepartureTime DATETIME,
    ArrivalTime DATETIME,
    OriginAirportID INTEGER,
    DestinationAirportID INTEGER,
    AssignedStaffID INTEGER,
    FOREIGN KEY (OriginAirportID) REFERENCES Airports(AirportID),
    FOREIGN KEY (DestinationAirportID) REFERENCES Airports(AirportID),
    FOREIGN KEY (AssignedStaffID) REFERENCES Staff(StaffID));
INSERT INTO "Flights" VALUES(1,'2023-01-01 08:00','2023-01-01 10:00',1,2,1);
INSERT INTO "Flights" VALUES(2,'2023-01-01 16:00','2023-01-01 18:00',2,1,2);
INSERT INTO "Flights" VALUES(3,'2023-01-02 09:00','2023-01-02 11:00',3,4,3);
INSERT INTO "Flights" VALUES(4,'2023-01-02 17:00','2023-01-02 19:00',4,3,4);
INSERT INTO "Flights" VALUES(5,'2023-01-03 08:00','2023-01-03 10:00',5,2,5);
CREATE TABLE Languages(
    LanguageID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT);
INSERT INTO "Languages" VALUES(1,'English');
INSERT INTO "Languages" VALUES(2,'Spanish');
INSERT INTO "Languages" VALUES(3,'French');
INSERT INTO "Languages" VALUES(4,'German');
INSERT INTO "Languages" VALUES(5,'Italian');
CREATE TABLE Staff(
    StaffID INTEGER PRIMARY KEY AUTOINCREMENT,
    Position VARCHAR(50) ,
    AirportID INTEGER,
    LanguageID INTEGER, FullName TEXT,
    FOREIGN KEY (AirportID) REFERENCES Airports(AirportID),
    FOREIGN KEY (LanguageID) REFERENCES Languages(LanguageID));
INSERT INTO "Staff" VALUES(1,'Gate Agent',1,1,'John Doe');
INSERT INTO "Staff" VALUES(2,'Pilot',2,3,'Jane Smith');
INSERT INTO "Staff" VALUES(3,'Gate Agent',3,2,'Emily Brown');
INSERT INTO "Staff" VALUES(4,'Pilot',4,1,'David White');
INSERT INTO "Staff" VALUES(5,'Gate Agent',5,4,'Alice Black');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Languages',5);
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Staff',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
COMMIT;
