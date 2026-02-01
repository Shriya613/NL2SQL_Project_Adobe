BEGIN TRANSACTION;
CREATE TABLE Accidents (
    AccidentID INTEGER PRIMARY KEY AUTOINCREMENT,
    Date DATETIME,
    LocationID INTEGER,
    Severity VARCHAR(50),
    FOREIGN KEY (LocationID) REFERENCES Streets (StreetID)
);
INSERT INTO "Accidents" VALUES(1,'2022-01-01 10:00:00',1,'Minor');
INSERT INTO "Accidents" VALUES(2,'2022-02-01 14:30:00',2,'Moderate');
INSERT INTO "Accidents" VALUES(3,'2022-03-01 19:00:00',3,'Major');
INSERT INTO "Accidents" VALUES(4,'2022-04-01 23:30:00',4,'Catastrophic');
INSERT INTO "Accidents" VALUES(5,'2022-05-01 01:00:00',5,'Fatal');
CREATE TABLE RoadWorks (
    WorkID INTEGER PRIMARY KEY AUTOINCREMENT,
    StartDate DATETIME,
    EndDate DATETIME,
    StreetID INTEGER,
    WorkType VARCHAR(50),
    FOREIGN KEY (StreetID) REFERENCES Streets (StreetID)
);
INSERT INTO "RoadWorks" VALUES(1,'2022-06-01 08:00:00','2022-06-02 18:00:00',1,'Repair');
INSERT INTO "RoadWorks" VALUES(2,'2022-07-01 08:00:00','2022-07-02 18:00:00',2,'Resurfacing');
INSERT INTO "RoadWorks" VALUES(3,'2022-08-01 08:00:00','2022-08-02 18:00:00',3,'Construction');
INSERT INTO "RoadWorks" VALUES(4,'2022-09-01 08:00:00','2022-09-02 18:00:00',4,'Drainage');
INSERT INTO "RoadWorks" VALUES(5,'2022-10-01 08:00:00','2022-10-02 18:00:00',5,'Landscaping');
CREATE TABLE SpeedingTickets (
    TicketID INTEGER PRIMARY KEY AUTOINCREMENT,
    IssueDate DATETIME,
    OffenderVehicleID INTEGER,
    FineAmount DECIMAL(8,2),
    FOREIGN KEY (OffenderVehicleID) REFERENCES Vehicles (VehicleID)
);
INSERT INTO "SpeedingTickets" VALUES(1,'2022-01-15 12:34:56',1,50);
INSERT INTO "SpeedingTickets" VALUES(2,'2022-02-15 12:34:56',2,100);
INSERT INTO "SpeedingTickets" VALUES(3,'2022-03-15 12:34:56',3,75);
INSERT INTO "SpeedingTickets" VALUES(4,'2022-04-15 12:34:56',4,125);
INSERT INTO "SpeedingTickets" VALUES(5,'2022-05-15 12:34:56',5,150);
CREATE TABLE Streets (
    StreetID INTEGER PRIMARY KEY AUTOINCREMENT,
    Street_Name TEXT,
    Block_Number INTEGER,
    Bus_Stop BOOLEAN DEFAULT FALSE,
    NumLanes INTEGER,
    MaxSpeed MPH
);
INSERT INTO "Streets" VALUES(1,'Main St',123,1,2,30);
INSERT INTO "Streets" VALUES(2,'Elm St',456,0,1,25);
INSERT INTO "Streets" VALUES(3,'Pine Ave',789,1,2,30);
INSERT INTO "Streets" VALUES(4,'Maple Dr',101,0,2,25);
INSERT INTO "Streets" VALUES(5,'Oak Blvd',123,1,4,40);
CREATE TABLE Users (
    UserID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Email VARCHAR(100),
    PhoneNumber VARCHAR(15)
);
INSERT INTO "Users" VALUES(1,'John','Doe','john.doe@example.com','+1234567890');
INSERT INTO "Users" VALUES(2,'Jane','Smith','jane.smith@example.com','+0987654321');
INSERT INTO "Users" VALUES(3,'Alice','Brown','alice.brown@example.com','+1123456789');
INSERT INTO "Users" VALUES(4,'Bob','White','bob.white@example.com','+0987654321');
INSERT INTO "Users" VALUES(5,'Charlie','Black','charlie.black@example.com','+1123456789');
CREATE TABLE Vehicles (
    VehicleID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type VARCHAR(20),
    LicensePlate CHAR(7),
    OwnerID INTEGER,
    FOREIGN KEY (OwnerID) REFERENCES Users (UserID)
);
INSERT INTO "Vehicles" VALUES(1,'Car','ABC123',1);
INSERT INTO "Vehicles" VALUES(2,'Truck','DEF456',2);
INSERT INTO "Vehicles" VALUES(3,'SUV','GHI789',3);
INSERT INTO "Vehicles" VALUES(4,'Motorcycle','JKL012',4);
INSERT INTO "Vehicles" VALUES(5,'Van','LMN321',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Streets',5);
INSERT INTO "sqlite_sequence" VALUES('Accidents',5);
INSERT INTO "sqlite_sequence" VALUES('Vehicles',5);
INSERT INTO "sqlite_sequence" VALUES('Users',5);
INSERT INTO "sqlite_sequence" VALUES('RoadWorks',5);
INSERT INTO "sqlite_sequence" VALUES('SpeedingTickets',5);
COMMIT;
