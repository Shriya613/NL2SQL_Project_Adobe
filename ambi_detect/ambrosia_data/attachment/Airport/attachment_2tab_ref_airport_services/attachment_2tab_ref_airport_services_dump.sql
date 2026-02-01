BEGIN TRANSACTION;
CREATE TABLE "Calendar"(
    CalendarID INTEGER PRIMARY KEY AUTOINCREMENT,
    Year INTEGER,
    Month INTEGER,
    Day INTEGER,
    Date TEXT);
INSERT INTO "Calendar" VALUES(1,2023,1,5,'2023-01-05');
INSERT INTO "Calendar" VALUES(2,2023,1,6,'2023-01-06');
INSERT INTO "Calendar" VALUES(3,2023,1,7,'2023-01-07');
INSERT INTO "Calendar" VALUES(4,2023,1,8,'2023-01-08');
INSERT INTO "Calendar" VALUES(5,2023,1,9,'2023-01-09');
CREATE TABLE Complaints(
    ComplaintID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT,
    Severity INT , -- 1=Low; 2=Medium; 3=High
    Resolved BOOLEAN DEFAULT FALSE,
    SubmissionDate DATE,
    CustomerId INTEGER,
    FlightNumber INTEGER,
    FOREIGN KEY(SubmissionDate) REFERENCES "Calendar"(Date),
    FOREIGN KEY(CustomerId) REFERENCES Customers(CustomerID),
    FOREIGN KEY(FlightNumber) REFERENCES Flights(FlightNumber));
INSERT INTO "Complaints" VALUES(1,'Delayed baggage',2,0,'2023-01-05',1,1);
INSERT INTO "Complaints" VALUES(2,'Rude staff',3,1,'2023-01-06',2,2);
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    EmailAddress TEXT UNIQUE);
INSERT INTO "Customers" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Customers" VALUES(3,'Alice','Brown','alice.brown@example.com');
INSERT INTO "Customers" VALUES(4,'Charlie','White','charlie.white@example.com');
INSERT INTO "Customers" VALUES(5,'David','Black','david.black@example.com');
CREATE TABLE Flights(
    FlightNumber INTEGER PRIMARY KEY AUTOINCREMENT,
    DepartureAirport TEXT,
    ArrivalAirport TEXT,
    DepartureTime TIMESTAMP,
    EstimatedArrivalTime TIMESTAMP);
INSERT INTO "Flights" VALUES(1,'LAX','ORD','2023-01-01 10:00:00','2023-01-01 18:00:00');
INSERT INTO "Flights" VALUES(2,'DFW','ATL','2023-01-01 16:00:00','2023-01-02 09:00:00');
INSERT INTO "Flights" VALUES(3,'EWR','MIA','2023-01-01 17:00:00','2023-01-03 08:00:00');
INSERT INTO "Flights" VALUES(4,'IAH','YYZ','2023-01-01 20:00:00','2023-01-04 15:00:00');
INSERT INTO "Flights" VALUES(5,'MSP','SEA','2023-01-01 22:00:00','2023-01-05 14:00:00');
CREATE TABLE LostProperties(
    PropertyID INTEGER PRIMARY KEY AUTOINCREMENT,
    ItemDescription TEXT,
    DateOfLoss DATE,
    Found BOOLEAN DEFAULT FALSE,
    FOREIGN KEY(DateOfLoss) REFERENCES "Calendar"(Date));
INSERT INTO "LostProperties" VALUES(1,'Phone','2023-01-07',1);
INSERT INTO "LostProperties" VALUES(2,'Keys','2023-01-08',0);
INSERT INTO "LostProperties" VALUES(3,'Passport','2023-01-09',1);
INSERT INTO "LostProperties" VALUES(4,'Wallet','2023-01-06',0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('LostProperties',5);
INSERT INTO "sqlite_sequence" VALUES('Complaints',5);
INSERT INTO "sqlite_sequence" VALUES('Calendar',5);
COMMIT;
