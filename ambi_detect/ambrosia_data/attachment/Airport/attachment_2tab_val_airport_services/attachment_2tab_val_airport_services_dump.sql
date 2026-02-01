BEGIN TRANSACTION;
CREATE TABLE Aircraft(
    Registration TEXT PRIMARY KEY,
    Model TEXT,
    Manufacturer TEXT,
    YearOfManufacture INT,
    SeatCapacity INT
);
INSERT INTO "Aircraft" VALUES('N123AB','Boeing 737-800','Boeing',2010,160);
INSERT INTO "Aircraft" VALUES('N345CD','Airbus A320','Airbus',2012,180);
CREATE TABLE Customer_Complaints(
    ComplaintID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerName TEXT,
    FlightNumber TEXT,
    Issue TEXT,
    Resolution TEXT,
    Date DATE
);
INSERT INTO "Customer_Complaints" VALUES(1,'John Doe','AA123','Late boarding','Apologized and provided meal voucher','January 5, 2024');
INSERT INTO "Customer_Complaints" VALUES(2,'Jane Smith','DL456','Luggage mishandled','Delivered luggage within two hours','December 30, 2023');
CREATE TABLE Employees(
    EmployeeID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    JobTitle TEXT,
    HireDate DATE
);
INSERT INTO "Employees" VALUES(1,'Alice','Smith','Security Officer','October 1, 2022');
INSERT INTO "Employees" VALUES(2,'Bob','Johnson','Pilot','September 1, 2022');
CREATE TABLE Flights(
    FlightNumber TEXT PRIMARY KEY,
    Airline TEXT,
    DepartureTime TIME,
    ArrivalTime TIME,
    OriginAirport TEXT,
    DestinationAirport TEXT
);
INSERT INTO "Flights" VALUES('AA123','American Airlines','08:00 AM','10:00 AM','LAX','ORD');
INSERT INTO "Flights" VALUES('DL456','Delta Airlines','09:00 AM','11:00 AM','ATL','MCO');
CREATE TABLE Lost_Property(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Item TEXT,
    FoundAt TEXT,
    Description TEXT,
    Date DATE
, Owner TEXT);
INSERT INTO "Lost_Property" VALUES(1,'Wristwatch','Gate A1','Silver wristwatch with black leather strap','January 5, 2024','John Brown');
INSERT INTO "Lost_Property" VALUES(2,'Mobile Phone','Food Court','iPhone X with a cracked screen','December 31, 2023','Max Black');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Lost_Property',2);
INSERT INTO "sqlite_sequence" VALUES('Customer_Complaints',2);
INSERT INTO "sqlite_sequence" VALUES('Employees',2);
COMMIT;
