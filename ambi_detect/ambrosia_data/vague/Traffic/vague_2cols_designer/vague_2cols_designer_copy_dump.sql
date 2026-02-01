BEGIN TRANSACTION;
CREATE TABLE Accidents(
    AccidentID INTEGER PRIMARY KEY AUTOINCREMENT,
    DateOfAccident DATE,
    TimeOfAccident TIME,
    LocationText TEXT,
    Latitude REAL,
    Longitude REAL,
    Severity TEXT);
INSERT INTO "Accidents" VALUES(1,'2021-08-01','10:30 AM','Main St & Oak Ave',40.0,-89.0,'Minor');
INSERT INTO "Accidents" VALUES(2,'2021-08-02','11:15 AM','Elm St & Pine Ave',41.0,-90.0,'Moderate');
CREATE TABLE "CivilEngineers" (
    ProjectID INTEGER PRIMARY KEY AUTOINCREMENT,
    Intersection TEXT,
    Designer TEXT,
    StartDate DATE,
    EndDate DATE,
    Budget REAL);
INSERT INTO "CivilEngineers" VALUES(1,'Main St & Oak Ave','John Doe','2021-06-01','2021-09-30',500000.0);
INSERT INTO "CivilEngineers" VALUES(2,'Elm St & Pine Ave','Jane Smith','2021-07-01','2021-10-31',600000.0);
CREATE TABLE Drivers(
    DriverID INTEGER PRIMARY KEY AUTOINCREMENT,
    LicenseNumber TEXT,
    Name TEXT,
    Address TEXT,
    PhoneNumber TEXT);
INSERT INTO "Drivers" VALUES(1,'A1234CD','Alice Johnson','123 Main St, Springfield, IL','555-123-4567');
INSERT INTO "Drivers" VALUES(2,'BCDEFGH','Bob Smith','456 Elm St, Chicago, IL','555-987-6543');
CREATE TABLE PoliceReports(
    ReportID INTEGER PRIMARY KEY AUTOINCREMENT,
    DateReported DATE,
    OfficerID INT,
    AccidentID INT,
    OffenseType TEXT,
    Penalty TEXT);
INSERT INTO "PoliceReports" VALUES(1,'2021-08-01',1,1,'Speeding','Warning');
INSERT INTO "PoliceReports" VALUES(2,'2021-08-02',2,2,'Running Red Light','$100 Fine');
CREATE TABLE Roads(
    RoadID INTEGER PRIMARY KEY AUTOINCREMENT,
    StreetName TEXT,
    City TEXT,
    State TEXT,
    ZipCode INT);
INSERT INTO "Roads" VALUES(1,'Bridge St','Springfield','IL',62704);
INSERT INTO "Roads" VALUES(2,'Parkway Dr','Chicago','IL',60616);
CREATE TABLE VehicleTypes(
    TypeID INTEGER PRIMARY KEY AUTOINCREMENT,
    Make TEXT,
    Model TEXT,
    Year INT);
INSERT INTO "VehicleTypes" VALUES(1,'Toyota','Corolla',2018);
INSERT INTO "VehicleTypes" VALUES(2,'Honda','Civic',2019);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Roads',2);
INSERT INTO "sqlite_sequence" VALUES('Accidents',2);
INSERT INTO "sqlite_sequence" VALUES('VehicleTypes',2);
INSERT INTO "sqlite_sequence" VALUES('Drivers',2);
INSERT INTO "sqlite_sequence" VALUES('PoliceReports',2);
INSERT INTO "sqlite_sequence" VALUES('CivilEngineers',2);
COMMIT;
