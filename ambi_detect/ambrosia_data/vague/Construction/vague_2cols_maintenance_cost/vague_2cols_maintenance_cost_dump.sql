BEGIN TRANSACTION;
CREATE TABLE Equipment(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    EquipName TEXT,
    PurchasePrice REAL,
    RentalRate REAL,
    DailyUsage INT);
INSERT INTO "Equipment" VALUES(1,'Excavator',20000.0,1000.0,8);
INSERT INTO "Equipment" VALUES(2,'Crane',30000.0,1500.0,6);
CREATE TABLE Materials(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    MaterialName TEXT,
    Quantity INT,
    UnitCost REAL);
INSERT INTO "Materials" VALUES(1,'Concrete Blocks',500,5.0);
INSERT INTO "Materials" VALUES(2,'Bricks',300,3.5);
CREATE TABLE Project (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    StartDate DATE,
    EndDate DATE,
    Monetary_Value REAL,
    Percentage_Of_Budget REAL,
    Infrastructure TEXT);
INSERT INTO "Project" VALUES(1,'Project1','2023-01-01','2023-06-01',500000.0,75.0,'Residential Building');
INSERT INTO "Project" VALUES(2,'Project2','2023-02-01','2023-08-01',900000.0,80.0,'Commercial Complex');
CREATE TABLE Worker(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    JobTitle TEXT);
INSERT INTO "Worker" VALUES(1,'John','Doe','Carpenter');
INSERT INTO "Worker" VALUES(2,'Jane','Smith','Electrician');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Project',2);
INSERT INTO "sqlite_sequence" VALUES('Worker',2);
INSERT INTO "sqlite_sequence" VALUES('Materials',2);
INSERT INTO "sqlite_sequence" VALUES('Equipment',2);
COMMIT;
