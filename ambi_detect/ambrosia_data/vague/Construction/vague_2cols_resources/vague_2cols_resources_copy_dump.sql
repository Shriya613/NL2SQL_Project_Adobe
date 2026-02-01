BEGIN TRANSACTION;
CREATE TABLE "Construction_Projects" (
    ProjectID INTEGER PRIMARY KEY AUTOINCREMENT,
    Resources REAL,
    Location TEXT,
    StartDate DATE,
    EndDate DATE);
INSERT INTO "Construction_Projects" VALUES(1,1000000.0,'New York','2023-01-01','2023-12-31');
INSERT INTO "Construction_Projects" VALUES(2,800000.0,'Los Angeles','2023-02-01','2023-11-30');
CREATE TABLE Contractors (
    ContractorID INTEGER PRIMARY KEY AUTOINCREMENT,
    CompanyName TEXT,
    ServicesOffered TEXT,
    LicenseNumber TEXT);
INSERT INTO "Contractors" VALUES(1,'PQR Construction Co.','General Contracting','Lic#123456');
INSERT INTO "Contractors" VALUES(2,'STU Building Solutions','Subcontracting','Lic#654321');
CREATE TABLE Employees (
    EmployeeID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Position TEXT,
    HireDate DATE);
INSERT INTO "Employees" VALUES(1,'John Doe','Project Manager','2022-06-01');
INSERT INTO "Employees" VALUES(2,'Jane Smith','Architect','2021-09-01');
CREATE TABLE Equipment (
    EquipmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    ModelName TEXT,
    RentalRate REAL,
    PurchasePrice REAL);
INSERT INTO "Equipment" VALUES(1,'Excavator X500',250.0,500000.0);
INSERT INTO "Equipment" VALUES(2,'Cranes MX700',300.0,600000.0);
CREATE TABLE Materials (
    MaterialID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT,
    Quantity REAL,
    UnitCost REAL);
INSERT INTO "Materials" VALUES(1,'Concrete',500.0,75.0);
INSERT INTO "Materials" VALUES(2,'Steel Rebars',300.0,100.0);
CREATE TABLE Suppliers (
    SupplierID INTEGER PRIMARY KEY AUTOINCREMENT,
    CompanyName TEXT,
    ContactInfo TEXT);
INSERT INTO "Suppliers" VALUES(1,'ABC Concrete Supply','(555) 123-4567');
INSERT INTO "Suppliers" VALUES(2,'XYZ Steel Distribution','(555) 765-4321');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Employees',2);
INSERT INTO "sqlite_sequence" VALUES('Materials',2);
INSERT INTO "sqlite_sequence" VALUES('Equipment',2);
INSERT INTO "sqlite_sequence" VALUES('Suppliers',2);
INSERT INTO "sqlite_sequence" VALUES('Contractors',2);
INSERT INTO "sqlite_sequence" VALUES('Construction_Projects',2);
COMMIT;
