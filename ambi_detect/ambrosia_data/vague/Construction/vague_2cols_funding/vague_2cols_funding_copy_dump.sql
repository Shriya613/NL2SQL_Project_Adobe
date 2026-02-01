BEGIN TRANSACTION;
CREATE TABLE Contractors (
    contractorID INTEGER PRIMARY KEY AUTOINCREMENT,
    companyName TEXT,
    address TEXT,
    phoneNumber CHAR(10),
    employeeCount INTEGER
);
INSERT INTO "Contractors" VALUES(1,'Contractor A','Address A','123-456-7890',10);
INSERT INTO "Contractors" VALUES(2,'Contractor B','Address B','234-567-8901',15);
INSERT INTO "Contractors" VALUES(3,'Contractor C','Address C','345-678-9012',20);
INSERT INTO "Contractors" VALUES(4,'Contractor D','Address D','456-789-0123',25);
INSERT INTO "Contractors" VALUES(5,'Contractor E','Address E','567-890-1234',30);
CREATE TABLE Employees (
    employeeID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    hireDate DATE,
    salary REAL,
    contractorID INTEGER,
    FOREIGN KEY(contractorID) REFERENCES Contractors(contractorID)
);
INSERT INTO "Employees" VALUES(1,'John','Doe','2022-01-01',50000.0,1);
INSERT INTO "Employees" VALUES(2,'Jane','Smith','2022-01-02',55000.0,1);
INSERT INTO "Employees" VALUES(3,'Mike','Brown','2022-01-03',60000.0,2);
INSERT INTO "Employees" VALUES(4,'Emily','Johnson','2022-01-04',65000.0,2);
INSERT INTO "Employees" VALUES(5,'Tom','Wilson','2022-01-05',70000.0,3);
CREATE TABLE "Funding" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Funding TEXT,
    publicPark TEXT,
    fundsAmount REAL,
    projectID INTEGER,
    FOREIGN KEY(projectID) REFERENCES Projects(projectID)
);
INSERT INTO "Funding" VALUES(1,'City Hall','Yes',50000.0,1);
INSERT INTO "Funding" VALUES(2,'State Department','No',75000.0,1);
INSERT INTO "Funding" VALUES(3,'National Park Service','Yes',100000.0,2);
INSERT INTO "Funding" VALUES(4,'Federal Grant Program','Yes',125000.0,2);
INSERT INTO "Funding" VALUES(5,'International Organization','Yes',150000.0,3);
CREATE TABLE Materials (
    materialID INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT,
    quantity INTEGER,
    unitCost REAL,
    projectID INTEGER,
    FOREIGN KEY(projectID) REFERENCES Projects(projectID)
);
INSERT INTO "Materials" VALUES(1,'Concrete',100,50.0,1);
INSERT INTO "Materials" VALUES(2,'Steel',50,75.0,1);
INSERT INTO "Materials" VALUES(3,'Glass',20,25.0,2);
INSERT INTO "Materials" VALUES(4,'Bricks',300,15.0,2);
INSERT INTO "Materials" VALUES(5,'Electrical Wiring',100,10.0,3);
CREATE TABLE Payments (
    paymentID INTEGER PRIMARY KEY AUTOINCREMENT,
    amount PRECISION(8,2),
    paymentDate DATE,
    projectID INTEGER,
    contractorID INTEGER,
    FOREIGN KEY(projectID) REFERENCES Projects(projectID),
    FOREIGN KEY(contractorID) REFERENCES Contractors(contractorID)
);
INSERT INTO "Payments" VALUES(1,10000,'2022-02-01',1,1);
INSERT INTO "Payments" VALUES(2,15000,'2022-02-02',1,1);
INSERT INTO "Payments" VALUES(3,20000,'2022-02-03',2,2);
INSERT INTO "Payments" VALUES(4,30000,'2022-02-04',2,2);
INSERT INTO "Payments" VALUES(5,40000,'2022-02-05',3,3);
CREATE TABLE Projects (
    projectID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    startDate DATE,
    endDate DATE,
    budget REAL,
    contractorID INTEGER,
    FOREIGN KEY(contractorID) REFERENCES Contractors(contractorID)
);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Contractors',5);
INSERT INTO "sqlite_sequence" VALUES('Employees',5);
INSERT INTO "sqlite_sequence" VALUES('Materials',5);
INSERT INTO "sqlite_sequence" VALUES('Payments',5);
INSERT INTO "sqlite_sequence" VALUES('Funding',5);
COMMIT;
