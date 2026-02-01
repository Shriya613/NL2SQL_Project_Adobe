BEGIN TRANSACTION;
CREATE TABLE Benefits(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employeeId INTEGER,
    healthInsurance BOOLEAN DEFAULT FALSE,
    retirementPlan BOOLEAN DEFAULT FALSE,
    vacationDays INT DEFAULT 0,
    FOREIGN KEY (employeeId) REFERENCES Employee(id)
);
INSERT INTO "Benefits" VALUES(1,1,1,1,15);
INSERT INTO "Benefits" VALUES(2,2,1,1,20);
INSERT INTO "Benefits" VALUES(3,3,1,0,25);
INSERT INTO "Benefits" VALUES(4,4,0,1,30);
INSERT INTO "Benefits" VALUES(5,5,1,1,22);
CREATE TABLE Company(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "Company" VALUES(1,'Events Unlimited');
INSERT INTO "Company" VALUES(2,'Grand Celebrations');
CREATE TABLE Employee(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Employee" VALUES(1,'John','Smith','johnsmith@gmail.com');
INSERT INTO "Employee" VALUES(2,'Jane','Doe','janedoe@gmail.com');
INSERT INTO "Employee" VALUES(3,'Alice','Brown','alicebrown@gmail.com');
INSERT INTO "Employee" VALUES(4,'Mike','White','mwhite@gmail.com');
INSERT INTO "Employee" VALUES(5,'Emily','Black','eblack@gmail.com');
CREATE TABLE "Event_Planner"(
    id INTEGER PRIMARY KEY AUTOINCREMENT, Position TEXT,
    companyId INTEGER,
    experience INT,
    skills TEXT,
    FOREIGN KEY (companyId) REFERENCES Company(id)
);
INSERT INTO "Event_Planner" VALUES(1,'Junior Planner',1,3,'Organization, Communication');
INSERT INTO "Event_Planner" VALUES(2,'Senior Planner',2,6,'Decorating, Coordination');
CREATE TABLE Salary(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employeeId INTEGER,
    salary DECIMAL(10,2),
    startDate DATE,
    endDate DATE,
    FOREIGN KEY (employeeId) REFERENCES Employee(id)
);
INSERT INTO "Salary" VALUES(1,1,80000,'2021-01-01','2023-12-31');
INSERT INTO "Salary" VALUES(2,2,90000,'2019-01-01','2022-12-31');
INSERT INTO "Salary" VALUES(3,3,85000,'2020-01-01','Present');
INSERT INTO "Salary" VALUES(4,4,82000,'2018-01-01','2021-12-31');
INSERT INTO "Salary" VALUES(5,5,88000,'2021-01-01','Present');
CREATE TABLE "Wedding_Coordinator"(
    id INTEGER PRIMARY KEY AUTOINCREMENT, Position TEXT,
    companyId INTEGER,
    experience INT,
    skills TEXT,
    FOREIGN KEY (companyId) REFERENCES Company(id)
);
INSERT INTO "Wedding_Coordinator" VALUES(1,'Assistant Coordinator',1,2,'Designing, Planning');
INSERT INTO "Wedding_Coordinator" VALUES(2,'Lead Coordinator',2,7,'Vendor Relationships, Budgeting');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Company',2);
INSERT INTO "sqlite_sequence" VALUES('Employee',5);
INSERT INTO "sqlite_sequence" VALUES('Salary',5);
INSERT INTO "sqlite_sequence" VALUES('Benefits',5);
INSERT INTO "sqlite_sequence" VALUES('Wedding_Coordinator',2);
INSERT INTO "sqlite_sequence" VALUES('Event_Planner',2);
COMMIT;
