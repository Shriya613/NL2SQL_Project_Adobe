BEGIN TRANSACTION;
CREATE TABLE Companies(
    CompanyID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    PhoneNumber TEXT
);
INSERT INTO "Companies" VALUES(1,'TechCorp','123 Tech St.','+123456789');
INSERT INTO "Companies" VALUES(2,'MedCo','456 Med Ave.','+987654321');
INSERT INTO "Companies" VALUES(3,'FinCorp','789 Fin Blvd.,','+321654897');
INSERT INTO "Companies" VALUES(4,'EduInc','123 Edu St.','+543219876');
INSERT INTO "Companies" VALUES(5,'GovAgency','456 Gov Plaza','+789654321');
CREATE TABLE Departments(
    DepartmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT,
    LocationID INTEGER,
    FOREIGN KEY(LocationID) REFERENCES Companies(CompanyID)
);
INSERT INTO "Departments" VALUES(1,'Engineering','Research and Development',1);
INSERT INTO "Departments" VALUES(2,'Marketing','Brand Management',1);
INSERT INTO "Departments" VALUES(3,'HR','Recruitment and Onboarding',2);
INSERT INTO "Departments" VALUES(4,'IT','System Administration',2);
INSERT INTO "Departments" VALUES(5,'Accounting','Budget and Audit',3);
INSERT INTO "Departments" VALUES(6,'Academics','Curriculum Design',4);
INSERT INTO "Departments" VALUES(7,'Policy','Legislative Affairs',5);
CREATE TABLE Employees(
    EmployeeID INTEGER PRIMARY KEY AUTOINCREMENT,
    BirthDate DATE,
    HireDate DATE,
    PositionID INTEGER,
    Mental_Health_Career TEXT,
    Clinic TEXT, FullName TEXT,
    FOREIGN KEY(PositionID) REFERENCES Positions(PositionID)
);
INSERT INTO "Employees" VALUES(1,'1990-01-01','2015-06-01',3,'Psychologist','Harmony Wellness Center','John Doe');
INSERT INTO "Employees" VALUES(2,'1985-01-01','2017-08-01',4,'Psychologist','Serenity Center','Jane Smith');
INSERT INTO "Employees" VALUES(3,'1988-01-01','2018-12-01',1,'Psychiatrist','Harmony Wellness Center','Alice Jackson');
INSERT INTO "Employees" VALUES(4,'1992-01-01','2019-05-01',2,'Psychiatrist','Peaceful Haven','Tom Lee');
CREATE TABLE Positions(
    PositionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Salary DECIMAL(10,2) ,
    DepartmentID INTEGER,
    FOREIGN KEY(DepartmentID) REFERENCES Departments(DepartmentID)
);
INSERT INTO "Positions" VALUES(1,'Software Engineer',80000,1);
INSERT INTO "Positions" VALUES(2,'Product Manager',90000,1);
INSERT INTO "Positions" VALUES(3,'HR Specialist',60000,2);
INSERT INTO "Positions" VALUES(4,'Network Administrator',70000,2);
INSERT INTO "Positions" VALUES(5,'Accountant',50000,3);
INSERT INTO "Positions" VALUES(6,'Professor',85000,4);
INSERT INTO "Positions" VALUES(7,'Policy Analyst',75000,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Companies',5);
INSERT INTO "sqlite_sequence" VALUES('Departments',7);
INSERT INTO "sqlite_sequence" VALUES('Positions',7);
INSERT INTO "sqlite_sequence" VALUES('Employees',4);
COMMIT;
