BEGIN TRANSACTION;
CREATE TABLE Companies (
    companyID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Companies" VALUES(1,'Company A','New York');
INSERT INTO "Companies" VALUES(2,'Company B','San Francisco');
INSERT INTO "Companies" VALUES(3,'Company C','Boston');
INSERT INTO "Companies" VALUES(4,'Company D','Chicago');
INSERT INTO "Companies" VALUES(5,'Company E','Los Angeles');
CREATE TABLE Departments (
    deptID INTEGER PRIMARY KEY AUTOINCREMENT,
    deptName TEXT,
    companyID INTEGER,
    FOREIGN KEY(companyID) REFERENCES Companies(companyID)
);
INSERT INTO "Departments" VALUES(1,'Engineering',1);
INSERT INTO "Departments" VALUES(2,'Marketing',2);
INSERT INTO "Departments" VALUES(3,'Human Resources',3);
INSERT INTO "Departments" VALUES(4,'Finance',4);
INSERT INTO "Departments" VALUES(5,'Product Development',5);
CREATE TABLE Employees (
    employeeID INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    phoneNumber TEXT,
    hireDate DATE,
    positionID INTEGER, fullname TEXT,
    FOREIGN KEY(positionID) REFERENCES Positions(positionID)
);
INSERT INTO "Employees" VALUES(1,'john.doe@example.com','123-456-7890','2021-01-15',1,'John Doe');
INSERT INTO "Employees" VALUES(2,'alice.johnson@example.com','098-765-4321','2020-02-10',2,'Alice Johnson');
INSERT INTO "Employees" VALUES(3,'tom.brown@example.com','555-555-5555','2019-03-05',3,'Tom Brown');
INSERT INTO "Employees" VALUES(4,'linda.green@example.com','555-123-4567','2018-04-01',4,'Linda Green');
INSERT INTO "Employees" VALUES(5,'kevin.black@example.com','111-222-3333','2017-05-15',5,'Kevin Black');
CREATE TABLE "Management" (
    roleID INTEGER PRIMARY KEY AUTOINCREMENT,
    PositionID INTEGER,
    Department_Head TEXT,
    Team_Leader TEXT,
    FOREIGN KEY(PositionID) REFERENCES Positions(positionID)
);
INSERT INTO "Management" VALUES(1,1,'Noah Davis','Jane Smith');
INSERT INTO "Management" VALUES(2,1,'Emma Garcia','Alex Brown');
INSERT INTO "Management" VALUES(3,2,'Sophia Lee','Mary White');
INSERT INTO "Management" VALUES(4,3,'Olivia Martinez','Kate Green');
INSERT INTO "Management" VALUES(5,4,'Liam Johnson','Robert Elliot');
CREATE TABLE Positions (
    positionID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    teamID INTEGER,
    FOREIGN KEY(teamID) REFERENCES Teams(teamID)
);
INSERT INTO "Positions" VALUES(1,'Software Engineer','Develop and maintain software applications.',1);
INSERT INTO "Positions" VALUES(2,'Android Developer','Design and develop Android apps.',2);
INSERT INTO "Positions" VALUES(3,'UX/UI Designer','Create user-centered designs.',2);
INSERT INTO "Positions" VALUES(4,'Data Scientist','Analyze data to provide insights.',4);
INSERT INTO "Positions" VALUES(5,'Accountant','Handle financial transactions and reports.',5);
CREATE TABLE Teams (
    teamID INTEGER PRIMARY KEY AUTOINCREMENT,
    teamName TEXT,
    deptID INTEGER,
    FOREIGN KEY(deptID) REFERENCES Departments(deptID)
);
INSERT INTO "Teams" VALUES(1,'Web Developers',1);
INSERT INTO "Teams" VALUES(2,'Mobile App Developers',1);
INSERT INTO "Teams" VALUES(3,'Graphic Designers',2);
INSERT INTO "Teams" VALUES(4,'Content Writers',2);
INSERT INTO "Teams" VALUES(5,'Data Analysts',4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Companies',5);
INSERT INTO "sqlite_sequence" VALUES('Departments',5);
INSERT INTO "sqlite_sequence" VALUES('Teams',5);
INSERT INTO "sqlite_sequence" VALUES('Positions',5);
INSERT INTO "sqlite_sequence" VALUES('Management',5);
INSERT INTO "sqlite_sequence" VALUES('Employees',5);
COMMIT;
