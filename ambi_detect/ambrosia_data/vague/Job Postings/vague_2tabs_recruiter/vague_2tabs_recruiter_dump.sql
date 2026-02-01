BEGIN TRANSACTION;
CREATE TABLE Applicant (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT,
    phoneNumber TEXT,
    positionId INTEGER,
    FOREIGN KEY (positionId) REFERENCES Position(id)
);
INSERT INTO "Applicant" VALUES(1,'John','Smith','johnsmith@example.com','+1-123-456-7890',1);
INSERT INTO "Applicant" VALUES(2,'Jane','Doe','janedoe@example.com','+1-987-654-3210',1);
CREATE TABLE Company (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Company" VALUES(1,'Tech Solutions','San Francisco');
INSERT INTO "Company" VALUES(2,'Software Inc.','New York');
CREATE TABLE Employee (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT,
    phoneNumber TEXT,
    role TEXT,
    hireDate TEXT
);
INSERT INTO "Employee" VALUES(1,'Alice','Wonderland','aliceland@example.com','+1-123-456-7890','HR Manager','2022-01-01');
INSERT INTO "Employee" VALUES(2,'Bob','Builder','bobbuilder@example.com','+1-987-654-3210','Engineering Director','2021-07-15');
CREATE TABLE Interview (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATETIME,
    time TEXT,
    applicantId INTEGER,
    interviewerId INTEGER,
    positionId INTEGER,
    status TEXT ,
    FOREIGN KEY (applicantId) REFERENCES Applicant(id),
    FOREIGN KEY (interviewerId) REFERENCES Employee(id),
    FOREIGN KEY (positionId) REFERENCES Position(id)
);
INSERT INTO "Interview" VALUES(1,'2023-01-15','10:00 AM',1,1,1,'Pending');
INSERT INTO "Interview" VALUES(2,'2023-01-16','11:00 AM',2,2,1,'Completed');
CREATE TABLE Position (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    salary DECIMAL(10,2) ,
    companyId INTEGER,
    recruitmentAgencyId INTEGER,
    FOREIGN KEY (companyId) REFERENCES Company(id),
    FOREIGN KEY (recruitmentAgencyId) REFERENCES Recruitment_Agency(id)
);
INSERT INTO "Position" VALUES(1,'Software Engineer','Develop and maintain software applications.',90000,1,1);
INSERT INTO "Position" VALUES(2,'Data Analyst','Analyze data to provide insights and recommendations.',80000,1,2);
CREATE TABLE Recruitment_Agency (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agencyName TEXT,
    contactEmail TEXT,
    contactPhone TEXT
);
INSERT INTO "Recruitment_Agency" VALUES(1,'Top Talent Agency','toptalent@example.com','+1-800-123-4567');
INSERT INTO "Recruitment_Agency" VALUES(2,'Job Connect','jobconnect@example.com','+1-900-987-6543');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Company',2);
INSERT INTO "sqlite_sequence" VALUES('Recruitment_Agency',2);
INSERT INTO "sqlite_sequence" VALUES('Position',2);
INSERT INTO "sqlite_sequence" VALUES('Applicant',2);
INSERT INTO "sqlite_sequence" VALUES('Interview',2);
INSERT INTO "sqlite_sequence" VALUES('Employee',2);
COMMIT;
