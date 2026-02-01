BEGIN TRANSACTION;
CREATE TABLE Applications (
    applicationId INTEGER PRIMARY KEY AUTOINCREMENT,
    applicantId INTEGER,
    jobId INTEGER,
    resumePath TEXT,
    coverLetterPath TEXT,
    status TEXT ,
    FOREIGN KEY(applicantId) REFERENCES Employees(id),
    FOREIGN KEY(jobId) REFERENCES Positions(position));
INSERT INTO "Applications" VALUES(1,1,1,'/path/to/resume1.pdf','/path/to/coverletter1.docx','Pending');
INSERT INTO "Applications" VALUES(2,2,2,'/path/to/resume2.pdf','/path/to/coverletter2.docx','Reviewed');
CREATE TABLE Departments (
    id INTEGER PRIMARY KEY,
    name TEXT,
    managerId INTEGER,
    FOREIGN KEY(managerId) REFERENCES Employees(id));
INSERT INTO "Departments" VALUES(1,'Engineering',1);
INSERT INTO "Departments" VALUES(2,'Human Resources',2);
CREATE TABLE Employees (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    email TEXT,
    phoneNumber VARCHAR(15),
    hireDate DATE);
INSERT INTO "Employees" VALUES(1,'John','Doe','john.doe@example.com','123-456-7890','2021-01-01');
INSERT INTO "Employees" VALUES(2,'Jane','Smith','jane.smith@example.com','098-765-4321','2021-02-01');
CREATE TABLE "Positions" (
    Recruiter TEXT,
    position TEXT,
    description TEXT,
    PRIMARY KEY(Recruiter, position));
INSERT INTO "Positions" VALUES('TechCorp','Software Engineer',NULL);
INSERT INTO "Positions" VALUES('HealthCo','Registered Nurse',NULL);
CREATE TABLE Salaries (
    employeeId INTEGER,
    salary DECIMAL(8,2),
    effectiveDate DATE,
    FOREIGN KEY(employeeId) REFERENCES Employees(id));
INSERT INTO "Salaries" VALUES(1,70000,'2021-01-01');
INSERT INTO "Salaries" VALUES(2,60000,'2021-02-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Applications',2);
COMMIT;
