BEGIN TRANSACTION;
CREATE TABLE Applicant (
    Id INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT,
    ResumeText BLOB
);
CREATE TABLE Application (
    Id INTEGER PRIMARY KEY,
    ApplicationDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    ApplicantId INTEGER,
    PositionId INTEGER,
    FOREIGN KEY(ApplicantId) REFERENCES Applicant(Id),
    FOREIGN KEY(PositionId) REFERENCES Position(Id)
);
INSERT INTO "Application" VALUES(1,'2024-02-19 12:51:33',NULL,1);
INSERT INTO "Application" VALUES(2,'2024-02-19 12:51:33',NULL,1);
CREATE TABLE ApplicationDeadline (
    Id INTEGER PRIMARY KEY,
    Deadline DATETIME,
    PositionId INTEGER,
    FOREIGN KEY(PositionId) REFERENCES Position(Id)
);
INSERT INTO "ApplicationDeadline" VALUES(1,'2023-01-31T00:00:00',1);
INSERT INTO "ApplicationDeadline" VALUES(2,'2023-02-15T00:00:00',1);
CREATE TABLE Employer (
    Id INTEGER PRIMARY KEY,
    Name TEXT,
    Email TEXT,
    PhoneNumber TEXT
);
INSERT INTO "Employer" VALUES(1,'Company A','companyA@example.com','123-456-7890');
INSERT INTO "Employer" VALUES(2,'Company B','companyB@example.com','098-765-4321');
CREATE TABLE Position (
    Id INTEGER PRIMARY KEY,
    Title TEXT,
    Description TEXT
);
INSERT INTO "Position" VALUES(1,'Software Developer','Develop and maintain software applications.');
INSERT INTO "Position" VALUES(2,'Data Analyst','Analyze data to provide insights and recommendations.');
CREATE TABLE "SalaryDetails" (
    Id INTEGER PRIMARY KEY,
    Compensation TEXT,
    PositionId INTEGER,
    FOREIGN KEY(PositionId) REFERENCES Position(Id)
);
INSERT INTO "SalaryDetails" VALUES(1,'$50k - $60k',1);
INSERT INTO "SalaryDetails" VALUES(2,'$70k - $80k',1);
COMMIT;
