BEGIN TRANSACTION;
CREATE TABLE Applicant (
    ApplicantID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Email TEXT,
    ResumePath TEXT
);
INSERT INTO "Applicant" VALUES(1,'John','Doe','1985-05-15','john.doe@email.com','/path/to/resume.pdf');
INSERT INTO "Applicant" VALUES(2,'Jane','Smith','1987-07-23','jane.smith@email.com','/path/to/resume.pdf');
CREATE TABLE Application (
    AppID INTEGER PRIMARY KEY AUTOINCREMENT,
    JobID INTEGER,
    ApplicantID INTEGER,
    SubmissionDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    CoverLetterPath TEXT,
    Status TEXT ,
    FOREIGN KEY (JobID) REFERENCES JobPosting(JobID),
    FOREIGN KEY (ApplicantID) REFERENCES Applicant(ApplicantID)
);
INSERT INTO "Application" VALUES(1,1,1,'2024-03-13 12:41:56','/path/to/coverletter.pdf',NULL);
INSERT INTO "Application" VALUES(2,2,2,'2024-03-13 12:41:56','/path/to/coverletter.pdf',NULL);
CREATE TABLE Company (
    CompanyID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    AddressLine1 TEXT,
    AddressLine2 TEXT,
    City TEXT,
    State TEXT,
    ZipCode INT,
    PhoneNumber VARCHAR(50),
    EmailAddress TEXT
);
INSERT INTO "Company" VALUES(1,'Company A','123 Main St',NULL,'Cityville','CA',98765,'123-456-7890','companyA@example.com');
INSERT INTO "Company" VALUES(2,'Company B','456 Elm St',NULL,'Downtown','NY',12345,'098-765-4321','companyB@example.com');
CREATE TABLE GeoArea (
    AreaID INTEGER PRIMARY KEY AUTOINCREMENT,
    LocationText TEXT UNIQUE
);
INSERT INTO "GeoArea" VALUES(1,'East End');
INSERT INTO "GeoArea" VALUES(2,'West Side');
CREATE TABLE HelpingProfession (
    ProfessionID INTEGER PRIMARY KEY AUTOINCREMENT,
    ProfessionName TEXT UNIQUE
);
INSERT INTO "HelpingProfession" VALUES(1,'Social Worker');
INSERT INTO "HelpingProfession" VALUES(2,'Counselor');
CREATE TABLE JobPosting (
    JobID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Description TEXT,
    PostedDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    DeadlineDate DATETIME,
    SalaryRange TEXT,
    Status TEXT ,
    ProfessionID INTEGER,
    AreaID INTEGER,
    FOREIGN KEY (ProfessionID) REFERENCES HelpingProfession(ProfessionID),
    FOREIGN KEY (AreaID) REFERENCES GeoArea(AreaID)
);
INSERT INTO "JobPosting" VALUES(1,'Social Worker Position','Help people improve their wellbeing.','2024-03-13 12:41:56',NULL,'$30k-$45k','Open',1,1);
INSERT INTO "JobPosting" VALUES(2,'Mental Health Counselor','Assist individuals with mental health issues.','2024-03-13 12:41:56',NULL,'$40k-$60k','Open',2,1);
INSERT INTO "JobPosting" VALUES(3,'Social Worker Position','Help people overcome challenges.','2024-03-13 12:41:56',NULL,'$30k-$45k','Open',1,2);
INSERT INTO "JobPosting" VALUES(4,'Mental Health Counselor','Guide clients towards emotional growth.','2024-03-13 12:41:56',NULL,'$40k-$60k','Open',2,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('HelpingProfession',2);
INSERT INTO "sqlite_sequence" VALUES('GeoArea',2);
INSERT INTO "sqlite_sequence" VALUES('Company',2);
INSERT INTO "sqlite_sequence" VALUES('JobPosting',4);
INSERT INTO "sqlite_sequence" VALUES('Applicant',2);
INSERT INTO "sqlite_sequence" VALUES('Application',2);
COMMIT;
