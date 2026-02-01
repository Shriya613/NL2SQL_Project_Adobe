BEGIN TRANSACTION;
CREATE TABLE Applicants(
    ApplicantID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT UNIQUE);
INSERT INTO "Applicants" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Applicants" VALUES(2,'Jane','Smith','jane.smith@example.com');
CREATE TABLE ApplicationJobs(
    ApplicationID INTEGER PRIMARY KEY,
    ApplicantID INTEGER,
    JobID INTEGER,
    SubmissionDate DATE,
    Status TEXT DEFAULT 'Pending',
    FOREIGN KEY (ApplicantID) REFERENCES Applicants(ApplicantID),
    FOREIGN KEY (JobID) REFERENCES Jobs(JobID));
INSERT INTO "ApplicationJobs" VALUES(1,1,1,'2023-01-01','Pending');
INSERT INTO "ApplicationJobs" VALUES(2,2,2,'2023-01-02','Pending');
CREATE TABLE Companies(
    CompanyID INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT);
INSERT INTO "Companies" VALUES(1,'ABC Corp.','New York');
INSERT INTO "Companies" VALUES(2,'XYZ Inc.','San Francisco');
CREATE TABLE DataTypes(
    DataTypeID INTEGER PRIMARY KEY,
    DataTypeName TEXT);
INSERT INTO "DataTypes" VALUES(1,'Demographics');
INSERT INTO "DataTypes" VALUES(2,'Financial Analysis');
CREATE TABLE Jobs(
    JobID INTEGER PRIMARY KEY,
    Title TEXT,
    Description TEXT,
    PostedDate DATE,
    CompanyID INTEGER,
    ROLEID INTEGER,
    DATAID INTEGER,
    FOREIGN KEY (CompanyID) REFERENCES Companies(CompanyID),
    FOREIGN KEY (ROLEID) REFERENCES Roles(RoleID),
    FOREIGN KEY (DATAID) REFERENCES DataTypes(DataTypeID));
INSERT INTO "Jobs" VALUES(1,'Economic Analyst','Analyze economic data...','2023-01-01',1,1,1);
INSERT INTO "Jobs" VALUES(2,'Senior Economic Analyst','Analyze economic data...','2023-01-02',2,1,2);
INSERT INTO "Jobs" VALUES(3,'Statistical Analyst','Analyze statistical data...','2023-01-01',1,2,1);
INSERT INTO "Jobs" VALUES(4,'Senior Statistical Analyst','Analyze statistical data...','2023-01-02',2,2,2);
CREATE TABLE Roles(
    RoleID INTEGER PRIMARY KEY,
    RoleName TEXT);
INSERT INTO "Roles" VALUES(1,'Economist');
INSERT INTO "Roles" VALUES(2,'Statistician');
COMMIT;
