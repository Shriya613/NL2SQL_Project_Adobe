BEGIN TRANSACTION;
CREATE TABLE Applicants(
    applicantId INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT);
INSERT INTO "Applicants" VALUES(1,'Alice','Smith','alice@example.com');
INSERT INTO "Applicants" VALUES(2,'Bob','Doe','bob@example.com');
INSERT INTO "Applicants" VALUES(3,'Charlie','Johnson','charlie@example.com');
INSERT INTO "Applicants" VALUES(4,'Diana','Williams','diana@example.com');
INSERT INTO "Applicants" VALUES(5,'Ethan','Martin','ethan@example.com');
CREATE TABLE Applications(
    applicationId INTEGER PRIMARY KEY AUTOINCREMENT,
    applicantId INTEGER,
    jobId INTEGER,
    dateApplied DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT ,
    FOREIGN KEY(applicantId) REFERENCES Applicants(applicantId),
    FOREIGN KEY(jobId) REFERENCES Jobs(jobId));
INSERT INTO "Applications" VALUES(1,1,1,'2024-01-30 19:36:07','Pending');
INSERT INTO "Applications" VALUES(2,2,2,'2024-01-30 19:36:07','In Progress');
INSERT INTO "Applications" VALUES(3,3,3,'2024-01-30 19:36:07','Rejected');
INSERT INTO "Applications" VALUES(4,4,4,'2024-01-30 19:36:07','Accepted');
INSERT INTO "Applications" VALUES(5,5,5,'2024-01-30 19:36:07','Interview Scheduled');
CREATE TABLE Functions(
    functionId INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT);
INSERT INTO "Functions" VALUES(1,'Recruitment');
INSERT INTO "Functions" VALUES(2,'Training');
INSERT INTO "Functions" VALUES(3,'Onboarding');
INSERT INTO "Functions" VALUES(4,'Offboarding');
INSERT INTO "Functions" VALUES(5,'Compliance');
CREATE TABLE Hr_Professionals(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT);
INSERT INTO "Hr_Professionals" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Hr_Professionals" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Hr_Professionals" VALUES(3,'Mike Johnson','mike@example.com');
INSERT INTO "Hr_Professionals" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Hr_Professionals" VALUES(5,'David Brown','david@example.com');
CREATE TABLE Hr_Professionals_Functions(
    hrProfessionalId INTEGER,
    functionId INTEGER,
    FOREIGN KEY(hrProfessionalId) REFERENCES Hr_Professionals(id),
    FOREIGN KEY(functionId) REFERENCES Functions(functionId));
INSERT INTO "Hr_Professionals_Functions" VALUES(1,1);
INSERT INTO "Hr_Professionals_Functions" VALUES(2,1);
INSERT INTO "Hr_Professionals_Functions" VALUES(3,1);
INSERT INTO "Hr_Professionals_Functions" VALUES(4,1);
INSERT INTO "Hr_Professionals_Functions" VALUES(5,1);
INSERT INTO "Hr_Professionals_Functions" VALUES(5,2);
CREATE TABLE Jobs(
    jobId INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT);
INSERT INTO "Jobs" VALUES(1,'Software Engineer','We are looking for a skilled Software Engineer...');
INSERT INTO "Jobs" VALUES(2,'Data Analyst','We need an experienced Data Analyst to join our team...');
INSERT INTO "Jobs" VALUES(3,'Project Manager','A Project Manager is needed to lead our projects...');
INSERT INTO "Jobs" VALUES(4,'HR Specialist','An HR Specialist will support our growing company...');
INSERT INTO "Jobs" VALUES(5,'UX Designer','We are seeking a UX Designer to create user-friendly interfaces...');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Hr_Professionals',5);
INSERT INTO "sqlite_sequence" VALUES('Functions',5);
INSERT INTO "sqlite_sequence" VALUES('Jobs',5);
INSERT INTO "sqlite_sequence" VALUES('Applicants',5);
INSERT INTO "sqlite_sequence" VALUES('Applications',5);
COMMIT;
