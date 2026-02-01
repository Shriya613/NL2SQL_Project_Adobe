BEGIN TRANSACTION;
CREATE TABLE Applicants (
    id INTEGER PRIMARY KEY,
    applicant_name TEXT,
    email TEXT,
    phone_number TEXT);
INSERT INTO "Applicants" VALUES(1,'John Doe','johndoe@example.com','+123-456-7890');
INSERT INTO "Applicants" VALUES(2,'Jane Smith','janesmith@example.com','+098-765-4321');
INSERT INTO "Applicants" VALUES(3,'Alice Brown','alicebrown@example.com','+456-789-0123');
INSERT INTO "Applicants" VALUES(4,'Bob Johnson','bobjohnson@example.com','+789-456-1230');
INSERT INTO "Applicants" VALUES(5,'Carol White','carolwhite@example.com','+123-098-7654');
CREATE TABLE Applications (
    application_id INTEGER PRIMARY KEY,
    applicant_id INTEGER,
    job_id INTEGER,
    status TEXT ,
    submission_date DATE,
    FOREIGN KEY(applicant_id) REFERENCES Applicants(id),
    FOREIGN KEY(job_id) REFERENCES Jobs(id));
INSERT INTO "Applications" VALUES(1,1,1,'Pending','2021-01-01');
INSERT INTO "Applications" VALUES(2,1,2,'Rejected','2021-01-02');
INSERT INTO "Applications" VALUES(3,2,3,'Interview Scheduled','2021-01-03');
INSERT INTO "Applications" VALUES(4,3,4,'Offered','2021-01-04');
INSERT INTO "Applications" VALUES(5,4,5,'Accepted','2021-01-05');
CREATE TABLE Companies (
    id INTEGER PRIMARY KEY,
    company_name TEXT,
    location TEXT,
    industry TEXT);
INSERT INTO "Companies" VALUES(1,'TechCorp','New York','Technology');
INSERT INTO "Companies" VALUES(2,'FinanceInc.','Los Angeles','Banking');
INSERT INTO "Companies" VALUES(3,'HealthCare','Chicago','Healthcare');
INSERT INTO "Companies" VALUES(4,'GreenEnergy','San Francisco','Renewable Energy');
INSERT INTO "Companies" VALUES(5,'LogisticsCo','Houston','Transportation');
CREATE TABLE CompanyJobs (
    company_id INTEGER,
    job_id INTEGER,
    FOREIGN KEY(company_id) REFERENCES Companies(id),
    FOREIGN KEY(job_id) REFERENCES Jobs(id),
    PRIMARY KEY(company_id, job_id));
INSERT INTO "CompanyJobs" VALUES(1,1);
INSERT INTO "CompanyJobs" VALUES(2,2);
INSERT INTO "CompanyJobs" VALUES(3,3);
INSERT INTO "CompanyJobs" VALUES(4,4);
INSERT INTO "CompanyJobs" VALUES(5,5);
CREATE TABLE "Degrees" (
    id INTEGER PRIMARY KEY,
    Qualifications TEXT,
    job_post TEXT);
INSERT INTO "Degrees" VALUES(1,'Bachelor','Software Developer');
INSERT INTO "Degrees" VALUES(2,'Masters','Data Analyst');
INSERT INTO "Degrees" VALUES(3,'PhD','Research Scientist');
INSERT INTO "Degrees" VALUES(4,'Associate','Network Engineer');
INSERT INTO "Degrees" VALUES(5,'Doctorate','AI Researcher');
CREATE TABLE Jobs (
    id INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    salary REAL,
    posted_date DATE);
INSERT INTO "Jobs" VALUES(1,'Software Developer','Design and develop software applications.',80000.0,'2021-01-01');
INSERT INTO "Jobs" VALUES(2,'Data Analyst','Analyze data to provide insights and support decision making.',60000.0,'2021-01-02');
INSERT INTO "Jobs" VALUES(3,'Research Scientist','Conduct research on cutting edge technologies.',100000.0,'2021-01-03');
INSERT INTO "Jobs" VALUES(4,'Network Engineer','Install, configure, and maintain computer networks.',70000.0,'2021-01-04');
INSERT INTO "Jobs" VALUES(5,'AI Researcher','Develop AI algorithms and models.',90000.0,'2021-01-05');
COMMIT;
