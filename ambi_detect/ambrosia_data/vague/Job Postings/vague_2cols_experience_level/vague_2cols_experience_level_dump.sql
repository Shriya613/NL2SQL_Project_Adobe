BEGIN TRANSACTION;
CREATE TABLE Applicants (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    phone TEXT,
    resume TEXT
);
INSERT INTO "Applicants" VALUES(1,'John Doe','john.doe@example.com','123-456-7890','path/to/resume');
INSERT INTO "Applicants" VALUES(2,'Jane Smith','jane.smith@example.com','098-765-4321','path/to/resume');
INSERT INTO "Applicants" VALUES(3,'Alice Johnson','alice.johnson@example.com','555-123-4567','path/to/resume');
INSERT INTO "Applicants" VALUES(4,'Bob Brown','bob.brown@example.com','111-222-3333','path/to/resume');
INSERT INTO "Applicants" VALUES(5,'Carol White','carol.white@example.com','444-555-6666','path/to/resume');
CREATE TABLE Companies (
    id INTEGER PRIMARY KEY,
    name TEXT,
    industry TEXT,
    size TEXT
);
INSERT INTO "Companies" VALUES(1,'TechCorp','Technology','Large');
INSERT INTO "Companies" VALUES(2,'FinancialGroup','Finance','Medium');
INSERT INTO "Companies" VALUES(3,'HealthCare','Healthcare','Small');
INSERT INTO "Companies" VALUES(4,'EcoFriendly','Environment','Large');
INSERT INTO "Companies" VALUES(5,'RetailChain','Retail','Medium');
CREATE TABLE CompanyJobs (
    company_id INTEGER,
    job_id INTEGER,
    FOREIGN KEY (company_id) REFERENCES Companies(id),
    FOREIGN KEY (job_id) REFERENCES Jobs(id),
    PRIMARY KEY (company_id, job_id)
);
CREATE TABLE Experiences (
    applicant_id INTEGER,
    years_of_experience INTEGER ,
    number_of_completed_projects INTEGER ,
    job TEXT,
    additional_skills TEXT,
    FOREIGN KEY (applicant_id) REFERENCES Applicants(id),
    PRIMARY KEY (applicant_id, job)
);
INSERT INTO "Experiences" VALUES(1,5,10,'Web Developer','HTML, CSS, JavaScript');
INSERT INTO "Experiences" VALUES(2,3,12,'Mobile App Developer','Swift, Android');
INSERT INTO "Experiences" VALUES(3,7,20,'Full Stack Developer','Node.js, Python');
INSERT INTO "Experiences" VALUES(4,10,25,'DevOps Engineer','AWS, Docker');
INSERT INTO "Experiences" VALUES(5,8,30,'Frontend Developer','React, Angular');
CREATE TABLE Jobs (
    id INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    location TEXT,
    salary REAL 
);
INSERT INTO "Jobs" VALUES(1,'Software Engineer','Develop and maintain software applications.','New York, NY',90000.0);
INSERT INTO "Jobs" VALUES(2,'Data Analyst','Analyze data to provide insights and recommendations.','Los Angeles, CA',80000.0);
INSERT INTO "Jobs" VALUES(3,'Product Manager','Lead product development from concept to launch.','San Francisco, CA',100000.0);
INSERT INTO "Jobs" VALUES(4,'UX Designer','Create user interfaces that enhance usability and user experience.','Chicago, IL',75000.0);
INSERT INTO "Jobs" VALUES(5,'Marketing Specialist','Plan and execute marketing campaigns to increase brand visibility.','Miami, FL',65000.0);
COMMIT;
