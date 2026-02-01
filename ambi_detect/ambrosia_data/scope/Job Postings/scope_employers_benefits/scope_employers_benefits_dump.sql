BEGIN TRANSACTION;
CREATE TABLE Applicants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number TEXT,
    resume TEXT
);
INSERT INTO "Applicants" VALUES(1,'John','Smith','johnsmith@email.com','555-123-4567','file://path/to/resume.pdf');
INSERT INTO "Applicants" VALUES(2,'Jane','Doe','janedoe@email.com','555-987-6543','file://path/to/resume.pdf');
INSERT INTO "Applicants" VALUES(3,'Michael','Johnson','michaeljohnson@email.com','555-321-7654','file://path/to/resume.pdf');
INSERT INTO "Applicants" VALUES(4,'Emily','Williams','emilywilliams@email.com','555-555-5555','file://path/to/resume.pdf');
INSERT INTO "Applicants" VALUES(5,'David','Brown','davidbrown@email.com','555-111-2222','file://path/to/resume.pdf');
CREATE TABLE ApplicationStatuses (
    application_id INTEGER,
    status TEXT ,
    interview_date DATETIME,
    feedback TEXT,
    FOREIGN KEY(application_id) REFERENCES Applicants(id)
);
INSERT INTO "ApplicationStatuses" VALUES(1,'Pending Interview',NULL,NULL);
INSERT INTO "ApplicationStatuses" VALUES(2,'Rejected',NULL,'Not a good fit.');
INSERT INTO "ApplicationStatuses" VALUES(3,'Accepted','2022-09-15 10:00:00','Excellent candidate!');
INSERT INTO "ApplicationStatuses" VALUES(4,'Interview Scheduled','2022-09-17 14:00:00','Thank you for applying.');
INSERT INTO "ApplicationStatuses" VALUES(5,'Offer Made',NULL,'We would like to offer you the position.');
CREATE TABLE Benefits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benefit_name TEXT,
    description TEXT
);
INSERT INTO "Benefits" VALUES(1,'Health Insurance','Comprehensive health insurance coverage for employees');
INSERT INTO "Benefits" VALUES(2,'Dental Coverage','Includes preventative care, basic restoratives, and major services');
INSERT INTO "Benefits" VALUES(3,'Vision Care','Covers annual eye exams, glasses, and contact lenses');
INSERT INTO "Benefits" VALUES(4,'Retirement Plan','Matching contributions up to 6% of employee earnings');
INSERT INTO "Benefits" VALUES(5,'Paid Time Off','Accrued PTO based on years of service');
CREATE TABLE Employers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Employers" VALUES(1,'Company A','companya@example.com');
INSERT INTO "Employers" VALUES(2,'Company B','companyb@example.com');
INSERT INTO "Employers" VALUES(3,'Company C','companyc@example.com');
INSERT INTO "Employers" VALUES(4,'Company D','companyd@example.com');
INSERT INTO "Employers" VALUES(5,'Company E','companye@example.com');
CREATE TABLE Employers_Benefits (
    employer_id INTEGER,
    benefit_id INTEGER,
    FOREIGN KEY(employer_id) REFERENCES Employers(id),
    FOREIGN KEY(benefit_id) REFERENCES Benefits(id),
    PRIMARY KEY(employer_id, benefit_id)
);
INSERT INTO "Employers_Benefits" VALUES(1,1);
INSERT INTO "Employers_Benefits" VALUES(2,1);
INSERT INTO "Employers_Benefits" VALUES(3,1);
INSERT INTO "Employers_Benefits" VALUES(4,1);
INSERT INTO "Employers_Benefits" VALUES(5,1);
INSERT INTO "Employers_Benefits" VALUES(5,5);
CREATE TABLE JobPostings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    location TEXT,
    salary DECIMAL(10,2) ,
    posting_date DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "JobPostings" VALUES(1,'Software Engineer','Design and develop software applications.','New York',90000,'CURRENT_TIMESTAMP');
INSERT INTO "JobPostings" VALUES(2,'Data Analyst','Analyze data sets to provide insights and recommendations.','Los Angeles',75000,'CURRENT_TIMESTAMP');
INSERT INTO "JobPostings" VALUES(3,'Marketing Manager','Lead marketing strategies and campaigns.','Chicago',80000,'CURRENT_TIMESTAMP');
INSERT INTO "JobPostings" VALUES(4,'Product Designer','Create user-centered designs for products.','San Francisco',85000,'CURRENT_TIMESTAMP');
INSERT INTO "JobPostings" VALUES(5,'HR Specialist','Manage recruitment and onboarding processes.','Boston',60000,'CURRENT_TIMESTAMP');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Employers',5);
INSERT INTO "sqlite_sequence" VALUES('Benefits',5);
INSERT INTO "sqlite_sequence" VALUES('JobPostings',5);
INSERT INTO "sqlite_sequence" VALUES('Applicants',5);
COMMIT;
