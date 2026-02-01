BEGIN TRANSACTION;
CREATE TABLE Applicant(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number TEXT,
    application_date DATETIME DEFAULT CURRENT_TIMESTAMP);
INSERT INTO "Applicant" VALUES(1,'John','Smith','johnsmith@email.com','555-123-4567','2024-02-19 12:44:27');
INSERT INTO "Applicant" VALUES(2,'Jane','Doe','janedoe@email.com','555-987-6543','2024-02-19 12:44:27');
INSERT INTO "Applicant" VALUES(3,'Alice','Williams','alicewilliams@email.com','555-098-7654','2024-02-19 12:44:27');
INSERT INTO "Applicant" VALUES(4,'Tom','Brown','tommybrown@email.com','555-135-7890','2024-02-19 12:44:27');
INSERT INTO "Applicant" VALUES(5,'Emily','Green','emilygreen@email.com','555-246-8790','2024-02-19 12:44:27');
CREATE TABLE Application(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_id INTEGER REFERENCES Applicant(id),
    job_post_id INTEGER REFERENCES Job_Post(id),
    resume_url TEXT,
    cover_letter TEXT,
    status TEXT );
INSERT INTO "Application" VALUES(1,1,1,'https://example.com/resume/johnsmith.pdf','https://example.com/coverletters/johnsmith.txt','Received');
INSERT INTO "Application" VALUES(2,1,2,'https://example.com/resume/johnsmith.pdf','https://example.com/coverletters/johnsmith.txt','Rejected');
INSERT INTO "Application" VALUES(3,2,1,'https://example.com/resume/janedoe.pdf','https://example.com/coverletters/janedoe.txt','Interview Scheduled');
INSERT INTO "Application" VALUES(4,2,2,'https://example.com/resume/janedoe.pdf','https://example.com/coverletters/janedoe.txt','Hired');
INSERT INTO "Application" VALUES(5,3,1,'https://example.com/resume/alicewilliams.pdf','https://example.com/coverletters/alicewilliams.txt','Waitlisted');
CREATE TABLE Company(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT,
    location TEXT);
INSERT INTO "Company" VALUES(1,'ABC Tech Corporation','New York');
INSERT INTO "Company" VALUES(2,'XYZ Software Solutions','San Francisco');
INSERT INTO "Company" VALUES(3,'MNO Consulting Group','Chicago');
INSERT INTO "Company" VALUES(4,'KLM International Inc.','Los Angeles');
INSERT INTO "Company" VALUES(5,'RST Media Networks','Washington');
CREATE TABLE "Job_Post"(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    job_type_id INTEGER REFERENCES Job_Type(id),
    degree_level_id INTEGER REFERENCES "Qualifications"(id),
    company_id INTEGER REFERENCES Company(id));
INSERT INTO "Job_Post" VALUES(1,'Software Engineer',1,3,1);
INSERT INTO "Job_Post" VALUES(2,'Data Analyst',2,1,2);
INSERT INTO "Job_Post" VALUES(3,'UX Designer',1,2,3);
INSERT INTO "Job_Post" VALUES(4,'Marketing Manager',1,1,4);
INSERT INTO "Job_Post" VALUES(5,'HR Specialist',2,1,5);
CREATE TABLE Job_Type(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_name TEXT,
    description TEXT);
INSERT INTO "Job_Type" VALUES(1,'Full Time','Permanent position offering regular hours and benefits.');
INSERT INTO "Job_Type" VALUES(2,'Part Time','Non-permanent position with limited hours and benefits.');
INSERT INTO "Job_Type" VALUES(3,'Contract','Temporary position for a fixed period of time.');
INSERT INTO "Job_Type" VALUES(4,'Internship','Short-term work experience opportunity often related to educational field.');
INSERT INTO "Job_Type" VALUES(5,'Freelance','Independent contractor providing services on a per-project basis.');
CREATE TABLE "Qualifications"(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level_name TEXT,
    description TEXT);
INSERT INTO "Qualifications" VALUES(1,'Bachelor','A bachelors degree is typically a four-year undergraduate program.');
INSERT INTO "Qualifications" VALUES(2,'Master','A masters degree is usually pursued after obtaining a bachelors degree and can take one to three years.');
INSERT INTO "Qualifications" VALUES(3,'PhD','A doctoral degree requires extensive research and study beyond a master''s degree.');
INSERT INTO "Qualifications" VALUES(4,'Associate','An associate degree is typically a two-year program that provides vocational training.');
INSERT INTO "Qualifications" VALUES(5,'High School Diploma','A high school diploma represents completion of secondary education.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Qualifications',5);
INSERT INTO "sqlite_sequence" VALUES('Job_Type',5);
INSERT INTO "sqlite_sequence" VALUES('Company',5);
INSERT INTO "sqlite_sequence" VALUES('Applicant',5);
INSERT INTO "sqlite_sequence" VALUES('Application',5);
INSERT INTO "sqlite_sequence" VALUES('Job_Post',5);
COMMIT;
