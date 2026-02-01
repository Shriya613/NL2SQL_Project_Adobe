BEGIN TRANSACTION;
CREATE TABLE Agency (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);
INSERT INTO "Agency" VALUES(1,'Expedia');
INSERT INTO "Agency" VALUES(2,'Booking.com');
INSERT INTO "Agency" VALUES(3,'Travelsupermarket.com');
INSERT INTO "Agency" VALUES(4,'Thomson Holidays');
INSERT INTO "Agency" VALUES(5,'Lastminute.com');
CREATE TABLE Applicant (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number VARCHAR(20)
);
INSERT INTO "Applicant" VALUES(1,'John','Doe','john@example.com','+1234567890');
INSERT INTO "Applicant" VALUES(2,'Jane','Smith','jane@example.com','+0987654321');
CREATE TABLE Application (
    application_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_posting_id INTEGER,
    applicant_id INTEGER,
    application_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT ,
    FOREIGN KEY(job_posting_id) REFERENCES JobPosting(id),
    FOREIGN KEY(applicant_id) REFERENCES Applicant(id)
);
INSERT INTO "Application" VALUES(1,1,1,'2024-03-13 13:06:58','Applied');
INSERT INTO "Application" VALUES(2,2,2,'2024-03-13 13:06:58','Rejected');
CREATE TABLE HospitalityJobType (
    id INTEGER PRIMARY KEY,
    job_type TEXT UNIQUE
);
INSERT INTO "HospitalityJobType" VALUES(1,'Tour Guide');
INSERT INTO "HospitalityJobType" VALUES(2,'Travel Agent');
CREATE TABLE JobPosting (
    id INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    posted_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agency_id INTEGER,
    hospitality_job_type_id INTEGER,
    FOREIGN KEY(agency_id) REFERENCES Agency(id),
    FOREIGN KEY(hospitality_job_type_id) REFERENCES HospitalityJobType(id)
);
INSERT INTO "JobPosting" VALUES(1,'Seeking Tour Guides','We need experienced tour guides to join our team.','2024-03-13 13:06:58',1,1);
INSERT INTO "JobPosting" VALUES(2,'Experienced Travel Agents Wanted','If you have experience in travel agency, apply now!','2024-03-13 13:06:58',1,2);
INSERT INTO "JobPosting" VALUES(3,'Join Our Team of Tour Guides','','2024-03-13 13:06:58',2,1);
INSERT INTO "JobPosting" VALUES(4,'Travel Agents Needed','','2024-03-13 13:06:58',3,2);
INSERT INTO "JobPosting" VALUES(5,'Looking For Enthusiastic Tour Guides','','2024-03-13 13:06:58',4,1);
INSERT INTO "JobPosting" VALUES(6,'Senior Travel Agents Required','','2024-03-13 13:06:58',5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Application',2);
COMMIT;
