BEGIN TRANSACTION;
CREATE TABLE Job_Postings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    salary REAL
);
INSERT INTO "Job_Postings" VALUES(1,'Software Engineer','Develop and maintain software applications.',80000.0);
INSERT INTO "Job_Postings" VALUES(2,'Data Analyst','Analyze data sets and provide insights.',60000.0);
INSERT INTO "Job_Postings" VALUES(3,'Graphic Designer','Create visual designs for marketing materials.',50000.0);
INSERT INTO "Job_Postings" VALUES(4,'Product Manager','Lead product development from conception to launch.',75000.0);
INSERT INTO "Job_Postings" VALUES(5,'HR Specialist','Recruit, train, and manage employees.',55000.0);
CREATE TABLE Job_Postings_Requirements (
    job_posting_id INTEGER,
    requirement_id INTEGER,
    FOREIGN KEY (job_posting_id) REFERENCES Job_Postings(id),
    FOREIGN KEY (requirement_id) REFERENCES Requirements(id),
    PRIMARY KEY (job_posting_id, requirement_id)
);
INSERT INTO "Job_Postings_Requirements" VALUES(1,1);
INSERT INTO "Job_Postings_Requirements" VALUES(2,1);
INSERT INTO "Job_Postings_Requirements" VALUES(3,2);
INSERT INTO "Job_Postings_Requirements" VALUES(4,3);
INSERT INTO "Job_Postings_Requirements" VALUES(5,4);
INSERT INTO "Job_Postings_Requirements" VALUES(1,5);
INSERT INTO "Job_Postings_Requirements" VALUES(2,5);
INSERT INTO "Job_Postings_Requirements" VALUES(3,1);
INSERT INTO "Job_Postings_Requirements" VALUES(4,1);
INSERT INTO "Job_Postings_Requirements" VALUES(5,1);
CREATE TABLE Requirements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    requirement_type TEXT,
    value TEXT
);
INSERT INTO "Requirements" VALUES(1,'Education','Bachelor''s degree');
INSERT INTO "Requirements" VALUES(2,'Experience','3 years');
INSERT INTO "Requirements" VALUES(3,'Skills','SQL, Python');
INSERT INTO "Requirements" VALUES(4,'Language','English');
INSERT INTO "Requirements" VALUES(5,'Certification','PMP');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Job_Postings',5);
INSERT INTO "sqlite_sequence" VALUES('Requirements',5);
COMMIT;
