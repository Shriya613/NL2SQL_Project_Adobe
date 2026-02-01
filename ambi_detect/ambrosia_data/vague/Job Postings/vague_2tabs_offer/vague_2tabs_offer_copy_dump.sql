BEGIN TRANSACTION;
CREATE TABLE Companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- unique identifier for company
    name TEXT,           -- name of the company
    location TEXT                        -- location of the company
);
INSERT INTO "Companies" VALUES(1,'TechCorp','San Francisco');
INSERT INTO "Companies" VALUES(2,'HealthPlus','New York');
INSERT INTO "Companies" VALUES(3,'EcoGreen','Los Angeles');
INSERT INTO "Companies" VALUES(4,'SoftWare','Chicago');
INSERT INTO "Companies" VALUES(5,'CyberSecure','Boston');
CREATE TABLE "Job_Post"(
    id INTEGER PRIMARY KEY AUTOINCREMENT,      -- unique identifier for job post
    title TEXT,                       -- title of the job posting
    description TEXT,                          -- detailed description of the job
    base_pay_id INTEGER,                -- foreign key referencing Benefits
    FOREIGN KEY (base_pay_id) REFERENCES "Offer"(id));
INSERT INTO "Job_Post" VALUES(1,'Software Engineer','Develop and maintain software applications.',1);
INSERT INTO "Job_Post" VALUES(2,'Data Scientist','Analyze complex datasets to drive business decisions.',3);
INSERT INTO "Job_Post" VALUES(3,'Marketing Manager','Lead marketing strategies and campaigns.',2);
INSERT INTO "Job_Post" VALUES(4,'Product Designer','Design user-centric products and interfaces.',4);
INSERT INTO "Job_Post" VALUES(5,'HR Specialist','Manage employee relations and HR processes.',5);
CREATE TABLE "Offer"(
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- unique identifier for pay range
    min_pay REAL,               -- minimum salary in the job posting
    max_pay REAL                 -- maximum salary in the job posting
);
INSERT INTO "Offer" VALUES(1,60000.0,80000.0);
INSERT INTO "Offer" VALUES(2,70000.0,90000.0);
INSERT INTO "Offer" VALUES(3,50000.0,70000.0);
INSERT INTO "Offer" VALUES(4,100000.0,120000.0);
INSERT INTO "Offer" VALUES(5,80000.0,100000.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Companies',5);
INSERT INTO "sqlite_sequence" VALUES('Offer',5);
INSERT INTO "sqlite_sequence" VALUES('Job_Post',5);
COMMIT;
