BEGIN TRANSACTION;
CREATE TABLE Base_Pay_Range(
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- unique identifier for pay range
    min_pay REAL,               -- minimum salary in the job posting
    max_pay REAL                 -- maximum salary in the job posting
);
INSERT INTO "Base_Pay_Range" VALUES(1,60000.0,80000.0);
INSERT INTO "Base_Pay_Range" VALUES(2,70000.0,90000.0);
INSERT INTO "Base_Pay_Range" VALUES(3,50000.0,70000.0);
INSERT INTO "Base_Pay_Range" VALUES(4,100000.0,120000.0);
INSERT INTO "Base_Pay_Range" VALUES(5,80000.0,100000.0);
CREATE TABLE Benefits(
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- unique identifier for benefits package
    health_insurance BOOLEAN DEFAULT FALSE,-- whether or not there is health insurance
    dental_insurance BOOLEAN DEFAULT FALSE, -- whether or not there is dental insurance
    vision_insurance BOOLEAN DEFAULT FALSE   -- whether or not there is vision insurance
);
INSERT INTO "Benefits" VALUES(1,1,1,1);
INSERT INTO "Benefits" VALUES(2,0,1,1);
INSERT INTO "Benefits" VALUES(3,1,0,1);
INSERT INTO "Benefits" VALUES(4,1,1,0);
INSERT INTO "Benefits" VALUES(5,0,0,0);
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
CREATE TABLE Job_Post(
    id INTEGER PRIMARY KEY AUTOINCREMENT,      -- unique identifier for job post
    title TEXT,                       -- title of the job posting
    description TEXT,                          -- detailed description of the job
    base_pay_id INTEGER,                       -- foreign key referencing Base_Pay_Range
    benefit_package_id INTEGER,                -- foreign key referencing Benefits
    FOREIGN KEY (base_pay_id) REFERENCES Base_Pay_Range(id), -- references Base_Pay_Range
    FOREIGN KEY (benefit_package_id) REFERENCES Benefits(id)  -- references Benefits
);
INSERT INTO "Job_Post" VALUES(1,'Software Engineer','Develop and maintain software applications.',1,1);
INSERT INTO "Job_Post" VALUES(2,'Data Scientist','Analyze complex datasets to drive business decisions.',3,2);
INSERT INTO "Job_Post" VALUES(3,'Marketing Manager','Lead marketing strategies and campaigns.',2,3);
INSERT INTO "Job_Post" VALUES(4,'Product Designer','Design user-centric products and interfaces.',4,4);
INSERT INTO "Job_Post" VALUES(5,'HR Specialist','Manage employee relations and HR processes.',5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Companies',5);
INSERT INTO "sqlite_sequence" VALUES('Base_Pay_Range',5);
INSERT INTO "sqlite_sequence" VALUES('Benefits',5);
INSERT INTO "sqlite_sequence" VALUES('Job_Post',5);
COMMIT;
