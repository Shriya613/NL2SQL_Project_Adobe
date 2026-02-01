BEGIN TRANSACTION;
CREATE TABLE Tech_Jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_title TEXT,
    description TEXT
);
INSERT INTO "Tech_Jobs" VALUES(1,'Product Manager','Managing product development from conception to launch.');
INSERT INTO "Tech_Jobs" VALUES(2,'Software Engineer','Developing software applications using various programming languages.');
INSERT INTO "Tech_Jobs" VALUES(3,'Data Scientist','Analyzing data to provide insights and make data-driven decisions.');
INSERT INTO "Tech_Jobs" VALUES(4,'Marketing Specialist','Creating marketing strategies to promote products and services.');
INSERT INTO "Tech_Jobs" VALUES(5,'Designer','Designing user interfaces and experiences for digital products.');
CREATE TABLE Tech_Startups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Tech_Startups" VALUES(1,'Startup1','San Francisco');
INSERT INTO "Tech_Startups" VALUES(2,'Startup2','New York');
INSERT INTO "Tech_Startups" VALUES(3,'Startup3','Los Angeles');
INSERT INTO "Tech_Startups" VALUES(4,'Startup4','Boston');
INSERT INTO "Tech_Startups" VALUES(5,'Startup5','Austin');
CREATE TABLE Tech_Startups_Tech_Jobs (
    tech_startup_id INTEGER REFERENCES Tech_Startups(id) ON DELETE CASCADE,
    tech_job_id INTEGER REFERENCES Tech_Jobs(id) ON DELETE CASCADE,
    PRIMARY KEY (tech_startup_id, tech_job_id)
);
INSERT INTO "Tech_Startups_Tech_Jobs" VALUES(1,1);
INSERT INTO "Tech_Startups_Tech_Jobs" VALUES(2,1);
INSERT INTO "Tech_Startups_Tech_Jobs" VALUES(3,1);
INSERT INTO "Tech_Startups_Tech_Jobs" VALUES(4,1);
INSERT INTO "Tech_Startups_Tech_Jobs" VALUES(5,1);
INSERT INTO "Tech_Startups_Tech_Jobs" VALUES(5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Tech_Startups',5);
INSERT INTO "sqlite_sequence" VALUES('Tech_Jobs',5);
COMMIT;
