BEGIN TRANSACTION;
CREATE TABLE Employee_Projects (
    employee_id INTEGER,
    project_id INTEGER,
    start_date DATE,
    end_date DATE,
    FOREIGN KEY(employee_id) REFERENCES Employees(id),
    FOREIGN KEY(project_id) REFERENCES Projects(id)
);
CREATE TABLE Employee_Skills (
    employee_id INTEGER,
    skill_id INTEGER,
    experience_level TEXT,
    FOREIGN KEY(employee_id) REFERENCES Employees(id),
    FOREIGN KEY(skill_id) REFERENCES Skills(id)
);
CREATE TABLE Job_Positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_title TEXT,
    description TEXT
);
INSERT INTO "Job_Positions" VALUES(1,'Software Engineer','Design and develop high-quality software solutions.');
INSERT INTO "Job_Positions" VALUES(2,'Data Scientist','Analyze data sets and provide insights.');
INSERT INTO "Job_Positions" VALUES(3,'Product Manager','Lead product development from concept to launch.');
INSERT INTO "Job_Positions" VALUES(4,'UX Designer','Create user interfaces that enhance usability.');
INSERT INTO "Job_Positions" VALUES(5,'QA Analyst','Identify defects and ensure product quality.');
CREATE TABLE Skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Skills" VALUES(1,'Programming');
INSERT INTO "Skills" VALUES(2,'Database Management');
INSERT INTO "Skills" VALUES(3,'Machine Learning');
INSERT INTO "Skills" VALUES(4,'Project Management');
INSERT INTO "Skills" VALUES(5,'User Experience Design');
CREATE TABLE Software_Companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT,
    location TEXT
);
INSERT INTO "Software_Companies" VALUES(1,'CompanyA','New York');
INSERT INTO "Software_Companies" VALUES(2,'CompanyB','San Francisco');
INSERT INTO "Software_Companies" VALUES(3,'CompanyC','London');
INSERT INTO "Software_Companies" VALUES(4,'CompanyD','Tokyo');
INSERT INTO "Software_Companies" VALUES(5,'CompanyE','Paris');
CREATE TABLE Software_Companies_Job_Positions (
    software_company_id INTEGER,
    job_position_id INTEGER,
    FOREIGN KEY(software_company_id) REFERENCES Software_Companies(id),
    FOREIGN KEY(job_position_id) REFERENCES Job_Positions(id),
    PRIMARY KEY(software_company_id, job_position_id)
);
INSERT INTO "Software_Companies_Job_Positions" VALUES(1,1);
INSERT INTO "Software_Companies_Job_Positions" VALUES(2,1);
INSERT INTO "Software_Companies_Job_Positions" VALUES(3,2);
INSERT INTO "Software_Companies_Job_Positions" VALUES(4,3);
INSERT INTO "Software_Companies_Job_Positions" VALUES(5,4);
INSERT INTO "Software_Companies_Job_Positions" VALUES(3,1);
INSERT INTO "Software_Companies_Job_Positions" VALUES(4,1);
INSERT INTO "Software_Companies_Job_Positions" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Software_Companies',5);
INSERT INTO "sqlite_sequence" VALUES('Job_Positions',5);
INSERT INTO "sqlite_sequence" VALUES('Skills',5);
COMMIT;
