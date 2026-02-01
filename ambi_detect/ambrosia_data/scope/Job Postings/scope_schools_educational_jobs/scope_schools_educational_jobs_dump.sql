BEGIN TRANSACTION;
CREATE TABLE Educational_Jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT
);
INSERT INTO "Educational_Jobs" VALUES(1,'Teacher','A teacher who teaches various subjects.');
INSERT INTO "Educational_Jobs" VALUES(2,'Principal','The principal manages a school.');
INSERT INTO "Educational_Jobs" VALUES(3,'Counselor','A counselor provides guidance to students.');
INSERT INTO "Educational_Jobs" VALUES(4,'Librarian','A librarian manages library resources.');
INSERT INTO "Educational_Jobs" VALUES(5,'IT Specialist','An IT specialist maintains computer systems.');
CREATE TABLE Employee_Education (
    employee_education_id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER,
    education_level TEXT,
    major TEXT,
    degree_year INTEGER,
    FOREIGN KEY(employee_id) REFERENCES School_Employees(employee_id)
);
INSERT INTO "Employee_Education" VALUES(1,1,'PhD','Mathematics',2019);
INSERT INTO "Employee_Education" VALUES(2,1,'Master','Physics',2017);
INSERT INTO "Employee_Education" VALUES(3,2,'Doctorate','English Literature',2020);
INSERT INTO "Employee_Education" VALUES(4,2,'Bachelor','History',2016);
INSERT INTO "Employee_Education" VALUES(5,3,'Master','Psychology',2018);
CREATE TABLE Jobs_at_School (
    school_job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    school_id INTEGER,
    job_id INTEGER,
    start_date DATE,
    end_date DATE,
    FOREIGN KEY(school_id) REFERENCES Schools(id),
    FOREIGN KEY(job_id) REFERENCES Educational_Jobs(job_id)
);
INSERT INTO "Jobs_at_School" VALUES(1,1,1,'2021-08-01',NULL);
INSERT INTO "Jobs_at_School" VALUES(2,1,2,'2021-08-01',NULL);
INSERT INTO "Jobs_at_School" VALUES(3,2,3,'2021-08-01',NULL);
INSERT INTO "Jobs_at_School" VALUES(4,2,4,'2021-08-01',NULL);
INSERT INTO "Jobs_at_School" VALUES(5,3,5,'2021-08-01',NULL);
INSERT INTO "Jobs_at_School" VALUES(8,4,5,'2022-08-01',NULL);
INSERT INTO "Jobs_at_School" VALUES(9,5,5,'2022-08-01',NULL);
INSERT INTO "Jobs_at_School" VALUES(10,2,5,'2022-08-01',NULL);
INSERT INTO "Jobs_at_School" VALUES(11,1,5,'2022-08-01',NULL);
CREATE TABLE School_Employees (
    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    job_title TEXT,
    school_id INTEGER,
    FOREIGN KEY(school_id) REFERENCES Schools(id)
);
INSERT INTO "School_Employees" VALUES(1,'John','Doe','Teacher',1);
INSERT INTO "School_Employees" VALUES(2,'Jane','Smith','Principal',2);
INSERT INTO "School_Employees" VALUES(3,'Mike','Brown','Counselor',3);
INSERT INTO "School_Employees" VALUES(4,'Alice','Green','Librarian',4);
INSERT INTO "School_Employees" VALUES(5,'Tom','Black','IT Specialist',5);
CREATE TABLE Schools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Schools" VALUES(1,'University A','City X');
INSERT INTO "Schools" VALUES(2,'High School B','City Y');
INSERT INTO "Schools" VALUES(3,'College C','City Z');
INSERT INTO "Schools" VALUES(4,'Primary School D','City W');
INSERT INTO "Schools" VALUES(5,'Technical School E','City V');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Schools',5);
INSERT INTO "sqlite_sequence" VALUES('Educational_Jobs',5);
INSERT INTO "sqlite_sequence" VALUES('School_Employees',5);
INSERT INTO "sqlite_sequence" VALUES('Jobs_at_School',12);
INSERT INTO "sqlite_sequence" VALUES('Employee_Education',5);
COMMIT;
