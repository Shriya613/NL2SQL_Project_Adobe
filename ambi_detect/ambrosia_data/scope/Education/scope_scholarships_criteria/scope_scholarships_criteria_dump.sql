BEGIN TRANSACTION;
CREATE TABLE Applications (
    application_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    scholarship_id INTEGER,
    submission_date DATE,
    status TEXT ,
    FOREIGN KEY(student_id) REFERENCES Students(student_id),
    FOREIGN KEY(scholarship_id) REFERENCES Scholarships(id)
);
INSERT INTO "Applications" VALUES(1,1,1,'2021-09-01','Pending');
INSERT INTO "Applications" VALUES(2,2,2,'2021-09-01','Approved');
INSERT INTO "Applications" VALUES(3,3,3,'2021-09-01','Rejected');
INSERT INTO "Applications" VALUES(4,4,4,'2021-09-01','Under Review');
INSERT INTO "Applications" VALUES(5,5,5,'2021-09-01','Withdrawn');
CREATE TABLE Criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    value TEXT
);
INSERT INTO "Criteria" VALUES(1,'Academic Achievement');
INSERT INTO "Criteria" VALUES(2,'Community Service');
INSERT INTO "Criteria" VALUES(3,'Leadership Skills');
INSERT INTO "Criteria" VALUES(4,'Financial Need');
INSERT INTO "Criteria" VALUES(5,'Extracurricular Activities');
CREATE TABLE Scholarship_Criteria (
    scholarship_id INTEGER,
    criterion_id INTEGER,
    FOREIGN KEY(scholarship_id) REFERENCES Scholarships(id),
    FOREIGN KEY(criterion_id) REFERENCES Criteria(id)
);
INSERT INTO "Scholarship_Criteria" VALUES(1,1);
INSERT INTO "Scholarship_Criteria" VALUES(2,1);
INSERT INTO "Scholarship_Criteria" VALUES(3,1);
INSERT INTO "Scholarship_Criteria" VALUES(4,1);
INSERT INTO "Scholarship_Criteria" VALUES(5,1);
INSERT INTO "Scholarship_Criteria" VALUES(5,2);
CREATE TABLE Scholarships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    amount REAL
);
INSERT INTO "Scholarships" VALUES(1,'Academic Excellence Scholarship',2000.0);
INSERT INTO "Scholarships" VALUES(2,'Service Leadership Scholarship',1500.0);
INSERT INTO "Scholarships" VALUES(3,'Young Leader Scholarship',1000.0);
INSERT INTO "Scholarships" VALUES(4,'Needy Student Grant',500.0);
INSERT INTO "Scholarships" VALUES(5,'Artistic Talent Award',750.0);
CREATE TABLE Students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number VARCHAR(15),
    admission_date DATE
);
INSERT INTO "Students" VALUES(1,'John','Doe','john.doe@example.com','+123-456-7890','2021-08-01');
INSERT INTO "Students" VALUES(2,'Jane','Smith','jane.smith@example.com','+123-456-7890','2021-08-01');
INSERT INTO "Students" VALUES(3,'Tom','Brown','tom.brown@example.com','+123-456-7890','2021-08-01');
INSERT INTO "Students" VALUES(4,'Alice','Green','alice.green@example.com','+123-456-7890','2021-08-01');
INSERT INTO "Students" VALUES(5,'Mike','Black','mike.black@example.com','+123-456-7890','2021-08-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Criteria',5);
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Scholarships',5);
INSERT INTO "sqlite_sequence" VALUES('Applications',5);
COMMIT;
