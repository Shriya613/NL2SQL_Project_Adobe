BEGIN TRANSACTION;
CREATE TABLE "Assignments"(
    assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    due_date INTEGER,
    description TEXT,
    FOREIGN KEY (due_date) REFERENCES "Deadline"(date_id));
INSERT INTO "Assignments" VALUES(1,1,'First project submission');
INSERT INTO "Assignments" VALUES(2,2,'Final exam preparation');
INSERT INTO "Assignments" VALUES(3,3,'Group presentation');
INSERT INTO "Assignments" VALUES(4,4,'Term paper');
INSERT INTO "Assignments" VALUES(5,5,'Book report');
CREATE TABLE Classes(
    class_id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_title TEXT,
    instructor TEXT
);
INSERT INTO "Classes" VALUES(1,'Introduction to Computer Science','Dr. Johnson');
INSERT INTO "Classes" VALUES(2,'Calculus I','Prof. Williams');
INSERT INTO "Classes" VALUES(3,'English Literature','Ms. Thompson');
INSERT INTO "Classes" VALUES(4,'Chemistry','Dr. Brown');
INSERT INTO "Classes" VALUES(5,'Physics','Prof. White');
CREATE TABLE "Deadline"(
    date_id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER,
    month INTEGER,
    day INTEGER
);
INSERT INTO "Deadline" VALUES(1,2021,9,15);
INSERT INTO "Deadline" VALUES(2,2021,10,30);
INSERT INTO "Deadline" VALUES(3,2021,11,7);
INSERT INTO "Deadline" VALUES(4,2021,12,18);
INSERT INTO "Deadline" VALUES(5,2022,1,6);
CREATE TABLE Enrollment(
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    class_id INTEGER,
    FOREIGN KEY (student_id) REFERENCES Students(id),
    FOREIGN KEY (class_id) REFERENCES Classes(class_id)
);
INSERT INTO "Enrollment" VALUES(1,1,1);
INSERT INTO "Enrollment" VALUES(2,2,2);
INSERT INTO "Enrollment" VALUES(3,3,3);
INSERT INTO "Enrollment" VALUES(4,4,4);
INSERT INTO "Enrollment" VALUES(5,5,5);
CREATE TABLE Students(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT
);
INSERT INTO "Students" VALUES(1,'John','Doe');
INSERT INTO "Students" VALUES(2,'Jane','Smith');
INSERT INTO "Students" VALUES(3,'Michael','Brown');
INSERT INTO "Students" VALUES(4,'Emily','White');
INSERT INTO "Students" VALUES(5,'William','Black');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Classes',5);
INSERT INTO "sqlite_sequence" VALUES('Deadline',5);
INSERT INTO "sqlite_sequence" VALUES('Enrollment',5);
INSERT INTO "sqlite_sequence" VALUES('Assignments',5);
COMMIT;
