BEGIN TRANSACTION;
CREATE TABLE Assessors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Assessors" VALUES(1,'Professor Smith');
INSERT INTO "Assessors" VALUES(2,'Dr. Johnson');
INSERT INTO "Assessors" VALUES(3,'Ms. Brown');
INSERT INTO "Assessors" VALUES(4,'Mr. White');
INSERT INTO "Assessors" VALUES(5,'Mrs. Green');
CREATE TABLE Exams(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    marks INTEGER,
    FOREIGN KEY (student_id) REFERENCES Students(id)
);
INSERT INTO "Exams" VALUES(1,1,85);
INSERT INTO "Exams" VALUES(2,2,90);
INSERT INTO "Exams" VALUES(3,3,75);
INSERT INTO "Exams" VALUES(4,4,65);
INSERT INTO "Exams" VALUES(5,5,80);
CREATE TABLE Quizzes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    points INTEGER,
    FOREIGN KEY (student_id) REFERENCES Students(id)
);
INSERT INTO "Quizzes" VALUES(1,1,70);
INSERT INTO "Quizzes" VALUES(2,2,85);
INSERT INTO "Quizzes" VALUES(3,3,60);
INSERT INTO "Quizzes" VALUES(4,4,55);
INSERT INTO "Quizzes" VALUES(5,5,75);
CREATE TABLE StudentSubjects(
    student_id INTEGER,
    subject_id INTEGER,
    FOREIGN KEY (student_id) REFERENCES Students(id),
    FOREIGN KEY (subject_id) REFERENCES Subjects(id),
    PRIMARY KEY (student_id, subject_id)
);
INSERT INTO "StudentSubjects" VALUES(1,1);
INSERT INTO "StudentSubjects" VALUES(2,2);
INSERT INTO "StudentSubjects" VALUES(3,3);
INSERT INTO "StudentSubjects" VALUES(4,4);
INSERT INTO "StudentSubjects" VALUES(5,5);
CREATE TABLE Students(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER
);
INSERT INTO "Students" VALUES(1,'Alice',20);
INSERT INTO "Students" VALUES(2,'Bob',21);
INSERT INTO "Students" VALUES(3,'Charlie',22);
INSERT INTO "Students" VALUES(4,'David',23);
INSERT INTO "Students" VALUES(5,'Eva',24);
CREATE TABLE Subjects(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_name TEXT,
    assessor_id INTEGER,
    FOREIGN KEY (assessor_id) REFERENCES Assessors(id)
);
INSERT INTO "Subjects" VALUES(1,'Mathematics',1);
INSERT INTO "Subjects" VALUES(2,'Physics',2);
INSERT INTO "Subjects" VALUES(3,'Chemistry',3);
INSERT INTO "Subjects" VALUES(4,'Biology',4);
INSERT INTO "Subjects" VALUES(5,'History',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Exams',5);
INSERT INTO "sqlite_sequence" VALUES('Quizzes',5);
INSERT INTO "sqlite_sequence" VALUES('Assessors',5);
INSERT INTO "sqlite_sequence" VALUES('Subjects',5);
COMMIT;
