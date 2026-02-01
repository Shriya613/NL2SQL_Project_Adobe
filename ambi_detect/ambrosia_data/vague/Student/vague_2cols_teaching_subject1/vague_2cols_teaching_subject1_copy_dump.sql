BEGIN TRANSACTION;
CREATE TABLE "Courses" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Teaching_Subject INTEGER REFERENCES Disciplines(id),
    professor_id INTEGER REFERENCES Professors(id)
);
INSERT INTO "Courses" VALUES(1,1,1);
INSERT INTO "Courses" VALUES(2,2,3);
INSERT INTO "Courses" VALUES(3,3,2);
INSERT INTO "Courses" VALUES(4,4,4);
INSERT INTO "Courses" VALUES(5,5,5);
CREATE TABLE Disciplines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "Disciplines" VALUES(1,'Mathematics');
INSERT INTO "Disciplines" VALUES(2,'Physics');
INSERT INTO "Disciplines" VALUES(3,'Chemistry');
INSERT INTO "Disciplines" VALUES(4,'Biology');
INSERT INTO "Disciplines" VALUES(5,'Computer Science');
CREATE TABLE Enrollments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER REFERENCES Students(id),
    course_id INTEGER REFERENCES Courses(id)
);
INSERT INTO "Enrollments" VALUES(1,1,1);
INSERT INTO "Enrollments" VALUES(2,2,2);
INSERT INTO "Enrollments" VALUES(3,3,3);
INSERT INTO "Enrollments" VALUES(4,4,4);
INSERT INTO "Enrollments" VALUES(5,5,5);
CREATE TABLE Grades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    enrollment_id INTEGER REFERENCES Enrollments(id),
    grade TEXT 
);
INSERT INTO "Grades" VALUES(1,1,'A+');
INSERT INTO "Grades" VALUES(2,2,'B-');
INSERT INTO "Grades" VALUES(3,3,'A');
INSERT INTO "Grades" VALUES(4,4,'B+');
INSERT INTO "Grades" VALUES(5,5,'A');
CREATE TABLE Professors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstname TEXT,
    lastname TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Professors" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Professors" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Professors" VALUES(3,'Michael','Johnson','michael.johnson@example.com');
INSERT INTO "Professors" VALUES(4,'Emily','Williams','emily.williams@example.com');
INSERT INTO "Professors" VALUES(5,'David','Brown','david.brown@example.com');
CREATE TABLE Students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstname TEXT,
    lastname TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Students" VALUES(1,'Tom','Robertson','tom.robertson@example.com');
INSERT INTO "Students" VALUES(2,'Lucy','Taylor','lucy.taylor@example.com');
INSERT INTO "Students" VALUES(3,'Chris','Wilson','chris.wilson@example.com');
INSERT INTO "Students" VALUES(4,'Alice','Thompson','alice.thompson@example.com');
INSERT INTO "Students" VALUES(5,'Daniel','Jackson','daniel.jackson@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Disciplines',5);
INSERT INTO "sqlite_sequence" VALUES('Professors',5);
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',5);
INSERT INTO "sqlite_sequence" VALUES('Grades',5);
INSERT INTO "sqlite_sequence" VALUES('Courses',5);
COMMIT;
