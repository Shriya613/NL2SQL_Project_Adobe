BEGIN TRANSACTION;
CREATE TABLE Courses(
    CourseID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Department TEXT
);
INSERT INTO "Courses" VALUES(1,'Computer Science','Engineering');
INSERT INTO "Courses" VALUES(2,'Mathematics','Science');
INSERT INTO "Courses" VALUES(3,'Physics','Science');
INSERT INTO "Courses" VALUES(4,'English Literature','Humanities');
INSERT INTO "Courses" VALUES(5,'History','Arts');
CREATE TABLE Enrollments(
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    CourseID INTEGER,
    Semester DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(CourseID) REFERENCES Courses(CourseID)
);
INSERT INTO "Enrollments" VALUES(1,1,1,'2023-01-01 00:00:00');
INSERT INTO "Enrollments" VALUES(2,2,2,'2023-01-01 00:00:00');
INSERT INTO "Enrollments" VALUES(3,3,3,'2023-01-01 00:00:00');
INSERT INTO "Enrollments" VALUES(4,4,4,'2023-01-01 00:00:00');
INSERT INTO "Enrollments" VALUES(5,5,5,'2023-01-01 00:00:00');
CREATE TABLE Grades(
    GradeID INTEGER PRIMARY KEY AUTOINCREMENT,
    EnrollmentID INTEGER,
    Grade CHAR(2) ,
    FOREIGN KEY(EnrollmentID) REFERENCES Enrollments(EnrollmentID)
);
INSERT INTO "Grades" VALUES(1,1,'A');
INSERT INTO "Grades" VALUES(2,2,'B');
INSERT INTO "Grades" VALUES(3,3,'C');
INSERT INTO "Grades" VALUES(4,4,'A');
INSERT INTO "Grades" VALUES(5,5,'B');
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    Academic_Writing TEXT ,
    Academic_Supervisor TEXT
, FullName TEXT);
INSERT INTO "Students" VALUES(1,'Dissertation','Dr. Jane Smith','John Doe');
INSERT INTO "Students" VALUES(2,'Dissertation','Dr. John Doe','Jane Smith');
INSERT INTO "Students" VALUES(3,'Thesis Proposal','Dr. Jane Smith','Mary Brown');
INSERT INTO "Students" VALUES(4,'Thesis Proposal','Dr. John Doe','Steve White');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',4);
INSERT INTO "sqlite_sequence" VALUES('Courses',5);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',5);
INSERT INTO "sqlite_sequence" VALUES('Grades',5);
COMMIT;
