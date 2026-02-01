BEGIN TRANSACTION;
CREATE TABLE Courses(
    CourseID INTEGER PRIMARY KEY AUTOINCREMENT,
    CourseName TEXT,
    TeacherName TEXT
);
INSERT INTO "Courses" VALUES(1,'English','Mr. Johnson');
INSERT INTO "Courses" VALUES(2,'Math','Ms. Williams');
INSERT INTO "Courses" VALUES(3,'History','Mr. Taylor');
INSERT INTO "Courses" VALUES(4,'Art','Ms. Davis');
INSERT INTO "Courses" VALUES(5,'Biology','Dr. Baker');
CREATE TABLE Enrollments(
    StudentID INTEGER,
    CourseID INTEGER,
    Grade INT DEFAULT -1,
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
);
INSERT INTO "Enrollments" VALUES(1,1,-1);
INSERT INTO "Enrollments" VALUES(2,1,-1);
INSERT INTO "Enrollments" VALUES(3,1,-1);
INSERT INTO "Enrollments" VALUES(4,1,-1);
INSERT INTO "Enrollments" VALUES(5,1,-1);
INSERT INTO "Enrollments" VALUES(5,2,-1);
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FullName TEXT);
INSERT INTO "Students" VALUES(1,'John Doe');
INSERT INTO "Students" VALUES(2,'Jane Smith');
INSERT INTO "Students" VALUES(3,'Michael Brown');
INSERT INTO "Students" VALUES(4,'Emily White');
INSERT INTO "Students" VALUES(5,'David Green');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Courses',5);
COMMIT;
