BEGIN TRANSACTION;
CREATE TABLE "Courses" (
    CourseID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Description TEXT,
    Career_Opportunities TEXT,
    Major TEXT,
    EnrollmentLimit INTEGER,
    CurrentEnrollment INTEGER  DEFAULT 0
);
INSERT INTO "Courses" VALUES(1,'Introduction to Programming','Learn how to code and build applications.','Software Developer','Computer Science',30,0);
INSERT INTO "Courses" VALUES(2,'Data Structures & Algorithms','Master data structures and algorithms with this course.','Software Engineer','Computer Science',30,0);
CREATE TABLE Enrollments (
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    CourseID INTEGER,
    Grade TEXT,
    FOREIGN KEY (StudentID) REFERENCES Students (StudentID),
    FOREIGN KEY (CourseID) REFERENCES Courses (CourseID)
);
INSERT INTO "Enrollments" VALUES(1,1,1,'A');
INSERT INTO "Enrollments" VALUES(2,2,1,'B');
CREATE TABLE Instructors (
    InstructorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT UNIQUE
);
INSERT INTO "Instructors" VALUES(1,'Tom','Brown','tom.brown@email.com');
INSERT INTO "Instructors" VALUES(2,'Emma','White','emma.white@email.com');
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE
);
INSERT INTO "Students" VALUES(1,'John','Doe','1986-02-05');
INSERT INTO "Students" VALUES(2,'Jane','Smith','1993-07-15');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',2);
INSERT INTO "sqlite_sequence" VALUES('Instructors',2);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',2);
INSERT INTO "sqlite_sequence" VALUES('Courses',2);
COMMIT;
