BEGIN TRANSACTION;
CREATE TABLE Courses (
    CourseID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    EducationalMethod TEXT ,
    CurriculumContent TEXT,
    InstructorID INTEGER,
    FOREIGN KEY (InstructorID) REFERENCES Instructors(InstructorID)
);
INSERT INTO "Courses" VALUES(1,'Calculus I','Online Learning','Mathematics',1);
INSERT INTO "Courses" VALUES(2,'World History','Classroom Teaching','History',5);
INSERT INTO "Courses" VALUES(3,'Introduction to Physics','Online Learning','Physics',2);
INSERT INTO "Courses" VALUES(4,'Discrete Mathematics','Classroom Teaching','Mathematics',1);
INSERT INTO "Courses" VALUES(5,'Creative Writing','Online Learning','English',4);
CREATE TABLE Enrollments (
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    CourseID INTEGER,
    StudentID INTEGER,
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID),
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID)
);
INSERT INTO "Enrollments" VALUES(1,1,1);
INSERT INTO "Enrollments" VALUES(2,3,4);
INSERT INTO "Enrollments" VALUES(3,2,2);
INSERT INTO "Enrollments" VALUES(4,4,3);
INSERT INTO "Enrollments" VALUES(5,5,5);
CREATE TABLE Instructors (
    InstructorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Department TEXT
);
INSERT INTO "Instructors" VALUES(1,'Dr.','Jones','Math');
INSERT INTO "Instructors" VALUES(2,'Prof.','Black','Physics');
INSERT INTO "Instructors" VALUES(3,'Assoc. Prof.','Gray','Computer Science');
INSERT INTO "Instructors" VALUES(4,'Lecturer','Hill','English');
INSERT INTO "Instructors" VALUES(5,'Adjunct Professor','Wilson','History');
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DOB TEXT
);
INSERT INTO "Students" VALUES(1,'John','Doe','1990-08-23');
INSERT INTO "Students" VALUES(2,'Jane','Smith','1995-06-17');
INSERT INTO "Students" VALUES(3,'Alice','Brown','1992-10-05');
INSERT INTO "Students" VALUES(4,'Michael','White','1993-09-22');
INSERT INTO "Students" VALUES(5,'Emily','Green','1991-11-11');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Instructors',5);
INSERT INTO "sqlite_sequence" VALUES('Courses',5);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',5);
COMMIT;
