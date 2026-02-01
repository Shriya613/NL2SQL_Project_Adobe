BEGIN TRANSACTION;
CREATE TABLE Awards(
    AwardType TEXT PRIMARY KEY  ,
    Description TEXT);
INSERT INTO "Awards" VALUES('Diploma','This award represents completion of a program.');
CREATE TABLE Courses(
    CourseID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    CreditHours INTEGER,
    CompletionRequirements TEXT, -- can contain multiple values like Passing exam, Project submission etc.
    AwardProgram TEXT , -- contains single value e.g., Diploma, Certificate etc.
    UniqueConstraint );
INSERT INTO "Courses" VALUES(1,'Introduction to Programming',3,'Passing Exam','Diploma',NULL);
INSERT INTO "Courses" VALUES(2,'Advanced Mathematics',4,'Project Submission','Diploma',NULL);
INSERT INTO "Courses" VALUES(3,'Web Development Basics',3,'Passing Exam','Certificate',NULL);
INSERT INTO "Courses" VALUES(4,'Database Management',4,'Project Submission','Certificate',NULL);
CREATE TABLE Enrollments(
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    CourseID INTEGER,
    StudentID INTEGER,
    TeacherID INTEGER,
    DateEnrolled DATE,
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(TeacherID) REFERENCES Teachers(TeacherID),
    FOREIGN KEY(CourseID) REFERENCES Courses(CourseID));
INSERT INTO "Enrollments" VALUES(1,1,1,2,'2020-08-15');
INSERT INTO "Enrollments" VALUES(2,2,1,2,'2020-08-15');
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    Email VARCHAR(255) UNIQUE);
INSERT INTO "Students" VALUES(1,'John','Doe','1990-01-01','john.doe@example.com');
CREATE TABLE Teachers(
    TeacherID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    HireDate DATE,
    Email VARCHAR(255) UNIQUE);
INSERT INTO "Teachers" VALUES(1,'Jane','Smith','2010-06-01','jane.smith@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Courses',4);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',2);
COMMIT;
