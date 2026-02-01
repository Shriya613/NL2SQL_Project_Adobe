BEGIN TRANSACTION;
CREATE TABLE Attendance(
    AttendenceID INTEGER PRIMARY KEY AUTOINCREMENT,
    EnrollmentID INT,
    NumberOfDaysPresent INT,
    TotalNumberOfClasses INT,
    ExcusedAbsences INT,
    UnexcusedAbsences INT,
    FOREIGN KEY (EnrollmentID) REFERENCES Enrollment(EnrollmentID));
INSERT INTO "Attendance" VALUES(1,1,20,25,2,1);
INSERT INTO "Attendance" VALUES(2,2,22,25,1,0);
CREATE TABLE "Courses"(
    CourseID INTEGER PRIMARY KEY AUTOINCREMENT,
    Curriculum_Topics TEXT,
    CourseName TEXT,
    TeachersName TEXT);
INSERT INTO "Courses" VALUES(1,'Introduction to Math','Math 101','Mr. Johnson');
INSERT INTO "Courses" VALUES(2,'English Literature','Engl 101','Mrs. Williams');
CREATE TABLE Enrollment(
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INT,
    CourseID INT,
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID));
CREATE TABLE Grades(
    GradeID INTEGER PRIMARY KEY AUTOINCREMENT,
    EnrollmentID INT,
    Grade CHAR(2),
    Comments TEXT,
    FOREIGN KEY (EnrollmentID) REFERENCES Enrollment(EnrollmentID));
INSERT INTO "Grades" VALUES(1,1,'A+','Excellent work on projects.');
INSERT INTO "Grades" VALUES(2,1,'A-','Great effort on final exam.');
INSERT INTO "Grades" VALUES(3,2,'B+','Improved a lot over the semester.');
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Address TEXT);
INSERT INTO "Students" VALUES(1,'John','Smith','1990-08-30','123 Main St');
INSERT INTO "Students" VALUES(2,'Jane','Doe','1992-06-07','456 Elm St');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',2);
INSERT INTO "sqlite_sequence" VALUES('Grades',3);
INSERT INTO "sqlite_sequence" VALUES('Attendance',2);
INSERT INTO "sqlite_sequence" VALUES('Courses',2);
COMMIT;
