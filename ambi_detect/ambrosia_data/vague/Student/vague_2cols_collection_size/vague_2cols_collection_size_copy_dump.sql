BEGIN TRANSACTION;
CREATE TABLE Courses (
    CourseID INTEGER PRIMARY KEY AUTOINCREMENT,
    CourseTitle TEXT,
    Instructor TEXT,
    StartDate DATE,
    EndDate DATE
);
INSERT INTO "Courses" VALUES(1,'Introduction to Programming','Dr. Jane Smith','2023-01-01','2023-05-31');
INSERT INTO "Courses" VALUES(2,'Advanced Database Management','Prof. John Doe','2023-01-15','2023-05-15');
CREATE TABLE Enrollments (
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    CourseID INTEGER,
    Semester TEXT,
    Year INTEGER,
    FOREIGN KEY (StudentID) REFERENCES Students(ID),
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
);
INSERT INTO "Enrollments" VALUES(1,1,1,'Fall',2023);
INSERT INTO "Enrollments" VALUES(2,2,2,'Spring',2023);
CREATE TABLE Grades (
    GradeID INTEGER PRIMARY KEY AUTOINCREMENT,
    EnrollmentID INTEGER,
    FinalGrade DECIMAL(2,1),
    FOREIGN KEY (EnrollmentID) REFERENCES Enrollments(EnrollmentID)
);
INSERT INTO "Grades" VALUES(1,1,88.5);
INSERT INTO "Grades" VALUES(2,2,92.3);
CREATE TABLE "Library" (
    LibraryID INTEGER PRIMARY KEY AUTOINCREMENT,
    Collection_Size INTEGER,
    Library TEXT,
    Accessible TEXT,
    OpeningHours TEXT
);
INSERT INTO "Library" VALUES(1,500,'University Main Library','Yes','Monday - Friday: 8am - 6pm, Saturday: 10am - 2pm, Sunday: Closed');
INSERT INTO "Library" VALUES(2,300,'Engineering Library','No','Monday - Thursday: 8am - 8pm, Friday: 8am - 6pm, Weekends: Closed');
CREATE TABLE Students (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Email TEXT,
    PhoneNumber VARCHAR(15)
);
INSERT INTO "Students" VALUES(1,'John','Doe','2003-06-18','john.doe@example.com','+1234567890');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2002-05-20','jane.smith@example.com','+0987654321');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',2);
INSERT INTO "sqlite_sequence" VALUES('Courses',2);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',2);
INSERT INTO "sqlite_sequence" VALUES('Grades',2);
INSERT INTO "sqlite_sequence" VALUES('Library',2);
COMMIT;
