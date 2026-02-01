BEGIN TRANSACTION;
CREATE TABLE Courses (
    CourseID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Description TEXT,
    DepartmentID INTEGER,
    Credits INTEGER,
    InstructorID INTEGER,
    FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID),
    FOREIGN KEY (InstructorID) REFERENCES Instructors(InstructorID));
INSERT INTO "Courses" VALUES(1,'Introduction to Computer Science','An introductory course covering programming fundamentals.',1,4,1);
INSERT INTO "Courses" VALUES(2,'Calculus I','A course covering fundamental calculus topics.',2,4,2);
CREATE TABLE Departments (
    DepartmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT,
    Location TEXT);
INSERT INTO "Departments" VALUES(1,'Computer Science','The department focuses on teaching and research in computer science.','Building A, Room 201');
INSERT INTO "Departments" VALUES(2,'Mathematics','The department offers a wide range of mathematical courses and research opportunities.','Building B, Room 101');
CREATE TABLE Enrollments (
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    CourseID INTEGER,
    Semester YEAR,
    Grade CHAR(2),
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID));
INSERT INTO "Enrollments" VALUES(1,1,1,2021,'A');
INSERT INTO "Enrollments" VALUES(2,2,2,2021,'B+');
CREATE TABLE Instructors (
    InstructorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    HireDate DATE,
    Email VARCHAR(255) UNIQUE);
INSERT INTO "Instructors" VALUES(1,'Michael','Johnson','2017-07-01','michael.johnson@example.com');
INSERT INTO "Instructors" VALUES(2,'Emma','Williams','2019-08-01','emma.williams@example.com');
CREATE TABLE Programs (
    ProgramID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT,
    Amount_Per_Semester REAL,
    Annual_Cost REAL,
    ProgramType TEXT );
INSERT INTO "Programs" VALUES(1,'Computer Science','A program focusing on computer science concepts and skills.',2000.0,8000.0,'Undergraduate');
INSERT INTO "Programs" VALUES(2,'Mathematics','A comprehensive mathematics program covering various branches.',2000.0,8000.0,'Undergraduate');
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Email VARCHAR(255) UNIQUE);
INSERT INTO "Students" VALUES(1,'John','Doe','1980-06-03','john.doe@example.com');
INSERT INTO "Students" VALUES(2,'Jane','Smith','1992-12-15','jane.smith@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',2);
INSERT INTO "sqlite_sequence" VALUES('Programs',2);
INSERT INTO "sqlite_sequence" VALUES('Departments',2);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',2);
INSERT INTO "sqlite_sequence" VALUES('Instructors',2);
INSERT INTO "sqlite_sequence" VALUES('Courses',2);
COMMIT;
