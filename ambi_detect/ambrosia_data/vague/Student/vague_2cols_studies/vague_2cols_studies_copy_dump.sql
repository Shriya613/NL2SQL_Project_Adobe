BEGIN TRANSACTION;
CREATE TABLE Courses(
    ID INTEGER PRIMARY KEY,
    Title TEXT,
    Description TEXT,
    Credit INT 
);
INSERT INTO "Courses" VALUES(1,'Introduction to Computer Programming','A fundamental course on programming concepts.',4);
INSERT INTO "Courses" VALUES(2,'Cellular Biology','An advanced course covering cell structure and function.',3);
INSERT INTO "Courses" VALUES(3,'Quantum Mechanics','An introductory course on quantum mechanics principles.',3);
INSERT INTO "Courses" VALUES(4,'Organic Chemistry','A comprehensive course on organic chemistry reactions.',4);
INSERT INTO "Courses" VALUES(5,'Abnormal Psychology','Exploring various psychological disorders and their treatment.',3);
CREATE TABLE Enrolments(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    CourseID INTEGER,
    Semester DATE,
    FOREIGN KEY(StudentID) REFERENCES Students(ID),
    FOREIGN KEY(CourseID) REFERENCES Courses(ID)
);
CREATE TABLE Marks(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    EnrollmentID INTEGER,
    Grade CHAR(2),
    DateAssigned DATE,
    FOREIGN KEY(EnrollmentID) REFERENCES Enrolments(ID)
);
CREATE TABLE "Students" (
    ID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    Studies TEXT,
    Email TEXT UNIQUE
);
INSERT INTO "Students" VALUES(1,'John','Doe','1990-05-15','Computer Science','john.doe@example.com');
INSERT INTO "Students" VALUES(2,'Jane','Smith','1988-06-23','Biology','jane.smith@example.com');
INSERT INTO "Students" VALUES(3,'Alice','Johnson','1992-07-01','Mathematics','alice.johnson@example.com');
INSERT INTO "Students" VALUES(4,'Tom','Williams','1994-08-10','Physics','tom.williams@example.com');
INSERT INTO "Students" VALUES(5,'Emma','Brown','1996-09-15','Psychology','emma.brown@example.com');
DELETE FROM "sqlite_sequence";
COMMIT;
