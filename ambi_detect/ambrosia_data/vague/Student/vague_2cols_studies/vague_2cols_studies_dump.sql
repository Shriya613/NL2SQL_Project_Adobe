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
CREATE TABLE Students (
    ID INTEGER PRIMARY KEY,
    BirthDate DATE,
    Field_Of_Study TEXT,
    Degree_Program TEXT,
    Email TEXT UNIQUE
, fullname TEXT);
INSERT INTO "Students" VALUES(1,'1990-05-15','Computer Science','BSc','john.doe@example.com','John Doe');
INSERT INTO "Students" VALUES(2,'1988-06-23','Biology','MSc','jane.smith@example.com','Jane Smith');
INSERT INTO "Students" VALUES(3,'1992-07-01','Mathematics','PhD','alice.johnson@example.com','Alice Johnson');
INSERT INTO "Students" VALUES(4,'1994-08-10','Physics','BSc','tom.williams@example.com','Tom Williams');
INSERT INTO "Students" VALUES(5,'1996-09-15','Psychology','BA','emma.brown@example.com','Emma Brown');
DELETE FROM "sqlite_sequence";
COMMIT;
