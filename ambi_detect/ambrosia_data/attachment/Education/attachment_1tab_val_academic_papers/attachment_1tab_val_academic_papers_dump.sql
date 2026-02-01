BEGIN TRANSACTION;
CREATE TABLE Conferences(
    ConferenceID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    Year INT
);
INSERT INTO "Conferences" VALUES(1,'Conference A','Location A',2021);
INSERT INTO "Conferences" VALUES(2,'Conference B','Location B',2022);
CREATE TABLE CourseEnrollments(
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    CourseID INTEGER,
    Grade CHAR(2),
    Semester YEAR,
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
);
INSERT INTO "CourseEnrollments" VALUES(1,1,1,'A',2020);
INSERT INTO "CourseEnrollments" VALUES(2,2,1,'B+',2021);
CREATE TABLE Courses(
    CourseID INTEGER PRIMARY KEY AUTOINCREMENT,
    CourseTitle TEXT,
    InstructorID INTEGER,
    FOREIGN KEY (InstructorID) REFERENCES Professors(ProfessorID)
);
INSERT INTO "Courses" VALUES(1,'Introduction to Computer Science',1);
INSERT INTO "Courses" VALUES(2,'Calculus I',2);
CREATE TABLE Papers(
    PaperID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    AbstractText TEXT,
    Academic_PaperType TEXT ,
    Citation_Style TEXT,
    AuthorID INTEGER,
    FOREIGN KEY (AuthorID) REFERENCES Students(StudentID),
    FOREIGN KEY (AuthorID) REFERENCES Professors(ProfessorID)
);
INSERT INTO "Papers" VALUES(1,'Sample Research Paper','This is a sample abstract.','Research Paper','APA',1);
INSERT INTO "Papers" VALUES(2,'Another Sample Research Paper','Here is another sample abstract.','Research Paper','Chicago',1);
INSERT INTO "Papers" VALUES(3,'Sample Review Article','Below is an example review article abstract.','Review Article','APA',2);
INSERT INTO "Papers" VALUES(4,'Yet Another Review Article','Check out this other review article abstract.','Review Article','MLA',2);
CREATE TABLE Presentations(
    PresentationID INTEGER PRIMARY KEY AUTOINCREMENT,
    PaperID INTEGER,
    ConferenceID INTEGER,
    Date DATE,
    FOREIGN KEY (PaperID) REFERENCES Papers(PaperID),
    FOREIGN KEY (ConferenceID) REFERENCES Conferences(ConferenceID)
);
INSERT INTO "Presentations" VALUES(1,1,1,'2021-06-15');
INSERT INTO "Presentations" VALUES(2,2,2,'2022-06-15');
CREATE TABLE Professors(
    ProfessorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Department VARCHAR(100)
);
INSERT INTO "Professors" VALUES(1,'Michael','Johnson','Physics');
INSERT INTO "Professors" VALUES(2,'Emily','Williams','Biology');
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Major VARCHAR(100)
);
INSERT INTO "Students" VALUES(1,'John','Doe','1987-06-15','Computer Science');
INSERT INTO "Students" VALUES(2,'Jane','Smith','1990-08-30','Mathematics');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',2);
INSERT INTO "sqlite_sequence" VALUES('Professors',2);
INSERT INTO "sqlite_sequence" VALUES('Papers',4);
INSERT INTO "sqlite_sequence" VALUES('Conferences',2);
INSERT INTO "sqlite_sequence" VALUES('Presentations',2);
INSERT INTO "sqlite_sequence" VALUES('CourseEnrollments',2);
INSERT INTO "sqlite_sequence" VALUES('Courses',2);
COMMIT;
