BEGIN TRANSACTION;
CREATE TABLE Authors(
    AuthorID INTEGER PRIMARY KEY,
    Name TEXT,
    DateOfBirth DATE,
    Country TEXT
);
INSERT INTO "Authors" VALUES(1,'Emily Dickinson','1830-12-10','USA');
INSERT INTO "Authors" VALUES(2,'John Doe','1975-06-22','Canada');
CREATE TABLE CourseTeachings(
    CourseTeachingID INTEGER PRIMARY KEY,
    CourseSubjectID INTEGER,
    TeacherID INTEGER,
    Semester CHAR(20),
    Year INTEGER,
    StartDate DATE,
    EndDate DATE,
    FOREIGN KEY(CourseSubjectID) REFERENCES Subjects(SubjectID),
    FOREIGN KEY(TeacherID) REFERENCES Students(StudentID)
);
INSERT INTO "CourseTeachings" VALUES(1,1,1,'Fall',2021,'2021-08-25','2021-12-21');
INSERT INTO "CourseTeachings" VALUES(2,2,2,'Spring',2022,'2022-01-13','2022-05-13');
CREATE TABLE Schools(
    SchoolID INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    City TEXT,
    State TEXT,
    ZipCode CHAR(5),
    FoundedYear INTEGER
);
INSERT INTO "Schools" VALUES(1,'Harvard University','Massachusetts Hall','Cambridge','MA','02138',1636);
INSERT INTO "Schools" VALUES(2,'Yale University','Old Campus','New Haven','CT','06510',1701);
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Birthdate DATE,
    EnrollmentDate DATE,
    SchoolID INTEGER,
    FOREIGN KEY(SchoolID) REFERENCES Schools(SchoolID)
);
INSERT INTO "Students" VALUES(1,'Alice','Smith','1990-02-15','2006-08-25',1);
INSERT INTO "Students" VALUES(2,'Charlie','Doe','1992-03-20','2008-08-25',2);
CREATE TABLE Subjects(
    SubjectID INTEGER PRIMARY KEY,
    SubjectName TEXT,
    Description TEXT
);
INSERT INTO "Subjects" VALUES(1,'Mathematics','The study of numbers, quantity, structure, space, and change.');
INSERT INTO "Subjects" VALUES(2,'Literature','The study of written works.');
CREATE TABLE WrittenWorks(
    WorkID INTEGER PRIMARY KEY,
    Title TEXT,
    Type TEXT, -- Can be Essay, Poem or others
    PublicationDate DATE,
    WordCount INTEGER,
    AuthorID INTEGER,
    FOREIGN KEY(AuthorID) REFERENCES Authors(AuthorID)
);
INSERT INTO "WrittenWorks" VALUES(1,'My Life','Essay','2000-01-01',1000,1);
INSERT INTO "WrittenWorks" VALUES(2,'Nature','Poetry','1890-01-01',200,1);
INSERT INTO "WrittenWorks" VALUES(3,'A Different Life','Essay','2010-01-01',1200,2);
COMMIT;
