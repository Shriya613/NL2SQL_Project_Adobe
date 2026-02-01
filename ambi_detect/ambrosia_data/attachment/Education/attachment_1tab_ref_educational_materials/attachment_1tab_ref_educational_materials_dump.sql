BEGIN TRANSACTION;
CREATE TABLE Courses(
    CourseID INTEGER PRIMARY KEY AUTOINCREMENT,
    CourseTitle TEXT,
    Department TEXT,
    TeacherID INTEGER,
    FOREIGN KEY(TeacherID) REFERENCES Teachers(TeacherID));
INSERT INTO "Courses" VALUES(1,'Calculus','Mathematics',1);
INSERT INTO "Courses" VALUES(2,'English Literature','Language Arts',2);
CREATE TABLE EducationalMaterials(
    MaterialID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Type TEXT ,
    ISBN TEXT,
    PublisherID INTEGER,
    FOREIGN KEY(PublisherID) REFERENCES Publishers(PublisherID));
INSERT INTO "EducationalMaterials" VALUES(1,'Math Book','Textbook','12345',1);
INSERT INTO "EducationalMaterials" VALUES(2,'Physics Book','Textbook','67890',2);
INSERT INTO "EducationalMaterials" VALUES(3,'Computer Science E-Book','E-Book','13579',1);
INSERT INTO "EducationalMaterials" VALUES(4,'Biology E-Book','E-Book','24680',2);
CREATE TABLE Enrollments(
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    CourseID INTEGER,
    StudentID INTEGER,
    Semester TEXT,
    Year INTEGER,
    Grade TEXT,
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(CourseID) REFERENCES Courses(CourseID));
INSERT INTO "Enrollments" VALUES(1,1,1,'Spring',2021,NULL);
INSERT INTO "Enrollments" VALUES(2,2,2,'Fall',2021,NULL);
CREATE TABLE Publishers(
    PublisherID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    Email TEXT,
    PhoneNumber TEXT);
INSERT INTO "Publishers" VALUES(1,'Pearson','New York','pearson@example.com','1234567890');
INSERT INTO "Publishers" VALUES(2,'Random House','Los Angeles','randomhouse@example.com','0987654321');
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Email TEXT UNIQUE);
INSERT INTO "Students" VALUES(1,'John','Doe','2000-01-01','john@example.com');
INSERT INTO "Students" VALUES(2,'Jane','Smith','1999-02-02','jane@example.com');
CREATE TABLE Teachers(
    TeacherID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT UNIQUE);
INSERT INTO "Teachers" VALUES(1,'Michael','Johnson','michael@example.com');
INSERT INTO "Teachers" VALUES(2,'Emily','Williams','emily@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Publishers',2);
INSERT INTO "sqlite_sequence" VALUES('EducationalMaterials',4);
INSERT INTO "sqlite_sequence" VALUES('Students',2);
INSERT INTO "sqlite_sequence" VALUES('Teachers',2);
INSERT INTO "sqlite_sequence" VALUES('Courses',2);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',2);
COMMIT;
