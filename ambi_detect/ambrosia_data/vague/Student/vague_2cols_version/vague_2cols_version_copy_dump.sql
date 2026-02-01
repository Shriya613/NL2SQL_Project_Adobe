BEGIN TRANSACTION;
CREATE TABLE BookCopies (
    CopyID INTEGER PRIMARY KEY AUTOINCREMENT,
    CourseOfferingID INTEGER REFERENCES CourseOfferings(OfferingID),
    TextbookEditionID INTEGER REFERENCES TextbookEditions(Version),
    Quantity INT );
INSERT INTO "BookCopies" VALUES(1,1,1,30);
INSERT INTO "BookCopies" VALUES(2,2,2,20);
CREATE TABLE CourseOfferings (
    OfferingID INTEGER PRIMARY KEY AUTOINCREMENT,
    CourseCode TEXT,
    Semester TEXT,
    Year INT ,
    InstructorID INTEGER REFERENCES Instructors(ID));
INSERT INTO "CourseOfferings" VALUES(1,'MATH-101','Fall',2021,1);
INSERT INTO "CourseOfferings" VALUES(2,'PHYS-101','Spring',2021,2);
CREATE TABLE Enrollments (
    RegistrationID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER REFERENCES Students(ID),
    CourseCode TEXT,
    Grade CHAR(2));
INSERT INTO "Enrollments" VALUES(1,1,'MATH-101','A');
INSERT INTO "Enrollments" VALUES(2,2,'PHYS-101','B');
CREATE TABLE Instructors (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Department TEXT);
INSERT INTO "Instructors" VALUES(1,'Alice','Johnson','Mathematics');
INSERT INTO "Instructors" VALUES(2,'Bob','Williams','Physics');
CREATE TABLE Students (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT UNIQUE);
INSERT INTO "Students" VALUES(1,'John','Doe','johndoe@example.com');
INSERT INTO "Students" VALUES(2,'Jane','Smith','janesmith@example.com');
CREATE TABLE "TextbookEditions" (
    Version INTEGER PRIMARY KEY AUTOINCREMENT,
    Publisher TEXT,
    ISBN TEXT,
    Textbooks TEXT);
INSERT INTO "TextbookEditions" VALUES(1,'Pearson Education','9780321606389','Introduction to Algorithms');
INSERT INTO "TextbookEditions" VALUES(2,'Wiley Publishing','9780470689571','Fundamentals of Physics');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',2);
INSERT INTO "sqlite_sequence" VALUES('Instructors',2);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',2);
INSERT INTO "sqlite_sequence" VALUES('CourseOfferings',2);
INSERT INTO "sqlite_sequence" VALUES('BookCopies',2);
INSERT INTO "sqlite_sequence" VALUES('TextbookEditions',2);
COMMIT;
