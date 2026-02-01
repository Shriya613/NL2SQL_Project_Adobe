BEGIN TRANSACTION;
CREATE TABLE Courses(
    CourseID INTEGER PRIMARY KEY,
    Title TEXT,
    Description TEXT,
    InstructorID INTEGER,
    FOREIGN KEY (InstructorID) REFERENCES Instructors(InstructorID)
);
INSERT INTO "Courses" VALUES(1,'Calculus I','Introduction to Calculus',1);
INSERT INTO "Courses" VALUES(2,'General Physics','Introduction to Classical Mechanics',2);
INSERT INTO "Courses" VALUES(3,'Introductory Biology','Cellular and Molecular Biology',3);
INSERT INTO "Courses" VALUES(4,'Modern English Literature','Survey of British Literature since 1900',4);
INSERT INTO "Courses" VALUES(5,'World History','Global Civilizations Since 1500 CE',5);
INSERT INTO "Courses" VALUES(6,'Human Geography','Regional Geographies of the World',6);
CREATE TABLE Enrollments(
    EnrollmentID INTEGER PRIMARY KEY,
    StudentID INTEGER,
    CourseID INTEGER,
    Grade TEXT ,
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
);
INSERT INTO "Enrollments" VALUES(1,1,1,'A');
INSERT INTO "Enrollments" VALUES(2,1,2,'B+');
INSERT INTO "Enrollments" VALUES(3,2,2,'A-');
INSERT INTO "Enrollments" VALUES(4,3,3,'B');
INSERT INTO "Enrollments" VALUES(5,4,4,'A');
INSERT INTO "Enrollments" VALUES(6,5,5,'A');
INSERT INTO "Enrollments" VALUES(7,6,6,'B');
CREATE TABLE Instructors(
    InstructorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Department TEXT
);
INSERT INTO "Instructors" VALUES(1,'Professor','Adams','Mathematics');
INSERT INTO "Instructors" VALUES(2,'Dr.','Johnson','Physics');
INSERT INTO "Instructors" VALUES(3,'Assistant','Williams','Biology');
INSERT INTO "Instructors" VALUES(4,'Associate','Taylor','English Literature');
INSERT INTO "Instructors" VALUES(5,'Professor','Thompson','History');
INSERT INTO "Instructors" VALUES(6,'Dr.','Martin','Geography');
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY,
    Personal_Computing_Devices TEXT,
    Operating_System TEXT,
    EmailAddress TEXT UNIQUE
, FullName TEXT);
INSERT INTO "Students" VALUES(1,'Tablet','Windows 10','john.doe@example.com','John Doe');
INSERT INTO "Students" VALUES(2,'Phone','iOS','jane.smith@example.com','Jane Smith');
INSERT INTO "Students" VALUES(3,'Laptop','Windows 10','mike.brown@example.com','Mike Brown');
INSERT INTO "Students" VALUES(4,'Desktop','macOS','emily.white@example.com','Emily White');
INSERT INTO "Students" VALUES(5,'Tablet','Android','david.green@example.com','David Green');
INSERT INTO "Students" VALUES(6,'Laptop','Linux','amy.black@example.com','Amy Black');
COMMIT;
