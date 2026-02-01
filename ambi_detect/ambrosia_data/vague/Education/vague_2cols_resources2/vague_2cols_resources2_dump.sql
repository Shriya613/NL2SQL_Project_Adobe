BEGIN TRANSACTION;
CREATE TABLE Courses (
    CourseID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Description TEXT,
    TeacherID INTEGER,
    FOREIGN KEY(TeacherID) REFERENCES Teachers(TeacherID)
);
INSERT INTO "Courses" VALUES(1,'Calculus I','Introductory Calculus with an emphasis on limits and derivatives.',1);
INSERT INTO "Courses" VALUES(2,'General Physics','Introduction to classical physics covering mechanics and thermodynamics.',2);
INSERT INTO "Courses" VALUES(3,'Biological Evolution','A comprehensive study of evolutionary processes in biology.',3);
INSERT INTO "Courses" VALUES(4,'Organic Chemistry I','Introduction to organic chemistry focusing on fundamental concepts and reactions.',4);
INSERT INTO "Courses" VALUES(5,'World Literature','Exploration of world literature from various cultures and time periods.',5);
CREATE TABLE Enrollments (
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    CourseID INTEGER,
    Grade TEXT,
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(CourseID) REFERENCES Courses(CourseID)
);
INSERT INTO "Enrollments" VALUES(1,1,1,'A');
INSERT INTO "Enrollments" VALUES(2,2,2,'B+');
INSERT INTO "Enrollments" VALUES(3,3,3,'A-');
INSERT INTO "Enrollments" VALUES(4,4,4,'B');
INSERT INTO "Enrollments" VALUES(5,5,5,'A');
CREATE TABLE Exams (
    ExamID INTEGER PRIMARY KEY AUTOINCREMENT,
    CourseID INTEGER,
    Date DATE,
    TotalMarks INT,
    PassMark INT,
    FOREIGN KEY(CourseID) REFERENCES Courses(CourseID)
);
INSERT INTO "Exams" VALUES(1,1,'2021-12-15',100,60);
INSERT INTO "Exams" VALUES(2,2,'2021-12-20',100,65);
INSERT INTO "Exams" VALUES(3,3,'2021-12-17',100,70);
INSERT INTO "Exams" VALUES(4,4,'2021-12-22',100,75);
INSERT INTO "Exams" VALUES(5,5,'2021-12-18',100,80);
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Age INT
);
INSERT INTO "Students" VALUES(1,'John','Doe',16);
INSERT INTO "Students" VALUES(2,'Jane','Smith',17);
INSERT INTO "Students" VALUES(3,'Tom','Brown',18);
INSERT INTO "Students" VALUES(4,'Alice','Green',19);
INSERT INTO "Students" VALUES(5,'Bob','White',20);
CREATE TABLE Teachers (
    TeacherID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    SubjectTaught TEXT
);
INSERT INTO "Teachers" VALUES(1,'Mr.','Davis','Mathematics');
INSERT INTO "Teachers" VALUES(2,'Ms.','Martin','Physics');
INSERT INTO "Teachers" VALUES(3,'Dr.','Parker','Biology');
INSERT INTO "Teachers" VALUES(4,'Professor','Lee','Chemistry');
INSERT INTO "Teachers" VALUES(5,'Mrs.','Hall','English Literature');
CREATE TABLE Textbooks_and_Software (
    ItemID INTEGER PRIMARY KEY AUTOINCREMENT,
    TextbookTitle TEXT,
    SoftwareName TEXT,
    CourseID INTEGER,
    Quantity INT,
    Price DECIMAL(10,2),
    FOREIGN KEY(CourseID) REFERENCES Courses(CourseID)
);
INSERT INTO "Textbooks_and_Software" VALUES(1,'Principia Mathematica',NULL,1,10,49.99);
INSERT INTO "Textbooks_and_Software" VALUES(2,'Fundamentals of Biochemistry',NULL,3,5,59.99);
INSERT INTO "Textbooks_and_Software" VALUES(3,'The Elements of Style','Grammarly Premium',5,15,39.99);
INSERT INTO "Textbooks_and_Software" VALUES(4,'Inorganic Chemistry: Structure and Reactivity',NULL,4,7,69.99);
INSERT INTO "Textbooks_and_Software" VALUES(5,'Anthropology: A Brief Introduction','Adobe Creative Cloud',2,8,79.99);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Teachers',5);
INSERT INTO "sqlite_sequence" VALUES('Courses',5);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',5);
INSERT INTO "sqlite_sequence" VALUES('Textbooks_and_Software',5);
INSERT INTO "sqlite_sequence" VALUES('Exams',5);
COMMIT;
