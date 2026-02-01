BEGIN TRANSACTION;
CREATE TABLE Courses (
    CourseID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Description TEXT,
    Credit INT );
INSERT INTO "Courses" VALUES(1,'Calculus I','Introduction to Calculus with focus on limits and derivatives.',3);
INSERT INTO "Courses" VALUES(2,'General Physics I','Fundamental principles of physics including mechanics, heat, sound, light, and electricity.',4);
INSERT INTO "Courses" VALUES(3,'Human Biology','An introduction to human biology covering cellular structure, genetics, physiological systems, and health issues.',3);
INSERT INTO "Courses" VALUES(4,'World History II','Exploration and colonization from 15th century to present day.',4);
INSERT INTO "Courses" VALUES(5,'American Literature','A survey of American literature from its beginnings through the early twentieth century.',3);
CREATE TABLE Enrollments (
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    CourseID INTEGER,
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(CourseID) REFERENCES Courses(CourseID));
INSERT INTO "Enrollments" VALUES(1,1,1);
INSERT INTO "Enrollments" VALUES(2,2,2);
INSERT INTO "Enrollments" VALUES(3,3,3);
INSERT INTO "Enrollments" VALUES(4,4,4);
INSERT INTO "Enrollments" VALUES(5,5,5);
CREATE TABLE Grades (
    GradeID INTEGER PRIMARY KEY AUTOINCREMENT,
    EnrollmentID INTEGER,
    FinalGrade DECIMAL(2,1),
    FOREIGN KEY(EnrollmentID) REFERENCES Enrollments(EnrollmentID));
INSERT INTO "Grades" VALUES(1,1,92);
INSERT INTO "Grades" VALUES(2,2,88);
INSERT INTO "Grades" VALUES(3,3,95);
INSERT INTO "Grades" VALUES(4,4,90);
INSERT INTO "Grades" VALUES(5,5,85);
CREATE TABLE "Programs" (
    ProgramID INTEGER PRIMARY KEY AUTOINCREMENT,
    ProgramName TEXT,
    Study_Plan TEXT,
    Requirements TEXT);
INSERT INTO "Programs" VALUES(1,'Computer Science','Introductory Computer Science, Data Structures, Algorithms, Operating Systems','Minimum cumulative GPA of 2.5, Minimum 2 years of college work.');
INSERT INTO "Programs" VALUES(2,'Mechanical Engineering','Engineering Mechanics, Materials Science, Thermodynamics, Fluid Mechanics','Minimum cumulative GPA of 2.7, Minimum 2 years of college work.');
INSERT INTO "Programs" VALUES(3,'Electrical Engineering','Circuit Analysis, Digital Logic Design, Signals and Systems, Power Electronics','Minimum cumulative GPA of 2.6, Minimum 2 years of college work.');
INSERT INTO "Programs" VALUES(4,'Civil Engineering','Structural Analysis, Geotechnical Engineering, Transportation Engineering, Environmental Engineering','Minimum cumulative GPA of 2.8, Minimum 2 years of college work.');
INSERT INTO "Programs" VALUES(5,'Business Administration','Principles of Accounting, Principles of Marketing, Organizational Behavior, Finance','Minimum cumulative GPA of 2.5, Minimum 2 years of college work.');
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE);
INSERT INTO "Students" VALUES(1,'John','Doe','2003-06-18');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2002-07-20');
INSERT INTO "Students" VALUES(3,'Michael','Johnson','2001-08-25');
INSERT INTO "Students" VALUES(4,'Emily','Williams','2000-09-10');
INSERT INTO "Students" VALUES(5,'David','Brown','1999-10-15');
CREATE TABLE Teachers (
    TeacherID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Subject TEXT);
INSERT INTO "Teachers" VALUES(1,'Mr.','Smith','Mathematics');
INSERT INTO "Teachers" VALUES(2,'Ms.','Johnson','Physics');
INSERT INTO "Teachers" VALUES(3,'Dr.','Williams','Biology');
INSERT INTO "Teachers" VALUES(4,'Professor','Baker','History');
INSERT INTO "Teachers" VALUES(5,'Miss','Thompson','English Literature');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Teachers',5);
INSERT INTO "sqlite_sequence" VALUES('Courses',5);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',5);
INSERT INTO "sqlite_sequence" VALUES('Grades',5);
INSERT INTO "sqlite_sequence" VALUES('Programs',5);
COMMIT;
