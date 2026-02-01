BEGIN TRANSACTION;
CREATE TABLE Courses (
    CourseID INTEGER PRIMARY KEY,
    Title TEXT,
    Description TEXT,
    Department TEXT,
    Credit Hours INTEGER );
INSERT INTO "Courses" VALUES(1,'Calculus I','Introduction to calculus with an emphasis on limits, derivatives, and integrals.','Math',4);
INSERT INTO "Courses" VALUES(2,'General Physics','This course covers classical mechanics, electricity and magnetism, optics, thermodynamics, and modern physics.','Physics',4);
INSERT INTO "Courses" VALUES(3,'American Literature','An examination of American literature through various time periods and genres.','English',3);
INSERT INTO "Courses" VALUES(4,'Human Biology','Exploration of human biology including cell structure and function, genetics, and physiological systems.','Biology',3);
INSERT INTO "Courses" VALUES(5,'World History II','Continuation of World History I, covering events from the fall of Rome until today.','History',3);
CREATE TABLE Enrollments (
    EnrollmentID INTEGER PRIMARY KEY,
    StudentID INTEGER,
    CourseID INTEGER,
    Grade TEXT,
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(CourseID) REFERENCES Courses(CourseID));
CREATE TABLE Professors (
    ProfessorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Department TEXT,
    OfficeNumber TEXT);
INSERT INTO "Professors" VALUES(1,'Dr.','White','Math','A101');
INSERT INTO "Professors" VALUES(2,'Dr.','Black','Physics','B101');
INSERT INTO "Professors" VALUES(3,'Dr.','Green','English','C101');
INSERT INTO "Professors" VALUES(4,'Dr.','Blue','Biology','D101');
INSERT INTO "Professors" VALUES(5,'Dr.','Red','History','E101');
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY,
    DateOfBirth DATE,
    HigherEducationInstitution TEXT,
    AcademicProgram TEXT, FullName TEXT);
INSERT INTO "Students" VALUES(1,'2000-09-15','University','Business Administration','John Doe');
INSERT INTO "Students" VALUES(2,'1998-10-23','University','Computer Science','Jane Smith');
INSERT INTO "Students" VALUES(3,'1996-11-27','College','Business Administration','Michael Johnson');
INSERT INTO "Students" VALUES(4,'1995-12-11','College','Mathematics','Emily Williams');
INSERT INTO "Students" VALUES(5,'1994-02-02','High School','Physics','David Brown');
COMMIT;
