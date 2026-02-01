BEGIN TRANSACTION;
CREATE TABLE Course(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Code TEXT,
    Description TEXT,
    Credit HONORS);
INSERT INTO "Course" VALUES(1,'CS101','Introduction to Computer Science',4);
INSERT INTO "Course" VALUES(2,'ENG101','English Language Skills',3);
INSERT INTO "Course" VALUES(3,'MAT201','Calculus I',4);
INSERT INTO "Course" VALUES(4,'SOC301','Sociology of Education',3);
INSERT INTO "Course" VALUES(5,'PHY101','General Physics',4);
CREATE TABLE Degree(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER REFERENCES Students(ID) ON DELETE CASCADE,
    DegreeType TEXT,
    GraduationDate DATE,
    Grade TEXT);
INSERT INTO "Degree" VALUES(1,1,'PhD','2021-12-01','A');
INSERT INTO "Degree" VALUES(2,2,'MSc','2022-12-01','A+');
INSERT INTO "Degree" VALUES(3,3,'MA','2023-12-01','A');
INSERT INTO "Degree" VALUES(4,4,'BA','2024-12-01','A+');
INSERT INTO "Degree" VALUES(5,5,'BSc','2025-12-01','A');
CREATE TABLE Dissertations(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Abstract TEXT,
    Date DATE,
    Academic_Supervisor TEXT);
INSERT INTO "Dissertations" VALUES(1,'Understanding Quantum Mechanics','Exploring the principles of quantum mechanics...','2021-01-01','Dr. Jane Smith');
INSERT INTO "Dissertations" VALUES(2,'Advances in Artificial Intelligence','Investigating the latest developments in AI technology...','2021-01-02','Dr. John Doe');
CREATE TABLE Student_Course(
    StudentID INTEGER REFERENCES Students(ID) ON DELETE CASCADE,
    CourseID INTEGER REFERENCES Course(ID) ON DELETE CASCADE,
    Marks INT ,
    PRIMARY KEY(StudentID, CourseID));
INSERT INTO "Student_Course" VALUES(1,1,85);
INSERT INTO "Student_Course" VALUES(1,2,90);
INSERT INTO "Student_Course" VALUES(1,3,88);
INSERT INTO "Student_Course" VALUES(2,4,92);
INSERT INTO "Student_Course" VALUES(2,5,80);
CREATE TABLE Student_Supervision(
    StudentID INTEGER REFERENCES Students(ID) ON DELETE CASCADE,
    SupervisorID INTEGER REFERENCES Supervisors(ID) ON DELETE CASCADE,
    DissertationID INTEGER REFERENCES Dissertations(ID) ON DELETE CASCADE,
    ThesisProposalID INTEGER REFERENCES Thesis_Proposals(ID) ON DELETE CASCADE,
    PRIMARY KEY(StudentID, SupervisorID, DissertationID, ThesisProposalID));
CREATE TABLE Students(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Email TEXT,
    PhoneNumber VARCHAR(15),
    EnrolmentDate DATE);
INSERT INTO "Students" VALUES(1,'John Doe','john@example.com','1234567890','2021-01-01');
INSERT INTO "Students" VALUES(2,'Jane Smith','jane@example.com','9876543210','2021-02-01');
INSERT INTO "Students" VALUES(3,'Alice Johnson','alice@example.com','0987654321','2021-03-01');
INSERT INTO "Students" VALUES(4,'Bob Brown','bob@example.com','1234567890','2021-04-01');
INSERT INTO "Students" VALUES(5,'Charlie Davis','charlie@example.com','0987654321','2021-05-01');
CREATE TABLE Supervisors(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FullName TEXT);
INSERT INTO "Supervisors" VALUES(1,'Dr. Jane Smith');
INSERT INTO "Supervisors" VALUES(2,'Dr. John Doe');
INSERT INTO "Supervisors" VALUES(3,'Dr. Alice Johnson');
INSERT INTO "Supervisors" VALUES(4,'Dr. Bob Brown');
INSERT INTO "Supervisors" VALUES(5,'Dr. Charlie Davis');
CREATE TABLE Thesis_Proposals(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Abstract TEXT,
    Date DATE,
    Academic_Supervisor TEXT);
INSERT INTO "Thesis_Proposals" VALUES(1,'Educational Technology Trends','An analysis of current trends in educational technology...','2021-02-01','Dr. Jane Smith');
INSERT INTO "Thesis_Proposals" VALUES(2,'Learning Styles and Effective Teaching Methods','Examining various learning styles and their impact on teaching methods...','2021-02-02','Dr. Alice Johnson');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Supervisors',5);
INSERT INTO "sqlite_sequence" VALUES('Dissertations',2);
INSERT INTO "sqlite_sequence" VALUES('Thesis_Proposals',2);
INSERT INTO "sqlite_sequence" VALUES('Degree',5);
INSERT INTO "sqlite_sequence" VALUES('Course',5);
COMMIT;
