BEGIN TRANSACTION;
CREATE TABLE "Assessments"(
    AssessmentID INTEGER PRIMARY KEY,
    AssessmentMethod TEXT ,
    Assessor TEXT,
    TotalMarks INTEGER,
    PassingMark INTEGER,
    DateOfAssessment DATETIME
);
INSERT INTO "Assessments" VALUES(1,'Exam','Professor Smith',100,60,'2024-03-07 23:00:42');
INSERT INTO "Assessments" VALUES(2,'Exam','Dr. Jones',100,65,'2024-03-07 23:00:42');
INSERT INTO "Assessments" VALUES(3,'Quiz','Professor Smith',20,10,'2024-03-07 23:00:42');
INSERT INTO "Assessments" VALUES(4,'Quiz','Dr. Brown',20,15,'2024-03-07 23:00:42');
INSERT INTO "Assessments" VALUES(5,'Project','Dr. Johnson',100,70,'2024-03-07 23:00:42');
CREATE TABLE Courses(
    CourseID INTEGER PRIMARY KEY,
    Title TEXT,
    Department TEXT
);
INSERT INTO "Courses" VALUES(1,'Mathematics','Math');
INSERT INTO "Courses" VALUES(2,'Physics','Science');
INSERT INTO "Courses" VALUES(3,'English Literature','Languages');
INSERT INTO "Courses" VALUES(4,'History','Humanities');
INSERT INTO "Courses" VALUES(5,'Computer Science','Technology');
CREATE TABLE Enrollments(
    EnrollmentID INTEGER PRIMARY KEY,
    StudentID INTEGER,
    CourseID INTEGER,
    Semester DATETIME DEFAULT (DATETIME('now', '+2 semesters')),
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(CourseID) REFERENCES Courses(CourseID)
);
INSERT INTO "Enrollments" VALUES(1,1,1,'2024-03-07 23:00:42');
INSERT INTO "Enrollments" VALUES(2,2,2,'2024-03-07 23:00:42');
INSERT INTO "Enrollments" VALUES(3,3,3,'2024-03-07 23:00:42');
INSERT INTO "Enrollments" VALUES(4,4,4,'2024-03-07 23:00:42');
INSERT INTO "Enrollments" VALUES(5,5,5,'2024-03-07 23:00:42');
CREATE TABLE Grades(
    GradeID INTEGER PRIMARY KEY,
    EnrollmentID INTEGER,
    FinalGrade CHAR(1) ,
    FOREIGN KEY(EnrollmentID) REFERENCES Enrollments(EnrollmentID)
);
INSERT INTO "Grades" VALUES(1,1,'A');
INSERT INTO "Grades" VALUES(2,2,'B');
INSERT INTO "Grades" VALUES(3,3,'A');
INSERT INTO "Grades" VALUES(4,4,'B');
INSERT INTO "Grades" VALUES(5,5,'A');
CREATE TABLE Scores(
    ScoreID INTEGER PRIMARY KEY,
    EnrollmentID INTEGER,
    AssessmentID INTEGER,
    Score INTEGER,
    FOREIGN KEY(EnrollmentID) REFERENCES Enrollments(EnrollmentID),
    FOREIGN KEY(AssessmentID) REFERENCES Assessments(AssessmentID)
);
INSERT INTO "Scores" VALUES(1,1,1,80);
INSERT INTO "Scores" VALUES(2,1,2,90);
INSERT INTO "Scores" VALUES(3,2,1,85);
INSERT INTO "Scores" VALUES(4,2,2,95);
INSERT INTO "Scores" VALUES(5,3,1,75);
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Age INTEGER 
);
INSERT INTO "Students" VALUES(1,'John','Doe',20);
INSERT INTO "Students" VALUES(2,'Jane','Smith',23);
INSERT INTO "Students" VALUES(3,'Bob','Johnson',21);
INSERT INTO "Students" VALUES(4,'Alice','Williams',22);
INSERT INTO "Students" VALUES(5,'Tom','Brown',20);
COMMIT;
