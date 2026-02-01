BEGIN TRANSACTION;
CREATE TABLE Certificates (
    CertificateID INTEGER PRIMARY KEY,
    Title TEXT,
    CompletionRequirements TEXT,
    Duration INTEGER
);
INSERT INTO "Certificates" VALUES(1,'Advanced Programming','Passing Exam',12);
INSERT INTO "Certificates" VALUES(2,'Data Analysis','Project Submission',10);
CREATE TABLE Diplomas (
    DiplomaID INTEGER PRIMARY KEY,
    Title TEXT,
    CompletionRequirements TEXT,
    UnitsRequired INTEGER
);
INSERT INTO "Diplomas" VALUES(1,'Bachelor of Science','Passing Exam',120);
INSERT INTO "Diplomas" VALUES(2,'Master of Arts','Submitting Thesis',60);
CREATE TABLE Enrolment (
    EnrollmentID INTEGER PRIMARY KEY,
    StudentID INTEGER,
    SubjectID INTEGER,
    StartDate DATE,
    EndDate DATE,
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(SubjectID) REFERENCES Subjects(SubjectID)
);
INSERT INTO "Enrolment" VALUES(1,1,1,'2022-09-01','2022-12-31');
INSERT INTO "Enrolment" VALUES(2,2,2,'2022-09-01','2022-12-31');
CREATE TABLE Grades (
    GradeID INTEGER PRIMARY KEY,
    EnrollmenID INTEGER,
    Score INTEGER,
    DateAwarded DATE,
    FOREIGN KEY(EnrollmenID) REFERENCES Enrolment(EnrollmentID)
);
INSERT INTO "Grades" VALUES(1,1,95,'2022-12-15');
INSERT INTO "Grades" VALUES(2,2,85,'2022-12-15');
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE
);
INSERT INTO "Students" VALUES(1,'John','Doe','2003-06-15');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2002-08-27');
CREATE TABLE Subjects (
    SubjectID INTEGER PRIMARY KEY,
    Name TEXT,
    Teacher TEXT,
    MaxScore INTEGER
);
INSERT INTO "Subjects" VALUES(1,'Algebra','Mr. Johnson',100);
INSERT INTO "Subjects" VALUES(2,'Calculus','Mrs. Smith',100);
COMMIT;
