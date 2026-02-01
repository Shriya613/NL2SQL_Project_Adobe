BEGIN TRANSACTION;
CREATE TABLE Classes(
    ClassID INTEGER PRIMARY KEY AUTOINCREMENT,
    ClassName TEXT,
    StartTime TIME,
    EndTime TIME,
    RoomNumber INT 
);
INSERT INTO "Classes" VALUES(1,'Calculus I','08:00:00','09:30:00',101);
INSERT INTO "Classes" VALUES(2,'Literary Analysis','09:45:00','11:15:00',102);
CREATE TABLE Course_Offerings(
    OfferingID INTEGER PRIMARY KEY AUTOINCREMENT,
    ClassID INTEGER,
    ProgramID INTEGER,
    InstructorID INTEGER,
    FOREIGN KEY (ClassID) REFERENCES Classes(ClassID),
    FOREIGN KEY (ProgramID) REFERENCES Education_Programs(ProgramID),
    FOREIGN KEY (InstructorID) REFERENCES Educators(EducatorID)
);
CREATE TABLE "Education_Programs"(
    ProgramID INTEGER PRIMARY KEY AUTOINCREMENT, `name` TEXT,
    "Type" TEXT,
    Description TEXT,
    Credits INT 
);
INSERT INTO "Education_Programs" VALUES(1,'IB Standard','International Baccalaureate Program','An internationally recognized pre-university program...',30);
INSERT INTO "Education_Programs" VALUES(2,'IB Extended','International Baccalaureate Program','This program has a slightly different curriculum structure...',31);
INSERT INTO "Education_Programs" VALUES(3,'AP College Board','Advanced Placement Program','The AP program offers college level courses...',30);
INSERT INTO "Education_Programs" VALUES(4,'General Program','Advanced Placement Program','These courses can earn college credit upon completion...',29);
CREATE TABLE Educators(
    EducatorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Subject TEXT
);
INSERT INTO "Educators" VALUES(1,'Alice','Brown','Mathematics');
INSERT INTO "Educators" VALUES(2,'Charlie','Green','English Literature');
CREATE TABLE Enrollments(
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    ClassID INTEGER,
    Semester TEXT,
    Grade CHAR(2),
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (ClassID) REFERENCES Classes(ClassID)
);
INSERT INTO "Enrollments" VALUES(1,1,1,'Fall 2021','A');
INSERT INTO "Enrollments" VALUES(2,2,2,'Winter 2022','B+');
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    Email VARCHAR(128) UNIQUE
);
INSERT INTO "Students" VALUES(1,'John','Doe','2000-01-01','john.doe@example.com');
INSERT INTO "Students" VALUES(2,'Jane','Smith','1997-06-28','jane.smith@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',2);
INSERT INTO "sqlite_sequence" VALUES('Educators',2);
INSERT INTO "sqlite_sequence" VALUES('Classes',2);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',2);
INSERT INTO "sqlite_sequence" VALUES('Education_Programs',4);
COMMIT;
