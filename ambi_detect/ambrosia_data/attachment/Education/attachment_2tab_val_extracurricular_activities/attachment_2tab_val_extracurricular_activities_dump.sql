BEGIN TRANSACTION;
CREATE TABLE Classes (
    ClassID INTEGER PRIMARY KEY,
    ClassroomNumber INTEGER,
    StartTime TIME,
    EndTime TIME,
    TeacherId INTEGER,
    FOREIGN KEY (TeacherId) REFERENCES Teachers(TeacherID)
);
INSERT INTO "Classes" VALUES(1,101,'08:00:00','09:00:00',1);
INSERT INTO "Classes" VALUES(2,201,'10:00:00','11:00:00',2);
CREATE TABLE Clubs (
    ClubID INTEGER PRIMARY KEY,
    Name VARCHAR(100),
    Description TEXT,
    Participants INTEGER
);
INSERT INTO "Clubs" VALUES(1,'Chess Club','A club dedicated to playing chess.',30);
INSERT INTO "Clubs" VALUES(2,'Coding Club','A club focused on coding and programming.',20);
CREATE TABLE Enrollments (
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    ClassId INTEGER,
    Grade TEXT,
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (ClassId) REFERENCES Classes(ClassID)
);
INSERT INTO "Enrollments" VALUES(1,1,1,'A');
INSERT INTO "Enrollments" VALUES(2,1,2,'B');
CREATE TABLE LeadershipPrograms (
    ProgramID INTEGER PRIMARY KEY,
    Name VARCHAR(100),
    Description TEXT,
    Participants INTEGER
);
INSERT INTO "LeadershipPrograms" VALUES(1,'Youth Leadership Academy','A program that helps young people develop leadership skills.',30);
INSERT INTO "LeadershipPrograms" VALUES(2,'Community Service Initiative','A program where students engage in community service projects.',25);
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    EmailAddress TEXT UNIQUE
);
INSERT INTO "Students" VALUES(1,'John','Doe','2000-01-01','john.doe@example.com');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2002-02-02','jane.smith@example.com');
CREATE TABLE Teachers (
    TeacherID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    SubjectTaught TEXT
);
INSERT INTO "Teachers" VALUES(1,'Michael','White','Mathematics');
INSERT INTO "Teachers" VALUES(2,'Emily','Green','English');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Enrollments',2);
COMMIT;
