BEGIN TRANSACTION;
CREATE TABLE "Classrooms" (
    ClassroomID INTEGER PRIMARY KEY,
    Name TEXT,
    Capacity INTEGER ,
    TypeOfEnvironment INTEGER,
    FOREIGN KEY (TypeOfEnvironment) REFERENCES LearningEnvironments(EnvironmentID)
);
INSERT INTO "Classrooms" VALUES(1,'Room A',30,1);
INSERT INTO "Classrooms" VALUES(2,'Room B',25,1);
INSERT INTO "Classrooms" VALUES(3,'Virtual Room C',30,2);
INSERT INTO "Classrooms" VALUES(4,'Virtual Room D',20,2);
INSERT INTO "Classrooms" VALUES(5,'Physical Room E',15,1);
CREATE TABLE Enrollments (
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    ClassroomID INTEGER,
    StartDate DATE,
    EndDate DATE,
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (ClassroomID) REFERENCES Classrooms(ClassroomID)
);
INSERT INTO "Enrollments" VALUES(1,1,1,'2022-09-01','2022-12-31');
INSERT INTO "Enrollments" VALUES(2,2,2,'2022-08-26','2022-12-17');
INSERT INTO "Enrollments" VALUES(3,3,3,'2022-09-01','2022-12-31');
INSERT INTO "Enrollments" VALUES(4,4,4,'2022-08-26','2022-12-17');
INSERT INTO "Enrollments" VALUES(5,5,5,'2022-09-01','2022-12-31');
CREATE TABLE LearningEnvironments (
    EnvironmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    Value TEXT UNIQUE
);
INSERT INTO "LearningEnvironments" VALUES(1,'Classroom');
INSERT INTO "LearningEnvironments" VALUES(2,'Virtual Room');
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Age INTEGER ,
    Email TEXT UNIQUE
);
INSERT INTO "Students" VALUES(1,'John','Doe',21,'john.doe@example.com');
INSERT INTO "Students" VALUES(2,'Jane','Smith',22,'jane.smith@example.com');
INSERT INTO "Students" VALUES(3,'Michael','Brown',23,'michael.brown@example.com');
INSERT INTO "Students" VALUES(4,'Emily','White',20,'emily.white@example.com');
INSERT INTO "Students" VALUES(5,'David','Green',24,'david.green@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('LearningEnvironments',2);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',5);
COMMIT;
