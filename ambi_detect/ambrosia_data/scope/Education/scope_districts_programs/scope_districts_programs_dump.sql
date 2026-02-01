BEGIN TRANSACTION;
CREATE TABLE Districts(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Population INT
);
INSERT INTO "Districts" VALUES(1,'District1',20000);
INSERT INTO "Districts" VALUES(2,'District2',30000);
INSERT INTO "Districts" VALUES(3,'District3',40000);
INSERT INTO "Districts" VALUES(4,'District4',50000);
INSERT INTO "Districts" VALUES(5,'District5',60000);
CREATE TABLE Districts_Programs(
    DistrictID INTEGER,
    ProgramID INTEGER,
    Implemented BOOLEAN DEFAULT FALSE,
    FOREIGN KEY(DistrictID) REFERENCES Districts(ID),
    FOREIGN KEY(ProgramID) REFERENCES Programs(ID)
);
INSERT INTO "Districts_Programs" VALUES(1,1,1);
INSERT INTO "Districts_Programs" VALUES(1,2,1);
INSERT INTO "Districts_Programs" VALUES(2,1,1);
INSERT INTO "Districts_Programs" VALUES(2,2,1);
INSERT INTO "Districts_Programs" VALUES(3,1,1);
INSERT INTO "Districts_Programs" VALUES(4,1,0);
INSERT INTO "Districts_Programs" VALUES(5,1,0);
CREATE TABLE Programs(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT
);
INSERT INTO "Programs" VALUES(1,'Special_Education','A program designed for students with special needs');
INSERT INTO "Programs" VALUES(2,'General_Education','Standard curriculum offered in most schools');
INSERT INTO "Programs" VALUES(3,'Vocational_Training','Career-focused education and training');
INSERT INTO "Programs" VALUES(4,'Art_and_Music','Program focused on creative arts');
INSERT INTO "Programs" VALUES(5,'Sports','Physical fitness and sports instruction');
CREATE TABLE School_Students(
    StudentID INTEGER,
    SchoolID INTEGER,
    EnrollmentDate DATE,
    FOREIGN KEY(StudentID) REFERENCES Students(ID),
    FOREIGN KEY(SchoolID) REFERENCES Schools(ID)
);
INSERT INTO "School_Students" VALUES(1,1,'2022-09-01');
INSERT INTO "School_Students" VALUES(2,1,'2022-09-01');
INSERT INTO "School_Students" VALUES(3,2,'2022-09-01');
INSERT INTO "School_Students" VALUES(4,2,'2022-09-01');
INSERT INTO "School_Students" VALUES(5,3,'2022-09-01');
CREATE TABLE School_Teachers(
    TeacherID INTEGER,
    SchoolID INTEGER,
    HireDate DATE,
    FOREIGN KEY(TeacherID) REFERENCES Teachers(ID),
    FOREIGN KEY(SchoolID) REFERENCES Schools(ID)
);
INSERT INTO "School_Teachers" VALUES(1,1,'2021-09-01');
INSERT INTO "School_Teachers" VALUES(2,1,'2021-09-01');
INSERT INTO "School_Teachers" VALUES(3,2,'2021-09-01');
INSERT INTO "School_Teachers" VALUES(4,2,'2021-09-01');
INSERT INTO "School_Teachers" VALUES(5,3,'2021-09-01');
CREATE TABLE Schools(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    DistrictID INTEGER,
    FOREIGN KEY(DistrictID) REFERENCES Districts(ID)
);
INSERT INTO "Schools" VALUES(1,'School1','Address1',1);
INSERT INTO "Schools" VALUES(2,'School2','Address2',2);
INSERT INTO "Schools" VALUES(3,'School3','Address3',3);
INSERT INTO "Schools" VALUES(4,'School4','Address4',4);
INSERT INTO "Schools" VALUES(5,'School5','Address5',5);
CREATE TABLE Students(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Age INT
);
INSERT INTO "Students" VALUES(1,'John','Doe',18);
INSERT INTO "Students" VALUES(2,'Jane','Smith',17);
INSERT INTO "Students" VALUES(3,'Mike','Brown',19);
INSERT INTO "Students" VALUES(4,'Emma','Williams',16);
INSERT INTO "Students" VALUES(5,'Jack','Johnson',20);
CREATE TABLE Students_Programs(
    StudentID INTEGER,
    ProgramID INTEGER,
    StartDate DATE,
    EndDate DATE,
    FOREIGN KEY(StudentID) REFERENCES Students(ID),
    FOREIGN KEY(ProgramID) REFERENCES Programs(ID)
);
INSERT INTO "Students_Programs" VALUES(1,1,'2022-08-01','2023-05-31');
INSERT INTO "Students_Programs" VALUES(2,1,'2022-08-01','2023-05-31');
INSERT INTO "Students_Programs" VALUES(3,2,'2022-08-01','2023-05-31');
INSERT INTO "Students_Programs" VALUES(4,2,'2022-08-01','2023-05-31');
INSERT INTO "Students_Programs" VALUES(5,1,'2022-08-01','2023-05-31');
CREATE TABLE Teachers(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Subject TEXT
);
INSERT INTO "Teachers" VALUES(1,'Mr.','Smith','Math');
INSERT INTO "Teachers" VALUES(2,'Ms.','Jones','English');
INSERT INTO "Teachers" VALUES(3,'Dr.','Black','History');
INSERT INTO "Teachers" VALUES(4,'Mr.','White','Biology');
INSERT INTO "Teachers" VALUES(5,'Mrs.','Green','Chemistry');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Districts',5);
INSERT INTO "sqlite_sequence" VALUES('Programs',5);
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Teachers',5);
INSERT INTO "sqlite_sequence" VALUES('Schools',5);
COMMIT;
