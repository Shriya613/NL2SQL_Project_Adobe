BEGIN TRANSACTION;
CREATE TABLE Case_Studies(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Topic TEXT,
    Description TEXT,
    Date DATETIME DEFAULT CURRENT_TIMESTAMP,
    Author TEXT);
INSERT INTO "Case_Studies" VALUES(1,'Environmental Science','A study on deforestation','2024-03-13 18:12:21',NULL);
INSERT INTO "Case_Studies" VALUES(2,'Mathematics','Application of maths in daily life','2024-03-13 18:12:21',NULL);
CREATE TABLE Classes(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    ClassroomNumber INT,
    MaxCapacity INT,
    SchoolYear INT);
INSERT INTO "Classes" VALUES(1,101,30,2021);
INSERT INTO "Classes" VALUES(2,102,30,2021);
CREATE TABLE Interviews(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Topic TEXT,
    Description TEXT,
    IntervieweeName TEXT,
    InterviewDate DATETIME DEFAULT CURRENT_TIMESTAMP);
INSERT INTO "Interviews" VALUES(1,'Environmental Science','Impact of climate change on biodiversity','John Doe','2024-03-13 18:12:21');
INSERT INTO "Interviews" VALUES(2,'Literature','Role of literature in society','Jane Smith','2024-03-13 18:12:21');
CREATE TABLE Students(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INT,
    GradeLevel INT);
INSERT INTO "Students" VALUES(1,'Alice Johnson',20,3);
INSERT INTO "Students" VALUES(2,'Bob Brown',22,3);
CREATE TABLE Teachers(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Subject TEXT,
    YearsOfExperience INT);
INSERT INTO "Teachers" VALUES(1,'Mike White','Physics',10);
INSERT INTO "Teachers" VALUES(2,'Emma Black','Chemistry',8);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Case_Studies',2);
INSERT INTO "sqlite_sequence" VALUES('Interviews',2);
INSERT INTO "sqlite_sequence" VALUES('Students',2);
INSERT INTO "sqlite_sequence" VALUES('Teachers',2);
INSERT INTO "sqlite_sequence" VALUES('Classes',2);
COMMIT;
