BEGIN TRANSACTION;
CREATE TABLE Exams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    studentId INTEGER,
    examDate DATE,
    minScoreId INTEGER,
    percentReqId INTEGER,
    FOREIGN KEY (studentId) REFERENCES Students(id),
    FOREIGN KEY (minScoreId) REFERENCES Minimum_Scores(id),
    FOREIGN KEY (percentReqId) REFERENCES Percentage_Requirements(id));
INSERT INTO "Exams" VALUES(1,1,'2021-05-10',1,1);
INSERT INTO "Exams" VALUES(2,2,'2021-06-15',2,2);
INSERT INTO "Exams" VALUES(3,3,'2021-07-20',3,3);
INSERT INTO "Exams" VALUES(4,4,'2021-08-25',4,4);
INSERT INTO "Exams" VALUES(5,5,'2021-09-08',5,5);
CREATE TABLE Minimum_Scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    score INT);
INSERT INTO "Minimum_Scores" VALUES(1,80);
INSERT INTO "Minimum_Scores" VALUES(2,85);
INSERT INTO "Minimum_Scores" VALUES(3,90);
INSERT INTO "Minimum_Scores" VALUES(4,95);
INSERT INTO "Minimum_Scores" VALUES(5,100);
CREATE TABLE Percentage_Requirements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    percentage DECIMAL(5,2));
INSERT INTO "Percentage_Requirements" VALUES(1,50);
INSERT INTO "Percentage_Requirements" VALUES(2,60);
INSERT INTO "Percentage_Requirements" VALUES(3,70);
INSERT INTO "Percentage_Requirements" VALUES(4,80);
INSERT INTO "Percentage_Requirements" VALUES(5,90);
CREATE TABLE Students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    birthDate DATE);
INSERT INTO "Students" VALUES(1,'John','Doe','1980-06-30');
INSERT INTO "Students" VALUES(2,'Jane','Smith','1991-12-17');
INSERT INTO "Students" VALUES(3,'Michael','Brown','1975-09-03');
INSERT INTO "Students" VALUES(4,'Emily','White','1995-05-12');
INSERT INTO "Students" VALUES(5,'David','Black','1983-01-15');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Minimum_Scores',5);
INSERT INTO "sqlite_sequence" VALUES('Percentage_Requirements',5);
INSERT INTO "sqlite_sequence" VALUES('Exams',5);
COMMIT;
