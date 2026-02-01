BEGIN TRANSACTION;
CREATE TABLE Classes (
    ClassID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    GradeLevel INT,
    TeacherID INTEGER,
    FOREIGN KEY(TeacherID) REFERENCES Teachers(TeacherID)
);
INSERT INTO "Classes" VALUES(1,'Algebra I',9,1);
INSERT INTO "Classes" VALUES(2,'World History',10,2);
INSERT INTO "Classes" VALUES(3,'American Literature',11,3);
INSERT INTO "Classes" VALUES(4,'Physics',10,4);
INSERT INTO "Classes" VALUES(5,'Advanced Art',12,5);
CREATE TABLE LearningActivities (
    ActivityID INTEGER PRIMARY KEY AUTOINCREMENT,
    ActivityType TEXT ,
    Purpose TEXT,
    Date DATE,
    ClassID INTEGER,
    FOREIGN KEY(ClassID) REFERENCES Classes(ClassID)
);
INSERT INTO "LearningActivities" VALUES(1,'Field Trip','Real-world Application','2023-03-15',1);
INSERT INTO "LearningActivities" VALUES(2,'Field Trip','Team Building','2023-03-22',2);
INSERT INTO "LearningActivities" VALUES(3,'Study Group','Real-world Application','2023-03-18',3);
INSERT INTO "LearningActivities" VALUES(4,'Study Group','Preparation for Test','2023-03-25',4);
INSERT INTO "LearningActivities" VALUES(5,'Online Lecture','Review Material','2023-03-20',5);
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE
);
INSERT INTO "Students" VALUES(1,'John','Doe','2003-10-07');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2002-08-21');
INSERT INTO "Students" VALUES(3,'Michael','Johnson','2005-06-29');
INSERT INTO "Students" VALUES(4,'Emily','Williams','2004-09-15');
INSERT INTO "Students" VALUES(5,'David','Brown','2006-07-23');
CREATE TABLE Teachers (
    TeacherID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    SubjectTaught TEXT
);
INSERT INTO "Teachers" VALUES(1,'Mr.','Davis','Math');
INSERT INTO "Teachers" VALUES(2,'Ms.','Martin','History');
INSERT INTO "Teachers" VALUES(3,'Mrs.','Wilson','English');
INSERT INTO "Teachers" VALUES(4,'Mr.','Hall','Science');
INSERT INTO "Teachers" VALUES(5,'Mrs.','Jackson','Art');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Teachers',5);
INSERT INTO "sqlite_sequence" VALUES('Classes',5);
INSERT INTO "sqlite_sequence" VALUES('LearningActivities',5);
COMMIT;
