BEGIN TRANSACTION;
CREATE TABLE Academic_Discipline(
    DisciplineID INTEGER PRIMARY KEY,
    Name TEXT,
    Description TEXT
);
INSERT INTO "Academic_Discipline" VALUES(1,'Mathematics','The study of numbers and their properties.');
INSERT INTO "Academic_Discipline" VALUES(2,'Physics','The study of matter and energy and how they interact.');
INSERT INTO "Academic_Discipline" VALUES(3,'Biology','The study of living organisms and their interactions with their environment.');
INSERT INTO "Academic_Discipline" VALUES(4,'Chemistry','The study of substances and their interactions.');
INSERT INTO "Academic_Discipline" VALUES(5,'History','The study of past events and their causes and effects.');
CREATE TABLE Grade_Level(
    LevelID INTEGER PRIMARY KEY,
    Name TEXT,
    MinAge INT,
    MaxAge INT
);
INSERT INTO "Grade_Level" VALUES(1,'PreKindergarten',3,4);
INSERT INTO "Grade_Level" VALUES(2,'Kindergarten',4,5);
INSERT INTO "Grade_Level" VALUES(3,'1st Grade',5,6);
INSERT INTO "Grade_Level" VALUES(4,'2nd Grade',6,7);
INSERT INTO "Grade_Level" VALUES(5,'3rd Grade',7,8);
CREATE TABLE Student(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    BirthDate DATE,
    GradeLevelID INTEGER,
    DisciplineID INTEGER, fullname TEXT,
    FOREIGN KEY(GradeLevelID) REFERENCES Grade_Level(LevelID),
    FOREIGN KEY(DisciplineID) REFERENCES Academic_Discipline(DisciplineID)
);
INSERT INTO "Student" VALUES(1,'2005-09-01',1,1,'John Doe');
INSERT INTO "Student" VALUES(2,'2006-04-15',2,2,'Jane Smith');
INSERT INTO "Student" VALUES(3,'2007-02-20',3,3,'Michael Johnson');
INSERT INTO "Student" VALUES(4,'2008-01-05',4,4,'Emily Williams');
INSERT INTO "Student" VALUES(5,'2009-03-10',5,5,'David Brown');
CREATE TABLE Teacher(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    HireDate DATE,
    GradeLevelID INTEGER,
    DisciplineID INTEGER, fullname TEXT,
    FOREIGN KEY(GradeLevelID) REFERENCES Grade_Level(LevelID),
    FOREIGN KEY(DisciplineID) REFERENCES Academic_Discipline(DisciplineID)
);
INSERT INTO "Teacher" VALUES(1,'2010-08-20',1,1,'Mr. Smith');
INSERT INTO "Teacher" VALUES(2,'2011-09-15',2,2,'Ms. Johnson');
INSERT INTO "Teacher" VALUES(3,'2012-11-01',3,3,'Dr. Williams');
INSERT INTO "Teacher" VALUES(4,'2013-02-10',4,4,'Ms. Baker');
INSERT INTO "Teacher" VALUES(5,'2014-03-20',5,5,'Mr. Thompson');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Student',5);
INSERT INTO "sqlite_sequence" VALUES('Teacher',5);
COMMIT;
