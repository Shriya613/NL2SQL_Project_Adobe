BEGIN TRANSACTION;
CREATE TABLE "Student"(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    DisciplineID INTEGER,
    FOREIGN KEY(DisciplineID) REFERENCES "Teaching_Subjects"(DisciplineID)
);
INSERT INTO "Student" VALUES(1,'John','Doe','2005-09-01',1);
INSERT INTO "Student" VALUES(2,'Jane','Smith','2006-04-15',2);
INSERT INTO "Student" VALUES(3,'Michael','Johnson','2007-02-20',3);
INSERT INTO "Student" VALUES(4,'Emily','Williams','2008-01-05',4);
INSERT INTO "Student" VALUES(5,'David','Brown','2009-03-10',5);
CREATE TABLE "Teacher"(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    HireDate DATE,
    DisciplineID INTEGER,
    FOREIGN KEY(DisciplineID) REFERENCES "Teaching_Subjects"(DisciplineID)
);
INSERT INTO "Teacher" VALUES(1,'Mr.','Smith','2010-08-20',1);
INSERT INTO "Teacher" VALUES(2,'Ms.','Johnson','2011-09-15',2);
INSERT INTO "Teacher" VALUES(3,'Dr.','Williams','2012-11-01',3);
INSERT INTO "Teacher" VALUES(4,'Ms.','Baker','2013-02-10',4);
INSERT INTO "Teacher" VALUES(5,'Mr.','Thompson','2014-03-20',5);
CREATE TABLE "Teaching_Subjects"(
    DisciplineID INTEGER PRIMARY KEY,
    Name TEXT,
    Description TEXT
);
INSERT INTO "Teaching_Subjects" VALUES(1,'Mathematics','The study of numbers and their properties.');
INSERT INTO "Teaching_Subjects" VALUES(2,'Physics','The study of matter and energy and how they interact.');
INSERT INTO "Teaching_Subjects" VALUES(3,'Biology','The study of living organisms and their interactions with their environment.');
INSERT INTO "Teaching_Subjects" VALUES(4,'Chemistry','The study of substances and their interactions.');
INSERT INTO "Teaching_Subjects" VALUES(5,'History','The study of past events and their causes and effects.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Student',5);
INSERT INTO "sqlite_sequence" VALUES('Teacher',5);
COMMIT;
