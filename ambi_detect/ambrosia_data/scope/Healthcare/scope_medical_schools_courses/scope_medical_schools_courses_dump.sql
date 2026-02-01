BEGIN TRANSACTION;
CREATE TABLE Courses(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    courseName TEXT,
    duration INT );
INSERT INTO "Courses" VALUES(1,'Anatomy',2);
INSERT INTO "Courses" VALUES(2,'Biology',2);
INSERT INTO "Courses" VALUES(3,'Physiology',1);
INSERT INTO "Courses" VALUES(4,'Pharmacology',3);
INSERT INTO "Courses" VALUES(5,'Pathology',3);
CREATE TABLE Medical_School_Courses(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schoolId INTEGER REFERENCES Medical_Schools(id) ON DELETE SET NULL,
    courseId INTEGER REFERENCES Courses(id) ON DELETE SET NULL);
INSERT INTO "Medical_School_Courses" VALUES(1,1,1);
INSERT INTO "Medical_School_Courses" VALUES(2,2,1);
INSERT INTO "Medical_School_Courses" VALUES(3,3,1);
INSERT INTO "Medical_School_Courses" VALUES(4,4,1);
INSERT INTO "Medical_School_Courses" VALUES(5,5,1);
INSERT INTO "Medical_School_Courses" VALUES(6,5,2);
CREATE TABLE Medical_Schools(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Medical_Schools" VALUES(1,'Harvard Medical School','Boston');
INSERT INTO "Medical_Schools" VALUES(2,'Johns Hopkins University School of Medicine','Baltimore');
INSERT INTO "Medical_Schools" VALUES(3,'Stanford University School of Medicine','Stanford');
INSERT INTO "Medical_Schools" VALUES(4,'University of California - San Francisco School of Medicine','San Francisco');
INSERT INTO "Medical_Schools" VALUES(5,'Yale School of Medicine','New Haven');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Medical_Schools',5);
INSERT INTO "sqlite_sequence" VALUES('Courses',5);
INSERT INTO "sqlite_sequence" VALUES('Medical_School_Courses',6);
COMMIT;
