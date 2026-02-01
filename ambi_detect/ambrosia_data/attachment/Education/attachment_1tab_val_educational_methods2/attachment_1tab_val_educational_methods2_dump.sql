BEGIN TRANSACTION;
CREATE TABLE Assignment(
    assignmentID INTEGER PRIMARY KEY,
    title TEXT,
    dueDate DATETIME,
    educationalMethod TEXT ,
    participants TEXT ,
    classID INTEGER REFERENCES Class(classID));
INSERT INTO "Assignment" VALUES(1,'Math Test','2022-06-01 14:00:00','Peer Tutoring','10 students',1);
INSERT INTO "Assignment" VALUES(2,'Literature Essay','2022-06-15 14:00:00','Peer Tutoring','5 students',2);
INSERT INTO "Assignment" VALUES(3,'Historical Project','2022-07-01 14:00:00','Group Work','10 students',3);
INSERT INTO "Assignment" VALUES(4,'Quiz on Chapters','2022-07-15 14:00:00','Group Work','8 students',1);
INSERT INTO "Assignment" VALUES(5,'Book Report Presentation','2022-08-01 14:00:00','Peer Tutoring','6 students',2);
CREATE TABLE Class(
    classID INTEGER PRIMARY KEY,
    subjectID INTEGER REFERENCES Subject(subjectID),
    teacherID INTEGER REFERENCES Teacher(teacherID));
INSERT INTO "Class" VALUES(1,1,1);
INSERT INTO "Class" VALUES(2,2,2);
INSERT INTO "Class" VALUES(3,3,3);
CREATE TABLE Student(
    studentID INTEGER PRIMARY KEY,
    name TEXT,
    grade INTEGER,
    teacherID INTEGER REFERENCES Teacher(teacherID));
INSERT INTO "Student" VALUES(1,'Alice Johnson',9,1);
INSERT INTO "Student" VALUES(2,'Bob Williams',10,2);
INSERT INTO "Student" VALUES(3,'Carol Lee',9,3);
CREATE TABLE Subject(
    subjectID INTEGER PRIMARY KEY,
    name TEXT);
INSERT INTO "Subject" VALUES(1,'Mathematics');
INSERT INTO "Subject" VALUES(2,'English');
INSERT INTO "Subject" VALUES(3,'History');
CREATE TABLE Teacher(
    teacherID INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    experience YEARS);
INSERT INTO "Teacher" VALUES(1,'John Doe',32,'5 years');
INSERT INTO "Teacher" VALUES(2,'Jane Smith',48,'10 years');
INSERT INTO "Teacher" VALUES(3,'Michael Brown',27,'3 years');
COMMIT;
