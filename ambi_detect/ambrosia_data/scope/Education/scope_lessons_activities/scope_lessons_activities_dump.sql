BEGIN TRANSACTION;
CREATE TABLE Activities(
    activityType TEXT PRIMARY KEY,
    description TEXT
);
INSERT INTO "Activities" VALUES('Group_Work','Collaborative learning task');
INSERT INTO "Activities" VALUES('Individual_Project','Personal project assignment');
INSERT INTO "Activities" VALUES('Quiz','Short test to assess understanding');
INSERT INTO "Activities" VALUES('Lecture','Instructor-led presentation on a topic');
INSERT INTO "Activities" VALUES('Discussion','Classwide conversation about a topic');
CREATE TABLE Enrollments(
    enrollmentId INTEGER PRIMARY KEY AUTOINCREMENT,
    studentId INTEGER,
    classId INTEGER,
    FOREIGN KEY (studentId) REFERENCES Students(studentId),
    FOREIGN KEY (classId) REFERENCES Lessons(id)
);
INSERT INTO "Enrollments" VALUES(1,1,1);
INSERT INTO "Enrollments" VALUES(2,2,2);
INSERT INTO "Enrollments" VALUES(3,3,3);
INSERT INTO "Enrollments" VALUES(4,4,4);
INSERT INTO "Enrollments" VALUES(5,5,5);
CREATE TABLE Lessons(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lessonName TEXT,
    teacherId INTEGER
);
INSERT INTO "Lessons" VALUES(1,'Introduction to Math',1);
INSERT INTO "Lessons" VALUES(2,'Beginner English Grammar',2);
INSERT INTO "Lessons" VALUES(3,'Ancient History Timeline',3);
INSERT INTO "Lessons" VALUES(4,'Cell Structure and Function',4);
INSERT INTO "Lessons" VALUES(5,'Basics of Physics',5);
CREATE TABLE Lessons_Activities(
    lessonsActivityId INTEGER PRIMARY KEY AUTOINCREMENT,
    lessonsId INTEGER,
    activitiesType TEXT,
    FOREIGN KEY (lessonsId) REFERENCES Lessons(id),
    FOREIGN KEY (activitiesType) REFERENCES Activities(activityType)
);
INSERT INTO "Lessons_Activities" VALUES(1,1,'Group_Work');
INSERT INTO "Lessons_Activities" VALUES(2,1,'Quiz');
INSERT INTO "Lessons_Activities" VALUES(3,2,'Individual_Project');
INSERT INTO "Lessons_Activities" VALUES(4,2,'Lecture');
INSERT INTO "Lessons_Activities" VALUES(5,3,'Quiz');
INSERT INTO "Lessons_Activities" VALUES(6,2,'Quiz');
INSERT INTO "Lessons_Activities" VALUES(7,4,'Quiz');
INSERT INTO "Lessons_Activities" VALUES(8,5,'Quiz');
CREATE TABLE Students(
    studentId INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER
);
INSERT INTO "Students" VALUES(1,'Alice',12);
INSERT INTO "Students" VALUES(2,'Bob',13);
INSERT INTO "Students" VALUES(3,'Charlie',14);
INSERT INTO "Students" VALUES(4,'David',15);
INSERT INTO "Students" VALUES(5,'Eva',16);
CREATE TABLE Teachers(
    teacherId INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    subject TEXT
);
INSERT INTO "Teachers" VALUES(1,'Mr. Smith','Math');
INSERT INTO "Teachers" VALUES(2,'Ms. Johnson','English');
INSERT INTO "Teachers" VALUES(3,'Mrs. Lee','History');
INSERT INTO "Teachers" VALUES(4,'Dr. Brown','Biology');
INSERT INTO "Teachers" VALUES(5,'Professor White','Physics');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Teachers',5);
INSERT INTO "sqlite_sequence" VALUES('Lessons',5);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',5);
INSERT INTO "sqlite_sequence" VALUES('Lessons_Activities',10);
COMMIT;
