BEGIN TRANSACTION;
CREATE TABLE Attendance (
    attendanceId INTEGER PRIMARY KEY,
    studentId INTEGER,
    classId INTEGER,
    dateAttended DATE,
    isPresent BOOLEAN,
    FOREIGN KEY(studentId) REFERENCES Students(id),
    FOREIGN KEY(classId) REFERENCES Classes(classId)
);
INSERT INTO "Attendance" VALUES(1,1,1,'2022-09-01',1);
INSERT INTO "Attendance" VALUES(2,2,2,'2022-09-02',0);
INSERT INTO "Attendance" VALUES(3,3,3,'2022-09-03',1);
INSERT INTO "Attendance" VALUES(4,4,4,'2022-09-04',1);
INSERT INTO "Attendance" VALUES(5,5,5,'2022-09-05',0);
CREATE TABLE Classes (
    classID INTEGER PRIMARY KEY,
    className TEXT,
    teacherName TEXT
);
INSERT INTO "Classes" VALUES(1,'Math','Mr. Thompson');
INSERT INTO "Classes" VALUES(2,'English','Ms. Johnson');
INSERT INTO "Classes" VALUES(3,'History','Mr. White');
INSERT INTO "Classes" VALUES(4,'Art','Ms. Green');
INSERT INTO "Classes" VALUES(5,'Physical Education','Coach Black');
CREATE TABLE DisciplineLogs (
    logId INTEGER PRIMARY KEY,
    studentId INTEGER,
    incidentDate DATE,
    reasonForDiscipline TEXT,
    disciplinaryAction TEXT,
    resolutionDate DATE,
    FOREIGN KEY(studentId) REFERENCES Students(id)
);
INSERT INTO "DisciplineLogs" VALUES(1,1,'2022-09-06','Talking during class','Detention','2022-09-08');
INSERT INTO "DisciplineLogs" VALUES(2,2,'2022-09-07','Cheating on a test','Suspension','2022-09-10');
INSERT INTO "DisciplineLogs" VALUES(3,3,'2022-09-08','Bullying another student','Apology letter','2022-09-10');
INSERT INTO "DisciplineLogs" VALUES(4,4,'2022-09-09','Damaging school property','Community service','2022-09-15');
INSERT INTO "DisciplineLogs" VALUES(5,5,'2022-09-10','Late assignment submission','Extra credit assignment','2022-09-15');
CREATE TABLE Enrollments (
    enrollmentId INTEGER PRIMARY KEY,
    studentId INTEGER,
    classId INTEGER,
    FOREIGN KEY(studentId) REFERENCES Students(id),
    FOREIGN KEY(classId) REFERENCES Classes(classId)
);
INSERT INTO "Enrollments" VALUES(1,1,1);
INSERT INTO "Enrollments" VALUES(2,2,2);
INSERT INTO "Enrollments" VALUES(3,3,3);
INSERT INTO "Enrollments" VALUES(4,4,4);
INSERT INTO "Enrollments" VALUES(5,5,5);
CREATE TABLE MilestoneRecords (
    recordId INTEGER PRIMARY KEY,
    studentId INTEGER,
    physicalMilestones TEXT,
    cognitiveMilestones TEXT,
    childAge INTEGER,
    otherDetails TEXT,
    FOREIGN KEY(studentId) REFERENCES Students(id)
);
INSERT INTO "MilestoneRecords" VALUES(1,1,'Walking and running','Counting to 10',8,'No additional details');
INSERT INTO "MilestoneRecords" VALUES(2,2,'Riding a bike','Reading simple sentences',9,'No additional details');
INSERT INTO "MilestoneRecords" VALUES(3,3,'Climbing stairs','Recognizing colors',7,'No additional details');
INSERT INTO "MilestoneRecords" VALUES(4,4,'Swimming','Understanding basic concepts',10,'No additional details');
INSERT INTO "MilestoneRecords" VALUES(5,5,'Playing sports','Writing short stories',11,'No additional details');
CREATE TABLE Students (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    dateOfBirth DATE
);
INSERT INTO "Students" VALUES(1,'John','Doe','2008-09-15');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2007-03-20');
INSERT INTO "Students" VALUES(3,'Michael','Johnson','2006-11-12');
INSERT INTO "Students" VALUES(4,'Emily','Williams','2005-06-01');
INSERT INTO "Students" VALUES(5,'David','Brown','2004-02-15');
COMMIT;
