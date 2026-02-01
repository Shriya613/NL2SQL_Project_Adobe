BEGIN TRANSACTION;
CREATE TABLE Attendance (
    AttendID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    Date DATE,
    Status CHAR(10),
    FOREIGN KEY (StudentID) REFERENCES Student (ID)
);
INSERT INTO "Attendance" VALUES(1,1,'2023-01-01','Present');
INSERT INTO "Attendance" VALUES(2,2,'2023-01-01','Absent');
INSERT INTO "Attendance" VALUES(3,1,'2023-01-02','Present');
INSERT INTO "Attendance" VALUES(4,2,'2023-01-02','Absent');
INSERT INTO "Attendance" VALUES(5,3,'2023-01-02','Present');
INSERT INTO "Attendance" VALUES(6,4,'2023-01-03','Present');
CREATE TABLE Classroom (
    ClassID INTEGER PRIMARY KEY AUTOINCREMENT,
    TeachID INTEGER,
    Location TEXT,
    FOREIGN KEY (TeachID) REFERENCES Teacher (TeachID)
);
INSERT INTO "Classroom" VALUES(1,1,'Room A');
INSERT INTO "Classroom" VALUES(2,2,'Room B');
INSERT INTO "Classroom" VALUES(3,3,'Room C');
INSERT INTO "Classroom" VALUES(4,4,'Room D');
INSERT INTO "Classroom" VALUES(5,5,'Room E');
CREATE TABLE Enrollment (
    EnrollID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    ClassID INTEGER,
    FOREIGN KEY (StudentID) REFERENCES Student (ID),
    FOREIGN KEY (ClassID) REFERENCES Classroom (ClassID)
);
INSERT INTO "Enrollment" VALUES(1,1,1);
INSERT INTO "Enrollment" VALUES(2,2,2);
INSERT INTO "Enrollment" VALUES(3,3,3);
INSERT INTO "Enrollment" VALUES(4,4,4);
INSERT INTO "Enrollment" VALUES(5,5,5);
CREATE TABLE "Grades" (
    GradeID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    Subject TEXT,
    Score TEXT ,
    FOREIGN KEY (StudentID) REFERENCES Student (ID)
);
INSERT INTO "Grades" VALUES(1,1,'Math','98');
INSERT INTO "Grades" VALUES(2,2,'English','87');
INSERT INTO "Grades" VALUES(3,1,'History','93');
INSERT INTO "Grades" VALUES(4,2,'Biology','76');
INSERT INTO "Grades" VALUES(5,3,'Chemistry','92');
INSERT INTO "Grades" VALUES(6,4,'Physics','97');
CREATE TABLE Student (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE
);
INSERT INTO "Student" VALUES(1,'Alice Johnson');
INSERT INTO "Student" VALUES(2,'Bob Smith');
INSERT INTO "Student" VALUES(3,'Carol Williams');
INSERT INTO "Student" VALUES(4,'David Brown');
INSERT INTO "Student" VALUES(5,'Eva Green');
CREATE TABLE Teacher (
    TeachID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE
);
INSERT INTO "Teacher" VALUES(1,'Mr. White');
INSERT INTO "Teacher" VALUES(2,'Mrs. Black');
INSERT INTO "Teacher" VALUES(3,'Ms. Blue');
INSERT INTO "Teacher" VALUES(4,'Dr. Red');
INSERT INTO "Teacher" VALUES(5,'Professor Green');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Student',5);
INSERT INTO "sqlite_sequence" VALUES('Attendance',6);
INSERT INTO "sqlite_sequence" VALUES('Teacher',5);
INSERT INTO "sqlite_sequence" VALUES('Classroom',5);
INSERT INTO "sqlite_sequence" VALUES('Enrollment',5);
INSERT INTO "sqlite_sequence" VALUES('Grades',7);
COMMIT;
