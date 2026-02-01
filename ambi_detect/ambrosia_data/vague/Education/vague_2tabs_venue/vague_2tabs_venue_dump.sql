BEGIN TRANSACTION;
CREATE TABLE Campus_Building (
    id INTEGER PRIMARY KEY,
    name TEXT,
    capacity INTEGER
);
INSERT INTO "Campus_Building" VALUES(1,'Main Building',1200);
INSERT INTO "Campus_Building" VALUES(2,'Library',800);
INSERT INTO "Campus_Building" VALUES(3,'Sports Center',600);
INSERT INTO "Campus_Building" VALUES(4,'Art Center',300);
INSERT INTO "Campus_Building" VALUES(5,'Conference Hall',100);
CREATE TABLE Course (
    id INTEGER PRIMARY KEY,
    courseCode TEXT,
    courseTitle TEXT,
    lecturesInCourse INTEGER DEFAULT 0,
    FOREIGN KEY (lecturesInCourse) REFERENCES Lecture(id)
);
INSERT INTO "Course" VALUES(1,'CS101','Fundamental of Computer Science',5);
INSERT INTO "Course" VALUES(2,'DS201','Applied Data Structures',4);
INSERT INTO "Course" VALUES(3,'ENG301','Technical Writing',3);
INSERT INTO "Course" VALUES(4,'MGMT401','Human Resource Management',3);
INSERT INTO "Course" VALUES(5,'IT202','Web Development',4);
CREATE TABLE Enrollment (
    studentId INTEGER REFERENCES Student(id),
    lectureId INTEGER REFERENCES Lecture(id),
    grade INTEGER 
);
INSERT INTO "Enrollment" VALUES(1,1,90);
INSERT INTO "Enrollment" VALUES(1,2,87);
INSERT INTO "Enrollment" VALUES(2,3,92);
INSERT INTO "Enrollment" VALUES(2,4,88);
INSERT INTO "Enrollment" VALUES(3,1,93);
CREATE TABLE Lecture (
    id INTEGER PRIMARY KEY,
    title TEXT,
    duration INTEGER,
    campusId INTEGER REFERENCES Campus_Building(id),
    onlineId INTEGER REFERENCES Online_Platform(id)
);
INSERT INTO "Lecture" VALUES(1,'Introduction to Databases',90,1,1);
INSERT INTO "Lecture" VALUES(2,'Advanced Programming Techniques',120,2,2);
INSERT INTO "Lecture" VALUES(3,'Data Science and Machine Learning',150,3,3);
INSERT INTO "Lecture" VALUES(4,'Network Security Fundamentals',100,4,4);
INSERT INTO "Lecture" VALUES(5,'Project Management Best Practices',110,5,5);
CREATE TABLE Online_Platform (
    id INTEGER PRIMARY KEY,
    platformName TEXT,
    URL TEXT UNIQUE
);
INSERT INTO "Online_Platform" VALUES(1,'EdX','https://www.edx.org');
INSERT INTO "Online_Platform" VALUES(2,'Coursera','https://www.coursera.org');
INSERT INTO "Online_Platform" VALUES(3,'Udacity','https://www.udacity.com');
INSERT INTO "Online_Platform" VALUES(4,'Codecademy','https://www.codecademy.com');
INSERT INTO "Online_Platform" VALUES(5,'LinkedIn Learning','https://www.linkedin.com/learning');
CREATE TABLE Student (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Student" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Student" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Student" VALUES(3,'Michael','Brown','michael.brown@example.com');
INSERT INTO "Student" VALUES(4,'Emily','White','emily.white@example.com');
INSERT INTO "Student" VALUES(5,'David','Green','david.green@example.com');
COMMIT;
