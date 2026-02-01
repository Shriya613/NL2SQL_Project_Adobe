BEGIN TRANSACTION;
CREATE TABLE Courses (
    id INTEGER PRIMARY KEY,
    courseCode TEXT,
    courseTitle TEXT,
    creditHours INTEGER,
    departmentId INTEGER,
    FOREIGN KEY(departmentId) REFERENCES Departments(id));
INSERT INTO "Courses" VALUES(1,'CS101','Introduction to Computer Science',4,1);
INSERT INTO "Courses" VALUES(2,'BIOL101','General Biology',4,2);
INSERT INTO "Courses" VALUES(3,'MATH101','Elementary Calculus',3,3);
INSERT INTO "Courses" VALUES(4,'CH101','General Chemistry',4,4);
INSERT INTO "Courses" VALUES(5,'PHYS101','Classical Mechanics',4,5);
CREATE TABLE Departments (
    id INTEGER PRIMARY KEY,
    departmentName TEXT,
    universityId INTEGER,
    FOREIGN KEY(universityId) REFERENCES Universities(id));
INSERT INTO "Departments" VALUES(1,'Department of Computer Science',1);
INSERT INTO "Departments" VALUES(2,'Department of Biology',2);
INSERT INTO "Departments" VALUES(3,'Department of Mathematics',3);
INSERT INTO "Departments" VALUES(4,'Department of Chemistry',4);
INSERT INTO "Departments" VALUES(5,'Department of Physics',5);
CREATE TABLE Enrollments (
    studentID INTEGER,
    courseID INTEGER,
    semester TEXT,
    grade CHAR(2),
    enrollmentDate DATE,
    FOREIGN KEY(studentID) REFERENCES Students(id),
    FOREIGN KEY(courseID) REFERENCES Courses(id));
INSERT INTO "Enrollments" VALUES(1,1,'Fall 2021','A','2021-08-25');
INSERT INTO "Enrollments" VALUES(2,2,'Spring 2022','B+','2022-01-10');
INSERT INTO "Enrollments" VALUES(3,3,'Summer 2022','A-','2022-05-20');
INSERT INTO "Enrollments" VALUES(4,4,'Fall 2022','A','2022-08-25');
INSERT INTO "Enrollments" VALUES(5,5,'Winter 2023','A','2023-01-10');
CREATE TABLE Students (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    major TEXT);
INSERT INTO "Students" VALUES(1,'John Doe',20,'Computer Science');
INSERT INTO "Students" VALUES(2,'Jane Smith',19,'Biology');
INSERT INTO "Students" VALUES(3,'Mike Johnson',21,'Mathematics');
INSERT INTO "Students" VALUES(4,'Emily Davis',22,'Chemistry');
INSERT INTO "Students" VALUES(5,'David Brown',23,'Physics');
CREATE TABLE Universities(
    id INTEGER PRIMARY KEY,
    universityName TEXT,
    admissionRate REAL,
    ranking INTEGER,
    location TEXT);
INSERT INTO "Universities" VALUES(1,'University of California',0.67,8,'Los Angeles');
INSERT INTO "Universities" VALUES(2,'Stanford University',0.53,1,'Palo Alto');
INSERT INTO "Universities" VALUES(3,'Harvard University',0.51,3,'Boston');
INSERT INTO "Universities" VALUES(4,'Massachusetts Institute of Technology',0.89,2,'Cambridge');
INSERT INTO "Universities" VALUES(5,'California Institute of Technology',0.11,6,'Pasadena');
COMMIT;
