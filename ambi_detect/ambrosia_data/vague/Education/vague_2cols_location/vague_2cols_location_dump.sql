BEGIN TRANSACTION;
CREATE TABLE Classes (
    ClassID INTEGER PRIMARY KEY,
    CourseTitle TEXT,
    CreditHours INT ,
    EnrollmentLimit INT,
    RoomNumber TEXT);
INSERT INTO "Classes" VALUES(1,'Introduction to English Literature',3,25,'Room 207');
INSERT INTO "Classes" VALUES(2,'World History I',4,30,'Room 210');
INSERT INTO "Classes" VALUES(3,'Calculus II',4,25,'Room 302');
INSERT INTO "Classes" VALUES(4,'General Physics',3,20,'Room 404');
INSERT INTO "Classes" VALUES(5,'Anatomy and Physiology',4,30,'Room 506');
CREATE TABLE Departments (
    DepartmentID INTEGER PRIMARY KEY,
    Name TEXT,
    DeanName TEXT,
    Building TEXT);
INSERT INTO "Departments" VALUES(1,'English Department','Dean Smith','Building A');
INSERT INTO "Departments" VALUES(2,'History Department','Dean Brown','Building B');
INSERT INTO "Departments" VALUES(3,'Math Department','Dean Johnson','Building C');
INSERT INTO "Departments" VALUES(4,'Physics Department','Dean Miller','Building D');
INSERT INTO "Departments" VALUES(5,'Biology Department','Dean Davis','Building E');
CREATE TABLE Facilities (
    FacilityID INTEGER PRIMARY KEY,
    Type TEXT, -- e.g., Library, Gymnasium, Science Lab etc.
    Capacity INT,
    OpeningTime TIME,
    ClosingTime TIME);
INSERT INTO "Facilities" VALUES(1,'Library',100,'08:00:00','22:00:00');
INSERT INTO "Facilities" VALUES(2,'Gymnasium',200,'06:00:00','22:00:00');
INSERT INTO "Facilities" VALUES(3,'Science Lab',50,'08:00:00','22:00:00');
INSERT INTO "Facilities" VALUES(4,'Art Studio',25,'08:00:00','22:00:00');
INSERT INTO "Facilities" VALUES(5,'Music Hall',100,'08:00:00','22:00:00');
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY,
    Major TEXT,
    City TEXT,
    Campus TEXT, fullname TEXT);
INSERT INTO "Students" VALUES(1,'Computer Science','New York','Uptown','John Doe');
INSERT INTO "Students" VALUES(2,'Mathematics','Los Angeles','Downtown','Jane Smith');
INSERT INTO "Students" VALUES(3,'Physics','Chicago','Lake Shore','Michael Brown');
INSERT INTO "Students" VALUES(4,'Chemistry','Houston','Medical Center','Emily White');
INSERT INTO "Students" VALUES(5,'Biology','Phoenix','Central','David Garcia');
CREATE TABLE Teachers (
    TeacherID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    SubjectTaught TEXT,
    HireDate DATE,
    OfficeLocation TEXT);
INSERT INTO "Teachers" VALUES(1,'Mr.','Green','English Literature','2019-08-15','Room 207');
INSERT INTO "Teachers" VALUES(2,'Ms.','Black','History','2018-09-01','Room 210');
INSERT INTO "Teachers" VALUES(3,'Dr.','Wilson','Calculus','2016-05-30','Room 302');
INSERT INTO "Teachers" VALUES(4,'Prof.','Martin','Physics','2015-10-12','Room 404');
INSERT INTO "Teachers" VALUES(5,'Dr.','Jackson','Biology','2017-12-20','Room 506');
COMMIT;
