BEGIN TRANSACTION;
CREATE TABLE Campus (
    id INTEGER PRIMARY KEY,
    campus_name TEXT,
    city_id INTEGER,
    state_id INTEGER,
    FOREIGN KEY(city_id) REFERENCES City(id),
    FOREIGN KEY(state_id) REFERENCES State(id)
);
INSERT INTO "Campus" VALUES(1,'New York University',1,1);
INSERT INTO "Campus" VALUES(2,'University of California Los Angeles',2,2);
INSERT INTO "Campus" VALUES(3,'University of Chicago',3,3);
INSERT INTO "Campus" VALUES(4,'University of Houston',4,4);
INSERT INTO "Campus" VALUES(5,'Arizona State University',5,5);
CREATE TABLE City (
    id INTEGER PRIMARY KEY,
    name TEXT,
    population INTEGER
);
INSERT INTO "City" VALUES(1,'New York',8398748);
INSERT INTO "City" VALUES(2,'Los Angeles',3960215);
INSERT INTO "City" VALUES(3,'Chicago',2697391);
INSERT INTO "City" VALUES(4,'Houston',2326146);
INSERT INTO "City" VALUES(5,'Phoenix',1681115);
CREATE TABLE State (
    id INTEGER PRIMARY KEY,
    full_name TEXT UNIQUE
);
INSERT INTO "State" VALUES(1,'New York');
INSERT INTO "State" VALUES(2,'California');
INSERT INTO "State" VALUES(3,'Illinois');
INSERT INTO "State" VALUES(4,'Texas');
INSERT INTO "State" VALUES(5,'Arizona');
CREATE TABLE Student (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Student" VALUES(1,'Emily','Davis','emily.davis@example.com');
INSERT INTO "Student" VALUES(2,'Alexander','Martinez','alexander.martinez@example.com');
INSERT INTO "Student" VALUES(3,'Olivia','Garcia','olivia.garcia@example.com');
INSERT INTO "Student" VALUES(4,'William','Anderson','william.anderson@example.com');
INSERT INTO "Student" VALUES(5,'Isabelle','Jackson','isabelle.jackson@example.com');
CREATE TABLE Teacher (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Teacher" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Teacher" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Teacher" VALUES(3,'Michael','Brown','michael.brown@example.com');
INSERT INTO "Teacher" VALUES(4,'Alice','Johnson','alice.johnson@example.com');
INSERT INTO "Teacher" VALUES(5,'David','Wilson','david.wilson@example.com');
CREATE TABLE enrolledIn (
    studentId INTEGER,
    campusId INTEGER,
    enrollmentStartDate DATE,
    FOREIGN KEY(studentId) REFERENCES Student(id),
    FOREIGN KEY(campusId) REFERENCES Campus(id)
);
INSERT INTO "enrolledIn" VALUES(1,1,'2021-01-01');
INSERT INTO "enrolledIn" VALUES(2,2,'2021-01-01');
INSERT INTO "enrolledIn" VALUES(3,3,'2021-01-01');
INSERT INTO "enrolledIn" VALUES(4,4,'2021-01-01');
INSERT INTO "enrolledIn" VALUES(5,5,'2021-01-01');
CREATE TABLE teachesAt (
    teacherId INTEGER,
    campusId INTEGER,
    startDate DATE,
    endDate DATE,
    FOREIGN KEY(teacherId) REFERENCES Teacher(id),
    FOREIGN KEY(campusId) REFERENCES Campus(id)
);
INSERT INTO "teachesAt" VALUES(1,1,'2021-01-01','2021-12-31');
INSERT INTO "teachesAt" VALUES(2,2,'2021-01-01','2021-12-31');
INSERT INTO "teachesAt" VALUES(3,3,'2021-01-01','2021-12-31');
INSERT INTO "teachesAt" VALUES(4,4,'2021-01-01','2021-12-31');
INSERT INTO "teachesAt" VALUES(5,5,'2021-01-01','2021-12-31');
COMMIT;
