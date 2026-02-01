BEGIN TRANSACTION;
CREATE TABLE courses(
    courseID INTEGER PRIMARY KEY AUTOINCREMENT,
    courseName TEXT,
    department TEXT,
    semester CHAR(7) ,
    year INTEGER 
);
INSERT INTO "courses" VALUES(1,'Introduction to Computer Science','CS','Fall',2022);
INSERT INTO "courses" VALUES(2,'Calculus I','Math','Spring',2023);
INSERT INTO "courses" VALUES(3,'Physics I','Physics','Fall',2022);
INSERT INTO "courses" VALUES(4,'Literature Survey','English','Winter',2023);
INSERT INTO "courses" VALUES(5,'Economic Principles','Business','Spring',2022);
CREATE TABLE group_work(
    assignmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    courseID INTEGER,
    dueDate DATETIME,
    maxScore INT,
    participants INT,
    FOREIGN KEY(courseID) REFERENCES courses(courseID)
);
INSERT INTO "group_work" VALUES(1,1,'2022-10-20 23:59:00',100,10);
INSERT INTO "group_work" VALUES(2,2,'2023-02-28 23:59:00',100,8);
INSERT INTO "group_work" VALUES(3,3,'2022-10-27 23:59:00',100,12);
INSERT INTO "group_work" VALUES(4,4,'2023-03-24 23:59:00',100,15);
INSERT INTO "group_work" VALUES(5,5,'2022-04-24 23:59:00',100,7);
CREATE TABLE instructors(
    instructorID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT,
    department TEXT
);
INSERT INTO "instructors" VALUES(1,'Dr.','Smith','dr.smith@example.com','CS');
INSERT INTO "instructors" VALUES(2,'Professor','Johnson','prof.johnson@example.com','Math');
INSERT INTO "instructors" VALUES(3,'Dr.','Williams','dr.williams@example.com','Physics');
INSERT INTO "instructors" VALUES(4,'Ms.','Garcia','ms.garcia@example.com','English');
INSERT INTO "instructors" VALUES(5,'Mr.','Martinez','mr.martinez@example.com','Economics');
CREATE TABLE peer_tutoring(
    sessionID INTEGER PRIMARY KEY AUTOINCREMENT,
    courseID INTEGER,
    startDateTime DATETIME,
    endDateTime DATETIME,
    participants INT,
    FOREIGN KEY(courseID) REFERENCES courses(courseID)
);
INSERT INTO "peer_tutoring" VALUES(1,1,'2022-10-10 18:00:00','2022-10-10 20:00:00',10);
INSERT INTO "peer_tutoring" VALUES(2,2,'2023-02-15 16:00:00','2023-02-15 18:00:00',8);
INSERT INTO "peer_tutoring" VALUES(3,3,'2022-10-12 14:00:00','2022-10-12 16:00:00',12);
INSERT INTO "peer_tutoring" VALUES(4,4,'2023-03-10 19:00:00','2023-03-10 21:00:00',15);
INSERT INTO "peer_tutoring" VALUES(5,5,'2022-04-20 13:00:00','2022-04-20 15:00:00',7);
CREATE TABLE students(
    studentID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT,
    major TEXT,
    enrollmentYear INTEGER 
);
INSERT INTO "students" VALUES(1,'John','Doe','john.doe@example.com','Computer Science',2020);
INSERT INTO "students" VALUES(2,'Jane','Smith','jane.smith@example.com','Mathematics',2021);
INSERT INTO "students" VALUES(3,'Michael','Brown','michael.brown@example.com','Physics',2020);
INSERT INTO "students" VALUES(4,'Emma','White','emma.white@example.com','English',2022);
INSERT INTO "students" VALUES(5,'David','Black','david.black@example.com','Economics',2021);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('courses',5);
INSERT INTO "sqlite_sequence" VALUES('students',5);
INSERT INTO "sqlite_sequence" VALUES('instructors',5);
INSERT INTO "sqlite_sequence" VALUES('peer_tutoring',5);
INSERT INTO "sqlite_sequence" VALUES('group_work',5);
COMMIT;
