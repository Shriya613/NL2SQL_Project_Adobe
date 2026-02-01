BEGIN TRANSACTION;
CREATE TABLE Enrollment (
    id INTEGER PRIMARY KEY,
    studentId INTEGER,
    projectId INTEGER,
    grade DECIMAL(3, 2),
    enrolledDate DATE,
    completionDate DATE,
    FOREIGN KEY(studentId) REFERENCES Student(id),
    FOREIGN KEY(projectId) REFERENCES Project(id)
);
INSERT INTO "Enrollment" VALUES(1,1,1,85,'2021-09-01','2021-12-01');
INSERT INTO "Enrollment" VALUES(2,2,2,90,'2021-09-01','2021-12-01');
INSERT INTO "Enrollment" VALUES(3,3,3,92.5,'2021-09-01','2021-12-01');
INSERT INTO "Enrollment" VALUES(4,4,4,87.5,'2021-09-01','2021-12-01');
INSERT INTO "Enrollment" VALUES(5,5,5,95,'2021-09-01','2021-12-01');
CREATE TABLE "Focus_Area" (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);
INSERT INTO "Focus_Area" VALUES(1,'Mathematics','The study of numbers and numerical relationships.');
INSERT INTO "Focus_Area" VALUES(2,'Physics','The natural science that involves the study of matter and energy.');
INSERT INTO "Focus_Area" VALUES(3,'Biology','The study of living organisms and their interactions with each other and their environments.');
INSERT INTO "Focus_Area" VALUES(4,'Chemistry','The study of substances at an atomic and molecular level.');
INSERT INTO "Focus_Area" VALUES(5,'Computer Science','The study of computation, algorithms, data structures, and computer systems.');
CREATE TABLE "Instructor" (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    email TEXT);
INSERT INTO "Instructor" VALUES(1,'Dr.','Jones','drjones@email.com');
INSERT INTO "Instructor" VALUES(2,'Prof.','Smith','profsmith@email.com');
INSERT INTO "Instructor" VALUES(3,'Assoc. Prof.','Davis','assocprofdavis@email.com');
INSERT INTO "Instructor" VALUES(4,'Asst. Prof.','Martin','asstprofmartin@email.com');
INSERT INTO "Instructor" VALUES(5,'Lecturer','Wilson','lecturerwilson@email.com');
CREATE TABLE "Project" (
    id INTEGER PRIMARY KEY,
    title TEXT,
    details TEXT,
    topicId INTEGER,
    FOREIGN KEY(topicId) REFERENCES "Focus_Area"(id));
INSERT INTO "Project" VALUES(1,'Optimization Algorithms','Developing new optimization techniques.',1);
INSERT INTO "Project" VALUES(2,'Quantum Mechanics Experiment','Testing quantum mechanics predictions.',2);
INSERT INTO "Project" VALUES(3,'Gene Editing','Exploring CRISPR gene editing technology.',3);
INSERT INTO "Project" VALUES(4,'Reaction Kinetics','Investigating reaction rates in organic chemistry.',4);
INSERT INTO "Project" VALUES(5,'Machine Learning for Finance','Applying machine learning to financial data analysis.',5);
CREATE TABLE "Student" (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    email TEXT);
INSERT INTO "Student" VALUES(1,'John','Smith','johnsmith@email.com');
INSERT INTO "Student" VALUES(2,'Jane','Doe','janedoe@email.com');
INSERT INTO "Student" VALUES(3,'Alice','Brown','alicebrown@email.com');
INSERT INTO "Student" VALUES(4,'Charlie','White','charliewhite@email.com');
INSERT INTO "Student" VALUES(5,'Emma','Green','emmagreen@email.com');
COMMIT;
