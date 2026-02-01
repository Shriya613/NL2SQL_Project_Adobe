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
CREATE TABLE Field_Of_Study (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);
INSERT INTO "Field_Of_Study" VALUES(1,'Applied Mathematics','A branch of mathematics concerned with practical applications.');
INSERT INTO "Field_Of_Study" VALUES(2,'Theoretical Physics','A branch of physics focused on developing mathematical models to explain physical phenomena.');
INSERT INTO "Field_Of_Study" VALUES(3,'Molecular Biology','The branch of biology dealing with molecules and chemical processes within living cells.');
INSERT INTO "Field_Of_Study" VALUES(4,'Organic Chemistry','The branch of chemistry dealing with carbon-containing compounds.');
INSERT INTO "Field_Of_Study" VALUES(5,'Artificial Intelligence','The simulation of human intelligence in machines programmed by humans.');
CREATE TABLE Instructor (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    email TEXT,
    fieldOfStudyId INTEGER,
    FOREIGN KEY(fieldOfStudyId) REFERENCES Field_Of_Study(id)
);
INSERT INTO "Instructor" VALUES(1,'Dr.','Jones','drjones@email.com',1);
INSERT INTO "Instructor" VALUES(2,'Prof.','Smith','profsmith@email.com',2);
INSERT INTO "Instructor" VALUES(3,'Assoc. Prof.','Davis','assocprofdavis@email.com',3);
INSERT INTO "Instructor" VALUES(4,'Asst. Prof.','Martin','asstprofmartin@email.com',4);
INSERT INTO "Instructor" VALUES(5,'Lecturer','Wilson','lecturerwilson@email.com',5);
CREATE TABLE Project (
    id INTEGER PRIMARY KEY,
    title TEXT,
    details TEXT,
    topicId INTEGER,
    fieldOfStudyId INTEGER,
    FOREIGN KEY(topicId) REFERENCES Topic(id),
    FOREIGN KEY(fieldOfStudyId) REFERENCES Field_Of_Study(id)
);
INSERT INTO "Project" VALUES(1,'Optimization Algorithms','Developing new optimization techniques.',1,1);
INSERT INTO "Project" VALUES(2,'Quantum Mechanics Experiment','Testing quantum mechanics predictions.',2,2);
INSERT INTO "Project" VALUES(3,'Gene Editing','Exploring CRISPR gene editing technology.',3,3);
INSERT INTO "Project" VALUES(4,'Reaction Kinetics','Investigating reaction rates in organic chemistry.',4,4);
INSERT INTO "Project" VALUES(5,'Machine Learning for Finance','Applying machine learning to financial data analysis.',5,5);
CREATE TABLE Student (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    email TEXT,
    majorId INTEGER,
    FOREIGN KEY(majorId) REFERENCES Field_Of_Study(id)
);
INSERT INTO "Student" VALUES(1,'John','Smith','johnsmith@email.com',1);
INSERT INTO "Student" VALUES(2,'Jane','Doe','janedoe@email.com',2);
INSERT INTO "Student" VALUES(3,'Alice','Brown','alicebrown@email.com',3);
INSERT INTO "Student" VALUES(4,'Charlie','White','charliewhite@email.com',4);
INSERT INTO "Student" VALUES(5,'Emma','Green','emmagreen@email.com',5);
CREATE TABLE Topic (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);
INSERT INTO "Topic" VALUES(1,'Mathematics','The study of numbers and numerical relationships.');
INSERT INTO "Topic" VALUES(2,'Physics','The natural science that involves the study of matter and energy.');
INSERT INTO "Topic" VALUES(3,'Biology','The study of living organisms and their interactions with each other and their environments.');
INSERT INTO "Topic" VALUES(4,'Chemistry','The study of substances at an atomic and molecular level.');
INSERT INTO "Topic" VALUES(5,'Computer Science','The study of computation, algorithms, data structures, and computer systems.');
COMMIT;
