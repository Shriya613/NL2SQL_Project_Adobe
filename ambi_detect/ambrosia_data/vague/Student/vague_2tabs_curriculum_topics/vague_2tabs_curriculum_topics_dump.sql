BEGIN TRANSACTION;
CREATE TABLE Advanced_Topics (
    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_name TEXT,
    difficulty LEVEL 
);
INSERT INTO "Advanced_Topics" VALUES(1,'Advanced Python','Intermediate');
INSERT INTO "Advanced_Topics" VALUES(2,'Multivariable Calculus','Advanced');
INSERT INTO "Advanced_Topics" VALUES(3,'Tensor Analysis','Expert');
CREATE TABLE Core_Concepts (
    concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_name TEXT,
    description TEXT
);
INSERT INTO "Core_Concepts" VALUES(1,'Python Programming','A high-level programming language');
INSERT INTO "Core_Concepts" VALUES(2,'Linear Algebra','The study of vector spaces and linear mappings between them');
INSERT INTO "Core_Concepts" VALUES(3,'Calculus','The study of change and motion');
CREATE TABLE EnrolledStudents (
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    subject_id INTEGER,
    grade DECIMAL(2,1), -- maximum two digits after decimal point representing percentage
    FOREIGN KEY (student_id) REFERENCES Students(id),
    FOREIGN KEY (subject_id) REFERENCES School_Subjects(subject_id)
);
INSERT INTO "EnrolledStudents" VALUES(1,1,1,87.6);
INSERT INTO "EnrolledStudents" VALUES(2,2,2,92.1);
INSERT INTO "EnrolledStudents" VALUES(3,3,3,81.5);
INSERT INTO "EnrolledStudents" VALUES(4,4,4,79.3);
INSERT INTO "EnrolledStudents" VALUES(5,5,5,98.5);
CREATE TABLE School_Subjects (
    subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_name TEXT,
    core_concept_id INTEGER,
    advanced_topic_id INTEGER,
    FOREIGN KEY (core_concept_id) REFERENCES Core_Concepts(concept_id),
    FOREIGN KEY (advanced_topic_id) REFERENCES Advanced_Topics(topic_id)
);
INSERT INTO "School_Subjects" VALUES(1,'Introduction to Python',1,1);
INSERT INTO "School_Subjects" VALUES(2,'Linear Algebra Applications',2,NULL);
INSERT INTO "School_Subjects" VALUES(3,'Single Variable Calculus',3,2);
INSERT INTO "School_Subjects" VALUES(4,'Multivariable Calculus Theory',3,3);
INSERT INTO "School_Subjects" VALUES(5,'Tensor Methods in Physics',4,4);
CREATE TABLE Students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INT
);
INSERT INTO "Students" VALUES(1,'Alice',20);
INSERT INTO "Students" VALUES(2,'Bob',23);
INSERT INTO "Students" VALUES(3,'Charlie',21);
INSERT INTO "Students" VALUES(4,'Diana',22);
INSERT INTO "Students" VALUES(5,'Evan',24);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Core_Concepts',3);
INSERT INTO "sqlite_sequence" VALUES('Advanced_Topics',3);
INSERT INTO "sqlite_sequence" VALUES('School_Subjects',5);
INSERT INTO "sqlite_sequence" VALUES('EnrolledStudents',5);
COMMIT;
