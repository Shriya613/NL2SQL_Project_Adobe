BEGIN TRANSACTION;
CREATE TABLE "Projects" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT,
    description TEXT,
    textbook_id INTEGER,
    FOREIGN KEY(textbook_id) REFERENCES "Resources"(id)
);
INSERT INTO "Projects" VALUES(1,'Electric Circuit Experiment','Designing an electric circuit using various components.',1);
INSERT INTO "Projects" VALUES(2,'Acids and Bases Laboratory','Identifying different types of acids and bases.',2);
INSERT INTO "Projects" VALUES(3,'Molecular Model Building','Creating molecular models based on chemical formulas.',3);
INSERT INTO "Projects" VALUES(4,'Optical Spectroscopy','Analyzing light absorption patterns of elements and compounds.',4);
INSERT INTO "Projects" VALUES(5,'Calorimetry Measurements','Measuring heat capacity using a calorimeter.',5);
CREATE TABLE "Resources" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    author TEXT
);
INSERT INTO "Resources" VALUES(1,'Introduction to Computer Science','John Doe');
INSERT INTO "Resources" VALUES(2,'Physics for Scientists and Engineers','Serway & Jewett');
INSERT INTO "Resources" VALUES(3,'Chemistry: The Central Science','Paula B. Welch');
INSERT INTO "Resources" VALUES(4,'Biology: The Dynamic Universe','James S. Aber');
INSERT INTO "Resources" VALUES(5,'Mathematical Methods for Physicists','Michael Abramowitz');
CREATE TABLE Students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name TEXT,
    age INTEGER,
    class_id INTEGER
);
INSERT INTO "Students" VALUES(1,'Alice Smith',16,1);
INSERT INTO "Students" VALUES(2,'Bob Johnson',17,1);
INSERT INTO "Students" VALUES(3,'Charlie Brown',16,2);
INSERT INTO "Students" VALUES(4,'Diana Prince',17,2);
INSERT INTO "Students" VALUES(5,'Emma Watson',16,3);
CREATE TABLE Teachers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_name TEXT,
    subject TEXT,
    class_id INTEGER,
    FOREIGN KEY(class_id) REFERENCES Students(id)
);
INSERT INTO "Teachers" VALUES(1,'Mr. Green','Physics',1);
INSERT INTO "Teachers" VALUES(2,'Ms. Blue','Chemistry',2);
INSERT INTO "Teachers" VALUES(3,'Dr. Red','Biology',3);
INSERT INTO "Teachers" VALUES(4,'Professor White','Mathematics',1);
INSERT INTO "Teachers" VALUES(5,'Assistant Professor Black','Computer Science',1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Resources',5);
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Teachers',5);
INSERT INTO "sqlite_sequence" VALUES('Projects',5);
COMMIT;
