BEGIN TRANSACTION;
CREATE TABLE Courses(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    courseName TEXT,
    instructorId INTEGER REFERENCES Instructors(id) ON DELETE SET NULL
);
INSERT INTO "Courses" VALUES(1,'Introduction to Computer Science',1);
INSERT INTO "Courses" VALUES(2,'Advanced Calculus',2);
INSERT INTO "Courses" VALUES(3,'General Physics',3);
INSERT INTO "Courses" VALUES(4,'Ecology',4);
INSERT INTO "Courses" VALUES(5,'Organic Chemistry',5);
CREATE TABLE Instructors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    department TEXT
);
INSERT INTO "Instructors" VALUES(1,'Dr. Smith','Computer Science');
INSERT INTO "Instructors" VALUES(2,'Prof. Johnson','Mathematics');
INSERT INTO "Instructors" VALUES(3,'Dr. Brown','Physics');
INSERT INTO "Instructors" VALUES(4,'Dr. Wilson','Biology');
INSERT INTO "Instructors" VALUES(5,'Dr. Taylor','Chemistry');
CREATE TABLE Projects(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    researchMethod TEXT ,
    topic TEXT ,
    studentId INTEGER REFERENCES Students(id) ON DELETE CASCADE,
    courseId INTEGER REFERENCES Courses(id) ON DELETE CASCADE
);
INSERT INTO "Projects" VALUES(1,'Understanding AI Ethics','Case Study','Artificial Intelligence',1,1);
INSERT INTO "Projects" VALUES(2,'Exploring Patterns in Mathematical Problems','Case Study','Mathematics',2,2);
INSERT INTO "Projects" VALUES(3,'The Impact of Light Pollution on Wildlife','Case Study','Environmental Science',3,3);
INSERT INTO "Projects" VALUES(4,'An Interview with Dr. Smith','Interview','Computer Science',4,6);
INSERT INTO "Projects" VALUES(5,'Examining Sustainable Practices in Ecotourism','Interview','Environmental Science',5,7);
INSERT INTO "Projects" VALUES(6,'Impacts of Climate Change on Biodiversity','Case Study','Environmental Science',6,8);
INSERT INTO "Projects" VALUES(7,'Investigating Learning Styles in STEM Fields','Case Study','Education',7,9);
INSERT INTO "Projects" VALUES(8,'Perspectives on Renewable Energy','Interview','Environmental Science',8,10);
INSERT INTO "Projects" VALUES(9,'A Survey of Public Opinion on Genetic Engineering','Interview','Biotechnology',9,11);
CREATE TABLE Students(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    major TEXT
);
INSERT INTO "Students" VALUES(1,'Alice',20,'Computer Science');
INSERT INTO "Students" VALUES(2,'Bob',21,'Mathematics');
INSERT INTO "Students" VALUES(3,'Charlie',22,'Physics');
INSERT INTO "Students" VALUES(4,'David',23,'Biology');
INSERT INTO "Students" VALUES(5,'Eva',24,'Chemistry');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Instructors',5);
INSERT INTO "sqlite_sequence" VALUES('Courses',5);
INSERT INTO "sqlite_sequence" VALUES('Projects',9);
COMMIT;
