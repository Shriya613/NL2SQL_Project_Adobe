BEGIN TRANSACTION;
CREATE TABLE Assessors(
    AssessorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Title TEXT,
    Department TEXT
);
INSERT INTO "Assessors" VALUES(1,'Professor Smith','Associate Professor','Literature');
INSERT INTO "Assessors" VALUES(2,'Dr. Johnson','Full Professor','History');
INSERT INTO "Assessors" VALUES(3,'Ms. Williams','Assistant Professor','Political Science');
INSERT INTO "Assessors" VALUES(4,'Mr. Lee','Adjunct Instructor','Economics');
INSERT INTO "Assessors" VALUES(5,'Dr. Jackson','Associate Professor','Philosophy');
CREATE TABLE Exams(
    ExamID INTEGER PRIMARY KEY AUTOINCREMENT,
    CourseTitle TEXT,
    TotalMarks INT,
    PassingMark INT,
    AssessorId INTEGER,
    FOREIGN KEY (AssessorId) REFERENCES Assessors(AssessorID)
);
INSERT INTO "Exams" VALUES(1,'Introduction to English Literature',100,60,1);
INSERT INTO "Exams" VALUES(2,'World History',100,60,2);
INSERT INTO "Exams" VALUES(3,'American Politics',100,60,3);
INSERT INTO "Exams" VALUES(4,'Macroeconomic Theory',100,60,4);
INSERT INTO "Exams" VALUES(5,'Contemporary Ethics',100,60,5);
CREATE TABLE Quizzes(
    QuizID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT,
    MaxScore INT,
    MinPassingScore INT,
    AssessorId INTEGER,
    FOREIGN KEY (AssessorId) REFERENCES Assessors(AssessorID)
);
INSERT INTO "Quizzes" VALUES(1,'Weekly quiz on literary themes',20,12,1);
INSERT INTO "Quizzes" VALUES(2,'Midterm exam review questions',30,20,2);
INSERT INTO "Quizzes" VALUES(3,'Current events analysis',20,12,3);
INSERT INTO "Quizzes" VALUES(4,'Final exam study guide',30,20,4);
INSERT INTO "Quizzes" VALUES(5,'Argumentative essay practice',20,12,5);
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    Major TEXT
);
INSERT INTO "Students" VALUES(1,'John','Doe','1998-06-15','Computer Science');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2000-03-25','Mathematics');
INSERT INTO "Students" VALUES(3,'Alice','Brown','1997-11-10','Physics');
INSERT INTO "Students" VALUES(4,'Bob','White','1999-07-22','Chemistry');
INSERT INTO "Students" VALUES(5,'Charlie','Green','2001-04-05','Biology');
INSERT INTO "Students" VALUES(6,'David','Black','1996-12-18','English Literature');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',6);
INSERT INTO "sqlite_sequence" VALUES('Assessors',5);
INSERT INTO "sqlite_sequence" VALUES('Exams',5);
INSERT INTO "sqlite_sequence" VALUES('Quizzes',5);
COMMIT;
