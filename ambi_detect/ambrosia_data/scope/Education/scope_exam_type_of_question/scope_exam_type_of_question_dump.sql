BEGIN TRANSACTION;
CREATE TABLE Exam (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    examName TEXT,
    date DATE
);
INSERT INTO "Exam" VALUES(1,'Math Test','2021-10-01');
INSERT INTO "Exam" VALUES(2,'English Test','2021-10-02');
INSERT INTO "Exam" VALUES(3,'History Test','2021-10-03');
INSERT INTO "Exam" VALUES(4,'Science Test','2021-10-04');
INSERT INTO "Exam" VALUES(5,'Art Test','2021-10-05');
CREATE TABLE Exam_Type_Of_Question (
    examID INTEGER,
    questionTypeID INTEGER,
    FOREIGN KEY(examID) REFERENCES Exam(id),
    FOREIGN KEY(questionTypeID) REFERENCES Type_Of_Question(id)
);
INSERT INTO "Exam_Type_Of_Question" VALUES(1,1);
INSERT INTO "Exam_Type_Of_Question" VALUES(2,1);
INSERT INTO "Exam_Type_Of_Question" VALUES(3,1);
INSERT INTO "Exam_Type_Of_Question" VALUES(4,1);
INSERT INTO "Exam_Type_Of_Question" VALUES(5,1);
INSERT INTO "Exam_Type_Of_Question" VALUES(5,5);
CREATE TABLE Student_Grades (
    studentId INTEGER,
    examId INTEGER,
    grade INT,
    FOREIGN KEY(studentId) REFERENCES Students(studentId),
    FOREIGN KEY(examId) REFERENCES Exam(id)
);
CREATE TABLE Students (
    studentId INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    enrollmentDate DATE
);
INSERT INTO "Students" VALUES(1,'John','Doe','2019-08-16');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2019-08-17');
INSERT INTO "Students" VALUES(3,'Alice','Brown','2019-08-18');
INSERT INTO "Students" VALUES(4,'Mike','Jackson','2019-08-19');
INSERT INTO "Students" VALUES(5,'Emily','White','2019-08-20');
CREATE TABLE Type_Of_Question (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    questionType TEXT
);
INSERT INTO "Type_Of_Question" VALUES(1,'Multiple_Choice');
INSERT INTO "Type_Of_Question" VALUES(2,'True_False');
INSERT INTO "Type_Of_Question" VALUES(3,'Short_Answer');
INSERT INTO "Type_Of_Question" VALUES(4,'Essay');
INSERT INTO "Type_Of_Question" VALUES(5,'Fill_In_The_Blank');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Type_Of_Question',5);
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Exam',5);
COMMIT;
