BEGIN TRANSACTION;
CREATE TABLE Assignments(
    assignmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT ,
    description TEXT);
INSERT INTO "Assignments" VALUES(1,'Homework','Weekly homework assignments');
INSERT INTO "Assignments" VALUES(2,'Quiz','Weekly quizzes');
INSERT INTO "Assignments" VALUES(3,'Project','Group project');
INSERT INTO "Assignments" VALUES(4,'Essay','Research essay');
INSERT INTO "Assignments" VALUES(5,'Exam','Final exam');
CREATE TABLE Courses(
    courseID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    credits INT ,
    professorID INTEGER,
    FOREIGN KEY(professorID) REFERENCES Professors(professorID)
);
INSERT INTO "Courses" VALUES(1,'Calculus I',4,1);
INSERT INTO "Courses" VALUES(2,'General Physics',3,2);
INSERT INTO "Courses" VALUES(3,'Introduction to Biology',3,3);
INSERT INTO "Courses" VALUES(4,'Discrete Mathematics',3,4);
INSERT INTO "Courses" VALUES(5,'World History',3,5);
CREATE TABLE Courses_Assignments(
    coursesAssignmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    courseID INTEGER,
    assignmentID INTEGER,
    FOREIGN KEY(courseID) REFERENCES Courses(courseID),
    FOREIGN KEY(assignmentID) REFERENCES Assignments(assignmentID)
);
INSERT INTO "Courses_Assignments" VALUES(1,1,1);
INSERT INTO "Courses_Assignments" VALUES(2,1,2);
INSERT INTO "Courses_Assignments" VALUES(3,2,3);
INSERT INTO "Courses_Assignments" VALUES(4,2,4);
INSERT INTO "Courses_Assignments" VALUES(5,3,5);
INSERT INTO "Courses_Assignments" VALUES(6,1,3);
INSERT INTO "Courses_Assignments" VALUES(7,3,3);
INSERT INTO "Courses_Assignments" VALUES(8,4,3);
INSERT INTO "Courses_Assignments" VALUES(9,5,3);
CREATE TABLE Departments(
    departmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    head TEXT
);
INSERT INTO "Departments" VALUES(1,'Mathematics','Dr. Johnson');
INSERT INTO "Departments" VALUES(2,'Physics','Prof. Williams');
INSERT INTO "Departments" VALUES(3,'Biology','Dr. Martinez');
INSERT INTO "Departments" VALUES(4,'Computer Science','Dr. Lee');
INSERT INTO "Departments" VALUES(5,'History','Prof. Clark');
CREATE TABLE Enrollments(
    enrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    studentID INTEGER,
    courseID INTEGER,
    grade DECIMAL(2,1) ,
    FOREIGN KEY(studentID) REFERENCES Students(studentID),
    FOREIGN KEY(courseID) REFERENCES Courses(courseID)
);
INSERT INTO "Enrollments" VALUES(1,1,1,NULL);
INSERT INTO "Enrollments" VALUES(2,2,1,NULL);
INSERT INTO "Enrollments" VALUES(3,3,2,NULL);
INSERT INTO "Enrollments" VALUES(4,4,2,NULL);
INSERT INTO "Enrollments" VALUES(5,5,3,NULL);
CREATE TABLE Professors(
    professorID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    departmentID INTEGER,
    FOREIGN KEY(departmentID) REFERENCES Departments(departmentID)
);
INSERT INTO "Professors" VALUES(1,'Dr.','Johnson',1);
INSERT INTO "Professors" VALUES(2,'Prof.','Williams',2);
INSERT INTO "Professors" VALUES(3,'Dr.','Martinez',3);
INSERT INTO "Professors" VALUES(4,'Dr.','Lee',4);
INSERT INTO "Professors" VALUES(5,'Prof.','Clark',5);
CREATE TABLE Students(
    studentID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT
);
INSERT INTO "Students" VALUES(1,'John','Doe');
INSERT INTO "Students" VALUES(2,'Jane','Smith');
INSERT INTO "Students" VALUES(3,'Michael','Brown');
INSERT INTO "Students" VALUES(4,'Emily','White');
INSERT INTO "Students" VALUES(5,'David','Green');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Departments',5);
INSERT INTO "sqlite_sequence" VALUES('Professors',5);
INSERT INTO "sqlite_sequence" VALUES('Courses',5);
INSERT INTO "sqlite_sequence" VALUES('Assignments',5);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',5);
INSERT INTO "sqlite_sequence" VALUES('Courses_Assignments',9);
COMMIT;
