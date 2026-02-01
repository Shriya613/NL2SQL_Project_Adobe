BEGIN TRANSACTION;
CREATE TABLE "Assignments" (
    assignment_id INTEGER PRIMARY KEY,
    "Timing" DATE,
    description TEXT,
    grade FLOAT
);
INSERT INTO "Assignments" VALUES(1,'2023-01-16','Write a paper on computer science history',9.0);
INSERT INTO "Assignments" VALUES(2,'2023-02-08','Complete calculus homework set 7',8.5);
INSERT INTO "Assignments" VALUES(3,'2023-02-22','Read and analyze two novels',8.0);
INSERT INTO "Assignments" VALUES(4,'2023-03-14','Present biology lab report',9.5);
INSERT INTO "Assignments" VALUES(5,'2023-03-28','Research and write an essay on global warming',9.0);
CREATE TABLE Courses (
    course_id INTEGER PRIMARY KEY,
    course_name TEXT,
    instructor_name TEXT
);
INSERT INTO "Courses" VALUES(1,'Introduction to Computer Science','Dr. Smith');
INSERT INTO "Courses" VALUES(2,'Calculus I','Professor Brown');
INSERT INTO "Courses" VALUES(3,'English Literature','Dr. Green');
INSERT INTO "Courses" VALUES(4,'World History','Professor White');
INSERT INTO "Courses" VALUES(5,'Biology','Dr. Doe');
CREATE TABLE Enrollments (
    enrollment_id INTEGER PRIMARY KEY,
    student_id INTEGER,
    course_id INTEGER,
    FOREIGN KEY(student_id) REFERENCES Students(id),
    FOREIGN KEY(course_id) REFERENCES Courses(course_id)
);
INSERT INTO "Enrollments" VALUES(1,1,1);
INSERT INTO "Enrollments" VALUES(2,2,3);
INSERT INTO "Enrollments" VALUES(3,3,2);
INSERT INTO "Enrollments" VALUES(4,4,4);
INSERT INTO "Enrollments" VALUES(5,5,5);
CREATE TABLE Grades (
    grade_id INTEGER PRIMARY KEY,
    student_id INTEGER,
    course_id INTEGER,
    final_grade FLOAT,
    semester TEXT,
    FOREIGN KEY(student_id) REFERENCES Students(id),
    FOREIGN KEY(course_id) REFERENCES Courses(course_id)
);
INSERT INTO "Grades" VALUES(1,1,1,8.5,'Fall 2022');
INSERT INTO "Grades" VALUES(2,2,3,8.8,'Spring 2023');
INSERT INTO "Grades" VALUES(3,3,2,7.9,'Fall 2022');
INSERT INTO "Grades" VALUES(4,4,4,9.3,'Winter 2023');
INSERT INTO "Grades" VALUES(5,5,5,9.7,'Spring 2023');
CREATE TABLE Students (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Students" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Students" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Students" VALUES(3,'Alice','Brown','alice.brown@example.com');
INSERT INTO "Students" VALUES(4,'Michael','Green','michael.green@example.com');
INSERT INTO "Students" VALUES(5,'Emma','White','emma.white@example.com');
CREATE TABLE Submissions (
    submission_id INTEGER PRIMARY KEY,
    student_id INTEGER,
    assignment_id INTEGER,
    file_path TEXT,
    score FLOAT,
    submitted_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(student_id) REFERENCES Students(id),
    FOREIGN KEY(assignment_id) REFERENCES Assignments(assignment_id)
);
INSERT INTO "Submissions" VALUES(1,1,1,'/home/user/submission1.pdf',9.5,'2024-02-19 14:44:34');
INSERT INTO "Submissions" VALUES(2,2,2,'/home/user/submission2.docx',8.8,'2024-02-19 14:44:34');
INSERT INTO "Submissions" VALUES(3,3,3,'/home/user/submission3.pdf',8.2,'2024-02-19 14:44:34');
INSERT INTO "Submissions" VALUES(4,4,4,'/home/user/submission4.pptx',9.3,'2024-02-19 14:44:34');
INSERT INTO "Submissions" VALUES(5,5,5,'/home/user/submission5.txt',9.7,'2024-02-19 14:44:34');
COMMIT;
