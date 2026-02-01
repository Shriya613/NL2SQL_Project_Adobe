BEGIN TRANSACTION;
CREATE TABLE Authors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Authors" VALUES(1,'John Doe','john.doe@example.com');
INSERT INTO "Authors" VALUES(2,'Jane Smith','jane.smith@example.com');
INSERT INTO "Authors" VALUES(3,'Michael Johnson','michael.johnson@example.com');
INSERT INTO "Authors" VALUES(4,'Emily Davis','emily.davis@example.com');
INSERT INTO "Authors" VALUES(5,'David Brown','david.brown@example.com');
CREATE TABLE "Courses" (
    id INTEGER PRIMARY KEY,
    course_name TEXT,
    textbook_id INTEGER,
    FOREIGN KEY(textbook_id) REFERENCES "Resources"(id));
INSERT INTO "Courses" VALUES(1,'Computer Science 101',1);
INSERT INTO "Courses" VALUES(2,'Engineering Mathematics',2);
INSERT INTO "Courses" VALUES(3,'Introductory Organic Chemistry',3);
INSERT INTO "Courses" VALUES(4,'Advanced Physics',4);
INSERT INTO "Courses" VALUES(5,'Data Structures and Algorithms',5);
CREATE TABLE Grades (
    id INTEGER PRIMARY KEY,
    grade INT ,
    student_id INTEGER,
    course_id INTEGER,
    FOREIGN KEY(student_id) REFERENCES Students(id),
    FOREIGN KEY(course_id) REFERENCES Courses(id)
);
INSERT INTO "Grades" VALUES(1,85,1,1);
INSERT INTO "Grades" VALUES(2,78,2,2);
INSERT INTO "Grades" VALUES(3,92,3,3);
INSERT INTO "Grades" VALUES(4,65,4,4);
INSERT INTO "Grades" VALUES(5,80,5,5);
CREATE TABLE "Resources" (
    id INTEGER PRIMARY KEY,
    title TEXT,
    author_id INTEGER,
    FOREIGN KEY(author_id) REFERENCES Authors(id)
);
INSERT INTO "Resources" VALUES(1,'Introduction to Computer Science',1);
INSERT INTO "Resources" VALUES(2,'Calculus for Engineers',2);
INSERT INTO "Resources" VALUES(3,'General Chemistry',3);
INSERT INTO "Resources" VALUES(4,'Organic Chemistry',4);
INSERT INTO "Resources" VALUES(5,'Physics for Scientists and Engineers',5);
CREATE TABLE Students (
    id INTEGER PRIMARY KEY,
    student_name TEXT,
    enrolled_course_id INTEGER,
    FOREIGN KEY(enrolled_course_id) REFERENCES Courses(id)
);
INSERT INTO "Students" VALUES(1,'Alice Smith',1);
INSERT INTO "Students" VALUES(2,'Bob Johnson',2);
INSERT INTO "Students" VALUES(3,'Carol Williams',3);
INSERT INTO "Students" VALUES(4,'Dave Brown',4);
INSERT INTO "Students" VALUES(5,'Eva Green',5);
COMMIT;
