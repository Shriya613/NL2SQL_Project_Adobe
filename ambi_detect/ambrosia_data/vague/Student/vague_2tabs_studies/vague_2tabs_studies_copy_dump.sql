BEGIN TRANSACTION;
CREATE TABLE Course (
    id INTEGER PRIMARY KEY,
    course_code TEXT,
    title TEXT,
    credit INT ,
    faculty_id INTEGER REFERENCES Faculty(id)
);
INSERT INTO "Course" VALUES(1,'CS101','Introduction to Programming',3,1);
INSERT INTO "Course" VALUES(2,'MATH201','Calculus I',4,2);
INSERT INTO "Course" VALUES(3,'PHYS101','General Physics',3,3);
CREATE TABLE Department (
    id INTEGER PRIMARY KEY,
    department_name TEXT,
    head_id INTEGER REFERENCES Employee(id)
);
INSERT INTO "Department" VALUES(1,'Computer Science',NULL);
INSERT INTO "Department" VALUES(2,'Mathematics',NULL);
INSERT INTO "Department" VALUES(3,'Physics',NULL);
CREATE TABLE Enrollment (
    id INTEGER PRIMARY KEY,
    student_id INTEGER REFERENCES Student(id),
    course_id INTEGER REFERENCES Course(id)
);
INSERT INTO "Enrollment" VALUES(1,1,1);
INSERT INTO "Enrollment" VALUES(2,2,2);
INSERT INTO "Enrollment" VALUES(3,3,3);
CREATE TABLE Faculty (
    id INTEGER PRIMARY KEY,
    faculty_name TEXT,
    department_id INTEGER REFERENCES Department(id)
);
INSERT INTO "Faculty" VALUES(1,'Dr. John Smith',1);
INSERT INTO "Faculty" VALUES(2,'Professor Jane Doe',2);
INSERT INTO "Faculty" VALUES(3,'Assistant Professor Bob Johnson',3);
CREATE TABLE "Student" (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    field_of_study_id INTEGER REFERENCES "Studies"(id));
INSERT INTO "Student" VALUES(1,'Alice','Smith','alice@example.com',1);
INSERT INTO "Student" VALUES(2,'Bob','Jones','bob@example.com',2);
INSERT INTO "Student" VALUES(3,'Carol','Williams','carol@example.com',3);
CREATE TABLE "Studies" (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);
INSERT INTO "Studies" VALUES(1,'Computer Science','The study of computers and computation.');
INSERT INTO "Studies" VALUES(2,'Mathematics','The study of numbers, quantity, structure, space, and change.');
INSERT INTO "Studies" VALUES(3,'Physics','The natural science that involves the study of matter and energy.');
COMMIT;
