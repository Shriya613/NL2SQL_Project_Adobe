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
CREATE TABLE Degree_program (
    id INTEGER PRIMARY KEY,
    degree_name TEXT,
    duration INT 
);
INSERT INTO "Degree_program" VALUES(1,'Bachelor of Computer Science',4);
INSERT INTO "Degree_program" VALUES(2,'Master of Mathematics',2);
INSERT INTO "Degree_program" VALUES(3,'PhD Physics',6);
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
CREATE TABLE Field_of_study (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);
INSERT INTO "Field_of_study" VALUES(1,'Computer Science','The study of computers and computation.');
INSERT INTO "Field_of_study" VALUES(2,'Mathematics','The study of numbers, quantity, structure, space, and change.');
INSERT INTO "Field_of_study" VALUES(3,'Physics','The natural science that involves the study of matter and energy.');
CREATE TABLE Student (
    id INTEGER PRIMARY KEY,
    email TEXT,
    field_of_study_id INTEGER REFERENCES Field_of_study(id),
    degree_program_id INTEGER REFERENCES Degree_program(id)
, full_name TEXT);
INSERT INTO "Student" VALUES(1,'alice@example.com',1,1,'Alice Smith');
INSERT INTO "Student" VALUES(2,'bob@example.com',2,2,'Bob Jones');
INSERT INTO "Student" VALUES(3,'carol@example.com',3,3,'Carol Williams');
COMMIT;
