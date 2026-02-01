BEGIN TRANSACTION;
CREATE TABLE Colleges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Colleges" VALUES(1,'Computer Science','San Francisco');
INSERT INTO "Colleges" VALUES(2,'Mathematics','Palo Alto');
CREATE TABLE CourseEnrollments (
    student_id INTEGER,
    course_id INTEGER,
    semester TEXT,
    year INTEGER,
    grade TEXT,
    FOREIGN KEY(student_id) REFERENCES Students(id),
    FOREIGN KEY(course_id) REFERENCES Courses(id)
);
INSERT INTO "CourseEnrollments" VALUES(1,1,'Spring',2021,'A');
INSERT INTO "CourseEnrollments" VALUES(2,2,'Fall',2021,'B+');
CREATE TABLE Courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    credits INTEGER 
);
INSERT INTO "Courses" VALUES(1,'Introduction to Computer Science','An overview of computer science concepts and programming languages.',4);
INSERT INTO "Courses" VALUES(2,'Calculus I','Basic principles of calculus including limits, derivatives, and integrals.',4);
CREATE TABLE Professors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    department TEXT,
    office_location TEXT
);
INSERT INTO "Professors" VALUES(1,'Dr.','Johnson','Computer Science','Room 123');
INSERT INTO "Professors" VALUES(2,'Professor','Williams','Mathematics','Room 456');
CREATE TABLE "Students" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number TEXT,
    Offering_Department TEXT);
INSERT INTO "Students" VALUES(1,'John','Doe','john.doe@example.com','555-123-4567','Computer Science');
INSERT INTO "Students" VALUES(2,'Jane','Smith','jane.smith@example.com','555-987-6543','Mathematics');
CREATE TABLE Universities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    city TEXT,
    state TEXT,
    country TEXT
);
INSERT INTO "Universities" VALUES(1,'University of California','Los Angeles','CA','USA');
INSERT INTO "Universities" VALUES(2,'Stanford University','Stanford','CA','USA');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Colleges',2);
INSERT INTO "sqlite_sequence" VALUES('Universities',2);
INSERT INTO "sqlite_sequence" VALUES('Courses',2);
INSERT INTO "sqlite_sequence" VALUES('Professors',2);
INSERT INTO "sqlite_sequence" VALUES('Students',2);
COMMIT;
