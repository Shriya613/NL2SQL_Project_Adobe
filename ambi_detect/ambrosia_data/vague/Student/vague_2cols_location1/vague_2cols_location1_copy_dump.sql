BEGIN TRANSACTION;
CREATE TABLE Courses (
    Course_id INT PRIMARY KEY,
    Title TEXT,
    Description TEXT,
    Credits INT,
    Teacher_id INT REFERENCES Teachers(Teacher_Id)
);
INSERT INTO "Courses" VALUES(1,'Introduction to Computer Science','An overview of computer science fundamentals',4,1);
INSERT INTO "Courses" VALUES(2,'Data Structures and Algorithms','Advanced study of data structures and algorithms',4,2);
INSERT INTO "Courses" VALUES(3,'Calculus I','Fundamental concepts of calculus',4,3);
INSERT INTO "Courses" VALUES(4,'Calculus II','More advanced topics in calculus',4,3);
INSERT INTO "Courses" VALUES(5,'General Physics','Basic principles of physics',4,5);
CREATE TABLE Departments (
    Department_ID INT PRIMARY KEY,
    Name TEXT,
    Location TEXT
);
INSERT INTO "Departments" VALUES(1,'Computer Science','Building A, Room 201');
INSERT INTO "Departments" VALUES(2,'Mathematics','Building B, Room 301');
INSERT INTO "Departments" VALUES(3,'Physics','Building C, Room 401');
INSERT INTO "Departments" VALUES(4,'Biology','Building D, Room 501');
INSERT INTO "Departments" VALUES(5,'History','Building E, Room 601');
CREATE TABLE Enrollments (
    Student_id INT REFERENCES Students(ID),
    Course_id INT REFERENCES Courses(Course_id),
    Semester_year YEAR,
    Grade CHAR(2) ,
    FOREIGN KEY (Student_id, Course_id, Semester_year) REFERENCES Schedule(Student_id, Course_id, Semester_year)
);
INSERT INTO "Enrollments" VALUES(1,1,2021,'A');
INSERT INTO "Enrollments" VALUES(1,2,2021,'B');
INSERT INTO "Enrollments" VALUES(2,3,2021,'A');
INSERT INTO "Enrollments" VALUES(2,4,2021,'B');
INSERT INTO "Enrollments" VALUES(3,5,2021,'A');
CREATE TABLE "Rooms" (
    Location INTEGER PRIMARY KEY,
    Capacity INT,
    Lectures TEXT
);
INSERT INTO "Rooms" VALUES(101,50,'CS101 - Introduction to Computer Science');
INSERT INTO "Rooms" VALUES(102,40,'CS102 - Data Structures and Algorithms');
INSERT INTO "Rooms" VALUES(103,30,'MA101 - Calculus I');
INSERT INTO "Rooms" VALUES(104,35,'MA102 - Calculus II');
INSERT INTO "Rooms" VALUES(105,25,'PH101 - General Physics');
CREATE TABLE Students (
    ID INT PRIMARY KEY,
    First_name TEXT,
    Last_name TEXT,
    Date_of_birth DATE,
    Email VARCHAR(100) UNIQUE
);
INSERT INTO "Students" VALUES(1,'John','Doe','1998-06-15','john.doe@example.com');
INSERT INTO "Students" VALUES(2,'Jane','Smith','1997-03-20','jane.smith@example.com');
INSERT INTO "Students" VALUES(3,'Michael','Brown','1996-09-05','michael.brown@example.com');
INSERT INTO "Students" VALUES(4,'Emily','White','1999-11-25','emily.white@example.com');
INSERT INTO "Students" VALUES(5,'William','Black','2000-02-10','william.black@example.com');
CREATE TABLE Teachers (
    Teacher_Id INT PRIMARY KEY,
    First_name TEXT,
    Last_name TEXT,
    Hire_date DATE,
    Office_room_number INTEGER REFERENCES Rooms(Location)
);
INSERT INTO "Teachers" VALUES(1,'Dr.','Smith','2015-05-01',101);
INSERT INTO "Teachers" VALUES(2,'Prof.','Jones','2010-08-01',102);
INSERT INTO "Teachers" VALUES(3,'Assoc. Prof.','Martin','2012-11-01',103);
INSERT INTO "Teachers" VALUES(4,'Instructor','Wilson','2018-02-01',104);
INSERT INTO "Teachers" VALUES(5,'Lecturer','Garcia','2013-06-01',105);
COMMIT;
