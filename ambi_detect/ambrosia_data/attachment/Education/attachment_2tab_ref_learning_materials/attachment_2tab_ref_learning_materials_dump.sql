BEGIN TRANSACTION;
CREATE TABLE Difficulty_Level(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Level TEXT UNIQUE);
INSERT INTO "Difficulty_Level" VALUES(1,'Beginner');
INSERT INTO "Difficulty_Level" VALUES(2,'Advanced');
INSERT INTO "Difficulty_Level" VALUES(3,'Expert');
CREATE TABLE EnrolledStudents(
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Age INTEGER,
    Email TEXT,
    PhoneNumber VARCHAR(15));
INSERT INTO "EnrolledStudents" VALUES(1,'Alice','Smith',20,'alicesmith@example.com','+1-123-456-7890');
INSERT INTO "EnrolledStudents" VALUES(2,'Bob','Johnson',21,'bobjohnson@example.com','+1-987-654-3210');
INSERT INTO "EnrolledStudents" VALUES(3,'Charlie','Williams',22,'charliewilliams@example.com','+1-555-123-4567');
INSERT INTO "EnrolledStudents" VALUES(4,'David','Martinez',23,'davidmartinez@example.com','+1-321-654-9870');
INSERT INTO "EnrolledStudents" VALUES(5,'Eva','Garcia',24,'evagarcia@example.com','+1-789-456-1234');
CREATE TABLE Online_Courses(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    CourseName TEXT,
    Instructor TEXT,
    Platform TEXT,
    URL TEXT,
    SubjectID INTEGER,
    FOREIGN KEY(SubjectID) REFERENCES Subjects(ID));
INSERT INTO "Online_Courses" VALUES(1,'Introductory Math','Prof. James Johnson','Udemy','https://www.udemy.com/introductory-math',1);
INSERT INTO "Online_Courses" VALUES(2,'Essential Physics','Dr. Emily White','Coursera','https://www.coursera.org/physics',2);
INSERT INTO "Online_Courses" VALUES(3,'Foundations of Biology','Dr. Robert Green','edX','https://www.edx.org/biology',3);
INSERT INTO "Online_Courses" VALUES(4,'Global History Survey','Dr. Susan King','FutureLearn','https://www.futurelearn.com/history',4);
INSERT INTO "Online_Courses" VALUES(5,'Great Literary Works','Ms. Karen Taylor','LinkedIn Learning','https://www.linkedinlearning.com/literature',5);
CREATE TABLE Subjects(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT,
    DifficultyID INTEGER,
    FOREIGN KEY(DifficultyID) REFERENCES Difficulty_Level(ID));
INSERT INTO "Subjects" VALUES(1,'Math','Introduction to Mathematics',1);
INSERT INTO "Subjects" VALUES(2,'Physics','Basic Principles of Physics',2);
INSERT INTO "Subjects" VALUES(3,'Biology','Basics of Biological Sciences',3);
INSERT INTO "Subjects" VALUES(4,'History','World History Overview',1);
INSERT INTO "Subjects" VALUES(5,'Literature','Classic Novels Analysis',2);
CREATE TABLE Textbooks(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Author TEXT,
    ISBN TEXT,
    SubjectID INTEGER,
    FOREIGN KEY(SubjectID) REFERENCES Subjects(ID));
INSERT INTO "Textbooks" VALUES(1,'Mathematics for Beginners','John Doe','978-0-123-456789',1);
INSERT INTO "Textbooks" VALUES(2,'The World of Physics','Jane Smith','978-0-091-234567',2);
INSERT INTO "Textbooks" VALUES(3,'An Introduction to Biology','Michael Brown','978-1-234-567890',3);
INSERT INTO "Textbooks" VALUES(4,'A Brief History of Time','Stephen Hawking','978-0-123-987654',4);
INSERT INTO "Textbooks" VALUES(5,'To Kill a Mockingbird','Harper Lee','978-0-123-123456',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Difficulty_Level',3);
INSERT INTO "sqlite_sequence" VALUES('Subjects',5);
INSERT INTO "sqlite_sequence" VALUES('Textbooks',5);
INSERT INTO "sqlite_sequence" VALUES('Online_Courses',5);
INSERT INTO "sqlite_sequence" VALUES('EnrolledStudents',5);
COMMIT;
