BEGIN TRANSACTION;
CREATE TABLE Online_Courses(
    id INTEGER PRIMARY KEY,
    Course_Name TEXT,
    Instructor TEXT,
    Topic TEXT,
    Difficulty_Level TEXT,
    URL TEXT UNIQUE);
INSERT INTO "Online_Courses" VALUES(1,'Computer Science','John Doe','Programming','Intermediate','https://www.example.com/cs101');
INSERT INTO "Online_Courses" VALUES(2,'Ancient History','Jane Smith','History','Advanced','https://www.example.com/history201');
INSERT INTO "Online_Courses" VALUES(3,'Graphic Design','Mike Johnson','Design','Beginner','https://www.example.com/design101');
INSERT INTO "Online_Courses" VALUES(4,'Quantum Mechanics','Emma Wilson','Physics','Intermediate','https://www.example.com/qm101');
INSERT INTO "Online_Courses" VALUES(5,'Mathematical Logic','William Brown','Maths','Advanced','https://www.example.com/ml101');
CREATE TABLE Schools(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    City TEXT,
    State TEXT);
INSERT INTO "Schools" VALUES(1,'Springfield High School','123 Main St','Springfield','IL');
INSERT INTO "Schools" VALUES(2,'Evergreen Middle School','456 Oak Ave','Evergreen','CO');
INSERT INTO "Schools" VALUES(3,'Sunrise Elementary','789 Pine St','Sunrise','CA');
INSERT INTO "Schools" VALUES(4,'Oakwood College','123 Maple Dr','Oakwood','TX');
INSERT INTO "Schools" VALUES(5,'Riverview Academy','456 Cedar Ln','Riverview','FL');
CREATE TABLE Students(
    id INTEGER PRIMARY KEY,
    First_Name TEXT,
    Last_Name TEXT,
    Age INTEGER ,
    Grade_Level INTEGER );
INSERT INTO "Students" VALUES(1,'John','Smith',15,'9th');
INSERT INTO "Students" VALUES(2,'Jane','Doe',16,'10th');
INSERT INTO "Students" VALUES(3,'Tom','Hanks',14,'8th');
INSERT INTO "Students" VALUES(4,'Lucy','Grey',17,'11th');
INSERT INTO "Students" VALUES(5,'Michael','Jackson',16,'10th');
CREATE TABLE Teachers(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    School_ID INTEGER REFERENCES Schools(id),
    Degree TEXT,
    Years_Experience INTEGER );
INSERT INTO "Teachers" VALUES(1,'Mr. Thompson',1,'PhD in Math',10);
INSERT INTO "Teachers" VALUES(2,'Ms. Williams',2,'MA in English',5);
INSERT INTO "Teachers" VALUES(3,'Dr. Baker',3,'PhD in Physics',15);
INSERT INTO "Teachers" VALUES(4,'Ms. Rodriguez',4,'BSc in Biology',8);
INSERT INTO "Teachers" VALUES(5,'Prof. Lee',5,'PhD in Computer Science',20);
CREATE TABLE Textbooks(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Author TEXT,
    Subject TEXT,
    Difficulty_Level TEXT,
    ISBN TEXT UNIQUE);
INSERT INTO "Textbooks" VALUES(1,'Calculus','Isaac Newton','Maths','Intermediate','9783652003');
INSERT INTO "Textbooks" VALUES(2,'Physics','Albert Einstein','Physics','Advanced','9781543220');
INSERT INTO "Textbooks" VALUES(3,'English Literature','Shakespeare','Literature','Beginner','9780141029');
INSERT INTO "Textbooks" VALUES(4,'Biology','Charles Darwin','Science','Intermediate','9780123456');
INSERT INTO "Textbooks" VALUES(5,'History','Herodotus','History','Advanced','9787654321');
COMMIT;
