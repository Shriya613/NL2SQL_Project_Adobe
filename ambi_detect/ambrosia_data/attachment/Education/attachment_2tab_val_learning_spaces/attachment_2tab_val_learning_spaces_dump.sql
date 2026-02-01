BEGIN TRANSACTION;
CREATE TABLE Art_Studio(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Capacity INTEGER,
    Available_Resources TEXT
);
INSERT INTO "Art_Studio" VALUES(1,'VanGogh Studio',15,'Paints');
INSERT INTO "Art_Studio" VALUES(2,'Dali Studio',20,'Computers');
INSERT INTO "Art_Studio" VALUES(3,'Warhol Studio',18,'Canvas');
CREATE TABLE Assigns(
    TeacherID INTEGER REFERENCES Teacher(ID),
    StudentID INTEGER REFERENCES Student(ID),
    ClassID INTEGER REFERENCES Class(ID)
);
INSERT INTO "Assigns" VALUES(1,1,1);
INSERT INTO "Assigns" VALUES(2,2,2);
INSERT INTO "Assigns" VALUES(3,3,3);
INSERT INTO "Assigns" VALUES(4,4,4);
INSERT INTO "Assigns" VALUES(5,5,5);
CREATE TABLE Class(
    ID INTEGER PRIMARY KEY,
    Classroom TEXT,
    Time INTEGER,
    Day TEXT
);
INSERT INTO "Class" VALUES(1,'Room A',9,'Monday');
INSERT INTO "Class" VALUES(2,'Room B',10,'Tuesday');
INSERT INTO "Class" VALUES(3,'Room C',11,'Wednesday');
INSERT INTO "Class" VALUES(4,'Room D',12,'Thursday');
INSERT INTO "Class" VALUES(5,'Room E',13,'Friday');
CREATE TABLE Science_Lab(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Capacity INTEGER,
    Available_Resources TEXT
);
INSERT INTO "Science_Lab" VALUES(1,'Newton Lab',20,'Microscopes');
INSERT INTO "Science_Lab" VALUES(2,'Einstein Lab',30,'Computers');
INSERT INTO "Science_Lab" VALUES(3,'Hawking Lab',25,'Chemicals');
CREATE TABLE Student(
    ID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Age INTEGER
);
INSERT INTO "Student" VALUES(1,'Tom','Jones',16);
INSERT INTO "Student" VALUES(2,'Emma','Williams',17);
INSERT INTO "Student" VALUES(3,'Lucas','Taylor',15);
INSERT INTO "Student" VALUES(4,'Olivia','Thompson',14);
INSERT INTO "Student" VALUES(5,'Noah','Jackson',13);
CREATE TABLE Teacher(
    ID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Subject TEXT
);
INSERT INTO "Teacher" VALUES(1,'John','Smith','Math');
INSERT INTO "Teacher" VALUES(2,'Jane','Doe','Physics');
INSERT INTO "Teacher" VALUES(3,'Alice','Brown','English');
INSERT INTO "Teacher" VALUES(4,'Bob','White','Biology');
INSERT INTO "Teacher" VALUES(5,'Charlie','Black','History');
COMMIT;
