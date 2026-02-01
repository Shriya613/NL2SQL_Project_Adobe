BEGIN TRANSACTION;
CREATE TABLE Classes(
    ClassID INTEGER PRIMARY KEY AUTOINCREMENT,
    TeacherID INTEGER,
    StudentID INTEGER,
    StartTime TIME,
    EndTime TIME,
    Classroom VARCHAR(10),
    FOREIGN KEY (TeacherID) REFERENCES Teachers(ID),
    FOREIGN KEY (StudentID) REFERENCES Students(ID)
);
INSERT INTO "Classes" VALUES(1,1,1,'10:00','11:00','Room A');
INSERT INTO "Classes" VALUES(2,2,2,'11:30','12:30','Room B');
INSERT INTO "Classes" VALUES(3,3,3,'1:00','2:00','Room C');
INSERT INTO "Classes" VALUES(4,4,4,'2:30','3:30','Room D');
INSERT INTO "Classes" VALUES(5,5,5,'4:00','5:00','Room E');
CREATE TABLE Illustrations(
    IllustID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT,
    PathToIllust BLOB,
    SubjectID INTEGER,
    FOREIGN KEY (SubjectID) REFERENCES Subjects(ID)
);
INSERT INTO "Illustrations" VALUES(1,'An illustration of photosynthesis.','path/to/illust1.png',1);
INSERT INTO "Illustrations" VALUES(2,'A map of ancient Rome.','path/to/illust2.png',2);
INSERT INTO "Illustrations" VALUES(3,'The periodic table of elements.','path/to/illust3.png',3);
INSERT INTO "Illustrations" VALUES(4,'A depiction of the Big Bang theory.','path/to/illust4.png',4);
INSERT INTO "Illustrations" VALUES(5,'Darwin''s theory of evolution.','path/to/illust5.png',5);
CREATE TABLE Photos(
    PhotoID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT,
    PathToPhoto BLOB,
    SubjectID INTEGER,
    FOREIGN KEY (SubjectID) REFERENCES Subjects(ID)
);
INSERT INTO "Photos" VALUES(1,'A beautiful sunrise over a lake.','path/to/photo1.jpg',1);
INSERT INTO "Photos" VALUES(2,'Autumn leaves falling in the park.','path/to/photo2.jpg',2);
INSERT INTO "Photos" VALUES(3,'A group of students learning about nature.','path/to/photo3.jpg',3);
INSERT INTO "Photos" VALUES(4,'Animals in their natural habitat.','path/to/photo4.jpg',4);
INSERT INTO "Photos" VALUES(5,'The wonders of outer space.','path/to/photo5.jpg',5);
CREATE TABLE Students(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INT,
    EnrollmentDate DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "Students" VALUES(1,'Alice',19,'2024-03-13 20:58:03');
INSERT INTO "Students" VALUES(2,'Bob',21,'2024-03-13 20:58:03');
INSERT INTO "Students" VALUES(3,'Charlie',20,'2024-03-13 20:58:03');
INSERT INTO "Students" VALUES(4,'David',18,'2024-03-13 20:58:03');
INSERT INTO "Students" VALUES(5,'Eva',22,'2024-03-13 20:58:03');
CREATE TABLE Subjects(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE
);
INSERT INTO "Subjects" VALUES(1,'Math');
INSERT INTO "Subjects" VALUES(2,'English');
INSERT INTO "Subjects" VALUES(3,'Physics');
INSERT INTO "Subjects" VALUES(4,'History');
INSERT INTO "Subjects" VALUES(5,'Chemistry');
CREATE TABLE Teachers(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INT,
    HiredDate DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "Teachers" VALUES(1,'John Doe',32,'2024-03-13 20:58:03');
INSERT INTO "Teachers" VALUES(2,'Jane Smith',38,'2024-03-13 20:58:03');
INSERT INTO "Teachers" VALUES(3,'Robert Johnson',47,'2024-03-13 20:58:03');
INSERT INTO "Teachers" VALUES(4,'Michael Brown',56,'2024-03-13 20:58:03');
INSERT INTO "Teachers" VALUES(5,'Emma Wilson',45,'2024-03-13 20:58:03');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Subjects',5);
INSERT INTO "sqlite_sequence" VALUES('Teachers',5);
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Photos',5);
INSERT INTO "sqlite_sequence" VALUES('Illustrations',5);
INSERT INTO "sqlite_sequence" VALUES('Classes',5);
COMMIT;
