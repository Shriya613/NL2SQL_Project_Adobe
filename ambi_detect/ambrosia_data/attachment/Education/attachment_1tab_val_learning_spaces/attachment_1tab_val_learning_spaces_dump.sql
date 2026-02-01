BEGIN TRANSACTION;
CREATE TABLE Classes(
    ClassID INTEGER PRIMARY KEY AUTOINCREMENT,
    ClassName TEXT,
    TeacherID INTEGER,
    FOREIGN KEY(TeacherID) REFERENCES Teachers(TeacherID));
INSERT INTO "Classes" VALUES(1,'Math',1);
INSERT INTO "Classes" VALUES(2,'Physics',2);
INSERT INTO "Classes" VALUES(3,'English',3);
INSERT INTO "Classes" VALUES(4,'History',4);
INSERT INTO "Classes" VALUES(5,'Chemistry',5);
CREATE TABLE "Learning_Spaces"(
    SpaceID INTEGER PRIMARY KEY AUTOINCREMENT, Name TEXT,
    Type TEXT ,
    AvailableResources TEXT ,
    Capacity INTEGER);
INSERT INTO "Learning_Spaces" VALUES(1,'Physics Lab','Science Lab','Computers',30);
INSERT INTO "Learning_Spaces" VALUES(2,'Biology Lab','Science Lab','Tables',25);
INSERT INTO "Learning_Spaces" VALUES(3,'Creative Art Studio','Art Studio','Computers',20);
INSERT INTO "Learning_Spaces" VALUES(4,'Painting Studio','Art Studio','Easels',25);
INSERT INTO "Learning_Spaces" VALUES(5,'Orchestra Room','Music Room','Pianos',15);
CREATE TABLE ResourceAllocation(
    AllocationID INTEGER PRIMARY KEY AUTOINCREMENT,
    ClassID INTEGER,
    ResourceType TEXT,
    Quantity INTEGER,
    FOREIGN KEY(ClassID) REFERENCES Classes(ClassID));
INSERT INTO "ResourceAllocation" VALUES(1,1,'Textbooks',30);
INSERT INTO "ResourceAllocation" VALUES(2,2,'Microscopes',15);
INSERT INTO "ResourceAllocation" VALUES(3,3,'Charts',20);
INSERT INTO "ResourceAllocation" VALUES(4,4,'Projectors',10);
INSERT INTO "ResourceAllocation" VALUES(5,5,'Flasks',25);
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    EmailAddress TEXT UNIQUE);
INSERT INTO "Students" VALUES(1,'John','Doe','1990-01-01','john.doe@example.com');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2000-06-30','jane.smith@example.com');
INSERT INTO "Students" VALUES(3,'Alice','Johnson','1987-02-20','alice.johnson@example.com');
INSERT INTO "Students" VALUES(4,'Bob','Williams','1992-12-15','bob.williams@example.com');
INSERT INTO "Students" VALUES(5,'Emma','Brown','1995-05-01','emma.brown@example.com');
CREATE TABLE Teachers(
    TeacherID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    EmailAddress TEXT UNIQUE);
INSERT INTO "Teachers" VALUES(1,'Michael','Black','1980-03-15','michael.black@example.com');
INSERT INTO "Teachers" VALUES(2,'Anna','White','1978-09-05','anna.white@example.com');
INSERT INTO "Teachers" VALUES(3,'David','Green','1991-11-20','david.green@example.com');
INSERT INTO "Teachers" VALUES(4,'Sarah','Blue','1985-07-31','sarah.blue@example.com');
INSERT INTO "Teachers" VALUES(5,'Tom','Red','1988-02-10','tom.red@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Teachers',5);
INSERT INTO "sqlite_sequence" VALUES('Classes',5);
INSERT INTO "sqlite_sequence" VALUES('ResourceAllocation',5);
INSERT INTO "sqlite_sequence" VALUES('Learning_Spaces',5);
COMMIT;
