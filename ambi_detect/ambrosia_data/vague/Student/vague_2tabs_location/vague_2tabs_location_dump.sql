BEGIN TRANSACTION;
CREATE TABLE Building_Name (
    id INTEGER PRIMARY KEY,
    name TEXT,
    address TEXT
);
INSERT INTO "Building_Name" VALUES(1,'Building A','123 Main St');
INSERT INTO "Building_Name" VALUES(2,'Building B','456 Elm St');
CREATE TABLE Buildings (
    id INTEGER PRIMARY KEY,
    building_type TEXT ,
    capacity INTEGER,
    address TEXT
);
INSERT INTO "Buildings" VALUES(1,'Administration Building',1000,'123 Main St');
INSERT INTO "Buildings" VALUES(2,'Engineering Building',500,'456 Elm St');
CREATE TABLE Classrooms (
    id INTEGER PRIMARY KEY,
    classroom_number INTEGER,
    building_id INTEGER,
    FOREIGN KEY(building_id) REFERENCES Buildings(id)
);
INSERT INTO "Classrooms" VALUES(1,101,1);
INSERT INTO "Classrooms" VALUES(2,102,1);
INSERT INTO "Classrooms" VALUES(3,103,1);
INSERT INTO "Classrooms" VALUES(4,104,2);
INSERT INTO "Classrooms" VALUES(5,105,2);
CREATE TABLE Libraries (
    id INTEGER PRIMARY KEY,
    library_name TEXT,
    building_id INTEGER,
    university_id INTEGER,
    FOREIGN KEY(building_id) REFERENCES Building_Name(id),
    FOREIGN KEY(university_id) REFERENCES University(id)
);
INSERT INTO "Libraries" VALUES(1,'Library A',1,1);
INSERT INTO "Libraries" VALUES(2,'Library B',2,1);
INSERT INTO "Libraries" VALUES(3,'Library C',1,2);
INSERT INTO "Libraries" VALUES(4,'Library D',2,2);
CREATE TABLE Students (
    id INTEGER PRIMARY KEY,
    student_name TEXT,
    birthdate DATE,
    major TEXT
);
INSERT INTO "Students" VALUES(1,'John Doe','2000-09-01','Computer Science');
INSERT INTO "Students" VALUES(2,'Jane Smith','2002-08-15','Biology');
INSERT INTO "Students" VALUES(3,'Mike Johnson','2003-10-20','Mathematics');
INSERT INTO "Students" VALUES(4,'Emily Davis','2004-11-25','English Literature');
INSERT INTO "Students" VALUES(5,'David Brown','2005-12-10','Physics');
CREATE TABLE University (
    id INTEGER PRIMARY KEY,
    university_name TEXT,
    website TEXT
);
INSERT INTO "University" VALUES(1,'University of Example','http://www.example.com/univ');
COMMIT;
