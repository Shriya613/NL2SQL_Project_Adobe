BEGIN TRANSACTION;
CREATE TABLE Classes(
    id INTEGER PRIMARY KEY,
    teacher_id INTEGER,
    class_name TEXT,
    grade_level INT,
    FOREIGN KEY (teacher_id) REFERENCES Teachers(id)
);
INSERT INTO "Classes" VALUES(1,1,'Algebra I',9);
INSERT INTO "Classes" VALUES(2,2,'World History',10);
CREATE TABLE "FieldTrips"(
    id INTEGER PRIMARY KEY,
    "trip" TEXT,
    destination TEXT,
    date DATE,
    budget REAL,
    Number_of_students INT
);
INSERT INTO "FieldTrips" VALUES(1,'Museum Visit','New York','2023-09-16',5000.0,50);
INSERT INTO "FieldTrips" VALUES(2,'Monument Tour','Washington DC','2023-08-20',7500.0,30);
CREATE TABLE Students(
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    studentId TEXT,
    birthDate TEXT,
    fieldtrip_participant BOOLEAN DEFAULT 1,
    studyhall_user BOOLEAN DEFAULT 1,
    class_id INTEGER,
    FOREIGN KEY (class_id) REFERENCES Classes(id)
);
INSERT INTO "Students" VALUES(1,'John','Doe','JD123','2005-05-15',1,1,1);
INSERT INTO "Students" VALUES(2,'Emma','Smith','ES456','2004-06-18',1,1,2);
CREATE TABLE StudyHalls(
    id INTEGER PRIMARY KEY,
    building TEXT,
    capacity INT,
    available_hours TEXT,
    number_of_students INT
);
INSERT INTO "StudyHalls" VALUES(1,'Building A',100,'08:00 - 22:00',50);
INSERT INTO "StudyHalls" VALUES(2,'Building B',150,'08:00 - 22:00',30);
CREATE TABLE Teachers(
    id INTEGER PRIMARY KEY,
    name TEXT,
    subject TEXT,
    years_experience INT,
    office_number TEXT
);
INSERT INTO "Teachers" VALUES(1,'Mr. Smith','Math',10,'123');
INSERT INTO "Teachers" VALUES(2,'Ms. Johnson','History',7,'456');
COMMIT;
