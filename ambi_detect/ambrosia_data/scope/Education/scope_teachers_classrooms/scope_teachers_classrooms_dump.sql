BEGIN TRANSACTION;
CREATE TABLE Classrooms (
    classroom_id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_number INT,
    capacity INT
);
INSERT INTO "Classrooms" VALUES(1,7,30);
INSERT INTO "Classrooms" VALUES(2,12,28);
INSERT INTO "Classrooms" VALUES(3,6,25);
INSERT INTO "Classrooms" VALUES(4,9,32);
INSERT INTO "Classrooms" VALUES(5,11,27);
CREATE TABLE Enrollments (
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    subject_id INTEGER,
    classroom_id INTEGER,
    grade REAL,
    FOREIGN KEY(student_id) REFERENCES Students(student_id),
    FOREIGN KEY(subject_id) REFERENCES Subjects(subject_id),
    FOREIGN KEY(classroom_id) REFERENCES Classrooms(classroom_id)
);
CREATE TABLE Students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    date_of_birth DATE,
    enrollment_date DATE
);
INSERT INTO "Students" VALUES(1,'Alice','Johnson','2003-05-15','2020-09-01');
INSERT INTO "Students" VALUES(2,'Bob','Williams','2004-06-20','2020-09-01');
INSERT INTO "Students" VALUES(3,'Carol','Martinez','2005-07-10','2020-09-01');
INSERT INTO "Students" VALUES(4,'Daniel','Garcia','2006-08-15','2020-09-01');
INSERT INTO "Students" VALUES(5,'Eva','Lee','2007-09-05','2020-09-01');
CREATE TABLE Subjects (
    subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT
);
INSERT INTO "Subjects" VALUES(1,'Mathematics','The study of numbers, quantities, shapes, patterns and structures.');
INSERT INTO "Subjects" VALUES(2,'English Literature','The study of written works and their social contexts.');
INSERT INTO "Subjects" VALUES(3,'Physical Education','The development of physical fitness and skills through sports and activities.');
INSERT INTO "Subjects" VALUES(4,'History','The study of past events and their impact on present-day society.');
INSERT INTO "Subjects" VALUES(5,'Art','The expression of creativity and imagination through various forms of visual arts.');
CREATE TABLE Teachers (
    teacher_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT);
INSERT INTO "Teachers" VALUES(1,'John Doe');
INSERT INTO "Teachers" VALUES(2,'Jane Smith');
INSERT INTO "Teachers" VALUES(3,'Michael Brown');
INSERT INTO "Teachers" VALUES(4,'Emily White');
INSERT INTO "Teachers" VALUES(5,'David Black');
CREATE TABLE Teachers_Classrooms (
    teachers_classrooms_id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id INTEGER,
    classroom_id INTEGER,
    start_date DATE,
    end_date DATE,
    FOREIGN KEY(teacher_id) REFERENCES Teachers(teacher_id),
    FOREIGN KEY(classroom_id) REFERENCES Classrooms(classroom_id)
);
INSERT INTO "Teachers_Classrooms" VALUES(1,1,1,'2021-08-30','2022-05-31');
INSERT INTO "Teachers_Classrooms" VALUES(2,2,2,'2021-08-30','2022-05-31');
INSERT INTO "Teachers_Classrooms" VALUES(3,3,3,'2021-08-30','2022-05-31');
INSERT INTO "Teachers_Classrooms" VALUES(4,4,4,'2021-08-30','2022-05-31');
INSERT INTO "Teachers_Classrooms" VALUES(5,5,5,'2021-08-30','2022-05-31');
INSERT INTO "Teachers_Classrooms" VALUES(6,2,1,'2021-08-30','2022-05-31');
INSERT INTO "Teachers_Classrooms" VALUES(7,3,1,'2021-08-30','2022-05-31');
INSERT INTO "Teachers_Classrooms" VALUES(8,4,1,'2021-08-30','2022-05-31');
INSERT INTO "Teachers_Classrooms" VALUES(9,5,1,'2021-08-30','2022-05-31');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Teachers',5);
INSERT INTO "sqlite_sequence" VALUES('Classrooms',5);
INSERT INTO "sqlite_sequence" VALUES('Subjects',5);
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Teachers_Classrooms',9);
COMMIT;
