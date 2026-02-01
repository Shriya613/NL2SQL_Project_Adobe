BEGIN TRANSACTION;
CREATE TABLE `Attendance` (
    `attendance_id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `student_id` INTEGER,
    `date_attended` DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(`student_id`) REFERENCES `Students`(student_id)
);
INSERT INTO "Attendance" VALUES(1,1,'2023-01-01 10:00:00');
INSERT INTO "Attendance" VALUES(2,2,'2023-01-01 10:00:00');
INSERT INTO "Attendance" VALUES(3,3,'2023-01-02 10:00:00');
INSERT INTO "Attendance" VALUES(4,4,'2023-01-02 10:00:00');
INSERT INTO "Attendance" VALUES(5,5,'2023-01-03 10:00:00');
CREATE TABLE `Classes` (
    `class_id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `course_name` TEXT,
    `teacher_name` TEXT
);
INSERT INTO "Classes" VALUES(1,'Math','Mr. Brown');
INSERT INTO "Classes" VALUES(2,'English','Mrs. Green');
INSERT INTO "Classes" VALUES(3,'History','Ms. Blue');
INSERT INTO "Classes" VALUES(4,'Science','Dr. Red');
INSERT INTO "Classes" VALUES(5,'Art','Prof. Yellow');
CREATE TABLE `Enrollments` (
    `enrollment_id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `student_id` INTEGER,
    `class_id` INTEGER,
    FOREIGN KEY(`student_id`) REFERENCES `Students`(student_id),
    FOREIGN KEY(`class_id`) REFERENCES `Classes`(class_id)
);
INSERT INTO "Enrollments" VALUES(1,1,1);
INSERT INTO "Enrollments" VALUES(2,2,2);
INSERT INTO "Enrollments" VALUES(3,3,3);
INSERT INTO "Enrollments" VALUES(4,4,4);
INSERT INTO "Enrollments" VALUES(5,5,5);
CREATE TABLE `Grades` (
    `grade_id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `student_id` INTEGER,
    `class_id` INTEGER,
    `grade` DECIMAL(2,1), -- assuming grades range from 0-9.9
    FOREIGN KEY(`student_id`) REFERENCES `Students`(student_id),
    FOREIGN KEY(`class_id`) REFERENCES `Classes`(class_id)
);
INSERT INTO "Grades" VALUES(1,1,1,8.5);
INSERT INTO "Grades" VALUES(2,2,2,7.5);
INSERT INTO "Grades" VALUES(3,3,3,6.5);
INSERT INTO "Grades" VALUES(4,4,4,9.5);
INSERT INTO "Grades" VALUES(5,5,5,7);
CREATE TABLE `Students` (
    `student_id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `first_name` TEXT,
    `last_name` TEXT
);
INSERT INTO "Students" VALUES(1,'Alice','Johnson');
INSERT INTO "Students" VALUES(2,'Bob','Smith');
INSERT INTO "Students" VALUES(3,'Charlie','Doe');
INSERT INTO "Students" VALUES(4,'David','Black');
INSERT INTO "Students" VALUES(5,'Eve','White');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Classes',5);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',5);
INSERT INTO "sqlite_sequence" VALUES('Attendance',5);
INSERT INTO "sqlite_sequence" VALUES('Grades',5);
COMMIT;
