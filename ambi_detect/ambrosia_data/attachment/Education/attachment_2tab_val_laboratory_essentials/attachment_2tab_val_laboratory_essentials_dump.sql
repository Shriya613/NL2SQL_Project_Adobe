BEGIN TRANSACTION;
CREATE TABLE Classes(
    classID INTEGER PRIMARY KEY AUTOINCREMENT,
    className TEXT,
    instructorID INTEGER,
    FOREIGN KEY(instructorID) REFERENCES Instructors(instructorID));
INSERT INTO "Classes" VALUES(1,'Introduction to Computer Science',1);
INSERT INTO "Classes" VALUES(2,'Advanced Calculus',2);
INSERT INTO "Classes" VALUES(3,'Organic Chemistry',3);
INSERT INTO "Classes" VALUES(4,'Physics I',4);
INSERT INTO "Classes" VALUES(5,'World History',5);
CREATE TABLE Instructors(
    instructorID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT UNIQUE);
INSERT INTO "Instructors" VALUES(1,'Dr.','Ava','avadoc@email.edu');
INSERT INTO "Instructors" VALUES(2,'Professor','Xander','xanderx@email.edu');
INSERT INTO "Instructors" VALUES(3,'Assistant','Penelope','pennyassis@email.edu');
INSERT INTO "Instructors" VALUES(4,'Adjunct','Robert','robertadjuct@email.edu');
INSERT INTO "Instructors" VALUES(5,'Teaching','Miles','milesteach@email.edu');
CREATE TABLE Lab_Equipments(
    equipmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    labID INTEGER,
    equipName TEXT,
    dateOfPurchase DATE, -- You should use DATE type for dates here instead of TEXT
    cost NUMERIC(10,2), -- Assuming that the cost might be a decimal number
    FOREIGN KEY(labID) REFERENCES Labs(labID));
INSERT INTO "Lab_Equipments" VALUES(1,1,'High Performance Desktop Computers','January 2023',8999.99);
INSERT INTO "Lab_Equipments" VALUES(2,1,'VR Headsets','February 2023',7999.99);
INSERT INTO "Lab_Equipments" VALUES(3,2,'Advanced Microscopes','April 2023',6999.99);
INSERT INTO "Lab_Equipments" VALUES(4,2,'Centrifuges','May 2023',5999.99);
INSERT INTO "Lab_Equipments" VALUES(5,3,'Electronic Balances','August 2023',4999.99);
CREATE TABLE Labs(
    labID INTEGER PRIMARY KEY AUTOINCREMENT,
    labName TEXT,
    buildingNumber TEXT,
    roomNumber TEXT);
INSERT INTO "Labs" VALUES(1,'Computer Lab A','Building 1','Room 101');
INSERT INTO "Labs" VALUES(2,'Biology Lab B','Building 2','Room 202');
INSERT INTO "Labs" VALUES(3,'Chemistry Lab C','Building 3','Room 303');
INSERT INTO "Labs" VALUES(4,'Physics Lab D','Building 4','Room 404');
INSERT INTO "Labs" VALUES(5,'Art Studio E','Building 5','Room 505');
CREATE TABLE Safety_Gears(
    safetyGearID INTEGER PRIMARY KEY AUTOINCREMENT,
    labID INTEGER,
    gearName TEXT,
    dateOfPurchase DATE,
    cost NUMERIC(10,2),
    FOREIGN KEY(labID) REFERENCES Labs(labID));
INSERT INTO "Safety_Gears" VALUES(1,1,'Fire Extinguishers','January 2023',1999.99);
INSERT INTO "Safety_Gears" VALUES(2,1,'First Aid Kits','February 2023',999.99);
INSERT INTO "Safety_Gears" VALUES(3,2,'Eye Wash Stations','April 2023',1499.99);
INSERT INTO "Safety_Gears" VALUES(4,2,'Spill Cleanup Kit','May 2023',899.99);
INSERT INTO "Safety_Gears" VALUES(5,3,'Respirators','August 2023',599.99);
CREATE TABLE Students(
    studentID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT UNIQUE);
INSERT INTO "Students" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Students" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Students" VALUES(3,'Alice','Johnson','alice.johnson@example.com');
INSERT INTO "Students" VALUES(4,'Bob','Williams','bob.williams@example.com');
INSERT INTO "Students" VALUES(5,'Charlie','Brown','charlie.brown@example.com');
CREATE TABLE StudentsEnrolledInClasses(
    enrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    studentID INTEGER,
    classID INTEGER,
    FOREIGN KEY(studentID) REFERENCES Students(studentID),
    FOREIGN KEY(classID) REFERENCES Classes(classID));
INSERT INTO "StudentsEnrolledInClasses" VALUES(1,1,1);
INSERT INTO "StudentsEnrolledInClasses" VALUES(2,2,2);
INSERT INTO "StudentsEnrolledInClasses" VALUES(3,3,3);
INSERT INTO "StudentsEnrolledInClasses" VALUES(4,4,4);
INSERT INTO "StudentsEnrolledInClasses" VALUES(5,5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Instructors',5);
INSERT INTO "sqlite_sequence" VALUES('Classes',5);
INSERT INTO "sqlite_sequence" VALUES('StudentsEnrolledInClasses',5);
INSERT INTO "sqlite_sequence" VALUES('Labs',5);
INSERT INTO "sqlite_sequence" VALUES('Lab_Equipments',5);
INSERT INTO "sqlite_sequence" VALUES('Safety_Gears',5);
COMMIT;
