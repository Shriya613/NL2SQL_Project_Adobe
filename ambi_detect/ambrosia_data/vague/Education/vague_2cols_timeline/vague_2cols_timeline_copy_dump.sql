BEGIN TRANSACTION;
CREATE TABLE Classes (
    ClassID INTEGER PRIMARY KEY,
    TeacherID INTEGER,
    RoomNumber INT,
    DayOfWeek TEXT,
    TimeStart TIME,
    FOREIGN KEY(TeacherID) REFERENCES Teachers(TeacherID)
);
INSERT INTO "Classes" VALUES(1,1,101,'Monday','09:00:00');
INSERT INTO "Classes" VALUES(2,2,201,'Wednesday','11:00:00');
INSERT INTO "Classes" VALUES(3,3,301,'Friday','14:00:00');
CREATE TABLE Enrollments (
    EnrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    ClassID INTEGER,
    Grade DECIMAL(2,1),
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(ClassID) REFERENCES Classes(ClassID)
);
INSERT INTO "Enrollments" VALUES(1,1,1,NULL);
INSERT INTO "Enrollments" VALUES(2,2,1,NULL);
INSERT INTO "Enrollments" VALUES(3,3,2,NULL);
INSERT INTO "Enrollments" VALUES(4,1,3,NULL);
INSERT INTO "Enrollments" VALUES(5,2,3,NULL);
CREATE TABLE Expenses (
    ExpenseID INTEGER PRIMARY KEY AUTOINCREMENT,
    MaterialID INTEGER,
    Supplier TEXT,
    Amount DECIMAL(8,2),
    FOREIGN KEY(MaterialID) REFERENCES Materials(MaterialID)
);
INSERT INTO "Expenses" VALUES(1,1,'Office Supply Store',150);
INSERT INTO "Expenses" VALUES(2,2,'Electronics Store',500);
INSERT INTO "Expenses" VALUES(3,3,'Bookstore',2550);
CREATE TABLE "Materials" (
    MaterialID INTEGER PRIMARY KEY,
    Timeline INTEGER,
    ResourceType TEXT,
    Quantity INTEGER,
    Price DECIMAL(8,2)
);
INSERT INTO "Materials" VALUES(1,1,'Textbooks',50,25.5);
INSERT INTO "Materials" VALUES(2,2,'Stationery',100,1.5);
INSERT INTO "Materials" VALUES(3,3,'Computers',10,500);
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE
);
INSERT INTO "Students" VALUES(1,'John','Doe','2003-06-15');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2004-07-22');
INSERT INTO "Students" VALUES(3,'Alice','Johnson','2005-09-10');
CREATE TABLE Teachers (
    TeacherID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Subject TEXtT
);
INSERT INTO "Teachers" VALUES(1,'Mr.','White','Mathematics');
INSERT INTO "Teachers" VALUES(2,'Mrs.','Green','English');
INSERT INTO "Teachers" VALUES(3,'Ms.','Black','History');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Enrollments',5);
INSERT INTO "sqlite_sequence" VALUES('Expenses',3);
COMMIT;
