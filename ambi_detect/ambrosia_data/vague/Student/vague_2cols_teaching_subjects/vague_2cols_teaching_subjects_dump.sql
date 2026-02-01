BEGIN TRANSACTION;
CREATE TABLE Academics (
    CourseID INTEGER PRIMARY KEY,
    Discipline TEXT,
    GradeLevel TEXT,
    Teacher TEXT,
    RoomNumber INTEGER,
    Capacity INTEGER
);
INSERT INTO "Academics" VALUES(1,'Mathematics','Algebra I','Ms. Smith',101,30);
INSERT INTO "Academics" VALUES(2,'English','Composition','Mr. Brown',201,25);
INSERT INTO "Academics" VALUES(3,'History','World History','Mrs. Johnson',301,35);
CREATE TABLE Addresses(
    AddressID INTEGER PRIMARY KEY,
    StreetAddress TEXT,
    City TEXT,
    State TEXT,
    PostalCode TEXT,
    Country TEXT,
    PhoneNumber VARCHAR(20)
);
INSERT INTO "Addresses" VALUES(1,'123 Main St.','Anytown','CA','12345','USA','555-1234');
INSERT INTO "Addresses" VALUES(2,'456 Elm St.','Othertown','NY','67890','USA','555-5678');
INSERT INTO "Addresses" VALUES(3,'789 Oak St.','Nowhere','TX','98765','USA','555-9012');
CREATE TABLE EmergencyContacts(
    ContactID INTEGER PRIMARY KEY,
    GuardianID INTEGER,
    Name TEXT,
    Relationship TEXT,
    PhoneNumber TEXT,
    FOREIGN KEY(GuardianID) REFERENCES Guardians(GuardianID)
);
INSERT INTO "EmergencyContacts" VALUES(1,1,'Uncle Bob','Uncle','555-9012');
INSERT INTO "EmergencyContacts" VALUES(2,2,'Dr. Jones','Family Friend','555-1234');
CREATE TABLE Enrollments(
    EnrollmentID INTEGER PRIMARY KEY,
    StudentID INTEGER,
    CourseID INTEGER,
    EnrolDate DATE,
    FinalGrade REAL ,
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(CourseID) REFERENCES Academics(CourseID)
);
INSERT INTO "Enrollments" VALUES(1,1,1,'2021-09-01',NULL);
INSERT INTO "Enrollments" VALUES(2,2,2,'2021-09-01',NULL);
INSERT INTO "Enrollments" VALUES(3,3,3,'2021-09-01',NULL);
CREATE TABLE Guardians(
    GuardianID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Relationship TEXT,
    HomePhone TEXT,
    WorkPhone TEXT,
    Email TEXT
);
INSERT INTO "Guardians" VALUES(1,'John','Doe','Parent','555-1234','','john.doe@example.com');
INSERT INTO "Guardians" VALUES(2,'Emily','Smith','Aunt','555-5678','','emily.smith@example.com');
CREATE TABLE Students (
    StudentID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE
);
INSERT INTO "Students" VALUES(1,'John','Doe','2003-06-18');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2004-09-07');
INSERT INTO "Students" VALUES(3,'Michael','Johnson','2002-11-22');
COMMIT;
