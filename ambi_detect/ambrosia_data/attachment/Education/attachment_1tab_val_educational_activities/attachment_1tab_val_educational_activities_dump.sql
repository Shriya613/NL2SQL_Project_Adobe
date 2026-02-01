BEGIN TRANSACTION;
CREATE TABLE Classes(
    ClassID INTEGER PRIMARY KEY,
    SchoolID INTEGER,
    TeacherName TEXT,
    Subject TEXT,
    FOREIGN KEY(SchoolID) REFERENCES Schools(SchoolID));
INSERT INTO "Classes" VALUES(1,1,'Mr. Smith','Math');
INSERT INTO "Classes" VALUES(2,1,'Mrs. Johnson','English');
INSERT INTO "Classes" VALUES(3,2,'Ms. Brown','History');
INSERT INTO "Classes" VALUES(4,2,'Mr. White','Science');
INSERT INTO "Classes" VALUES(5,3,'Mrs. Black','Art');
CREATE TABLE Enrollments(
    EnrollmentID INTEGER PRIMARY KEY,
    StudentID INTEGER,
    ClassID INTEGER,
    Educational_Activity VARCHAR(255), -- This can contain multiple activities separated by comma like 'Field trip', 'Study hall' etc.
    Number_of_students INTEGER,
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(ClassID) REFERENCES Classes(ClassID));
INSERT INTO "Enrollments" VALUES(1,1,1,'Field Trip',50);
INSERT INTO "Enrollments" VALUES(2,1,2,'Field Trip',30);
INSERT INTO "Enrollments" VALUES(3,2,3,'Interactive Webinar',50);
INSERT INTO "Enrollments" VALUES(4,2,4,'Interactive Webinar',40);
INSERT INTO "Enrollments" VALUES(5,3,5,'Project Day',25);
INSERT INTO "Enrollments" VALUES(6,4,1,'Lecture Series',20);
INSERT INTO "Enrollments" VALUES(7,5,2,'Workshop',22);
CREATE TABLE Schools(
    SchoolID INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    PhoneText);
INSERT INTO "Schools" VALUES(1,'ABC High School','123 Street View','+987654321');
INSERT INTO "Schools" VALUES(2,'XYZ Elementary School','456 Boulevard Place','+765432198');
INSERT INTO "Schools" VALUES(3,'LMN Middle School','789 Avenue Drive','+543219876');
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Age INTEGER,
    EnrollmentDate DATE);
INSERT INTO "Students" VALUES(1,'John','Doe',15,'2021-09-01');
INSERT INTO "Students" VALUES(2,'Jane','Smith',14,'2021-09-01');
INSERT INTO "Students" VALUES(3,'Alice','Brown',16,'2021-09-01');
INSERT INTO "Students" VALUES(4,'Bob','White',17,'2021-09-01');
INSERT INTO "Students" VALUES(5,'Charlie','Black',13,'2021-09-01');
COMMIT;
