BEGIN TRANSACTION;
CREATE TABLE Course (
    ID INTEGER PRIMARY KEY,
    Title TEXT,
    Subject TEXT,
    TeacherID INT,
    FOREIGN KEY(TeacherID) REFERENCES Teacher(ID)
);
INSERT INTO "Course" VALUES(1,'Algebra I','Mathematics',1);
INSERT INTO "Course" VALUES(2,'Physics','Science',2);
INSERT INTO "Course" VALUES(3,'English Literature','Language Arts',3);
INSERT INTO "Course" VALUES(4,'World History','History',4);
INSERT INTO "Course" VALUES(5,'Art Appreciation','Fine Arts',5);
CREATE TABLE Enrollment (
    StudentID INT,
    CourseID INT,
    DateEnrolled DATE,
    Grades INT,
    FOREIGN KEY(StudentID) REFERENCES Student(ID),
    FOREIGN KEY(CourseID) REFERENCES Course(ID)
);
INSERT INTO "Enrollment" VALUES(1,1,'2022-09-01',85);
INSERT INTO "Enrollment" VALUES(2,2,'2022-09-01',90);
INSERT INTO "Enrollment" VALUES(3,3,'2022-09-01',80);
INSERT INTO "Enrollment" VALUES(4,4,'2022-09-01',75);
INSERT INTO "Enrollment" VALUES(5,5,'2022-09-01',95);
CREATE TABLE LearningActivity (
    ID INTEGER PRIMARY KEY,
    Description TEXT,
    ActivityType TEXT ,
    PurposeID INTEGER,
    FOREIGN KEY(PurposeID) REFERENCES Purpose(ID)
);
INSERT INTO "LearningActivity" VALUES(1,'Visit a local museum','Field Trip',1);
INSERT INTO "LearningActivity" VALUES(2,'Visit a science center','Field Trip',2);
INSERT INTO "LearningActivity" VALUES(3,'Organize a group project','Study Group',1);
INSERT INTO "LearningActivity" VALUES(4,'Host a debate on current events','Study Group',2);
CREATE TABLE Purpose (
    ID INTEGER PRIMARY KEY,
    Name TEXT
);
INSERT INTO "Purpose" VALUES(1,'Real-world Application');
INSERT INTO "Purpose" VALUES(2,'Problem Solving Skills Development');
CREATE TABLE Student (
    ID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Age INT,
    GradeLevel INT
);
INSERT INTO "Student" VALUES(1,'John','Doe',16,10);
INSERT INTO "Student" VALUES(2,'Jane','Smith',17,12);
INSERT INTO "Student" VALUES(3,'Emma','Brown',18,11);
INSERT INTO "Student" VALUES(4,'Michael','Johnson',19,10);
INSERT INTO "Student" VALUES(5,'Olivia','Williams',16,9);
CREATE TABLE Teacher (
    ID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Department TEXT
);
INSERT INTO "Teacher" VALUES(1,'Mr.','White','Mathematics');
INSERT INTO "Teacher" VALUES(2,'Ms.','Green','Science');
INSERT INTO "Teacher" VALUES(3,'Dr.','Black','Language Arts');
INSERT INTO "Teacher" VALUES(4,'Mrs.','Blue','History');
INSERT INTO "Teacher" VALUES(5,'Ms.','Red','Fine Arts');
COMMIT;
