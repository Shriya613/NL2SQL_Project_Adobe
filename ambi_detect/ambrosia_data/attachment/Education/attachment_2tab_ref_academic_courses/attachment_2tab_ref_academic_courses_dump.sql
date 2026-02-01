BEGIN TRANSACTION;
CREATE TABLE CourseEnrollments(
    StudentID INTEGER,
    EngineeringClassId INTEGER,
    DesignClassId INTEGER,
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(EngineeringClassId) REFERENCES EngineeringClasses(ClassID),
    FOREIGN KEY(DesignClassId) REFERENCES DesignClasses(ClassID)
);
INSERT INTO "CourseEnrollments" VALUES(1,1,1);
INSERT INTO "CourseEnrollments" VALUES(2,2,3);
INSERT INTO "CourseEnrollments" VALUES(3,3,4);
INSERT INTO "CourseEnrollments" VALUES(4,4,2);
INSERT INTO "CourseEnrollments" VALUES(5,5,5);
CREATE TABLE DesignClasses(
    ClassID INTEGER PRIMARY KEY,
    ClassName TEXT
);
INSERT INTO "DesignClasses" VALUES(1,'Sustainable Architecture');
INSERT INTO "DesignClasses" VALUES(2,'Urban Planning');
INSERT INTO "DesignClasses" VALUES(3,'Graphic Design');
INSERT INTO "DesignClasses" VALUES(4,'Product Design');
INSERT INTO "DesignClasses" VALUES(5,'Industrial Design');
CREATE TABLE EngineeringClasses(
    ClassID INTEGER PRIMARY KEY,
    ClassName TEXT
);
INSERT INTO "EngineeringClasses" VALUES(1,'Sustainability Project');
INSERT INTO "EngineeringClasses" VALUES(2,'Robotics');
INSERT INTO "EngineeringClasses" VALUES(3,'Automation');
INSERT INTO "EngineeringClasses" VALUES(4,'Material Science');
INSERT INTO "EngineeringClasses" VALUES(5,'Electronics');
CREATE TABLE EnrolledStudents(
    StudentID INTEGER,
    ProjectID INTEGER,
    Grade TEXT ,
    FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY(ProjectID) REFERENCES Projects(ProjectID)
);
INSERT INTO "EnrolledStudents" VALUES(1,1,'A+');
INSERT INTO "EnrolledStudents" VALUES(2,2,'B+');
INSERT INTO "EnrolledStudents" VALUES(3,3,'C+');
INSERT INTO "EnrolledStudents" VALUES(4,4,'A');
INSERT INTO "EnrolledStudents" VALUES(5,5,'B');
CREATE TABLE Projects(
    ProjectID INTEGER PRIMARY KEY,
    ProjectName TEXT,
    EngineeringClassId INTEGER,
    DesignClassId INTEGER,
    FOREIGN KEY(EngineeringClassId) REFERENCES EngineeringClasses(ClassID),
    FOREIGN KEY(DesignClassId) REFERENCES DesignClasses(ClassID)
);
INSERT INTO "Projects" VALUES(1,'Green Building',1,1);
INSERT INTO "Projects" VALUES(2,'Autonomous Vehicle',2,3);
INSERT INTO "Projects" VALUES(3,'Waste Management System',3,4);
INSERT INTO "Projects" VALUES(4,'Smart Home Devices',4,2);
INSERT INTO "Projects" VALUES(5,'Nanotechnology Applications',5,5);
CREATE TABLE Students(
    StudentID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE
);
INSERT INTO "Students" VALUES(1,'John','Doe','1990-01-01');
INSERT INTO "Students" VALUES(2,'Jane','Smith','1987-02-15');
INSERT INTO "Students" VALUES(3,'Michael','Johnson','1992-03-20');
INSERT INTO "Students" VALUES(4,'Emily','Williams','1996-04-10');
INSERT INTO "Students" VALUES(5,'David','Brown','1989-05-25');
COMMIT;
