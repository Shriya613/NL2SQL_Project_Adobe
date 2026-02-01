BEGIN TRANSACTION;
CREATE TABLE Data_Analyst (
    id INTEGER PRIMARY KEY,
    employeeId INTEGER,
    experience INT,
    analyticalTools TEXT,
    dataSources TEXT,
    FOREIGN KEY(employeeId) REFERENCES Employee(id)
);
INSERT INTO "Data_Analyst" VALUES(1,1,5,'R, SAS, Excel','SQL Server, MySQL, Oracle');
INSERT INTO "Data_Analyst" VALUES(2,2,9,'Matlab, Stata, SPSS','SAP HANA, Teradata, MongoDB');
INSERT INTO "Data_Analyst" VALUES(3,3,7,'Tableau, PowerBI, QlikView','Google Analytics, Adobe Analytics, Salesforce');
CREATE TABLE Department (
    id INTEGER PRIMARY KEY,
    departmentName TEXT
);
INSERT INTO "Department" VALUES(1,'IT');
INSERT INTO "Department" VALUES(2,'Finance');
INSERT INTO "Department" VALUES(3,'Human Resources');
CREATE TABLE Employee (
    id INTEGER PRIMARY KEY,
    gradeId INTEGER,
    departmentId INTEGER,
    positionId INTEGER, FullName TEXT,
    FOREIGN KEY(gradeId) REFERENCES Grade(id),
    FOREIGN KEY(departmentId) REFERENCES Department(id),
    FOREIGN KEY(positionId) REFERENCES Position(id)
);
INSERT INTO "Employee" VALUES(1,1,1,2,'John Smith');
INSERT INTO "Employee" VALUES(2,2,1,3,'Jane Doe');
INSERT INTO "Employee" VALUES(3,3,2,1,'Bob Builder');
INSERT INTO "Employee" VALUES(4,1,3,2,'Alice Wonderland');
INSERT INTO "Employee" VALUES(5,2,3,1,'Charlie Brown');
CREATE TABLE Grade (
    id INTEGER PRIMARY KEY,
    name TEXT
);
INSERT INTO "Grade" VALUES(1,'Junior');
INSERT INTO "Grade" VALUES(2,'Middle');
INSERT INTO "Grade" VALUES(3,'Senior');
CREATE TABLE Position (
    id INTEGER PRIMARY KEY,
    positionName TEXT
);
INSERT INTO "Position" VALUES(1,'Manager');
INSERT INTO "Position" VALUES(2,'Engineer');
INSERT INTO "Position" VALUES(3,'Specialist');
CREATE TABLE Software_Developer (
    id INTEGER PRIMARY KEY,
    employeeId INTEGER,
    experience INT,
    programmingLanguages TEXT,
    toolsUsed TEXT,
    FOREIGN KEY(employeeId) REFERENCES Employee(id)
);
INSERT INTO "Software_Developer" VALUES(4,4,5,'Kotlin, Rust, Scala','Android Studio, Vim');
INSERT INTO "Software_Developer" VALUES(5,5,7,'Perl, Objective-C, Fortran','Emacs, Xcode');
COMMIT;
