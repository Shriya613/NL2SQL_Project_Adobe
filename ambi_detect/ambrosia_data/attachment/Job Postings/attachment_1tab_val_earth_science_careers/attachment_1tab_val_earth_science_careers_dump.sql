BEGIN TRANSACTION;
CREATE TABLE Companies (
    id INTEGER PRIMARY KEY,
    name TEXT,
    address TEXT);
INSERT INTO "Companies" VALUES(1,'Company A','123 Main St');
INSERT INTO "Companies" VALUES(2,'Company B','456 Elm St');
INSERT INTO "Companies" VALUES(3,'Company C','789 Oak St');
INSERT INTO "Companies" VALUES(4,'Company D','098 Maple Ave');
INSERT INTO "Companies" VALUES(5,'Company E','543 Pine St');
CREATE TABLE Departments (
    id INTEGER PRIMARY KEY,
    name TEXT);
INSERT INTO "Departments" VALUES(1,'Department of Geosciences');
INSERT INTO "Departments" VALUES(2,'Department of Chemistry');
INSERT INTO "Departments" VALUES(3,'Department of Biology');
INSERT INTO "Departments" VALUES(4,'Department of Physics');
INSERT INTO "Departments" VALUES(5,'Department of Mathematics');
CREATE TABLE EarthSciencesJobs (
    id INTEGER PRIMARY KEY,
    jobTitle TEXT,
    departmentId INTEGER,
    companyId INTEGER,
    FOREIGN KEY(departmentId) REFERENCES Departments(id),
    FOREIGN KEY(companyId) REFERENCES Companies(id));
INSERT INTO "EarthSciencesJobs" VALUES(1,'Research Scientist',1,1);
INSERT INTO "EarthSciencesJobs" VALUES(2,'Laboratory Technician',1,2);
INSERT INTO "EarthSciencesJobs" VALUES(3,'Field Geologist',2,3);
INSERT INTO "EarthSciencesJobs" VALUES(4,'Teaching Assistant',3,4);
INSERT INTO "EarthSciencesJobs" VALUES(5,'Data Analyst',4,5);
CREATE TABLE Education (
    employeeId INTEGER,
    degree TEXT,
    major TEXT,
    institution TEXT,
    yearGraduated INTEGER,
    FOREIGN KEY(employeeId) REFERENCES Employees(id));
INSERT INTO "Education" VALUES(NULL,'PhD','Chemistry','Harvard',2008);
INSERT INTO "Education" VALUES(NULL,'MS','Physics','Yale',2010);
INSERT INTO "Education" VALUES(NULL,'BS','Biology','Princeton',2012);
INSERT INTO "Education" VALUES(NULL,'BA','Mathematics','Columbia',2014);
INSERT INTO "Education" VALUES(NULL,'MA','Computational Science','Berkeley',2016);
CREATE TABLE Employees (
    id INTEGER PRIMARY KEY,
    title TEXT,
    startDate DATE,
    endDate DATE,
    university TEXT, FullName TEXT);
INSERT INTO "Employees" VALUES(1,'Chemist','2010-01-01',NULL,'MIT','John Doe');
INSERT INTO "Employees" VALUES(2,'Chemist','2015-01-01',NULL,'UCLA','Jane Smith');
INSERT INTO "Employees" VALUES(3,'Geoscientist','2010-01-01',NULL,'MIT','Michael Brown');
INSERT INTO "Employees" VALUES(4,'Geoscientist','2015-01-01',NULL,'Stanford','Emily White');
INSERT INTO "Employees" VALUES(5,'Engineer','2010-01-01',NULL,'Caltech','David Black');
COMMIT;
