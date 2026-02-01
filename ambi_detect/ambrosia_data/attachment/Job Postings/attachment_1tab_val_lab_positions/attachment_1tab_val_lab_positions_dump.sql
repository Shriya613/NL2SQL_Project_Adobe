BEGIN TRANSACTION;
CREATE TABLE Company(
    id INTEGER PRIMARY KEY,
    name TEXT);
INSERT INTO "Company" VALUES(1,'TechCorp');
INSERT INTO "Company" VALUES(2,'HealthPlus');
INSERT INTO "Company" VALUES(3,'GreenTech');
INSERT INTO "Company" VALUES(4,'SoftWare');
INSERT INTO "Company" VALUES(5,'MedicalPioneers');
CREATE TABLE Department(
    id INTEGER PRIMARY KEY,
    companyId INTEGER,
    departmentName TEXT,
    FOREIGN KEY (companyId) REFERENCES Company(id));
INSERT INTO "Department" VALUES(1,1,'Biotechnology');
INSERT INTO "Department" VALUES(2,2,'Chemistry');
INSERT INTO "Department" VALUES(3,3,'Physics');
INSERT INTO "Department" VALUES(4,4,'Computer Science');
INSERT INTO "Department" VALUES(5,5,'Mathematics');
CREATE TABLE Employee(
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    dateOfBirth DATE,
    email TEXT,
    phoneNumber VARCHAR(15),
    hireDate DATE,
    jobTitle TEXT,
    departmentId INTEGER,
    gender TEXT,
    FOREIGN KEY (departmentId) REFERENCES Department(id));
INSERT INTO "Employee" VALUES(1,'John','Smith','1990-01-15','johnsmith@email.com','+1 (234) 567-8910','2021-06-01','Research Scientist',1,'Male');
INSERT INTO "Employee" VALUES(2,'Jane','Doe','1988-03-20','janedoe@email.com','+1 (123) 456-7890','2021-07-01','Laboratory Technician',2,'Female');
INSERT INTO "Employee" VALUES(3,'Tom','Hanks','1985-09-05','thomhanks@email.com','+1 (345) 678-9012','2021-08-01','Engineer',3,'Male');
INSERT INTO "Employee" VALUES(4,'Alice','Williams','1992-11-10','alicewilliams@email.com','+1 (456) 789-0123','2021-09-01','Analyst',4,'Female');
INSERT INTO "Employee" VALUES(5,'Chris','Evans','1989-06-25','chrisevans@email.com','+1 (567) 890-1234','2021-10-01','Manager',5,'Male');
CREATE TABLE LabPosition(
    id INTEGER PRIMARY KEY,
    positionTypeId INTEGER,
    labPosition TEXT,
    gender TEXT,
    minimumEducation TEXT,
    experienceInYears INTEGER,
    salaryRange TEXT,
    FOREIGN KEY (positionTypeId) REFERENCES PositionType(id));
INSERT INTO "LabPosition" VALUES(1,1,'Research Scientist','Female','PhD',5,'60k - 90k');
INSERT INTO "LabPosition" VALUES(2,1,'Research Scientist','Male','PhD',7,'80k - 110k');
INSERT INTO "LabPosition" VALUES(3,2,'Laboratory Technician','Female','BSc',3,'40k - 60k');
INSERT INTO "LabPosition" VALUES(4,2,'Laboratory Technician','Male','BSc',4,'50k - 70k');
INSERT INTO "LabPosition" VALUES(5,3,'Data Analyst','Non-binary','MS',2,'60k - 80k');
CREATE TABLE PositionType(
    id INTEGER PRIMARY KEY,
    typeName TEXT);
INSERT INTO "PositionType" VALUES(1,'Full Time');
INSERT INTO "PositionType" VALUES(2,'Part Time');
INSERT INTO "PositionType" VALUES(3,'Contractual');
INSERT INTO "PositionType" VALUES(4,'Internship');
INSERT INTO "PositionType" VALUES(5,'Volunteering');
COMMIT;
