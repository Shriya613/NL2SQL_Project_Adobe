BEGIN TRANSACTION;
CREATE TABLE Applicants(
    applicantID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT,
    phoneNumber CHAR(10),
    resumePath TEXT);
INSERT INTO "Applicants" VALUES(1,'John','Doe','john.doe@example.com','+1 (123) 456-7890','https://www.example.com/resume/johndoe.pdf');
INSERT INTO "Applicants" VALUES(2,'Jane','Smith','jane.smith@example.com','+1 (123) 456-7891','https://www.example.com/resume/janesmith.pdf');
INSERT INTO "Applicants" VALUES(3,'Alice','Johnson','alice.johnson@example.com','+1 (123) 456-7892','https://www.example.com/resume/alicejohnson.pdf');
INSERT INTO "Applicants" VALUES(4,'Tom','Williams','tom.williams@example.com','+1 (123) 456-7893','https://www.example.com/resume/tomwilliams.pdf');
INSERT INTO "Applicants" VALUES(5,'Lucy','Brown','lucy.brown@example.com','+1 (123) 456-7894','https://www.example.com/resume/lucybrown.pdf');
CREATE TABLE Construction_Jobs(
    constructionJobID INTEGER PRIMARY KEY AUTOINCREMENT,
    jobTitle TEXT,
    jobDescription TEXT,
    requiredSkills TEXT,
    constructionJobType TEXT,
    designSoftware TEXT,
    FOREIGN KEY(designSoftware) REFERENCES Design_Software_Expertise(softwareName));
INSERT INTO "Construction_Jobs" VALUES(1,'Senior Civil Engineer','Plan, design, and supervise construction projects.','Strong math and communication skills; Knowledge of building codes and regulations.','Civil Engineering','AutoCAD');
INSERT INTO "Construction_Jobs" VALUES(2,'Civil Engineer','Plan, design, and supervise construction projects.','Strong math and communication skills; Knowledge of building codes and regulations.','Civil Engineering','Revit');
INSERT INTO "Construction_Jobs" VALUES(3,'Senior Architect','Design buildings and spaces, ensuring functionality and aesthetics.','Creative thinking, attention to detail, and knowledge of architectural principles.','Architecture','AutoCAD');
INSERT INTO "Construction_Jobs" VALUES(4,'Architect','Design buildings and spaces, ensuring functionality and aesthetics.','Creative thinking, attention to detail, and knowledge of architectural principles.','Architecture','Rhino');
CREATE TABLE Design_Software_Expertise(
    softwareID INTEGER PRIMARY KEY AUTOINCREMENT,
    softwareName TEXT UNIQUE);
INSERT INTO "Design_Software_Expertise" VALUES(1,'AutoCAD');
INSERT INTO "Design_Software_Expertise" VALUES(2,'Revit');
INSERT INTO "Design_Software_Expertise" VALUES(3,'Rhino');
INSERT INTO "Design_Software_Expertise" VALUES(4,'SketchUp');
INSERT INTO "Design_Software_Expertise" VALUES(5,'Adobe Illustrator');
CREATE TABLE Employed_in_Construction_Jobs(
    employedID INTEGER PRIMARY KEY AUTOINCREMENT,
    applicantID INTEGER,
    constructionJobID INTEGER,
    startDate DATE,
    endDate DATE,
    FOREIGN KEY(applicantID) REFERENCES Applicants(applicantID),
    FOREIGN KEY(constructionJobID) REFERENCES Construction_Jobs(constructionJobID));
INSERT INTO "Employed_in_Construction_Jobs" VALUES(1,1,1,'2023-01-15','2023-12-31');
INSERT INTO "Employed_in_Construction_Jobs" VALUES(2,2,2,'2023-02-15','2023-12-31');
INSERT INTO "Employed_in_Construction_Jobs" VALUES(3,3,3,'2023-03-15','2023-12-31');
INSERT INTO "Employed_in_Construction_Jobs" VALUES(4,4,4,'2023-04-15','2023-12-31');
INSERT INTO "Employed_in_Construction_Jobs" VALUES(5,5,5,'2023-05-15','2023-12-31');
CREATE TABLE Employers(
    employerID INTEGER PRIMARY KEY AUTOINCREMENT,
    companyName TEXT,
    location TEXT);
INSERT INTO "Employers" VALUES(1,'CompanyA','New York');
INSERT INTO "Employers" VALUES(2,'CompanyB','Los Angeles');
INSERT INTO "Employers" VALUES(3,'CompanyC','Chicago');
INSERT INTO "Employers" VALUES(4,'CompanyD','Miami');
INSERT INTO "Employers" VALUES(5,'CompanyE','Houston');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Employers',5);
INSERT INTO "sqlite_sequence" VALUES('Applicants',5);
INSERT INTO "sqlite_sequence" VALUES('Construction_Jobs',4);
INSERT INTO "sqlite_sequence" VALUES('Design_Software_Expertise',5);
INSERT INTO "sqlite_sequence" VALUES('Employed_in_Construction_Jobs',5);
COMMIT;
