BEGIN TRANSACTION;
CREATE TABLE AgeGroup (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    AgeRange TEXT,
    PopulationInThisAgeRange INTEGER 
);
INSERT INTO "AgeGroup" VALUES(1,'0-18',120);
INSERT INTO "AgeGroup" VALUES(2,'19-35',100);
INSERT INTO "AgeGroup" VALUES(3,'36-50',70);
INSERT INTO "AgeGroup" VALUES(4,'51-65',50);
INSERT INTO "AgeGroup" VALUES(5,'65+',30);
CREATE TABLE EducationLevel (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Level TEXT 
);
INSERT INTO "EducationLevel" VALUES(1,'No formal education');
INSERT INTO "EducationLevel" VALUES(2,'Primary school');
INSERT INTO "EducationLevel" VALUES(3,'Secondary school');
INSERT INTO "EducationLevel" VALUES(4,'High school diploma');
INSERT INTO "EducationLevel" VALUES(5,'Some college');
CREATE TABLE EmploymentStatus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Status TEXT 
);
INSERT INTO "EmploymentStatus" VALUES(1,'Unemployed');
INSERT INTO "EmploymentStatus" VALUES(2,'Employed full-time');
INSERT INTO "EmploymentStatus" VALUES(3,'Employed part-time');
INSERT INTO "EmploymentStatus" VALUES(4,'Self-employed');
INSERT INTO "EmploymentStatus" VALUES(5,'Retired');
CREATE TABLE GenderDemographics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Gender TEXT ,
    Count INTEGER 
);
INSERT INTO "GenderDemographics" VALUES(1,'Male',200);
INSERT INTO "GenderDemographics" VALUES(2,'Female',100);
INSERT INTO "GenderDemographics" VALUES(3,'Other',5);
INSERT INTO "GenderDemographics" VALUES(4,'Nonbinary',3);
INSERT INTO "GenderDemographics" VALUES(5,'Prefer not to say',2);
CREATE TABLE MaritalStatus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Status TEXT 
);
INSERT INTO "MaritalStatus" VALUES(1,'Single');
INSERT INTO "MaritalStatus" VALUES(2,'Divorced');
INSERT INTO "MaritalStatus" VALUES(3,'Widowed');
INSERT INTO "MaritalStatus" VALUES(4,'Separated');
INSERT INTO "MaritalStatus" VALUES(5,'Married');
CREATE TABLE "VillageDemographics" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Village TEXT,
    Divorce_Rate REAL ,
    TotalPopulation INTEGER ,
    MaleCount INTEGER ,
    FemaleCount INTEGER 
);
INSERT INTO "VillageDemographics" VALUES(1,'Village1',0.8,300,160,140);
INSERT INTO "VillageDemographics" VALUES(2,'Village2',0.7,290,150,140);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('AgeGroup',5);
INSERT INTO "sqlite_sequence" VALUES('GenderDemographics',5);
INSERT INTO "sqlite_sequence" VALUES('MaritalStatus',5);
INSERT INTO "sqlite_sequence" VALUES('EmploymentStatus',5);
INSERT INTO "sqlite_sequence" VALUES('EducationLevel',5);
INSERT INTO "sqlite_sequence" VALUES('VillageDemographics',2);
COMMIT;
