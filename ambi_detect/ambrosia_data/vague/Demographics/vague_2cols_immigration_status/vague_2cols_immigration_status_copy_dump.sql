BEGIN TRANSACTION;
CREATE TABLE EducationLevel (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Age_Group TEXT,
    Graduation_Rate DECIMAL(5,2),
    HighSchool_Diploma_or_Higher INT,
    College_Degree INT
);
INSERT INTO "EducationLevel" VALUES(1,'Age Group 1',90,7000,4500);
INSERT INTO "EducationLevel" VALUES(2,'Age Group 2',85,6000,4000);
INSERT INTO "EducationLevel" VALUES(3,'Age Group 3',80,5000,3500);
INSERT INTO "EducationLevel" VALUES(4,'Age Group 4',75,4000,3000);
INSERT INTO "EducationLevel" VALUES(5,'Age Group 5',70,3000,2500);
CREATE TABLE EmploymentStatus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Age_Group TEXT,
    Unemployed INT,
    Employed INT,
    Labor_Force_Participation_Rate DECIMAL(5,2)
);
INSERT INTO "EmploymentStatus" VALUES(1,'Age Group 1',500,8000,45);
INSERT INTO "EmploymentStatus" VALUES(2,'Age Group 2',400,6500,40);
INSERT INTO "EmploymentStatus" VALUES(3,'Age Group 3',300,5000,35);
INSERT INTO "EmploymentStatus" VALUES(4,'Age Group 4',200,4500,30);
INSERT INTO "EmploymentStatus" VALUES(5,'Age Group 5',100,4000,25);
CREATE TABLE HealthStatistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Age_Group TEXT,
    Obesity_Rate DECIMAL(5,2),
    Smoking_Prevalence DECIMAL(5,2),
    LifeExpectancy INT
);
INSERT INTO "HealthStatistics" VALUES(1,'Age Group 1',20,20,75);
INSERT INTO "HealthStatistics" VALUES(2,'Age Group 2',25,25,74);
INSERT INTO "HealthStatistics" VALUES(3,'Age Group 3',30,30,73);
INSERT INTO "HealthStatistics" VALUES(4,'Age Group 4',35,35,72);
INSERT INTO "HealthStatistics" VALUES(5,'Age Group 5',40,40,71);
CREATE TABLE IncomeLevels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Age_Group TEXT,
    Median_Income DECIMAL(10,2),
    Poverty_Rate DECIMAL(5,2),
    Low_Income_Percentage INT
);
INSERT INTO "IncomeLevels" VALUES(1,'Age Group 1',50000,10,500);
INSERT INTO "IncomeLevels" VALUES(2,'Age Group 2',60000,12,600);
INSERT INTO "IncomeLevels" VALUES(3,'Age Group 3',70000,15,750);
INSERT INTO "IncomeLevels" VALUES(4,'Age Group 4',80000,18,850);
INSERT INTO "IncomeLevels" VALUES(5,'Age Group 5',90000,20,900);
CREATE TABLE "Population" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Total_Population INT,
    Immigration_Status INT,
    Municipality TEXT
);
INSERT INTO "Population" VALUES(1,100000,5000,'New York City');
INSERT INTO "Population" VALUES(2,80000,3500,'Los Angeles');
INSERT INTO "Population" VALUES(3,70000,2500,'Chicago');
INSERT INTO "Population" VALUES(4,60000,2000,'Houston');
INSERT INTO "Population" VALUES(5,50000,1500,'Philadelphia');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('EducationLevel',5);
INSERT INTO "sqlite_sequence" VALUES('EmploymentStatus',5);
INSERT INTO "sqlite_sequence" VALUES('HealthStatistics',5);
INSERT INTO "sqlite_sequence" VALUES('IncomeLevels',5);
INSERT INTO "sqlite_sequence" VALUES('Population',5);
COMMIT;
