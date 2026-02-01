BEGIN TRANSACTION;
CREATE TABLE CityPopulationByDemographics (
    locationID INTEGER REFERENCES LocationData(locationID),
    ageRange TEXT,
    gender TEXT,
    ethnicity TEXT,
    population INTEGER);
INSERT INTO "CityPopulationByDemographics" VALUES(1,'18-35','Female','White',4000000);
INSERT INTO "CityPopulationByDemographics" VALUES(1,'36+','Male','Black',2500000);
CREATE TABLE EducationLevel (
    educationLevel TEXT,
    avgYearsOfEducation REAL ,
    avgAnnualIncome DECIMAL(9,2));
INSERT INTO "EducationLevel" VALUES('PhD',8.0,90000);
INSERT INTO "EducationLevel" VALUES('Masters',6.0,75000);
CREATE TABLE LocationData (
    locationID INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT,
    state TEXT,
    country TEXT,
    numPeople INTEGER);
INSERT INTO "LocationData" VALUES(1,'New York','NY','USA',8000000);
INSERT INTO "LocationData" VALUES(2,'Los Angeles','CA','USA',3800000);
CREATE TABLE OccupationalStats (
    occupation TEXT,
    percentageGap REAL ,
    differenceInSalaries DECIMAL(8,2),
    jobSatisfactionIndex INT ,
    FOREIGN KEY (occupation) REFERENCES People(occupation));
INSERT INTO "OccupationalStats" VALUES('Engineering',20.0,25000,80);
INSERT INTO "OccupationalStats" VALUES('Teaching',10.0,10000,70);
CREATE TABLE People (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    dateOfBirth DATE,
    occupation TEXT);
INSERT INTO "People" VALUES(1,'John','Doe','1970-06-01','Engineering');
INSERT INTO "People" VALUES(2,'Jane','Smith','1983-08-15','Teaching');
CREATE TABLE SalaryData (
    personId INTEGER REFERENCES People(id),
    salary DECIMAL(10,2),
    hireDate DATE);
INSERT INTO "SalaryData" VALUES(1,90000,'2010-01-01');
INSERT INTO "SalaryData" VALUES(2,60000,'2012-05-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('LocationData',2);
COMMIT;
