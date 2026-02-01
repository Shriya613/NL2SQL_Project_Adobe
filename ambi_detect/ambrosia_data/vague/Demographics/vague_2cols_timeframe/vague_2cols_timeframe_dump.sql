BEGIN TRANSACTION;
CREATE TABLE Age_Groups(
    [AgeGroupID] INTEGER PRIMARY KEY AUTOINCREMENT,
    [CityName] TEXT,
    [PercentageOfTotalPopulation] REAL 
);
INSERT INTO "Age_Groups" VALUES(1,'New York City',0.15);
INSERT INTO "Age_Groups" VALUES(2,'Los Angeles',0.1);
INSERT INTO "Age_Groups" VALUES(3,'Chicago',0.08);
INSERT INTO "Age_Groups" VALUES(4,'Houston',0.06);
INSERT INTO "Age_Groups" VALUES(5,'Philadelphia',0.05);
CREATE TABLE Annual_Demographics(
    [Year] INTEGER PRIMARY KEY,
    [Month] INTEGER,
    [Census] TEXT,
    [Population] INTEGER,
    [Area] REAL,
    [Density] REAL
);
INSERT INTO "Annual_Demographics" VALUES(2020,1,'US Census',331489267,9833517.0,334.0);
INSERT INTO "Annual_Demographics" VALUES(2021,2,'Estimate',333000000,9876543.0,337.0);
INSERT INTO "Annual_Demographics" VALUES(2022,3,'Estimate',334000000,9899432.0,339.0);
INSERT INTO "Annual_Demographics" VALUES(2023,4,'Estimate',335000000,9922321.0,340.0);
INSERT INTO "Annual_Demographics" VALUES(2024,5,'Estimate',336000000,9945201.0,341.0);
CREATE TABLE Educational_Levels(
    [EducationLevelID] INTEGER PRIMARY KEY AUTOINCREMENT,
    [CityName] TEXT,
    [BachelorsOrHigher] INTEGER,
    [HighSchoolGraduate] INTEGER,
    [LessThanHighSchoolDiploma] INTEGER
);
INSERT INTO "Educational_Levels" VALUES(1,'New York City',70,20,10);
INSERT INTO "Educational_Levels" VALUES(2,'Los Angeles',60,25,15);
INSERT INTO "Educational_Levels" VALUES(3,'Chicago',55,25,20);
INSERT INTO "Educational_Levels" VALUES(4,'Houston',45,30,25);
INSERT INTO "Educational_Levels" VALUES(5,'Philadelphia',50,25,25);
CREATE TABLE Ethnicities(
    [EthnicityID] INTEGER PRIMARY KEY AUTOINCREMENT,
    [CityName] TEXT,
    [White] INTEGER,
    [BlackOrAfricanAmerican] INTEGER,
    [HispanicOrLatino] INTEGER,
    [Asian] INTEGER,
    [NativeAmericanAlaskanNative] INTEGER,
    [OtherPacificIslander] INTEGER,
    [TwoOrMoreRaces] INTEGER
);
INSERT INTO "Ethnicities" VALUES(1,'New York City',45,20,25,10,2,1,2);
INSERT INTO "Ethnicities" VALUES(2,'Los Angeles',25,10,40,15,2,1,3);
INSERT INTO "Ethnicities" VALUES(3,'Chicago',30,25,25,10,3,1,2);
INSERT INTO "Ethnicities" VALUES(4,'Houston',20,25,40,10,3,1,2);
INSERT INTO "Ethnicities" VALUES(5,'Philadelphia',35,20,25,10,2,1,2);
CREATE TABLE Income_Brackets(
    [IncomeBracketID] INTEGER PRIMARY KEY AUTOINCREMENT,
    [CityName] TEXT,
    [Under50k] INTEGER,
    [Between50And100K] INTEGER,
    [Above100k] INTEGER
);
INSERT INTO "Income_Brackets" VALUES(1,'New York City',35,40,25);
INSERT INTO "Income_Brackets" VALUES(2,'Los Angeles',30,45,25);
INSERT INTO "Income_Brackets" VALUES(3,'Chicago',25,40,35);
INSERT INTO "Income_Brackets" VALUES(4,'Houston',20,50,30);
INSERT INTO "Income_Brackets" VALUES(5,'Philadelphia',25,40,35);
CREATE TABLE Occupations(
    [OccupationID] INTEGER PRIMARY KEY AUTOINCREMENT,
    [CityName] TEXT,
    [ManagementOrProfessional] INTEGER,
    [ServiceIndustry] INTEGER,
    [Sales] INTEGER,
    [OfficeAdminSupport] INTEGER,
    [FarmingFishing] INTEGER,
    [ConstructionTrades] INTEGER,
    [ProductionTransportation] INTEGER,
    [MaterialMoving] INTEGER,
    [PersonalServices] INTEGER,
    [HealthcareTechnician] INTEGER,
    [Teachers] INTEGER,
    [ArmedForces] INTEGER,
    [None] INTEGER
);
INSERT INTO "Occupations" VALUES(1,'New York City',25,20,15,10,5,10,5,5,10,5,5,5,2);
INSERT INTO "Occupations" VALUES(2,'Los Angeles',20,25,15,10,5,10,5,5,10,5,5,5,3);
INSERT INTO "Occupations" VALUES(3,'Chicago',15,20,15,10,5,10,5,5,10,5,5,5,4);
INSERT INTO "Occupations" VALUES(4,'Houston',10,20,15,10,5,10,5,5,10,5,5,5,5);
INSERT INTO "Occupations" VALUES(5,'Philadelphia',15,20,15,10,5,10,5,5,10,5,5,5,6);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Age_Groups',5);
INSERT INTO "sqlite_sequence" VALUES('Educational_Levels',5);
INSERT INTO "sqlite_sequence" VALUES('Income_Brackets',5);
INSERT INTO "sqlite_sequence" VALUES('Ethnicities',5);
INSERT INTO "sqlite_sequence" VALUES('Occupations',5);
COMMIT;
