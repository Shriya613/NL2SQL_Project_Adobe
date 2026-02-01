BEGIN TRANSACTION;
CREATE TABLE AgeGroup (
    id INTEGER PRIMARY KEY,
    ageFrom INTEGER,
    ageTo INTEGER,
    population BIGINT,
    FOREIGN KEY(population) REFERENCES Population(id)
);
INSERT INTO "AgeGroup" VALUES(1,0,17,1000000);
INSERT INTO "AgeGroup" VALUES(2,18,64,2000000);
INSERT INTO "AgeGroup" VALUES(3,65,74,500000);
INSERT INTO "AgeGroup" VALUES(4,75,84,250000);
INSERT INTO "AgeGroup" VALUES(5,85,99,100000);
CREATE TABLE Country (
    id INTEGER PRIMARY KEY,
    countryName TEXT,
    continent TEXT
);
INSERT INTO "Country" VALUES(1,'USA','North America');
INSERT INTO "Country" VALUES(2,'Canada','North America');
INSERT INTO "Country" VALUES(3,'Mexico','North America');
INSERT INTO "Country" VALUES(4,'France','Europe');
INSERT INTO "Country" VALUES(5,'Germany','Europe');
CREATE TABLE EducationLevel (
    id INTEGER PRIMARY KEY,
    levelName TEXT,
    percentage FLOAT ,
    FOREIGN KEY(levelName) REFERENCES Ethnicity(ethnicGroupName)
);
INSERT INTO "EducationLevel" VALUES(1,'High School Diploma',10.0);
INSERT INTO "EducationLevel" VALUES(2,'Bachelor Degree',25.0);
INSERT INTO "EducationLevel" VALUES(3,'Master Degree',15.0);
INSERT INTO "EducationLevel" VALUES(4,'Doctorate',5.0);
INSERT INTO "EducationLevel" VALUES(5,'No College',25.0);
CREATE TABLE Ethnicity (
    id INTEGER PRIMARY KEY,
    ethnicGroupName TEXT,
    population BIGINT,
    FOREIGN KEY(population) REFERENCES Population(id)
);
INSERT INTO "Ethnicity" VALUES(1,'White',2000000);
INSERT INTO "Ethnicity" VALUES(2,'Black',500000);
INSERT INTO "Ethnicity" VALUES(3,'Asian',300000);
INSERT INTO "Ethnicity" VALUES(4,'Hispanic',400000);
INSERT INTO "Ethnicity" VALUES(5,'Other',100000);
CREATE TABLE ImmigrationStats (
    id INTEGER PRIMARY KEY,
    numberOfImmigrants INTEGER,
    immigrantPercentage FLOAT ,
    countryId INTEGER,
    year BIGINT,
    FOREIGN KEY(countryId) REFERENCES Country(id)
);
INSERT INTO "ImmigrationStats" VALUES(1,100000,3.5,1,2019);
INSERT INTO "ImmigrationStats" VALUES(2,120000,4.0,1,2020);
INSERT INTO "ImmigrationStats" VALUES(3,140000,5.5,4,2021);
INSERT INTO "ImmigrationStats" VALUES(4,160000,5.0,1,2022);
INSERT INTO "ImmigrationStats" VALUES(5,180000,4.5,2,2023);
CREATE TABLE Population (
    id INTEGER PRIMARY KEY,
    totalPopulation BIGINT,
    year BIGINT
);
INSERT INTO "Population" VALUES(1,30000000,2019);
INSERT INTO "Population" VALUES(2,31000000,2020);
INSERT INTO "Population" VALUES(3,32000000,2021);
INSERT INTO "Population" VALUES(4,33000000,2022);
INSERT INTO "Population" VALUES(5,34000000,2023);
COMMIT;
