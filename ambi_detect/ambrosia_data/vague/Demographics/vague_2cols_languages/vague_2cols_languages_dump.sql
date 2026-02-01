BEGIN TRANSACTION;
CREATE TABLE Cities (
    CityID INTEGER PRIMARY KEY,
    Name TEXT,
    Province TEXT,
    Population INTEGER,
    RegionID INTEGER,
    FOREIGN KEY(RegionID) REFERENCES Regions(RegionID));
CREATE TABLE Countries (
    CountryID INTEGER PRIMARY KEY,
    Name TEXT,
    Capital TEXT,
    Population INTEGER,
    Official_Language TEXT,
    Spoken_Language TEXT,
    Community TEXT);
INSERT INTO "Countries" VALUES(1,'United States','Washington D.C.',328200000,'English','English','Community A');
INSERT INTO "Countries" VALUES(2,'Canada','Ottawa',37742000,'French','English','Community B');
CREATE TABLE EducationLevel (
    LevelID INTEGER PRIMARY KEY,
    Description TEXT,
    Percentage INTEGER);
INSERT INTO "EducationLevel" VALUES(1,'Less than High School',1000000);
INSERT INTO "EducationLevel" VALUES(2,'High School Graduate',2000000);
INSERT INTO "EducationLevel" VALUES(3,'Some College',2500000);
INSERT INTO "EducationLevel" VALUES(4,'College Degree',3000000);
INSERT INTO "EducationLevel" VALUES(5,'Postgraduate Degree',500000);
CREATE TABLE EthnicGroups (
    GroupID INTEGER PRIMARY KEY,
    Name TEXT,
    Population INTEGER);
INSERT INTO "EthnicGroups" VALUES(1,'White American',246000000);
INSERT INTO "EthnicGroups" VALUES(2,'Black Americans',43000000);
CREATE TABLE Regions (
    RegionID INTEGER PRIMARY KEY,
    Name TEXT,
    Area FLOAT,
    PopulationCount INTEGER);
INSERT INTO "Regions" VALUES(1,'North America',9560000.0,377420000);
INSERT INTO "Regions" VALUES(2,'South America',17840000.0,428200000);
CREATE TABLE Religion (
    ID INTEGER PRIMARY KEY,
    Denomination TEXT,
    Adherents INTEGER);
INSERT INTO "Religion" VALUES(1,'Christianity',240000000);
INSERT INTO "Religion" VALUES(2,'Islam',8000000);
INSERT INTO "Religion" VALUES(3,'Hinduism',1000000);
INSERT INTO "Religion" VALUES(4,'Buddhism',2000000);
INSERT INTO "Religion" VALUES(5,'Judaism',1500000);
COMMIT;
