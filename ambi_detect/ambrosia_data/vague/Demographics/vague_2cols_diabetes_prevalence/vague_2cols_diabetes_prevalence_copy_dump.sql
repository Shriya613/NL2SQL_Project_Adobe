BEGIN TRANSACTION;
CREATE TABLE Cities(
    CityID INTEGER PRIMARY KEY AUTOINCREMENT,
    StateID INTEGER,
    Name TEXT,
    FOREIGN KEY (StateID) REFERENCES States(StateID));
INSERT INTO "Cities" VALUES(1,1,'Los Angeles');
INSERT INTO "Cities" VALUES(2,1,'San Francisco');
INSERT INTO "Cities" VALUES(3,1,'Chicago');
INSERT INTO "Cities" VALUES(4,2,'Montreal');
INSERT INTO "Cities" VALUES(5,2,'Toronto');
CREATE TABLE Countries(
    CountryID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Population INTEGER);
INSERT INTO "Countries" VALUES(1,'USA',328000000);
INSERT INTO "Countries" VALUES(2,'Canada',37600000);
CREATE TABLE "DiabetesData"(
    DataID INTEGER PRIMARY KEY AUTOINCREMENT,
    Diabetes_Prevalence INTEGER,
    Population INTEGER,
    CityId INTEGER,
    FOREIGN KEY (CityId) REFERENCES Cities(CityID));
INSERT INTO "DiabetesData" VALUES(1,5,10000,1);
INSERT INTO "DiabetesData" VALUES(2,3,10000,2);
INSERT INTO "DiabetesData" VALUES(3,4,10000,3);
INSERT INTO "DiabetesData" VALUES(4,6,10000,4);
INSERT INTO "DiabetesData" VALUES(5,2,10000,5);
CREATE TABLE HealthCareFacilities(
    FacilityID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    PhoneNumber TEXT,
    CityId INTEGER,
    FOREIGN KEY (CityId) REFERENCES Cities(CityID));
INSERT INTO "HealthCareFacilities" VALUES(1,'St. John Hospital','123 Main St.','555-1234',1);
INSERT INTO "HealthCareFacilities" VALUES(2,'Mount Sinai Medical Center','456 Elm St.','555-5678',2);
INSERT INTO "HealthCareFacilities" VALUES(3,'Sunnyview Hospital','789 Oak St.','555-9012',3);
INSERT INTO "HealthCareFacilities" VALUES(4,'Riverside Clinic','101 Pine St.','555-1357',4);
INSERT INTO "HealthCareFacilities" VALUES(5,'Community Care Center','202 Fir St.','555-1415',5);
CREATE TABLE People(
    PersonID INTEGER PRIMARY KEY AUTOINCREMENT,
    Age INTEGER,
    Sex CHAR(1), -- 'M' for Male, 'F' for Female, etc.
    Occupation TEXT,
    ResidenceCityId INTEGER,
    FOREIGN KEY (ResidenceCityId) REFERENCES Cities(CityID));
INSERT INTO "People" VALUES(1,30,'M','Engineer',1);
INSERT INTO "People" VALUES(2,45,'F','Doctor',2);
INSERT INTO "People" VALUES(3,50,'M','Teacher',3);
INSERT INTO "People" VALUES(4,25,'F','Lawyer',4);
INSERT INTO "People" VALUES(5,35,'M','Accountant',5);
CREATE TABLE States(
    StateID INTEGER PRIMARY KEY AUTOINCREMENT,
    CountryID INTEGER,
    Name TEXT,
    FOREIGN KEY (CountryID) REFERENCES Countries(CountryID));
INSERT INTO "States" VALUES(1,1,'California');
INSERT INTO "States" VALUES(2,1,'New York');
INSERT INTO "States" VALUES(3,2,'Quebec');
INSERT INTO "States" VALUES(4,2,'Ontario');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Countries',2);
INSERT INTO "sqlite_sequence" VALUES('States',4);
INSERT INTO "sqlite_sequence" VALUES('Cities',5);
INSERT INTO "sqlite_sequence" VALUES('People',5);
INSERT INTO "sqlite_sequence" VALUES('HealthCareFacilities',5);
INSERT INTO "sqlite_sequence" VALUES('DiabetesData',5);
COMMIT;
