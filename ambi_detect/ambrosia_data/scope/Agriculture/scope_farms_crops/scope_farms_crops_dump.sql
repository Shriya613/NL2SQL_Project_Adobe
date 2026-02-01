BEGIN TRANSACTION;
CREATE TABLE Crops(
    CropID INTEGER PRIMARY KEY AUTOINCREMENT,
    CropName TEXT,
    YieldPerAcre REAL);
INSERT INTO "Crops" VALUES(1,'Wheat',6000.0);
INSERT INTO "Crops" VALUES(2,'Corn',18000.0);
INSERT INTO "Crops" VALUES(3,'Rice',3500.0);
INSERT INTO "Crops" VALUES(4,'Barley',2700.0);
INSERT INTO "Crops" VALUES(5,'Oats',1900.0);
CREATE TABLE CropsWeather(
    CropID INTEGER,
    ConditionID INTEGER,
    FOREIGN KEY(CropID) REFERENCES Crops(CropID),
    FOREIGN KEY(ConditionID) REFERENCES WeatherConditions(ConditionID));
INSERT INTO "CropsWeather" VALUES(1,1);
INSERT INTO "CropsWeather" VALUES(1,2);
INSERT INTO "CropsWeather" VALUES(2,3);
INSERT INTO "CropsWeather" VALUES(2,4);
INSERT INTO "CropsWeather" VALUES(3,5);
CREATE TABLE Farms(
    FarmID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT);
INSERT INTO "Farms" VALUES(1,'Farm A','Location A');
INSERT INTO "Farms" VALUES(2,'Farm B','Location B');
INSERT INTO "Farms" VALUES(3,'Farm C','Location C');
INSERT INTO "Farms" VALUES(4,'Farm D','Location D');
INSERT INTO "Farms" VALUES(5,'Farm E','Location E');
CREATE TABLE FarmsCrops(
    FarmID INTEGER,
    CropID INTEGER,
    Quantity INTEGER DEFAULT 0,
    PlantingDate DATE,
    HarvestDate DATE,
    FOREIGN KEY(FarmID) REFERENCES Farms(FarmID),
    FOREIGN KEY(CropID) REFERENCES Crops(CropID));
INSERT INTO "FarmsCrops" VALUES(1,1,500,'2022-04-01','2022-09-01');
INSERT INTO "FarmsCrops" VALUES(1,2,300,'2022-04-01','2022-09-01');
INSERT INTO "FarmsCrops" VALUES(2,3,200,'2022-04-01','2022-09-01');
INSERT INTO "FarmsCrops" VALUES(3,1,400,'2022-04-01','2022-09-01');
INSERT INTO "FarmsCrops" VALUES(4,2,100,'2022-04-01','2022-09-01');
INSERT INTO "FarmsCrops" VALUES(2,1,200,'2022-04-01','2022-09-01');
INSERT INTO "FarmsCrops" VALUES(4,1,100,'2022-04-01','2022-09-01');
INSERT INTO "FarmsCrops" VALUES(5,1,0,NULL,NULL);
CREATE TABLE FarmsSoil(
    FarmID INTEGER,
    SoilTypeID INTEGER,
    FOREIGN KEY(FarmID) REFERENCES Farms(FarmID),
    FOREIGN KEY(SoilTypeID) REFERENCES SoilTypes(TypeID));
INSERT INTO "FarmsSoil" VALUES(1,1);
INSERT INTO "FarmsSoil" VALUES(1,2);
INSERT INTO "FarmsSoil" VALUES(2,3);
INSERT INTO "FarmsSoil" VALUES(2,4);
INSERT INTO "FarmsSoil" VALUES(3,5);
CREATE TABLE SoilTypes(
    TypeID INTEGER PRIMARY KEY AUTOINCREMENT,
    TypeName TEXT);
INSERT INTO "SoilTypes" VALUES(1,'Clay Loam');
INSERT INTO "SoilTypes" VALUES(2,'Sandy Loam');
INSERT INTO "SoilTypes" VALUES(3,'Silty Clay Loam');
INSERT INTO "SoilTypes" VALUES(4,'Gravelly Sandy Loam');
INSERT INTO "SoilTypes" VALUES(5,'Loamy Sand');
CREATE TABLE WeatherConditions(
    ConditionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT);
INSERT INTO "WeatherConditions" VALUES(1,'Sunny');
INSERT INTO "WeatherConditions" VALUES(2,'Cloudy');
INSERT INTO "WeatherConditions" VALUES(3,'Rainy');
INSERT INTO "WeatherConditions" VALUES(4,'Windy');
INSERT INTO "WeatherConditions" VALUES(5,'Stormy');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Farms',5);
INSERT INTO "sqlite_sequence" VALUES('Crops',5);
INSERT INTO "sqlite_sequence" VALUES('SoilTypes',5);
INSERT INTO "sqlite_sequence" VALUES('WeatherConditions',5);
COMMIT;
