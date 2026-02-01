BEGIN TRANSACTION;
CREATE TABLE Components(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    componentName TEXT);
INSERT INTO "Components" VALUES(1,'Nitrogen');
INSERT INTO "Components" VALUES(2,'Phosphorus');
INSERT INTO "Components" VALUES(3,'Potassium');
INSERT INTO "Components" VALUES(4,'Calcium');
INSERT INTO "Components" VALUES(5,'Magnesium');
CREATE TABLE Crops(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cropName TEXT,
    yield REAL);
INSERT INTO "Crops" VALUES(1,'Wheat',7000.0);
INSERT INTO "Crops" VALUES(2,'Corn',8000.0);
INSERT INTO "Crops" VALUES(3,'Rice',6000.0);
INSERT INTO "Crops" VALUES(4,'Barley',5000.0);
INSERT INTO "Crops" VALUES(5,'Oats',4000.0);
CREATE TABLE FarmerInfo(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    farmerName TEXT,
    farmLocation TEXT);
INSERT INTO "FarmerInfo" VALUES(1,'Farmer1','New York');
INSERT INTO "FarmerInfo" VALUES(2,'Farmer2','Chicago');
INSERT INTO "FarmerInfo" VALUES(3,'Farmer3','Los Angeles');
INSERT INTO "FarmerInfo" VALUES(4,'Farmer4','Houston');
INSERT INTO "FarmerInfo" VALUES(5,'Farmer5','Philadelphia');
CREATE TABLE Farmers_Crops(
    farmerId INTEGER,
    cropId INTEGER,
    FOREIGN KEY (farmerId) REFERENCES FarmerInfo(id),
    FOREIGN KEY (cropId) REFERENCES Crops(id));
INSERT INTO "Farmers_Crops" VALUES(1,1);
INSERT INTO "Farmers_Crops" VALUES(1,2);
INSERT INTO "Farmers_Crops" VALUES(2,1);
INSERT INTO "Farmers_Crops" VALUES(2,3);
INSERT INTO "Farmers_Crops" VALUES(3,1);
CREATE TABLE Fertilizers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    quantity REAL);
INSERT INTO "Fertilizers" VALUES(1,'Fertilizer1',100.0);
INSERT INTO "Fertilizers" VALUES(2,'Fertilizer2',150.0);
INSERT INTO "Fertilizers" VALUES(3,'Fertilizer3',200.0);
INSERT INTO "Fertilizers" VALUES(4,'Fertilizer4',250.0);
INSERT INTO "Fertilizers" VALUES(5,'Fertilizer5',300.0);
CREATE TABLE Fertilizers_Components(
    fertilizerId INTEGER,
    componentId INTEGER,
    FOREIGN KEY (fertilizerId) REFERENCES Fertilizers(id),
    FOREIGN KEY (componentId) REFERENCES Components(id));
INSERT INTO "Fertilizers_Components" VALUES(1,1);
INSERT INTO "Fertilizers_Components" VALUES(1,2);
INSERT INTO "Fertilizers_Components" VALUES(2,1);
INSERT INTO "Fertilizers_Components" VALUES(2,3);
INSERT INTO "Fertilizers_Components" VALUES(3,1);
INSERT INTO "Fertilizers_Components" VALUES(4,1);
INSERT INTO "Fertilizers_Components" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Fertilizers',5);
INSERT INTO "sqlite_sequence" VALUES('Components',5);
INSERT INTO "sqlite_sequence" VALUES('FarmerInfo',5);
INSERT INTO "sqlite_sequence" VALUES('Crops',5);
COMMIT;
