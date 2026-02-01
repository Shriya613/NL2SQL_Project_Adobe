BEGIN TRANSACTION;
CREATE TABLE Animals (
    AnimalID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT,
    Quantity INTEGER 
);
INSERT INTO "Animals" VALUES(1,'Cow',20);
INSERT INTO "Animals" VALUES(2,'Pig',30);
INSERT INTO "Animals" VALUES(3,'Chicken',100);
INSERT INTO "Animals" VALUES(4,'Sheep',25);
INSERT INTO "Animals" VALUES(5,'Goat',15);
CREATE TABLE Crops (
    CropID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    HarvestDate DATE
);
INSERT INTO "Crops" VALUES(1,'Wheat','2023-08-01');
INSERT INTO "Crops" VALUES(2,'Corn','2023-09-01');
INSERT INTO "Crops" VALUES(3,'Barley','2023-07-01');
INSERT INTO "Crops" VALUES(4,'Rice','2023-10-01');
INSERT INTO "Crops" VALUES(5,'Oats','2023-06-01');
CREATE TABLE Equipment (
    EquipID INTEGER PRIMARY KEY AUTOINCREMENT,
    Model TEXT,
    PurchaseDate DATE
);
INSERT INTO "Equipment" VALUES(1,'Tractor A','2020-01-01');
INSERT INTO "Equipment" VALUES(2,'Combine','2021-03-01');
INSERT INTO "Equipment" VALUES(3,'Plow','2019-05-01');
INSERT INTO "Equipment" VALUES(4,'Truck','2022-02-01');
INSERT INTO "Equipment" VALUES(5,'Irrigation System','2020-04-01');
CREATE TABLE Farms (
    FarmID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT
);
INSERT INTO "Farms" VALUES(1,'Farm A','California');
INSERT INTO "Farms" VALUES(2,'Farm B','Texas');
INSERT INTO "Farms" VALUES(3,'Farm C','Washington');
INSERT INTO "Farms" VALUES(4,'Farm D','Florida');
INSERT INTO "Farms" VALUES(5,'Farm E','North Dakota');
CREATE TABLE Farms_Animals (
    FarmID INTEGER,
    AnimalID INTEGER,
    FOREIGN KEY (FarmID) REFERENCES Farms(FarmID),
    FOREIGN KEY (AnimalID) REFERENCES Animals(AnimalID),
    PRIMARY KEY (FarmID, AnimalID)
);
INSERT INTO "Farms_Animals" VALUES(1,1);
INSERT INTO "Farms_Animals" VALUES(2,1);
INSERT INTO "Farms_Animals" VALUES(3,1);
INSERT INTO "Farms_Animals" VALUES(4,1);
INSERT INTO "Farms_Animals" VALUES(5,1);
INSERT INTO "Farms_Animals" VALUES(5,4);
CREATE TABLE Farms_Crops (
    FarmID INTEGER,
    CropID INTEGER,
    PlantingDate DATE,
    HarvestEstimate DATE,
    FOREIGN KEY (FarmID) REFERENCES Farms(FarmID),
    FOREIGN KEY (CropID) REFERENCES Crops(CropID),
    PRIMARY KEY (FarmID, CropID)
);
CREATE TABLE Farms_Equipment (
    FarmID INTEGER,
    EquipID INTEGER,
    AcquisitionDate DATE,
    MaintenanceSchedule TEXT,
    FOREIGN KEY (FarmID) REFERENCES Farms(FarmID),
    FOREIGN KEY (EquipID) REFERENCES Equipment(EquipID),
    PRIMARY KEY (FarmID, EquipID)
);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Farms',5);
INSERT INTO "sqlite_sequence" VALUES('Animals',5);
INSERT INTO "sqlite_sequence" VALUES('Crops',5);
INSERT INTO "sqlite_sequence" VALUES('Equipment',5);
COMMIT;
