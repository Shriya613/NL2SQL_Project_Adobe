BEGIN TRANSACTION;
CREATE TABLE Countries(
    countryId INTEGER PRIMARY KEY AUTOINCREMENT,
    countryName TEXT);
INSERT INTO "Countries" VALUES(1,'France');
INSERT INTO "Countries" VALUES(2,'Italy');
INSERT INTO "Countries" VALUES(3,'Spain');
CREATE TABLE Regions(
    regionId INTEGER PRIMARY KEY AUTOINCREMENT,
    regionName TEXT,
    countryId INTEGER,
    FOREIGN KEY(countryId) REFERENCES Countries(countryId));
INSERT INTO "Regions" VALUES(1,'Bordeaux',1);
INSERT INTO "Regions" VALUES(2,'Tuscany',2);
INSERT INTO "Regions" VALUES(3,'La Rioja',3);
CREATE TABLE VineyardTypes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    typeName TEXT);
INSERT INTO "VineyardTypes" VALUES(1,'Red');
INSERT INTO "VineyardTypes" VALUES(2,'White');
INSERT INTO "VineyardTypes" VALUES(3,'Rosé');
CREATE TABLE Vineyards(
    vineyardId INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    areaSize REAL,
    typeId INTEGER,
    regionId INTEGER,
    FOREIGN KEY(typeId) REFERENCES VineyardTypes(id),
    FOREIGN KEY(regionId) REFERENCES Regions(regionId));
INSERT INTO "Vineyards" VALUES(1,'Château Lafite Rothschild',500.0,1,1);
INSERT INTO "Vineyards" VALUES(2,'Antinori Toscano',400.0,2,2);
INSERT INTO "Vineyards" VALUES(3,'Marqués de Riscal',600.0,1,3);
INSERT INTO "Vineyards" VALUES(4,'Domaine Leflaive',100.0,3,NULL);
INSERT INTO "Vineyards" VALUES(5,'Felsina',150.0,2,2);
CREATE TABLE Vineyards_Wines(
    vineyardId INTEGER,
    wineId INTEGER,
    yearProduced INTEGER,
    FOREIGN KEY(vineyardId) REFERENCES Vineyards(vineyardId),
    FOREIGN KEY(wineId) REFERENCES Wines(wineId),
    PRIMARY KEY(vineyardId, wineId, yearProduced));
INSERT INTO "Vineyards_Wines" VALUES(1,1,2018);
INSERT INTO "Vineyards_Wines" VALUES(2,2,2020);
INSERT INTO "Vineyards_Wines" VALUES(2,2,2021);
INSERT INTO "Vineyards_Wines" VALUES(3,3,2017);
INSERT INTO "Vineyards_Wines" VALUES(4,3,2016);
INSERT INTO "Vineyards_Wines" VALUES(2,1,2020);
INSERT INTO "Vineyards_Wines" VALUES(3,1,2017);
INSERT INTO "Vineyards_Wines" VALUES(4,1,2016);
INSERT INTO "Vineyards_Wines" VALUES(5,1,NULL);
CREATE TABLE Wines(
    wineId INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    variety TEXT);
INSERT INTO "Wines" VALUES(1,'Cabernet Sauvignon','Cabernet Sauvignon');
INSERT INTO "Wines" VALUES(2,'Merlot','Merlot');
INSERT INTO "Wines" VALUES(3,'Pinot Noir','Pinot Noir');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('VineyardTypes',3);
INSERT INTO "sqlite_sequence" VALUES('Countries',3);
INSERT INTO "sqlite_sequence" VALUES('Regions',3);
INSERT INTO "sqlite_sequence" VALUES('Vineyards',5);
INSERT INTO "sqlite_sequence" VALUES('Wines',3);
COMMIT;
