BEGIN TRANSACTION;
CREATE TABLE Wine_Regions(
    RegionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT);
INSERT INTO "Wine_Regions" VALUES(1,'Napa Valley','A famous wine region known for its Cabernet Sauvignon.');
INSERT INTO "Wine_Regions" VALUES(2,'Bordeaux','Famous for its red wines, primarily blends of several grape varieties.');
INSERT INTO "Wine_Regions" VALUES(3,'Champagne','Known for producing some of the world''s finest sparkling wines.');
INSERT INTO "Wine_Regions" VALUES(4,'Barossa Valley','Australia''s most renowned wine region, producing a wide variety of styles.');
INSERT INTO "Wine_Regions" VALUES(5,'Marlborough','New Zealand''s largest wine-producing region, specializing in Sauvignon Blanc.');
CREATE TABLE Wine_Regions_Winery_Tours(
    WineRegionID INTEGER,
    TourType TEXT,
    FOREIGN KEY(WineRegionID) REFERENCES Wine_Regions(RegionID),
    FOREIGN KEY(TourType) REFERENCES Winery_Tours(Type));
INSERT INTO "Wine_Regions_Winery_Tours" VALUES(1,'Self_guided_Tours');
INSERT INTO "Wine_Regions_Winery_Tours" VALUES(1,'Group_tours');
INSERT INTO "Wine_Regions_Winery_Tours" VALUES(2,'Self_guided_Tours');
INSERT INTO "Wine_Regions_Winery_Tours" VALUES(2,'Group_tours');
INSERT INTO "Wine_Regions_Winery_Tours" VALUES(3,'Self_guided_Tours');
INSERT INTO "Wine_Regions_Winery_Tours" VALUES(3,'Group_tours');
INSERT INTO "Wine_Regions_Winery_Tours" VALUES(4,'Self_guided_Tours');
INSERT INTO "Wine_Regions_Winery_Tours" VALUES(4,'Group_tours');
INSERT INTO "Wine_Regions_Winery_Tours" VALUES(5,'Self_guided_Tours');
INSERT INTO "Wine_Regions_Winery_Tours" VALUES(5,'Group_tours');
CREATE TABLE Winery_Tours(
    TourID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT,
    Duration REAL );
INSERT INTO "Winery_Tours" VALUES(1,'Self_guided_Tours',2.0);
INSERT INTO "Winery_Tours" VALUES(2,'Group_tours',3.0);
INSERT INTO "Winery_Tours" VALUES(3,'VIP_tours',4.0);
INSERT INTO "Winery_Tours" VALUES(4,'Custom_tours',5.0);
INSERT INTO "Winery_Tours" VALUES(5,'Educational_tours',6.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Wine_Regions',5);
INSERT INTO "sqlite_sequence" VALUES('Winery_Tours',5);
COMMIT;
