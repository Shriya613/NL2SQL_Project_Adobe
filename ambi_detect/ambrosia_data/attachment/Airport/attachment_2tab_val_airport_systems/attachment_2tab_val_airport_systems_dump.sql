BEGIN TRANSACTION;
CREATE TABLE Communication_Systems(
    SystemID INTEGER PRIMARY KEY,
    Airport TEXT,
    TypeOfSystem TEXT,
    FrequencyInUse BOOLEAN ,
    LastUpgradeYear INTEGER);
INSERT INTO "Communication_Systems" VALUES(1,'Heathrow','VHF Omnidirectional Range',1,2016);
INSERT INTO "Communication_Systems" VALUES(2,'JFK','Distance Measuring Equipment',0,2019);
CREATE TABLE Navigation_Systems(
    SystemID INTEGER PRIMARY KEY,
    Airport TEXT,
    TypeOfSystem TEXT,
    YearInstalled INTEGER,
    MaintenanceFrequency TEXT);
INSERT INTO "Navigation_Systems" VALUES(1,'Heathrow','VOR/DME',2008,'Annually');
INSERT INTO "Navigation_Systems" VALUES(2,'JFK','ILS',2013,'Biannual');
CREATE TABLE Runways(
    ID INTEGER PRIMARY KEY,
    Airport TEXT,
    Length REAL,
    Width REAL,
    SurfaceType TEXT);
INSERT INTO "Runways" VALUES(1,'Heathrow',3200.0,45.0,'Asphalt');
INSERT INTO "Runways" VALUES(2,'JFK',10000.0,150.0,'Concrete');
CREATE TABLE Terminals(
    ID INTEGER PRIMARY KEY,
    Airport TEXT,
    Name TEXT,
    NumberOfGates INTEGER,
    ServicesOffered TEXT);
INSERT INTO "Terminals" VALUES(1,'Heathrow','Terminal 5',27,'Shopping, Dining, Baggage Claim');
INSERT INTO "Terminals" VALUES(2,'JFK','Terminal 1',30,'Shopping, Dining, Baggage Claim');
COMMIT;
