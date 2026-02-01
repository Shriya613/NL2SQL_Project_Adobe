BEGIN TRANSACTION;
CREATE TABLE Platforms(
    PlatformID INTEGER PRIMARY KEY AUTOINCREMENT,
    "Name" TEXT);
INSERT INTO "Platforms" VALUES(1,'Platform_1');
INSERT INTO "Platforms" VALUES(2,'Platform_2');
INSERT INTO "Platforms" VALUES(3,'Platform_3');
INSERT INTO "Platforms" VALUES(4,'Platform_4');
INSERT INTO "Platforms" VALUES(5,'Platform_5');
CREATE TABLE Train_Stations(
    StationID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT);
INSERT INTO "Train_Stations" VALUES(1,'Central','Downtown');
INSERT INTO "Train_Stations" VALUES(2,'Parkville','East Side');
INSERT INTO "Train_Stations" VALUES(3,'Washington','West Side');
INSERT INTO "Train_Stations" VALUES(4,'Lakeview','North Side');
INSERT INTO "Train_Stations" VALUES(5,'Riverfront','South Side');
CREATE TABLE Train_Stations_Platforms(
    StationID INTEGER,
    PlatformID INTEGER,
    FOREIGN KEY(StationID) REFERENCES Train_Stations(StationID),
    FOREIGN KEY(PlatformID) REFERENCES Platforms(PlatformID));
INSERT INTO "Train_Stations_Platforms" VALUES(1,1);
INSERT INTO "Train_Stations_Platforms" VALUES(1,2);
INSERT INTO "Train_Stations_Platforms" VALUES(2,1);
INSERT INTO "Train_Stations_Platforms" VALUES(2,2);
INSERT INTO "Train_Stations_Platforms" VALUES(3,1);
INSERT INTO "Train_Stations_Platforms" VALUES(3,3);
INSERT INTO "Train_Stations_Platforms" VALUES(4,4);
INSERT INTO "Train_Stations_Platforms" VALUES(4,5);
INSERT INTO "Train_Stations_Platforms" VALUES(5,1);
INSERT INTO "Train_Stations_Platforms" VALUES(5,3);
INSERT INTO "Train_Stations_Platforms" VALUES(4,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Train_Stations',5);
INSERT INTO "sqlite_sequence" VALUES('Platforms',5);
COMMIT;
