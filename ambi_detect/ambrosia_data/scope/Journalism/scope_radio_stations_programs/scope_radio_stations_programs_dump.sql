BEGIN TRANSACTION;
CREATE TABLE Programs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    programType TEXT
);
INSERT INTO "Programs" VALUES(1,'Talk Show');
INSERT INTO "Programs" VALUES(2,'News Update');
INSERT INTO "Programs" VALUES(3,'Music Hour');
INSERT INTO "Programs" VALUES(4,'Interview Series');
INSERT INTO "Programs" VALUES(5,'Podcast');
CREATE TABLE Radio_Stations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stationName TEXT,
    location TEXT
);
INSERT INTO "Radio_Stations" VALUES(1,'ABC Radio','New York');
INSERT INTO "Radio_Stations" VALUES(2,'Def Radio','Los Angeles');
INSERT INTO "Radio_Stations" VALUES(3,'GHI Radio','Chicago');
INSERT INTO "Radio_Stations" VALUES(4,'JKL Radio','Boston');
INSERT INTO "Radio_Stations" VALUES(5,'MNO Radio','Dallas');
CREATE TABLE Radio_Stations_Programs(
    radioStationId INTEGER,
    programId INTEGER,
    FOREIGN KEY(radioStationId) REFERENCES Radio_Stations(id),
    FOREIGN KEY(programId) REFERENCES Programs(id)
);
INSERT INTO "Radio_Stations_Programs" VALUES(1,1);
INSERT INTO "Radio_Stations_Programs" VALUES(2,2);
INSERT INTO "Radio_Stations_Programs" VALUES(3,3);
INSERT INTO "Radio_Stations_Programs" VALUES(4,4);
INSERT INTO "Radio_Stations_Programs" VALUES(5,5);
INSERT INTO "Radio_Stations_Programs" VALUES(2,1);
INSERT INTO "Radio_Stations_Programs" VALUES(3,1);
INSERT INTO "Radio_Stations_Programs" VALUES(4,1);
INSERT INTO "Radio_Stations_Programs" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Radio_Stations',5);
INSERT INTO "sqlite_sequence" VALUES('Programs',5);
COMMIT;
