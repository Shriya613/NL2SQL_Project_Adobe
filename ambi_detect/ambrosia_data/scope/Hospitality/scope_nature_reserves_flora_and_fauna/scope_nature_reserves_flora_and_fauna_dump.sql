BEGIN TRANSACTION;
CREATE TABLE Flora_and_fauna(
    FaunaID INTEGER PRIMARY KEY AUTOINCREMENT,
    SpeciesName TEXT);
INSERT INTO "Flora_and_fauna" VALUES(1,'Deer');
INSERT INTO "Flora_and_fauna" VALUES(2,'Elk');
INSERT INTO "Flora_and_fauna" VALUES(3,'Bison');
INSERT INTO "Flora_and_fauna" VALUES(4,'Coyote');
INSERT INTO "Flora_and_fauna" VALUES(5,'Raccoon');
CREATE TABLE Nature_Reserves(
    ReserveID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT);
INSERT INTO "Nature_Reserves" VALUES(1,'Yellowstone','Wyoming');
INSERT INTO "Nature_Reserves" VALUES(2,'Grand Canyon','Arizona');
INSERT INTO "Nature_Reserves" VALUES(3,'Yosemite','California');
INSERT INTO "Nature_Reserves" VALUES(4,'Zion National Park','Utah');
INSERT INTO "Nature_Reserves" VALUES(5,'Great Smoky Mountains','Tennessee');
CREATE TABLE Nature_reserves_flora_and_fauna(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    NatureReserveId INTEGER,
    FaunaId INTEGER,
    FOREIGN KEY (NatureReserveId) REFERENCES Nature_Reserves(ReserveID),
    FOREIGN KEY (FaunaId) REFERENCES Flora_and_fauna(FaunaID));
INSERT INTO "Nature_reserves_flora_and_fauna" VALUES(1,1,1);
INSERT INTO "Nature_reserves_flora_and_fauna" VALUES(2,2,1);
INSERT INTO "Nature_reserves_flora_and_fauna" VALUES(3,3,1);
INSERT INTO "Nature_reserves_flora_and_fauna" VALUES(4,4,1);
INSERT INTO "Nature_reserves_flora_and_fauna" VALUES(5,5,1);
INSERT INTO "Nature_reserves_flora_and_fauna" VALUES(6,5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Nature_Reserves',5);
INSERT INTO "sqlite_sequence" VALUES('Flora_and_fauna',5);
INSERT INTO "sqlite_sequence" VALUES('Nature_reserves_flora_and_fauna',6);
COMMIT;
