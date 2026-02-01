BEGIN TRANSACTION;
CREATE TABLE Restrictions(
    restrictionType TEXT,
    value REAL
);
INSERT INTO "Restrictions" VALUES('Weight Limit',7.5);
INSERT INTO "Restrictions" VALUES('Speed Limit',60.0);
INSERT INTO "Restrictions" VALUES('Height Clearance',4.0);
INSERT INTO "Restrictions" VALUES('Width Clearance',8.0);
INSERT INTO "Restrictions" VALUES('Length Clearance',15.0);
CREATE TABLE Routes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    routeName TEXT,
    distance REAL
);
INSERT INTO "Routes" VALUES(1,'Route1',100.0);
INSERT INTO "Routes" VALUES(2,'Route2',200.0);
INSERT INTO "Routes" VALUES(3,'Route3',300.0);
INSERT INTO "Routes" VALUES(4,'Route4',400.0);
INSERT INTO "Routes" VALUES(5,'Route5',500.0);
CREATE TABLE Routes_Restrictions(
    routesId INTEGER,
    restrictionsType TEXT,
    FOREIGN KEY(routesId) REFERENCES Routes(id) ON DELETE CASCADE,
    PRIMARY KEY(routesId, restrictionsType)
);
INSERT INTO "Routes_Restrictions" VALUES(1,'Weight Limit');
INSERT INTO "Routes_Restrictions" VALUES(1,'Speed Limit');
INSERT INTO "Routes_Restrictions" VALUES(2,'Height Clearance');
INSERT INTO "Routes_Restrictions" VALUES(2,'Width Clearance');
INSERT INTO "Routes_Restrictions" VALUES(3,'Length Clearance');
INSERT INTO "Routes_Restrictions" VALUES(2,'Speed Limit');
INSERT INTO "Routes_Restrictions" VALUES(3,'Speed Limit');
INSERT INTO "Routes_Restrictions" VALUES(4,'Speed Limit');
INSERT INTO "Routes_Restrictions" VALUES(5,'Speed Limit');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Routes',5);
COMMIT;
