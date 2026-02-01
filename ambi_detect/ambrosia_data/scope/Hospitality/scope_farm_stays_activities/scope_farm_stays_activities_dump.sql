BEGIN TRANSACTION;
CREATE TABLE Activities (
    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT
);
INSERT INTO "Activities" VALUES(1,'Milking cows');
INSERT INTO "Activities" VALUES(2,'Feeding animals');
INSERT INTO "Activities" VALUES(3,'Egg collection');
INSERT INTO "Activities" VALUES(4,'Baking bread from freshly harvested grain');
INSERT INTO "Activities" VALUES(5,'Harvesting fruits');
CREATE TABLE Farm_Stays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Farm_Stays" VALUES(1,'Green Valley','Rural Area');
INSERT INTO "Farm_Stays" VALUES(2,'Sunrise Farms','Mountainous Region');
INSERT INTO "Farm_Stays" VALUES(3,'Oak Tree Lodge','Urban Suburbia');
INSERT INTO "Farm_Stays" VALUES(4,'Country Retreat','Vineyard Estate');
INSERT INTO "Farm_Stays" VALUES(5,'Cozy Cottage','Lake Side');
CREATE TABLE Farm_Stays_Activities (
    farm_stay_id INTEGER,
    activity_id INTEGER,
    FOREIGN KEY(farm_stay_id) REFERENCES Farm_Stays(id),
    FOREIGN KEY(activity_id) REFERENCES Activities(activity_id),
    PRIMARY KEY(farm_stay_id, activity_id)
);
INSERT INTO "Farm_Stays_Activities" VALUES(1,1);
INSERT INTO "Farm_Stays_Activities" VALUES(2,1);
INSERT INTO "Farm_Stays_Activities" VALUES(3,1);
INSERT INTO "Farm_Stays_Activities" VALUES(4,1);
INSERT INTO "Farm_Stays_Activities" VALUES(5,1);
INSERT INTO "Farm_Stays_Activities" VALUES(5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Farm_Stays',5);
INSERT INTO "sqlite_sequence" VALUES('Activities',5);
COMMIT;
