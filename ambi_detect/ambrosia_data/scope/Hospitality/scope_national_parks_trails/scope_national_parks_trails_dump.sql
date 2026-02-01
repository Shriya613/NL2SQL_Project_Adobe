BEGIN TRANSACTION;
CREATE TABLE National_Parks (
    park_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "National_Parks" VALUES(1,'Yellowstone','Wyoming');
INSERT INTO "National_Parks" VALUES(2,'Grand Canyon','Arizona');
INSERT INTO "National_Parks" VALUES(3,'Rocky Mountain','Colorado');
INSERT INTO "National_Parks" VALUES(4,'Everglades','Florida');
INSERT INTO "National_Parks" VALUES(5,'Great Smoky Mountains','Tennessee');
CREATE TABLE National_Parks_Trails (
    national_park_id INTEGER,
    trail_id INTEGER,
    FOREIGN KEY (national_park_id) REFERENCES National_Parks(park_id),
    FOREIGN KEY (trail_id) REFERENCES Trails(trail_id),
    PRIMARY KEY (national_park_id, trail_id)
);
INSERT INTO "National_Parks_Trails" VALUES(1,1);
INSERT INTO "National_Parks_Trails" VALUES(2,1);
INSERT INTO "National_Parks_Trails" VALUES(3,1);
INSERT INTO "National_Parks_Trails" VALUES(4,1);
INSERT INTO "National_Parks_Trails" VALUES(5,1);
INSERT INTO "National_Parks_Trails" VALUES(5,2);
CREATE TABLE Trails (
    trail_id INTEGER PRIMARY KEY AUTOINCREMENT,
    difficulty TEXT,
    length REAL
);
INSERT INTO "Trails" VALUES(1,'Beginner_Trail',1.0);
INSERT INTO "Trails" VALUES(2,'Moderate_Trail',3.0);
INSERT INTO "Trails" VALUES(3,'Difficult_Trail',6.0);
INSERT INTO "Trails" VALUES(4,'Expert_Trail',9.0);
INSERT INTO "Trails" VALUES(5,'Family_Friendly_Trail',2.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('National_Parks',5);
INSERT INTO "sqlite_sequence" VALUES('Trails',5);
COMMIT;
