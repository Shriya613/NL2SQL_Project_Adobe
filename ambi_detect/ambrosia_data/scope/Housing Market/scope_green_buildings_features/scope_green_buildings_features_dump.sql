BEGIN TRANSACTION;
CREATE TABLE Features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255)
);
INSERT INTO "Features" VALUES(1,'Solar Panels');
INSERT INTO "Features" VALUES(2,'Energy Efficient Windows');
INSERT INTO "Features" VALUES(3,'Rainwater Harvesting System');
INSERT INTO "Features" VALUES(4,'Garden Roof');
INSERT INTO "Features" VALUES(5,'Geothermal Heating/Cooling');
CREATE TABLE Green_Buildings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT,
    price REAL
);
INSERT INTO "Green_Buildings" VALUES(1,'123 Main St',300000.0);
INSERT INTO "Green_Buildings" VALUES(2,'456 Elm St',400000.0);
INSERT INTO "Green_Buildings" VALUES(3,'789 Oak St',500000.0);
INSERT INTO "Green_Buildings" VALUES(4,'101 Maple Ave',600000.0);
INSERT INTO "Green_Buildings" VALUES(5,'222 Pine Blvd',700000.0);
CREATE TABLE Green_Buildings_Features (
    green_building_id INTEGER,
    feature_id INTEGER,
    FOREIGN KEY (green_building_id) REFERENCES Green_Buildings (id),
    FOREIGN KEY (feature_id) REFERENCES Features (feature_id),
    PRIMARY KEY (green_building_id, feature_id)
);
INSERT INTO "Green_Buildings_Features" VALUES(1,1);
INSERT INTO "Green_Buildings_Features" VALUES(1,2);
INSERT INTO "Green_Buildings_Features" VALUES(2,3);
INSERT INTO "Green_Buildings_Features" VALUES(2,4);
INSERT INTO "Green_Buildings_Features" VALUES(3,5);
INSERT INTO "Green_Buildings_Features" VALUES(3,1);
INSERT INTO "Green_Buildings_Features" VALUES(4,2);
INSERT INTO "Green_Buildings_Features" VALUES(4,3);
INSERT INTO "Green_Buildings_Features" VALUES(5,4);
INSERT INTO "Green_Buildings_Features" VALUES(5,5);
INSERT INTO "Green_Buildings_Features" VALUES(2,1);
INSERT INTO "Green_Buildings_Features" VALUES(4,1);
INSERT INTO "Green_Buildings_Features" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Features',5);
INSERT INTO "sqlite_sequence" VALUES('Green_Buildings',5);
COMMIT;
