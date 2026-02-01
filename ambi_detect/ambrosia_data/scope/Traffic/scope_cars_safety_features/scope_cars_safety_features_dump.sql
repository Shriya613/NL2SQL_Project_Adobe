BEGIN TRANSACTION;
CREATE TABLE Cars(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    make TEXT,
    model TEXT);
INSERT INTO "Cars" VALUES(1,'Toyota','Corolla');
INSERT INTO "Cars" VALUES(2,'Honda','Accord');
INSERT INTO "Cars" VALUES(3,'Ford','Mustang');
INSERT INTO "Cars" VALUES(4,'Chevrolet','Camaro');
INSERT INTO "Cars" VALUES(5,'Dodge','Challenger');
CREATE TABLE Cars_Safety_Features(
    carId INTEGER,
    featureId INTEGER,
    FOREIGN KEY (carId) REFERENCES Cars(id),
    FOREIGN KEY (featureId) REFERENCES Safety_Features(featureId));
INSERT INTO "Cars_Safety_Features" VALUES(1,1);
INSERT INTO "Cars_Safety_Features" VALUES(2,1);
INSERT INTO "Cars_Safety_Features" VALUES(3,1);
INSERT INTO "Cars_Safety_Features" VALUES(4,1);
INSERT INTO "Cars_Safety_Features" VALUES(5,1);
INSERT INTO "Cars_Safety_Features" VALUES(1,2);
INSERT INTO "Cars_Safety_Features" VALUES(2,2);
INSERT INTO "Cars_Safety_Features" VALUES(3,2);
INSERT INTO "Cars_Safety_Features" VALUES(4,2);
INSERT INTO "Cars_Safety_Features" VALUES(5,2);
CREATE TABLE Safety_Features(
    featureId INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Safety_Features" VALUES(1,'Airbags');
INSERT INTO "Safety_Features" VALUES(2,'Anti-lock Braking System');
INSERT INTO "Safety_Features" VALUES(3,'Traction Control');
INSERT INTO "Safety_Features" VALUES(4,'Electronic Stability Control');
INSERT INTO "Safety_Features" VALUES(5,'Blind Spot Monitoring');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Cars',5);
INSERT INTO "sqlite_sequence" VALUES('Safety_Features',5);
COMMIT;
