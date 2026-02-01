BEGIN TRANSACTION;
CREATE TABLE Features(
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT);
INSERT INTO "Features" VALUES(1,'Movies');
INSERT INTO "Features" VALUES(2,'Music');
INSERT INTO "Features" VALUES(3,'Games');
INSERT INTO "Features" VALUES(4,'WiFi');
INSERT INTO "Features" VALUES(5,'News');
CREATE TABLE Onboard_Entertainment_System(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    airline TEXT,
    aircraft_model TEXT);
INSERT INTO "Onboard_Entertainment_System" VALUES(1,'Airline1','Model1');
INSERT INTO "Onboard_Entertainment_System" VALUES(2,'Airline2','Model2');
INSERT INTO "Onboard_Entertainment_System" VALUES(3,'Airline3','Model3');
INSERT INTO "Onboard_Entertainment_System" VALUES(4,'Airline4','Model4');
INSERT INTO "Onboard_Entertainment_System" VALUES(5,'Airline5','Model5');
CREATE TABLE Onboard_Entertainment_System_Features(
    onboard_entertainment_system_id INTEGER,
    features_feature_id INTEGER,
    FOREIGN KEY(onboard_entertainment_system_id) REFERENCES Onboard_Entertainment_System(id),
    FOREIGN KEY(features_feature_id) REFERENCES Features(feature_id));
INSERT INTO "Onboard_Entertainment_System_Features" VALUES(1,1);
INSERT INTO "Onboard_Entertainment_System_Features" VALUES(2,1);
INSERT INTO "Onboard_Entertainment_System_Features" VALUES(3,1);
INSERT INTO "Onboard_Entertainment_System_Features" VALUES(4,1);
INSERT INTO "Onboard_Entertainment_System_Features" VALUES(5,1);
INSERT INTO "Onboard_Entertainment_System_Features" VALUES(5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Onboard_Entertainment_System',5);
INSERT INTO "sqlite_sequence" VALUES('Features',5);
COMMIT;
