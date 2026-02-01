BEGIN TRANSACTION;
CREATE TABLE Camera_Road_Link(
    cameraId INTEGER,
    roadId INTEGER,
    FOREIGN KEY (cameraId) REFERENCES Traffic_Camera(id),
    FOREIGN KEY (roadId) REFERENCES Road_Condition(id));
INSERT INTO "Camera_Road_Link" VALUES(1,1);
INSERT INTO "Camera_Road_Link" VALUES(2,2);
INSERT INTO "Camera_Road_Link" VALUES(3,3);
INSERT INTO "Camera_Road_Link" VALUES(4,4);
INSERT INTO "Camera_Road_Link" VALUES(5,5);
CREATE TABLE Function(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    functionName TEXT);
INSERT INTO "Function" VALUES(1,'Speeding');
INSERT INTO "Function" VALUES(2,'Accident Detection');
INSERT INTO "Function" VALUES(3,'Lane Departure Warning');
INSERT INTO "Function" VALUES(4,'Weather Monitoring');
INSERT INTO "Function" VALUES(5,'Pedestrian Detection');
CREATE TABLE Road_Condition(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roadId INTEGER,
    condition TEXT);
INSERT INTO "Road_Condition" VALUES(1,1,NULL);
INSERT INTO "Road_Condition" VALUES(2,2,NULL);
INSERT INTO "Road_Condition" VALUES(3,3,NULL);
INSERT INTO "Road_Condition" VALUES(4,4,NULL);
INSERT INTO "Road_Condition" VALUES(5,5,NULL);
CREATE TABLE Traffic_Camera(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Traffic_Camera" VALUES(1,'Cam1','Downtown');
INSERT INTO "Traffic_Camera" VALUES(2,'Cam2','Highway Entrance');
INSERT INTO "Traffic_Camera" VALUES(3,'Cam3','Highway Exit');
INSERT INTO "Traffic_Camera" VALUES(4,'Cam4','Main Street');
INSERT INTO "Traffic_Camera" VALUES(5,'Cam5','Side Street');
CREATE TABLE Traffic_Camera_Function(
    cameraId INTEGER,
    functionId INTEGER,
    FOREIGN KEY (cameraId) REFERENCES Traffic_Camera(id),
    FOREIGN KEY (functionId) REFERENCES Function(id));
INSERT INTO "Traffic_Camera_Function" VALUES(1,1);
INSERT INTO "Traffic_Camera_Function" VALUES(2,1);
INSERT INTO "Traffic_Camera_Function" VALUES(3,1);
INSERT INTO "Traffic_Camera_Function" VALUES(4,1);
INSERT INTO "Traffic_Camera_Function" VALUES(5,1);
INSERT INTO "Traffic_Camera_Function" VALUES(5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Traffic_Camera',5);
INSERT INTO "sqlite_sequence" VALUES('Function',5);
INSERT INTO "sqlite_sequence" VALUES('Road_Condition',5);
COMMIT;
