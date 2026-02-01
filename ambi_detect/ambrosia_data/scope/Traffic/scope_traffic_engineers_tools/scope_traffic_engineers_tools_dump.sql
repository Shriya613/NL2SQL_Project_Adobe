BEGIN TRANSACTION;
CREATE TABLE `Tools`(
    tool_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT,
    type TEXT);
INSERT INTO "Tools" VALUES(1,'Software_Modeling','software');
INSERT INTO "Tools" VALUES(2,'Road_Simulation','hardware');
INSERT INTO "Tools" VALUES(3,'Vehicle_Tracking','hardware');
INSERT INTO "Tools" VALUES(4,'Accident_Analysis','software');
INSERT INTO "Tools" VALUES(5,'Signal_Optimization','software');
CREATE TABLE `Traffic_Engineers`(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    years_of_experience INT);
INSERT INTO "Traffic_Engineers" VALUES(1,'John',20);
INSERT INTO "Traffic_Engineers" VALUES(2,'Anna',30);
INSERT INTO "Traffic_Engineers" VALUES(3,'James',25);
INSERT INTO "Traffic_Engineers" VALUES(4,'Emma',28);
INSERT INTO "Traffic_Engineers" VALUES(5,'Michael',27);
CREATE TABLE `Traffic_Engineers_Tools`(
    engineer_id INTEGER,
    tool_id INTEGER,
    FOREIGN KEY (engineer_id) REFERENCES `Traffic_Engineers`(id),
    FOREIGN KEY (tool_id) REFERENCES `Tools`(tool_id));
INSERT INTO "Traffic_Engineers_Tools" VALUES(1,1);
INSERT INTO "Traffic_Engineers_Tools" VALUES(1,2);
INSERT INTO "Traffic_Engineers_Tools" VALUES(2,1);
INSERT INTO "Traffic_Engineers_Tools" VALUES(2,3);
INSERT INTO "Traffic_Engineers_Tools" VALUES(3,1);
INSERT INTO "Traffic_Engineers_Tools" VALUES(3,2);
INSERT INTO "Traffic_Engineers_Tools" VALUES(4,2);
INSERT INTO "Traffic_Engineers_Tools" VALUES(4,5);
INSERT INTO "Traffic_Engineers_Tools" VALUES(5,3);
INSERT INTO "Traffic_Engineers_Tools" VALUES(5,4);
INSERT INTO "Traffic_Engineers_Tools" VALUES(4,1);
INSERT INTO "Traffic_Engineers_Tools" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Traffic_Engineers',5);
INSERT INTO "sqlite_sequence" VALUES('Tools',5);
COMMIT;
