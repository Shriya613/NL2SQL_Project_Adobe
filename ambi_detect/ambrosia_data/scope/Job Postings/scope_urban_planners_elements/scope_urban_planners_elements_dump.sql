BEGIN TRANSACTION;
CREATE TABLE Elements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    element_name TEXT
);
INSERT INTO "Elements" VALUES(1,'Green_Space');
INSERT INTO "Elements" VALUES(2,'Transportation_Infrastructure');
INSERT INTO "Elements" VALUES(3,'Landscape_Architecture');
INSERT INTO "Elements" VALUES(4,'Water_Management');
INSERT INTO "Elements" VALUES(5,'Public_Spaces');
CREATE TABLE Urban_Planners (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    experience INTEGER
);
INSERT INTO "Urban_Planners" VALUES(1,'John Doe',3);
INSERT INTO "Urban_Planners" VALUES(2,'Jane Smith',2);
INSERT INTO "Urban_Planners" VALUES(3,'Michael Johnson',5);
INSERT INTO "Urban_Planners" VALUES(4,'Emily Williams',10);
INSERT INTO "Urban_Planners" VALUES(5,'David Brown',8);
CREATE TABLE Urban_Planners_Elements (
    urban_planner_id INTEGER,
    element_id INTEGER,
    FOREIGN KEY(urban_planner_id) REFERENCES Urban_Planners(id),
    FOREIGN KEY(element_id) REFERENCES Elements(id),
    PRIMARY KEY(urban_planner_id, element_id)
);
INSERT INTO "Urban_Planners_Elements" VALUES(1,1);
INSERT INTO "Urban_Planners_Elements" VALUES(2,1);
INSERT INTO "Urban_Planners_Elements" VALUES(3,1);
INSERT INTO "Urban_Planners_Elements" VALUES(4,1);
INSERT INTO "Urban_Planners_Elements" VALUES(5,1);
INSERT INTO "Urban_Planners_Elements" VALUES(5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Urban_Planners',5);
INSERT INTO "sqlite_sequence" VALUES('Elements',5);
COMMIT;
