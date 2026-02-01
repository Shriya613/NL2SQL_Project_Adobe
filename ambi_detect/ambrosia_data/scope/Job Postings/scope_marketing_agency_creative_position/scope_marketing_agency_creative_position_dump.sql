BEGIN TRANSACTION;
CREATE TABLE Creative_Position (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position TEXT);
INSERT INTO "Creative_Position" VALUES(1,'Graphic Designer');
INSERT INTO "Creative_Position" VALUES(2,'Copywriter');
INSERT INTO "Creative_Position" VALUES(3,'Photographer');
INSERT INTO "Creative_Position" VALUES(4,'Web Developer');
INSERT INTO "Creative_Position" VALUES(5,'Content Creator');
CREATE TABLE Marketing_Agency (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Marketing_Agency" VALUES(1,'Advertising Agency A','New York');
INSERT INTO "Marketing_Agency" VALUES(2,'Digital Media Company B','Los Angeles');
INSERT INTO "Marketing_Agency" VALUES(3,'PR Firm C','Chicago');
INSERT INTO "Marketing_Agency" VALUES(4,'Media Consultancy D','Boston');
INSERT INTO "Marketing_Agency" VALUES(5,'Branding Agency E','Dallas');
CREATE TABLE Marketing_Agency_Creative_Positions (
    agency_id INTEGER,
    position_id INTEGER,
    FOREIGN KEY(agency_id) REFERENCES Marketing_Agency(id),
    FOREIGN KEY(position_id) REFERENCES Creative_Position(id));
INSERT INTO "Marketing_Agency_Creative_Positions" VALUES(1,1);
INSERT INTO "Marketing_Agency_Creative_Positions" VALUES(2,1);
INSERT INTO "Marketing_Agency_Creative_Positions" VALUES(3,1);
INSERT INTO "Marketing_Agency_Creative_Positions" VALUES(4,1);
INSERT INTO "Marketing_Agency_Creative_Positions" VALUES(5,1);
INSERT INTO "Marketing_Agency_Creative_Positions" VALUES(5,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Marketing_Agency',5);
INSERT INTO "sqlite_sequence" VALUES('Creative_Position',5);
COMMIT;
