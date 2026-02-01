BEGIN TRANSACTION;
CREATE TABLE Communities(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Communities" VALUES(1,'Sunnyvale','California');
INSERT INTO "Communities" VALUES(2,'Silicon Valley','California');
INSERT INTO "Communities" VALUES(3,'Tech City','Texas');
INSERT INTO "Communities" VALUES(4,'Software Park','New York');
INSERT INTO "Communities" VALUES(5,'Innovation Hub','Colorado');
CREATE TABLE Communities_Features(
    community_id INTEGER REFERENCES Communities(id) ON DELETE CASCADE,
    feature_id INTEGER REFERENCES Features(id));
INSERT INTO "Communities_Features" VALUES(1,1);
INSERT INTO "Communities_Features" VALUES(2,1);
INSERT INTO "Communities_Features" VALUES(3,1);
INSERT INTO "Communities_Features" VALUES(4,1);
INSERT INTO "Communities_Features" VALUES(5,1);
INSERT INTO "Communities_Features" VALUES(5,2);
CREATE TABLE Features(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT);
INSERT INTO "Features" VALUES(1,'Golf Course');
INSERT INTO "Features" VALUES(2,'Park');
INSERT INTO "Features" VALUES(3,'School');
INSERT INTO "Features" VALUES(4,'Library');
INSERT INTO "Features" VALUES(5,'Hospital');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Communities',5);
INSERT INTO "sqlite_sequence" VALUES('Features',5);
COMMIT;
