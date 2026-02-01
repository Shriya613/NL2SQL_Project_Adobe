BEGIN TRANSACTION;
CREATE TABLE Communities(
    community_id INTEGER PRIMARY KEY AUTOINCREMENT,
    community_name TEXT,
    location TEXT);
INSERT INTO "Communities" VALUES(1,'Green Valley','California');
INSERT INTO "Communities" VALUES(2,'Sunny Hills','Florida');
INSERT INTO "Communities" VALUES(3,'Lakeview Heights','Texas');
INSERT INTO "Communities" VALUES(4,'Riverbend','New York');
INSERT INTO "Communities" VALUES(5,'Mountain View','Colorado');
CREATE TABLE Communities_Features(
    cf_id INTEGER PRIMARY KEY AUTOINCREMENT,
    community_id INTEGER,
    feature_id INTEGER,
    detail_description TEXT,
    FOREIGN KEY(community_id) REFERENCES Communities(community_id),
    FOREIGN KEY(feature_id) REFERENCES Features(feature_id));
INSERT INTO "Communities_Features" VALUES(1,1,1,'The Golf Course is located within walking distance.');
INSERT INTO "Communities_Features" VALUES(2,1,2,'There is a beautiful park near the community.');
INSERT INTO "Communities_Features" VALUES(3,2,1,'A golf course can also be seen from this community.');
INSERT INTO "Communities_Features" VALUES(4,2,3,'Nearby schools provide excellent education options.');
INSERT INTO "Communities_Features" VALUES(5,3,2,'This community has its own playground in the park.');
INSERT INTO "Communities_Features" VALUES(6,3,4,'An emergency hospital is just around the corner.');
INSERT INTO "Communities_Features" VALUES(7,4,5,'The shopping center offers various shops and restaurants.');
INSERT INTO "Communities_Features" VALUES(8,4,1,'A stunning golf course lies adjacent to the community.');
INSERT INTO "Communities_Features" VALUES(9,5,3,'A school bus stop is conveniently located here.');
INSERT INTO "Communities_Features" VALUES(10,5,4,'The local hospital provides quick medical assistance.');
INSERT INTO "Communities_Features" VALUES(11,3,1,'This community has its own playground in the park.');
INSERT INTO "Communities_Features" VALUES(12,5,1,'A school bus stop is conveniently located here.');
CREATE TABLE Features(
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_value TEXT);
INSERT INTO "Features" VALUES(1,'Golf Course');
INSERT INTO "Features" VALUES(2,'Park');
INSERT INTO "Features" VALUES(3,'School');
INSERT INTO "Features" VALUES(4,'Hospital');
INSERT INTO "Features" VALUES(5,'Shopping Center');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Features',5);
INSERT INTO "sqlite_sequence" VALUES('Communities',5);
INSERT INTO "sqlite_sequence" VALUES('Communities_Features',12);
COMMIT;
