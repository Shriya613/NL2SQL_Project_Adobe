BEGIN TRANSACTION;
CREATE TABLE Agricultural_Research_Institutes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Agricultural_Research_Institutes" VALUES(1,'Institute A','Location A');
INSERT INTO "Agricultural_Research_Institutes" VALUES(2,'Institute B','Location B');
INSERT INTO "Agricultural_Research_Institutes" VALUES(3,'Institute C','Location C');
INSERT INTO "Agricultural_Research_Institutes" VALUES(4,'Institute D','Location D');
INSERT INTO "Agricultural_Research_Institutes" VALUES(5,'Institute E','Location E');
CREATE TABLE Agricultural_Research_Institutes_Focus_Areas (
    institute_id INTEGER,
    focus_area_id INTEGER,
    FOREIGN KEY(institute_id) REFERENCES Agricultural_Research_Institutes(id),
    FOREIGN KEY(focus_area_id) REFERENCES Focus_Areas(id),
    PRIMARY KEY(institute_id, focus_area_id)
);
INSERT INTO "Agricultural_Research_Institutes_Focus_Areas" VALUES(1,1);
INSERT INTO "Agricultural_Research_Institutes_Focus_Areas" VALUES(2,1);
INSERT INTO "Agricultural_Research_Institutes_Focus_Areas" VALUES(3,1);
INSERT INTO "Agricultural_Research_Institutes_Focus_Areas" VALUES(4,1);
INSERT INTO "Agricultural_Research_Institutes_Focus_Areas" VALUES(5,1);
INSERT INTO "Agricultural_Research_Institutes_Focus_Areas" VALUES(5,3);
CREATE TABLE Focus_Areas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    area_name TEXT
);
INSERT INTO "Focus_Areas" VALUES(1,'Sustainability Research');
INSERT INTO "Focus_Areas" VALUES(2,'Soil Management');
INSERT INTO "Focus_Areas" VALUES(3,'Pest Control');
INSERT INTO "Focus_Areas" VALUES(4,'Crop Rotation');
INSERT INTO "Focus_Areas" VALUES(5,'Fertilization Techniques');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Agricultural_Research_Institutes',5);
INSERT INTO "sqlite_sequence" VALUES('Focus_Areas',5);
COMMIT;
