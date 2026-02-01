BEGIN TRANSACTION;
CREATE TABLE Equipment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT);
INSERT INTO "Equipment" VALUES(1,'Microscope');
INSERT INTO "Equipment" VALUES(2,'Laser Pointer');
INSERT INTO "Equipment" VALUES(3,'Projector');
INSERT INTO "Equipment" VALUES(4,'Smart Board');
INSERT INTO "Equipment" VALUES(5,'Headphones');
CREATE TABLE Labs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Labs" VALUES(1,'Biology Lab','Room 203');
INSERT INTO "Labs" VALUES(2,'Chemistry Lab','Room 206');
INSERT INTO "Labs" VALUES(3,'Physics Lab','Room 212');
INSERT INTO "Labs" VALUES(4,'Computer Lab','Room 307');
INSERT INTO "Labs" VALUES(5,'Art Lab','Room 314');
CREATE TABLE Labs_Equipment (
    labs_id INTEGER,
    equipment_id INTEGER,
    FOREIGN KEY(labs_id) REFERENCES Labs(id),
    FOREIGN KEY(equipment_id) REFERENCES Equipment(id));
INSERT INTO "Labs_Equipment" VALUES(1,3);
INSERT INTO "Labs_Equipment" VALUES(2,3);
INSERT INTO "Labs_Equipment" VALUES(3,3);
INSERT INTO "Labs_Equipment" VALUES(4,3);
INSERT INTO "Labs_Equipment" VALUES(5,3);
INSERT INTO "Labs_Equipment" VALUES(5,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Labs',5);
INSERT INTO "sqlite_sequence" VALUES('Equipment',5);
COMMIT;
