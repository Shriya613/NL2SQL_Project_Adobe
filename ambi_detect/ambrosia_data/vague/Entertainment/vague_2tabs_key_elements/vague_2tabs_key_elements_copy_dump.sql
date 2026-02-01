BEGIN TRANSACTION;
CREATE TABLE "Fashion_Design" (
    design_id INTEGER PRIMARY KEY AUTOINCREMENT,
    design_name TEXT,
    color_id INTEGER,
    FOREIGN KEY(color_id) REFERENCES "Key_Elements"(color_id));
INSERT INTO "Fashion_Design" VALUES(1,'Classic Black Suit',1);
INSERT INTO "Fashion_Design" VALUES(2,'Denim Jacket',2);
INSERT INTO "Fashion_Design" VALUES(3,'Knitted Sweater',3);
INSERT INTO "Fashion_Design" VALUES(4,'Summer Dress',4);
INSERT INTO "Fashion_Design" VALUES(5,'Party Gown',5);
CREATE TABLE "Key_Elements" (
    color_id INTEGER PRIMARY KEY AUTOINCREMENT,
    color_name TEXT UNIQUE
);
INSERT INTO "Key_Elements" VALUES(1,'Red');
INSERT INTO "Key_Elements" VALUES(2,'Blue');
INSERT INTO "Key_Elements" VALUES(3,'Green');
INSERT INTO "Key_Elements" VALUES(4,'Yellow');
INSERT INTO "Key_Elements" VALUES(5,'White');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Key_Elements',5);
INSERT INTO "sqlite_sequence" VALUES('Fashion_Design',5);
COMMIT;
