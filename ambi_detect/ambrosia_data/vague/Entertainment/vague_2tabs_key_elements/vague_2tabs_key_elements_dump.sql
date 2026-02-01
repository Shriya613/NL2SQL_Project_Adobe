BEGIN TRANSACTION;
CREATE TABLE Color_Palette (
    color_id INTEGER PRIMARY KEY AUTOINCREMENT,
    color_name TEXT UNIQUE
);
INSERT INTO "Color_Palette" VALUES(1,'Red');
INSERT INTO "Color_Palette" VALUES(2,'Blue');
INSERT INTO "Color_Palette" VALUES(3,'Green');
INSERT INTO "Color_Palette" VALUES(4,'Yellow');
INSERT INTO "Color_Palette" VALUES(5,'White');
CREATE TABLE Fabric_Choice (
    fabric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fabric_type TEXT UNIQUE
);
INSERT INTO "Fabric_Choice" VALUES(1,'Silk');
INSERT INTO "Fabric_Choice" VALUES(2,'Cotton');
INSERT INTO "Fabric_Choice" VALUES(3,'Wool');
INSERT INTO "Fabric_Choice" VALUES(4,'Linen');
INSERT INTO "Fabric_Choice" VALUES(5,'Polyester');
CREATE TABLE Fashion_Design (
    design_id INTEGER PRIMARY KEY AUTOINCREMENT,
    design_name TEXT,
    color_id INTEGER,
    fabric_id INTEGER,
    FOREIGN KEY(color_id) REFERENCES Color_Palette(color_id),
    FOREIGN KEY(fabric_id) REFERENCES Fabric_Choice(fabric_id)
);
INSERT INTO "Fashion_Design" VALUES(1,'Classic Black Suit',1,1);
INSERT INTO "Fashion_Design" VALUES(2,'Denim Jacket',2,3);
INSERT INTO "Fashion_Design" VALUES(3,'Knitted Sweater',3,4);
INSERT INTO "Fashion_Design" VALUES(4,'Summer Dress',4,5);
INSERT INTO "Fashion_Design" VALUES(5,'Party Gown',5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Color_Palette',5);
INSERT INTO "sqlite_sequence" VALUES('Fabric_Choice',5);
INSERT INTO "sqlite_sequence" VALUES('Fashion_Design',5);
COMMIT;
