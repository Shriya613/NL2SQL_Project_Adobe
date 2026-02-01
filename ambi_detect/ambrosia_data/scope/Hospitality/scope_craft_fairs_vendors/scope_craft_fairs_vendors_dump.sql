BEGIN TRANSACTION;
CREATE TABLE Craft_Fairs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT,
    date DATE);
INSERT INTO "Craft_Fairs" VALUES(1,'Spring Fling','Central Park','2023-04-15');
INSERT INTO "Craft_Fairs" VALUES(2,'Holiday Bazaar','Downtown Marketplace','2023-12-09');
INSERT INTO "Craft_Fairs" VALUES(3,'Artisan Expo','City Hall Plaza','2023-06-23');
INSERT INTO "Craft_Fairs" VALUES(4,'Gift Fair','Convention Center','2023-11-18');
INSERT INTO "Craft_Fairs" VALUES(5,'Handmade Festival','Park Avenue','2023-07-30');
CREATE TABLE Craft_Fairs_Vendors(
    craft_fair_id INTEGER,
    vendor_type_id INTEGER,
    FOREIGN KEY (craft_fair_id) REFERENCES Craft_Fairs(id),
    FOREIGN KEY (vendor_type_id) REFERENCES Vendors(id));
INSERT INTO "Craft_Fairs_Vendors" VALUES(1,1);
INSERT INTO "Craft_Fairs_Vendors" VALUES(2,1);
INSERT INTO "Craft_Fairs_Vendors" VALUES(3,1);
INSERT INTO "Craft_Fairs_Vendors" VALUES(4,1);
INSERT INTO "Craft_Fairs_Vendors" VALUES(5,1);
INSERT INTO "Craft_Fairs_Vendors" VALUES(5,5);
CREATE TABLE Vendors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT);
INSERT INTO "Vendors" VALUES(1,'Jewelry_Makers');
INSERT INTO "Vendors" VALUES(2,'Pottery Artisans');
INSERT INTO "Vendors" VALUES(3,'Clothing Designers');
INSERT INTO "Vendors" VALUES(4,'Home Decorators');
INSERT INTO "Vendors" VALUES(5,'Accessory Makers');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Craft_Fairs',5);
INSERT INTO "sqlite_sequence" VALUES('Vendors',5);
COMMIT;
