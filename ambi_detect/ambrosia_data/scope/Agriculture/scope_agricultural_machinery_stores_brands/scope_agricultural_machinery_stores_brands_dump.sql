BEGIN TRANSACTION;
CREATE TABLE Agricultural_Machinery_Stores(
    store_id INTEGER PRIMARY KEY AUTOINCREMENT,
    store_name TEXT,
    location TEXT
);
INSERT INTO "Agricultural_Machinery_Stores" VALUES(1,'Store A','123 Main St');
INSERT INTO "Agricultural_Machinery_Stores" VALUES(2,'Store B','456 Elm St');
INSERT INTO "Agricultural_Machinery_Stores" VALUES(3,'Store C','789 Oak St');
INSERT INTO "Agricultural_Machinery_Stores" VALUES(4,'Store D','012 Pine St');
INSERT INTO "Agricultural_Machinery_Stores" VALUES(5,'Store E','345 Maple St');
CREATE TABLE Brands(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand_name TEXT
);
INSERT INTO "Brands" VALUES(1,'John Deere');
INSERT INTO "Brands" VALUES(2,'Caterpillar');
INSERT INTO "Brands" VALUES(3,'New Holland');
INSERT INTO "Brands" VALUES(4,'Case IH');
INSERT INTO "Brands" VALUES(5,'Fendt');
CREATE TABLE Machines(
    machine_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_number TEXT,
    description TEXT
);
INSERT INTO "Machines" VALUES(1,'Model X','Tractor Model X from John Deere.');
INSERT INTO "Machines" VALUES(2,'Model Y','Combine Harvester Model Y by New Holland.');
INSERT INTO "Machines" VALUES(3,'Model Z','Plow Model Z made by Case IH.');
INSERT INTO "Machines" VALUES(4,'Model W','Tiller Model W produced by Fendt.');
INSERT INTO "Machines" VALUES(5,'Model V','Seeder Model V created by Caterpillar.');
CREATE TABLE Stores_Brands(
    store_id INTEGER,
    brand_id INTEGER,
    FOREIGN KEY (store_id) REFERENCES Agricultural_Machinery_Stores(store_id),
    FOREIGN KEY (brand_id) REFERENCES Brands(id)
);
INSERT INTO "Stores_Brands" VALUES(1,1);
INSERT INTO "Stores_Brands" VALUES(2,1);
INSERT INTO "Stores_Brands" VALUES(3,1);
INSERT INTO "Stores_Brands" VALUES(4,1);
INSERT INTO "Stores_Brands" VALUES(5,1);
INSERT INTO "Stores_Brands" VALUES(5,2);
CREATE TABLE Stores_Machines(
    store_id INTEGER,
    machine_id INTEGER,
    FOREIGN KEY (store_id) REFERENCES Agricultural_Machinery_Stores(store_id),
    FOREIGN KEY (machine_id) REFERENCES Machines(machine_id)
);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Brands',5);
INSERT INTO "sqlite_sequence" VALUES('Agricultural_Machinery_Stores',5);
INSERT INTO "sqlite_sequence" VALUES('Machines',5);
COMMIT;
