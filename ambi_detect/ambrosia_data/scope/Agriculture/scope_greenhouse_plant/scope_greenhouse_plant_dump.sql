BEGIN TRANSACTION;
CREATE TABLE Greenhouse (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Greenhouse" VALUES(1,'Greenhouse A','New York');
INSERT INTO "Greenhouse" VALUES(2,'Greenhouse B','California');
INSERT INTO "Greenhouse" VALUES(3,'Greenhouse C','Texas');
INSERT INTO "Greenhouse" VALUES(4,'Greenhouse D','Florida');
INSERT INTO "Greenhouse" VALUES(5,'Greenhouse E','Washington');
CREATE TABLE Greenhouse_Plant (
    greenhouse_id INTEGER,
    plant_id INTEGER,
    quantity INTEGER,
    FOREIGN KEY(greenhouse_id) REFERENCES Greenhouse(id),
    FOREIGN KEY(plant_id) REFERENCES Plant(id),
    PRIMARY KEY(greenhouse_id, plant_id)
);
INSERT INTO "Greenhouse_Plant" VALUES(1,1,NULL);
INSERT INTO "Greenhouse_Plant" VALUES(2,1,NULL);
INSERT INTO "Greenhouse_Plant" VALUES(3,1,NULL);
INSERT INTO "Greenhouse_Plant" VALUES(4,1,NULL);
INSERT INTO "Greenhouse_Plant" VALUES(5,1,NULL);
INSERT INTO "Greenhouse_Plant" VALUES(5,3,NULL);
CREATE TABLE Plant (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    type TEXT
);
INSERT INTO "Plant" VALUES(1,'Tomato','Fruit');
INSERT INTO "Plant" VALUES(2,'Cucumber','Vegetable');
INSERT INTO "Plant" VALUES(3,'Pepper','Vegetable');
INSERT INTO "Plant" VALUES(4,'Lettuce','Leafy Vegetable');
INSERT INTO "Plant" VALUES(5,'Carrot','Root Vegetable');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Greenhouse',5);
INSERT INTO "sqlite_sequence" VALUES('Plant',5);
COMMIT;
