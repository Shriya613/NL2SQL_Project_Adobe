BEGIN TRANSACTION;
CREATE TABLE Cuisine_Types(
    cuisine_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT);
INSERT INTO "Cuisine_Types" VALUES(1,'French_Cuisine');
INSERT INTO "Cuisine_Types" VALUES(2,'Italian_Cuisine');
INSERT INTO "Cuisine_Types" VALUES(3,'Japanese_Cuisine');
INSERT INTO "Cuisine_Types" VALUES(4,'Mexican_Cuisine');
INSERT INTO "Cuisine_Types" VALUES(5,'Spanish_Cuisine');
CREATE TABLE Culinary_Schools(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Culinary_Schools" VALUES(1,'Le Cordon Bleu','Paris');
INSERT INTO "Culinary_Schools" VALUES(2,'Institute of Culinary Education','New York City');
INSERT INTO "Culinary_Schools" VALUES(3,'The Culinary Institute of America','Hyde Park');
INSERT INTO "Culinary_Schools" VALUES(4,'Escoffier School of Culinary Arts','Austin');
INSERT INTO "Culinary_Schools" VALUES(5,'International Culinary Center','New York City');
CREATE TABLE Culinary_Schools_Cuisine_Types(
    culinary_school_id INTEGER,
    cuisine_type_id INTEGER,
    FOREIGN KEY (culinary_school_id) REFERENCES Culinary_Schools(id),
    FOREIGN KEY (cuisine_type_id) REFERENCES Cuisine_Types(cuisine_id));
INSERT INTO "Culinary_Schools_Cuisine_Types" VALUES(1,1);
INSERT INTO "Culinary_Schools_Cuisine_Types" VALUES(2,1);
INSERT INTO "Culinary_Schools_Cuisine_Types" VALUES(3,1);
INSERT INTO "Culinary_Schools_Cuisine_Types" VALUES(4,1);
INSERT INTO "Culinary_Schools_Cuisine_Types" VALUES(5,1);
INSERT INTO "Culinary_Schools_Cuisine_Types" VALUES(5,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Culinary_Schools',5);
INSERT INTO "sqlite_sequence" VALUES('Cuisine_Types',5);
COMMIT;
