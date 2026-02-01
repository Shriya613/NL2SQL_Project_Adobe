BEGIN TRANSACTION;
CREATE TABLE Ingredients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingredient_name TEXT);
INSERT INTO "Ingredients" VALUES(1,'Malathion');
INSERT INTO "Ingredients" VALUES(2,'Pyrethrin');
INSERT INTO "Ingredients" VALUES(3,'Neem Oil');
INSERT INTO "Ingredients" VALUES(4,'Rotenone');
INSERT INTO "Ingredients" VALUES(5,'Spinosad');
CREATE TABLE Pesticides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    usage_instructions TEXT);
INSERT INTO "Pesticides" VALUES(1,'PestiX','Apply once a month');
INSERT INTO "Pesticides" VALUES(2,'EcoGuard','Use sparingly on sensitive crops');
INSERT INTO "Pesticides" VALUES(3,'BugOff','Apply only when needed');
INSERT INTO "Pesticides" VALUES(4,'CropShield','Apply before pests become a problem');
INSERT INTO "Pesticides" VALUES(5,'PlantProtect','Reapply after heavy rainfall');
CREATE TABLE Pesticides_Ingredients (
    pesticide_id INTEGER,
    ingredient_id INTEGER,
    FOREIGN KEY(pesticide_id) REFERENCES Pesticides(id),
    FOREIGN KEY(ingredient_id) REFERENCES Ingredients(id));
INSERT INTO "Pesticides_Ingredients" VALUES(1,1);
INSERT INTO "Pesticides_Ingredients" VALUES(2,2);
INSERT INTO "Pesticides_Ingredients" VALUES(3,3);
INSERT INTO "Pesticides_Ingredients" VALUES(4,4);
INSERT INTO "Pesticides_Ingredients" VALUES(5,5);
INSERT INTO "Pesticides_Ingredients" VALUES(2,1);
INSERT INTO "Pesticides_Ingredients" VALUES(3,1);
INSERT INTO "Pesticides_Ingredients" VALUES(4,1);
INSERT INTO "Pesticides_Ingredients" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Pesticides',5);
INSERT INTO "sqlite_sequence" VALUES('Ingredients',5);
COMMIT;
