BEGIN TRANSACTION;
CREATE TABLE Eco_Lodges(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Eco_Lodges" VALUES(1,'Green Valley Lodge','Amazon Rainforest');
INSERT INTO "Eco_Lodges" VALUES(2,'Ocean Breeze Lodge','Costa Rica Coastline');
INSERT INTO "Eco_Lodges" VALUES(3,'Mountain Retreat Lodge','Appalachian Mountains');
INSERT INTO "Eco_Lodges" VALUES(4,'Desert Oasis Lodge','Mojave Desert');
INSERT INTO "Eco_Lodges" VALUES(5,'Arctic Igloo Lodge','North Pole');
CREATE TABLE Eco_Lodges_Sustainable_Practices(
    eco_lodge_id INTEGER,
    practice_id INTEGER,
    FOREIGN KEY(eco_lodge_id) REFERENCES Eco_Lodges(id),
    FOREIGN KEY(practice_id) REFERENCES Sustainable_Practices(practice_id));
INSERT INTO "Eco_Lodges_Sustainable_Practices" VALUES(1,1);
INSERT INTO "Eco_Lodges_Sustainable_Practices" VALUES(1,2);
INSERT INTO "Eco_Lodges_Sustainable_Practices" VALUES(2,3);
INSERT INTO "Eco_Lodges_Sustainable_Practices" VALUES(3,4);
INSERT INTO "Eco_Lodges_Sustainable_Practices" VALUES(4,5);
INSERT INTO "Eco_Lodges_Sustainable_Practices" VALUES(2,1);
INSERT INTO "Eco_Lodges_Sustainable_Practices" VALUES(3,1);
INSERT INTO "Eco_Lodges_Sustainable_Practices" VALUES(4,1);
INSERT INTO "Eco_Lodges_Sustainable_Practices" VALUES(5,1);
CREATE TABLE Sustainable_Practices(
    practice_id INTEGER PRIMARY KEY AUTOINCREMENT,
    practice_name TEXT);
INSERT INTO "Sustainable_Practices" VALUES(1,'Solar Power');
INSERT INTO "Sustainable_Practices" VALUES(2,'Water Conservation');
INSERT INTO "Sustainable_Practices" VALUES(3,'Waste Management');
INSERT INTO "Sustainable_Practices" VALUES(4,'Local Sourcing');
INSERT INTO "Sustainable_Practices" VALUES(5,'Biodiversity Protection');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Sustainable_Practices',5);
INSERT INTO "sqlite_sequence" VALUES('Eco_Lodges',5);
COMMIT;
