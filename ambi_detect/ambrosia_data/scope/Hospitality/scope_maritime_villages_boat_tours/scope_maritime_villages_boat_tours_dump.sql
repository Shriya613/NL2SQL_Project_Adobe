BEGIN TRANSACTION;
CREATE TABLE Boat_Tours (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tour_name TEXT,
    description TEXT
);
INSERT INTO "Boat_Tours" VALUES(1,'Whale Watching','A tour to watch whales.');
INSERT INTO "Boat_Tours" VALUES(2,'Fishing Tour','Go on a fishing trip.');
INSERT INTO "Boat_Tours" VALUES(3,'Cruise','Relaxing cruise on the sea.');
INSERT INTO "Boat_Tours" VALUES(4,'Dolphin Spotting','Spot dolphins in their natural habitat.');
INSERT INTO "Boat_Tours" VALUES(5,'Sunset Cruise','Enjoy sunset over the ocean.');
CREATE TABLE Maritime_Villages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Maritime_Villages" VALUES(1,'Village1','Location1');
INSERT INTO "Maritime_Villages" VALUES(2,'Village2','Location2');
INSERT INTO "Maritime_Villages" VALUES(3,'Village3','Location3');
INSERT INTO "Maritime_Villages" VALUES(4,'Village4','Location4');
INSERT INTO "Maritime_Villages" VALUES(5,'Village5','Location5');
CREATE TABLE Maritime_Villages_Boat_Tours (
    maritime_village_id INTEGER,
    boat_tour_id INTEGER,
    FOREIGN KEY(maritime_village_id) REFERENCES Maritime_Villages(id),
    FOREIGN KEY(boat_tour_id) REFERENCES Boat_Tours(id)
);
INSERT INTO "Maritime_Villages_Boat_Tours" VALUES(1,1);
INSERT INTO "Maritime_Villages_Boat_Tours" VALUES(2,1);
INSERT INTO "Maritime_Villages_Boat_Tours" VALUES(3,1);
INSERT INTO "Maritime_Villages_Boat_Tours" VALUES(4,1);
INSERT INTO "Maritime_Villages_Boat_Tours" VALUES(5,1);
INSERT INTO "Maritime_Villages_Boat_Tours" VALUES(5,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Maritime_Villages',5);
INSERT INTO "sqlite_sequence" VALUES('Boat_Tours',5);
COMMIT;
