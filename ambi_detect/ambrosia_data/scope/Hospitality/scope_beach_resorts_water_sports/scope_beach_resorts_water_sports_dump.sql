BEGIN TRANSACTION;
CREATE TABLE Beach_Resorts(
    ResortID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT
);
INSERT INTO "Beach_Resorts" VALUES(1,'Paradise Beach','California');
INSERT INTO "Beach_Resorts" VALUES(2,'Sunset Shores','Florida');
INSERT INTO "Beach_Resorts" VALUES(3,'Tropical Paradise','Hawaii');
INSERT INTO "Beach_Resorts" VALUES(4,'Seashell Cove','Caribbean');
INSERT INTO "Beach_Resorts" VALUES(5,'Wave Rider Bay','Australia');
CREATE TABLE Beach_Resorts_Water_Sports(
    ResortID INTEGER,
    SportID INTEGER,
    FOREIGN KEY (ResortID) REFERENCES Beach_Resorts(ResortID),
    FOREIGN KEY (SportID) REFERENCES Water_Sports(SportID),
    PRIMARY KEY (ResortID, SportID)
);
INSERT INTO "Beach_Resorts_Water_Sports" VALUES(1,1);
INSERT INTO "Beach_Resorts_Water_Sports" VALUES(2,2);
INSERT INTO "Beach_Resorts_Water_Sports" VALUES(3,3);
INSERT INTO "Beach_Resorts_Water_Sports" VALUES(4,4);
INSERT INTO "Beach_Resorts_Water_Sports" VALUES(5,5);
INSERT INTO "Beach_Resorts_Water_Sports" VALUES(2,1);
INSERT INTO "Beach_Resorts_Water_Sports" VALUES(3,1);
INSERT INTO "Beach_Resorts_Water_Sports" VALUES(4,1);
INSERT INTO "Beach_Resorts_Water_Sports" VALUES(5,1);
CREATE TABLE Water_Sports(
    SportID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT
);
INSERT INTO "Water_Sports" VALUES(1,'Surfing','A water sport involving the riding of ocean waves on a surfboard.');
INSERT INTO "Water_Sports" VALUES(2,'Snorkeling','An activity where you swim underwater while wearing a mask and fins.');
INSERT INTO "Water_Sports" VALUES(3,'Scuba Diving','An underwater diving activity that allows divers to breathe underwater using a tank containing compressed air.');
INSERT INTO "Water_Sports" VALUES(4,'Kayaking','Paddling a small narrow boat on rivers or coastal waters.');
INSERT INTO "Water_Sports" VALUES(5,'Swimming','The act of propelling oneself through water by moving the limbs.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Beach_Resorts',5);
INSERT INTO "sqlite_sequence" VALUES('Water_Sports',5);
COMMIT;
