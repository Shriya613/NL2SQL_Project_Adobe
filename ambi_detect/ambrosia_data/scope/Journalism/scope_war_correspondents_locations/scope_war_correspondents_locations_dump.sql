BEGIN TRANSACTION;
CREATE TABLE Locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    locationName TEXT);
INSERT INTO "Locations" VALUES(1,'Battlefield');
INSERT INTO "Locations" VALUES(2,'Capital City');
INSERT INTO "Locations" VALUES(3,'Refugee Camp');
INSERT INTO "Locations" VALUES(4,'Military Base');
INSERT INTO "Locations" VALUES(5,'Government Building');
CREATE TABLE War_Correspondents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    organization TEXT);
INSERT INTO "War_Correspondents" VALUES(1,'John Smith','ABC News');
INSERT INTO "War_Correspondents" VALUES(2,'Jane Doe','CNN');
INSERT INTO "War_Correspondents" VALUES(3,'Michael Brown','Fox News');
INSERT INTO "War_Correspondents" VALUES(4,'Emily Johnson','BBC');
INSERT INTO "War_Correspondents" VALUES(5,'David Wilson','Al Jazeera');
CREATE TABLE War_Correspondents_Locations (
    warCorrespondentId INTEGER,
    locationId INTEGER,
    assignmentDate DATE,
    FOREIGN KEY(warCorrespondentId) REFERENCES War_Correspondents(id),
    FOREIGN KEY(locationId) REFERENCES Locations(id));
INSERT INTO "War_Correspondents_Locations" VALUES(1,1,'2023-01-01');
INSERT INTO "War_Correspondents_Locations" VALUES(1,2,'2023-02-01');
INSERT INTO "War_Correspondents_Locations" VALUES(2,1,'2023-03-01');
INSERT INTO "War_Correspondents_Locations" VALUES(2,3,'2023-04-01');
INSERT INTO "War_Correspondents_Locations" VALUES(3,2,'2023-05-01');
INSERT INTO "War_Correspondents_Locations" VALUES(3,4,'2023-06-01');
INSERT INTO "War_Correspondents_Locations" VALUES(4,1,'2023-07-01');
INSERT INTO "War_Correspondents_Locations" VALUES(4,5,'2023-08-01');
INSERT INTO "War_Correspondents_Locations" VALUES(5,3,'2023-09-01');
INSERT INTO "War_Correspondents_Locations" VALUES(5,4,'2023-10-01');
INSERT INTO "War_Correspondents_Locations" VALUES(3,1,'2023-05-01');
INSERT INTO "War_Correspondents_Locations" VALUES(5,1,'2023-09-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('War_Correspondents',5);
INSERT INTO "sqlite_sequence" VALUES('Locations',5);
COMMIT;
