BEGIN TRANSACTION;
CREATE TABLE Equipment(
    equipment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT
);
INSERT INTO "Equipment" VALUES(1,'Dslr_Camera');
INSERT INTO "Equipment" VALUES(2,'Lens');
INSERT INTO "Equipment" VALUES(3,'Tripod');
INSERT INTO "Equipment" VALUES(4,'Flash');
INSERT INTO "Equipment" VALUES(5,'Lighting_Kit');
CREATE TABLE Photojournalist_Equipment(
    photojournalist_id INTEGER,
    equipment_id INTEGER,
    FOREIGN KEY(photojournalist_id) REFERENCES Photojournalists(id),
    FOREIGN KEY(equipment_id) REFERENCES Equipment(equipment_id),
    PRIMARY KEY(photojournalist_id, equipment_id)
);
INSERT INTO "Photojournalist_Equipment" VALUES(1,1);
INSERT INTO "Photojournalist_Equipment" VALUES(2,1);
INSERT INTO "Photojournalist_Equipment" VALUES(3,1);
INSERT INTO "Photojournalist_Equipment" VALUES(4,1);
INSERT INTO "Photojournalist_Equipment" VALUES(5,1);
INSERT INTO "Photojournalist_Equipment" VALUES(5,2);
CREATE TABLE Photojournalists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Photojournalists" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Photojournalists" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Photojournalists" VALUES(3,'Michael Johnson','michael@example.com');
INSERT INTO "Photojournalists" VALUES(4,'Emily Williams','emily@example.com');
INSERT INTO "Photojournalists" VALUES(5,'David Brown','david@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Photojournalists',5);
INSERT INTO "sqlite_sequence" VALUES('Equipment',5);
COMMIT;
