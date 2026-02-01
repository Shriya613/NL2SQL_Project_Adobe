BEGIN TRANSACTION;
CREATE TABLE Geologists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Geologists" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Geologists" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Geologists" VALUES(3,'Michael Johnson','michael@example.com');
INSERT INTO "Geologists" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Geologists" VALUES(5,'David Brown','david@example.com');
CREATE TABLE Geologists_Specimens (
    geologistId INTEGER,
    specimenType TEXT,
    FOREIGN KEY (geologistId) REFERENCES Geologists(id),
    FOREIGN KEY (specimenType) REFERENCES Specimens(type)
);
INSERT INTO "Geologists_Specimens" VALUES(1,'Rock');
INSERT INTO "Geologists_Specimens" VALUES(2,'Rock');
INSERT INTO "Geologists_Specimens" VALUES(3,'Rock');
INSERT INTO "Geologists_Specimens" VALUES(4,'Rock');
INSERT INTO "Geologists_Specimens" VALUES(5,'Rock');
INSERT INTO "Geologists_Specimens" VALUES(1,'Fossil');
INSERT INTO "Geologists_Specimens" VALUES(2,'Fossil');
INSERT INTO "Geologists_Specimens" VALUES(3,'Fossil');
INSERT INTO "Geologists_Specimens" VALUES(4,'Fossil');
INSERT INTO "Geologists_Specimens" VALUES(5,'Fossil');
CREATE TABLE Specimens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT ,
    description TEXT
);
INSERT INTO "Specimens" VALUES(1,'Rock','Igneous rock formed by cooling of magma.');
INSERT INTO "Specimens" VALUES(2,'Fossil','Remains or impressions of prehistoric life preserved in petrified form or as a mold.');
INSERT INTO "Specimens" VALUES(3,'Mineral','Naturally occurring solid substance with a definite chemical composition.');
INSERT INTO "Specimens" VALUES(4,'Gemstone','Polished stone used to make jewelry or decorative objects.');
INSERT INTO "Specimens" VALUES(5,'Metal Ore','Natural occurrence of metal-containing minerals from which metals are extracted.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Geologists',5);
INSERT INTO "sqlite_sequence" VALUES('Specimens',5);
COMMIT;
