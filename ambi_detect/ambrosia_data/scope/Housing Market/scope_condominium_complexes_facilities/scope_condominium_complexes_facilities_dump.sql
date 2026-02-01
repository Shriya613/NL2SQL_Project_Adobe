BEGIN TRANSACTION;
CREATE TABLE Condominium_Complexes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Condominium_Complexes" VALUES(1,'Parkside Towers','Downtown');
INSERT INTO "Condominium_Complexes" VALUES(2,'Riverview Estates','Suburb');
INSERT INTO "Condominium_Complexes" VALUES(3,'City Heights','Uptown');
INSERT INTO "Condominium_Complexes" VALUES(4,'Lakeshore Villas','Waterfront');
INSERT INTO "Condominium_Complexes" VALUES(5,'Green Meadows','Country');
INSERT INTO "Condominium_Complexes" VALUES(6,'Mountain View Residence','Hills');
CREATE TABLE Condominium_Complexes_Facilities(
    condoId INTEGER REFERENCES Condominium_Complexes(id),
    facilityId INTEGER REFERENCES Facilities(id));
INSERT INTO "Condominium_Complexes_Facilities" VALUES(1,1);
INSERT INTO "Condominium_Complexes_Facilities" VALUES(2,3);
INSERT INTO "Condominium_Complexes_Facilities" VALUES(3,2);
INSERT INTO "Condominium_Complexes_Facilities" VALUES(4,1);
INSERT INTO "Condominium_Complexes_Facilities" VALUES(5,4);
INSERT INTO "Condominium_Complexes_Facilities" VALUES(6,7);
INSERT INTO "Condominium_Complexes_Facilities" VALUES(2,1);
INSERT INTO "Condominium_Complexes_Facilities" VALUES(3,1);
INSERT INTO "Condominium_Complexes_Facilities" VALUES(5,1);
INSERT INTO "Condominium_Complexes_Facilities" VALUES(6,1);
CREATE TABLE Facilities(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    facilityName TEXT);
INSERT INTO "Facilities" VALUES(1,'Fitness Center');
INSERT INTO "Facilities" VALUES(2,'Swimming Pool');
INSERT INTO "Facilities" VALUES(3,'Tennis Court');
INSERT INTO "Facilities" VALUES(4,'Playground');
INSERT INTO "Facilities" VALUES(5,'Clubhouse');
INSERT INTO "Facilities" VALUES(6,'Community Garden');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Condominium_Complexes',6);
INSERT INTO "sqlite_sequence" VALUES('Facilities',6);
COMMIT;
