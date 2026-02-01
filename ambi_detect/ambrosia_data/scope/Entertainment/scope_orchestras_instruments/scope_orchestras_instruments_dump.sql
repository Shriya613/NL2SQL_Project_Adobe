BEGIN TRANSACTION;
CREATE TABLE Conductors(
    id INTEGER PRIMARY KEY,
    name TEXT,
    birthdate DATE);
INSERT INTO "Conductors" VALUES(1,'Jane Doe','2000-01-01');
INSERT INTO "Conductors" VALUES(2,'John Smith','1987-03-26');
INSERT INTO "Conductors" VALUES(3,'Emily Johnson','1995-08-12');
INSERT INTO "Conductors" VALUES(4,'Michael Brown','1978-10-05');
INSERT INTO "Conductors" VALUES(5,'Rachel White','1992-06-09');
CREATE TABLE Instruments(
    id INTEGER PRIMARY KEY,
    instrument_type TEXT);
INSERT INTO "Instruments" VALUES(1,'Violin');
INSERT INTO "Instruments" VALUES(2,'Cello');
INSERT INTO "Instruments" VALUES(3,'Piano');
INSERT INTO "Instruments" VALUES(4,'Flute');
INSERT INTO "Instruments" VALUES(5,'Trumpet');
CREATE TABLE Orchestras(
    id INTEGER PRIMARY KEY,
    orchestra_name TEXT,
    conductor_id INTEGER REFERENCES Conductors(id));
INSERT INTO "Orchestras" VALUES(1,'New York Philharmonic',1);
INSERT INTO "Orchestras" VALUES(2,'Boston Symphony Orchestra',2);
INSERT INTO "Orchestras" VALUES(3,'Chicago Symphony Orchestra',3);
INSERT INTO "Orchestras" VALUES(4,'Los Angeles Philharmonic',4);
INSERT INTO "Orchestras" VALUES(5,'Berlin Philharmonic',5);
CREATE TABLE Orchestras_Instruments(
    id INTEGER PRIMARY KEY,
    orchestra_id INTEGER REFERENCES Orchestras(id),
    instrument_id INTEGER REFERENCES Instruments(id));
INSERT INTO "Orchestras_Instruments" VALUES(1,1,1);
INSERT INTO "Orchestras_Instruments" VALUES(2,1,2);
INSERT INTO "Orchestras_Instruments" VALUES(3,2,1);
INSERT INTO "Orchestras_Instruments" VALUES(4,2,2);
INSERT INTO "Orchestras_Instruments" VALUES(5,3,3);
INSERT INTO "Orchestras_Instruments" VALUES(6,3,4);
INSERT INTO "Orchestras_Instruments" VALUES(7,4,4);
INSERT INTO "Orchestras_Instruments" VALUES(8,4,5);
INSERT INTO "Orchestras_Instruments" VALUES(9,5,1);
INSERT INTO "Orchestras_Instruments" VALUES(10,5,2);
INSERT INTO "Orchestras_Instruments" VALUES(11,3,1);
INSERT INTO "Orchestras_Instruments" VALUES(12,4,1);
COMMIT;
