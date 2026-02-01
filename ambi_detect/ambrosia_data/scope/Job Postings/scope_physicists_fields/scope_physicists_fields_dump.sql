BEGIN TRANSACTION;
CREATE TABLE Fields(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fieldName TEXT);
INSERT INTO "Fields" VALUES(1,'Quantum_Mechanics');
INSERT INTO "Fields" VALUES(2,'Electromagnetism');
INSERT INTO "Fields" VALUES(3,'Relativity');
INSERT INTO "Fields" VALUES(4,'Particle_Physics');
INSERT INTO "Fields" VALUES(5,'Cosmology');
CREATE TABLE Physicist_Fields(
    physicistId INTEGER,
    fieldId INTEGER,
    FOREIGN KEY (physicistId) REFERENCES Physicists(id),
    FOREIGN KEY (fieldId) REFERENCES Fields(id));
INSERT INTO "Physicist_Fields" VALUES(1,1);
INSERT INTO "Physicist_Fields" VALUES(2,1);
INSERT INTO "Physicist_Fields" VALUES(3,1);
INSERT INTO "Physicist_Fields" VALUES(4,1);
INSERT INTO "Physicist_Fields" VALUES(5,1);
INSERT INTO "Physicist_Fields" VALUES(5,3);
CREATE TABLE Physicists(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT);
INSERT INTO "Physicists" VALUES(1,'Alice','alice@email.com');
INSERT INTO "Physicists" VALUES(2,'Bob','bob@email.com');
INSERT INTO "Physicists" VALUES(3,'Carol','carol@email.com');
INSERT INTO "Physicists" VALUES(4,'Dave','dave@email.com');
INSERT INTO "Physicists" VALUES(5,'Eve','eve@email.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Fields',5);
INSERT INTO "sqlite_sequence" VALUES('Physicists',5);
COMMIT;
