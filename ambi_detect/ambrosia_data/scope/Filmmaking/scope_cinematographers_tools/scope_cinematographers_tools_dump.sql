BEGIN TRANSACTION;
CREATE TABLE Cinematographers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthdate DATE
);
INSERT INTO "Cinematographers" VALUES(1,'John Doe','1980-01-01');
INSERT INTO "Cinematographers" VALUES(2,'Jane Smith','1978-02-02');
INSERT INTO "Cinematographers" VALUES(3,'Mike Johnson','1990-03-03');
INSERT INTO "Cinematographers" VALUES(4,'Emily Davis','1985-04-04');
INSERT INTO "Cinematographers" VALUES(5,'William Brown','1976-05-05');
CREATE TABLE Cinematographers_Tools (
    cinematographerId INTEGER,
    toolId INTEGER,
    FOREIGN KEY(cinematographerId) REFERENCES Cinematographers(id),
    FOREIGN KEY(toolId) REFERENCES Tools(id),
    PRIMARY KEY(cinematographerId, toolId)
);
INSERT INTO "Cinematographers_Tools" VALUES(2,1);
INSERT INTO "Cinematographers_Tools" VALUES(3,1);
INSERT INTO "Cinematographers_Tools" VALUES(4,1);
INSERT INTO "Cinematographers_Tools" VALUES(5,1);
INSERT INTO "Cinematographers_Tools" VALUES(1,2);
INSERT INTO "Cinematographers_Tools" VALUES(3,2);
INSERT INTO "Cinematographers_Tools" VALUES(4,2);
INSERT INTO "Cinematographers_Tools" VALUES(5,2);
INSERT INTO "Cinematographers_Tools" VALUES(1,3);
INSERT INTO "Cinematographers_Tools" VALUES(2,3);
INSERT INTO "Cinematographers_Tools" VALUES(4,3);
INSERT INTO "Cinematographers_Tools" VALUES(5,3);
INSERT INTO "Cinematographers_Tools" VALUES(1,4);
INSERT INTO "Cinematographers_Tools" VALUES(2,4);
INSERT INTO "Cinematographers_Tools" VALUES(3,4);
INSERT INTO "Cinematographers_Tools" VALUES(5,4);
INSERT INTO "Cinematographers_Tools" VALUES(1,5);
INSERT INTO "Cinematographers_Tools" VALUES(2,5);
INSERT INTO "Cinematographers_Tools" VALUES(3,5);
INSERT INTO "Cinematographers_Tools" VALUES(4,5);
INSERT INTO "Cinematographers_Tools" VALUES(4,4);
INSERT INTO "Cinematographers_Tools" VALUES(1,1);
INSERT INTO "Cinematographers_Tools" VALUES(2,2);
INSERT INTO "Cinematographers_Tools" VALUES(3,3);
INSERT INTO "Cinematographers_Tools" VALUES(5,5);
CREATE TABLE Tools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    toolName TEXT
);
INSERT INTO "Tools" VALUES(1,'Camera');
INSERT INTO "Tools" VALUES(2,'Tripod');
INSERT INTO "Tools" VALUES(3,'Lighting Kit');
INSERT INTO "Tools" VALUES(4,'Dolly');
INSERT INTO "Tools" VALUES(5,'Gimbal');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Cinematographers',5);
INSERT INTO "sqlite_sequence" VALUES('Tools',5);
COMMIT;
