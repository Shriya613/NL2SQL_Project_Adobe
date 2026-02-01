BEGIN TRANSACTION;
CREATE TABLE Cinematographies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    release_year INTEGER
);
INSERT INTO "Cinematographies" VALUES(1,'Movie1',2000);
INSERT INTO "Cinematographies" VALUES(2,'Movie2',2001);
INSERT INTO "Cinematographies" VALUES(3,'Movie3',2002);
INSERT INTO "Cinematographies" VALUES(4,'Movie4',2003);
INSERT INTO "Cinematographies" VALUES(5,'Movie5',2004);
CREATE TABLE Cinematographies_Techniques (
    cinematography_id INTEGER,
    technique_id INTEGER,
    FOREIGN KEY(cinematography_id) REFERENCES Cinematographies(id),
    FOREIGN KEY(technique_id) REFERENCES Techniques(id),
    PRIMARY KEY(cinematography_id, technique_id)
);
INSERT INTO "Cinematographies_Techniques" VALUES(1,1);
INSERT INTO "Cinematographies_Techniques" VALUES(1,2);
INSERT INTO "Cinematographies_Techniques" VALUES(2,1);
INSERT INTO "Cinematographies_Techniques" VALUES(2,3);
INSERT INTO "Cinematographies_Techniques" VALUES(3,2);
INSERT INTO "Cinematographies_Techniques" VALUES(3,4);
INSERT INTO "Cinematographies_Techniques" VALUES(4,1);
INSERT INTO "Cinematographies_Techniques" VALUES(4,5);
INSERT INTO "Cinematographies_Techniques" VALUES(5,2);
INSERT INTO "Cinematographies_Techniques" VALUES(5,3);
INSERT INTO "Cinematographies_Techniques" VALUES(3,1);
INSERT INTO "Cinematographies_Techniques" VALUES(5,1);
CREATE TABLE Techniques (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    technique_name TEXT
);
INSERT INTO "Techniques" VALUES(1,'Wide_Shot');
INSERT INTO "Techniques" VALUES(2,'Close_Up');
INSERT INTO "Techniques" VALUES(3,'Low_Angle');
INSERT INTO "Techniques" VALUES(4,'High_Angle');
INSERT INTO "Techniques" VALUES(5,'Overhead');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Cinematographies',5);
INSERT INTO "sqlite_sequence" VALUES('Techniques',5);
COMMIT;
