BEGIN TRANSACTION;
CREATE TABLE Missions (
    mission_id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_name TEXT
);
INSERT INTO "Missions" VALUES(1,'Mars_Exploration');
INSERT INTO "Missions" VALUES(2,'Lunar_Exploration');
INSERT INTO "Missions" VALUES(3,'Asteroid_Study');
INSERT INTO "Missions" VALUES(4,'Planetary_Research');
INSERT INTO "Missions" VALUES(5,'Deep_Space_Probes');
CREATE TABLE Space_Reporters (
    reporter_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT);
INSERT INTO "Space_Reporters" VALUES(1,'John Doe');
INSERT INTO "Space_Reporters" VALUES(2,'Jane Smith');
INSERT INTO "Space_Reporters" VALUES(3,'Alex Brown');
INSERT INTO "Space_Reporters" VALUES(4,'Emily Green');
INSERT INTO "Space_Reporters" VALUES(5,'William Black');
CREATE TABLE Space_Reporters_Missions (
    space_reporter_id INTEGER,
    mission_id INTEGER,
    assignment_date DATETIME,
    FOREIGN KEY (space_reporter_id) REFERENCES Space_Reporters(reporter_id),
    FOREIGN KEY (mission_id) REFERENCES Missions(mission_id)
);
INSERT INTO "Space_Reporters_Missions" VALUES(1,1,'2031-07-01');
INSERT INTO "Space_Reporters_Missions" VALUES(2,2,'2032-12-01');
INSERT INTO "Space_Reporters_Missions" VALUES(3,3,'2033-06-01');
INSERT INTO "Space_Reporters_Missions" VALUES(4,4,'2034-11-01');
INSERT INTO "Space_Reporters_Missions" VALUES(5,5,'2035-05-01');
INSERT INTO "Space_Reporters_Missions" VALUES(2,1,'2032-12-01');
INSERT INTO "Space_Reporters_Missions" VALUES(3,1,'2033-06-01');
INSERT INTO "Space_Reporters_Missions" VALUES(4,1,'2034-11-01');
INSERT INTO "Space_Reporters_Missions" VALUES(5,1,'2035-05-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Missions',5);
INSERT INTO "sqlite_sequence" VALUES('Space_Reporters',5);
COMMIT;
