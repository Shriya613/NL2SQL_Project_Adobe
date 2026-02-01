BEGIN TRANSACTION;
CREATE TABLE Elections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    election_type TEXT,
    start_date DATE,
    end_date DATE
);
INSERT INTO "Elections" VALUES(1,'Presidential_Campaigns','2021-01-01','2021-12-31');
INSERT INTO "Elections" VALUES(2,'Local_Council_elections','2022-01-01','2022-12-31');
INSERT INTO "Elections" VALUES(3,'State_Assembly_elections','2023-01-01','2023-12-31');
INSERT INTO "Elections" VALUES(4,'Referendums','2024-01-01','2024-12-31');
INSERT INTO "Elections" VALUES(5,'By-Elections','2025-01-01','2025-12-31');
CREATE TABLE Political_Reporters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Political_Reporters" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Political_Reporters" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Political_Reporters" VALUES(3,'Alice Johnson','alice@example.com');
INSERT INTO "Political_Reporters" VALUES(4,'Bob Brown','bob@example.com');
INSERT INTO "Political_Reporters" VALUES(5,'Charlie Green','charlie@example.com');
CREATE TABLE Political_Reporters_Elections (
    reporter_id INTEGER REFERENCES Political_Reporters(id),
    election_id INTEGER REFERENCES Elections(id),
    FOREIGN KEY (reporter_id) REFERENCES Political_Reporters(id),
    FOREIGN KEY (election_id) REFERENCES Elections(id)
);
INSERT INTO "Political_Reporters_Elections" VALUES(1,1);
INSERT INTO "Political_Reporters_Elections" VALUES(1,2);
INSERT INTO "Political_Reporters_Elections" VALUES(2,1);
INSERT INTO "Political_Reporters_Elections" VALUES(2,3);
INSERT INTO "Political_Reporters_Elections" VALUES(3,1);
INSERT INTO "Political_Reporters_Elections" VALUES(4,1);
INSERT INTO "Political_Reporters_Elections" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Political_Reporters',5);
INSERT INTO "sqlite_sequence" VALUES('Elections',5);
COMMIT;
