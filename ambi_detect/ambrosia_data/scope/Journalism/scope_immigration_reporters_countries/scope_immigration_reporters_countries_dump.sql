BEGIN TRANSACTION;
CREATE TABLE Countries(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    countryName TEXT
);
INSERT INTO "Countries" VALUES(1,'Mexico');
INSERT INTO "Countries" VALUES(2,'Canada');
INSERT INTO "Countries" VALUES(3,'United States');
INSERT INTO "Countries" VALUES(4,'Germany');
INSERT INTO "Countries" VALUES(5,'France');
CREATE TABLE Immigration_Reporters(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reporterName TEXT,
    email TEXT
);
INSERT INTO "Immigration_Reporters" VALUES(1,'John Doe','johndoe@example.com');
INSERT INTO "Immigration_Reporters" VALUES(2,'Jane Smith','janesmith@example.com');
INSERT INTO "Immigration_Reporters" VALUES(3,'Alice Brown','alicebrown@example.com');
INSERT INTO "Immigration_Reporters" VALUES(4,'Bob Green','bogreen@example.com');
INSERT INTO "Immigration_Reporters" VALUES(5,'Charlie White','charliewhite@example.com');
CREATE TABLE Immigration_Reporters_Countries(
    immigrationReporterId INTEGER,
    countryId INTEGER,
    FOREIGN KEY(immigrationReporterId) REFERENCES Immigration_Reporters(id),
    FOREIGN KEY(countryId) REFERENCES Countries(id)
);
INSERT INTO "Immigration_Reporters_Countries" VALUES(1,1);
INSERT INTO "Immigration_Reporters_Countries" VALUES(2,1);
INSERT INTO "Immigration_Reporters_Countries" VALUES(3,1);
INSERT INTO "Immigration_Reporters_Countries" VALUES(4,1);
INSERT INTO "Immigration_Reporters_Countries" VALUES(5,1);
INSERT INTO "Immigration_Reporters_Countries" VALUES(5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Countries',5);
INSERT INTO "sqlite_sequence" VALUES('Immigration_Reporters',5);
COMMIT;
