BEGIN TRANSACTION;
CREATE TABLE sports_team (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    sport TEXT ,
    country TEXT,
    founded_year INT
);
INSERT INTO "sports_team" VALUES(1,'Toronto Blue Jays','Baseball Team','Canada',1976);
INSERT INTO "sports_team" VALUES(2,'Los Angeles Dodgers','Baseball Team','USA',1883);
INSERT INTO "sports_team" VALUES(3,'Houston Rockets','Basketball Team','USA',1967);
INSERT INTO "sports_team" VALUES(4,'Golden State Warriors','Basketball Team','USA',1946);
INSERT INTO "sports_team" VALUES(5,'Manchester United FC','Football Club','UK',1878);
INSERT INTO "sports_team" VALUES(6,'FC Barcelona','Football Club','Spain',1899);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('sports_team',6);
COMMIT;
