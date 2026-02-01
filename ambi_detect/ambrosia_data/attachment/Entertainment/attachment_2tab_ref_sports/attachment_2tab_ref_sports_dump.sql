BEGIN TRANSACTION;
CREATE TABLE BaseballTeams(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teamName TEXT,
    countryId INTEGER,
    FOREIGN KEY(countryId) REFERENCES Countries(id));
INSERT INTO "BaseballTeams" VALUES(1,'New York Yankees',1);
INSERT INTO "BaseballTeams" VALUES(2,'San Francisco Giants',2);
INSERT INTO "BaseballTeams" VALUES(3,'Adelaide Strikers',3);
INSERT INTO "BaseballTeams" VALUES(4,'Barcelona',4);
INSERT INTO "BaseballTeams" VALUES(5,'Hiroshima Toyo Carp',5);
CREATE TABLE BasketballTeams(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teamName TEXT,
    countryId INTEGER,
    FOREIGN KEY(countryId) REFERENCES Countries(id));
INSERT INTO "BasketballTeams" VALUES(1,'Golden State Warriors',1);
INSERT INTO "BasketballTeams" VALUES(2,'Chicago Bulls',2);
INSERT INTO "BasketballTeams" VALUES(3,'Perth Wildcats',3);
INSERT INTO "BasketballTeams" VALUES(4,'FC Barcelona',4);
INSERT INTO "BasketballTeams" VALUES(5,'Yokohama B-Corsairs',5);
CREATE TABLE Countries(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
INSERT INTO "Countries" VALUES(1,'Canada');
INSERT INTO "Countries" VALUES(2,'United States');
INSERT INTO "Countries" VALUES(3,'Australia');
INSERT INTO "Countries" VALUES(4,'Spain');
INSERT INTO "Countries" VALUES(5,'Japan');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Countries',5);
INSERT INTO "sqlite_sequence" VALUES('BaseballTeams',5);
INSERT INTO "sqlite_sequence" VALUES('BasketballTeams',5);
COMMIT;
