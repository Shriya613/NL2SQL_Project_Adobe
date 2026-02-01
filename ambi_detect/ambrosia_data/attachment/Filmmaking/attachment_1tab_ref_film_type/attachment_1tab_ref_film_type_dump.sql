BEGIN TRANSACTION;
CREATE TABLE budget(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount REAL);
INSERT INTO "budget" VALUES(1,500000.0);
INSERT INTO "budget" VALUES(2,700000.0);
CREATE TABLE casts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filmmakingId INTEGER,
    characterName TEXT,
    actorName TEXT,
    FOREIGN KEY(filmmakingId) REFERENCES filmmaking(id));
INSERT INTO "casts" VALUES(1,1,'Himself','John Doe');
INSERT INTO "casts" VALUES(2,2,'Herself','Jane Smith');
INSERT INTO "casts" VALUES(3,3,'Child Prodigy','Michael Johnson');
INSERT INTO "casts" VALUES(4,4,'Leading Role','Emily Brown');
CREATE TABLE crew_members(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    memberName TEXT);
INSERT INTO "crew_members" VALUES(1,'Director','John Doe');
INSERT INTO "crew_members" VALUES(2,'Producer','Alice Davis');
INSERT INTO "crew_members" VALUES(3,'Cinematographer','Charlie Green');
INSERT INTO "crew_members" VALUES(4,'Editor','David Black');
CREATE TABLE film_details(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filmmakingId INTEGER,
    film_typeId INTEGER,
    budgetId INTEGER,
    FOREIGN KEY(filmmakingId) REFERENCES filmmaking(id),
    FOREIGN KEY(film_typeId) REFERENCES film_type(id),
    FOREIGN KEY(budgetId) REFERENCES budget(id));
INSERT INTO "film_details" VALUES(1,1,1,1);
INSERT INTO "film_details" VALUES(2,2,1,2);
INSERT INTO "film_details" VALUES(3,3,2,1);
INSERT INTO "film_details" VALUES(4,4,2,2);
CREATE TABLE film_type(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT);
INSERT INTO "film_type" VALUES(1,'Documentary');
INSERT INTO "film_type" VALUES(2,'Short Film');
CREATE TABLE filmmaking(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    directorName TEXT,
    releaseDate DATE);
INSERT INTO "filmmaking" VALUES(1,'The Documentary','John Doe','2021-09-01');
INSERT INTO "filmmaking" VALUES(2,'Another Documentary','Jane Smith','2021-08-30');
INSERT INTO "filmmaking" VALUES(3,'A Short Film','Michael Johnson','2021-07-26');
INSERT INTO "filmmaking" VALUES(4,'Yet Another Short Film','Emily Brown','2021-07-25');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('filmmaking',4);
INSERT INTO "sqlite_sequence" VALUES('film_type',2);
INSERT INTO "sqlite_sequence" VALUES('budget',2);
INSERT INTO "sqlite_sequence" VALUES('film_details',4);
INSERT INTO "sqlite_sequence" VALUES('crew_members',4);
INSERT INTO "sqlite_sequence" VALUES('casts',4);
COMMIT;
