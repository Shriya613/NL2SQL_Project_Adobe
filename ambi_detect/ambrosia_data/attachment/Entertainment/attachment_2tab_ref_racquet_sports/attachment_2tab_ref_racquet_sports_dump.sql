BEGIN TRANSACTION;
CREATE TABLE Golf_Players(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    playerName TEXT,
    golfTournaments INT REFERENCES Golf_Tournaments(id));
INSERT INTO "Golf_Players" VALUES(1,'Tiger Woods',79);
INSERT INTO "Golf_Players" VALUES(2,'Phil Mickelson',46);
INSERT INTO "Golf_Players" VALUES(3,'Rory McIlroy',19);
INSERT INTO "Golf_Players" VALUES(4,'Jordan Spieth',14);
INSERT INTO "Golf_Players" VALUES(5,'Justin Rose',20);
CREATE TABLE Golf_Tournaments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournamentName TEXT,
    sponsorID INT REFERENCES Sponsors(id));
INSERT INTO "Golf_Tournaments" VALUES(1,'Masters Tournament',1);
INSERT INTO "Golf_Tournaments" VALUES(2,'US Open',2);
INSERT INTO "Golf_Tournaments" VALUES(3,'The Open Championship',1);
INSERT INTO "Golf_Tournaments" VALUES(4,'PGA Championship',2);
INSERT INTO "Golf_Tournaments" VALUES(5,'BMW PGA Championship',1);
CREATE TABLE Matches(
    matchId INTEGER PRIMARY KEY AUTOINCREMENT,
    tennisPlayerID INT REFERENCES Tennis_Players(id),
    golferID INT REFERENCES Golf_Players(id),
    score TEXT,
    date DATE);
INSERT INTO "Matches" VALUES(1,1,1,'6-3, 7-5','2021-09-11');
INSERT INTO "Matches" VALUES(2,2,2,'6-4, 6-7, 6-4','2021-09-12');
INSERT INTO "Matches" VALUES(3,3,3,'6-3, 6-2','2021-09-13');
INSERT INTO "Matches" VALUES(4,4,4,'6-4, 7-6','2021-09-14');
INSERT INTO "Matches" VALUES(5,5,5,'6-3, 6-3','2021-09-15');
CREATE TABLE Sponsors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
INSERT INTO "Sponsors" VALUES(1,'Nike');
INSERT INTO "Sponsors" VALUES(2,'Adidas');
CREATE TABLE Tennis_Players(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    playerName TEXT,
    tennisTournaments INT REFERENCES Tennis_Tournaments(id));
INSERT INTO "Tennis_Players" VALUES(1,'Roger Federer',10);
INSERT INTO "Tennis_Players" VALUES(2,'Novak Djokovic',20);
INSERT INTO "Tennis_Players" VALUES(3,'Stan Wawrinka',15);
INSERT INTO "Tennis_Players" VALUES(4,'Andy Murray',30);
INSERT INTO "Tennis_Players" VALUES(5,'Dominic Thiem',18);
CREATE TABLE Tennis_Tournaments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournamentName TEXT,
    sponsorID INT REFERENCES Sponsors(id));
INSERT INTO "Tennis_Tournaments" VALUES(1,'US Open',1);
INSERT INTO "Tennis_Tournaments" VALUES(2,'Wimbledon',2);
INSERT INTO "Tennis_Tournaments" VALUES(3,'Australian Open',1);
INSERT INTO "Tennis_Tournaments" VALUES(4,'French Open',2);
INSERT INTO "Tennis_Tournaments" VALUES(5,'ATP Finals',1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Sponsors',2);
INSERT INTO "sqlite_sequence" VALUES('Tennis_Players',5);
INSERT INTO "sqlite_sequence" VALUES('Golf_Players',5);
INSERT INTO "sqlite_sequence" VALUES('Tennis_Tournaments',5);
INSERT INTO "sqlite_sequence" VALUES('Golf_Tournaments',5);
INSERT INTO "sqlite_sequence" VALUES('Matches',5);
COMMIT;
