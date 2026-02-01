BEGIN TRANSACTION;
CREATE TABLE Games(
    gameID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    genre TEXT,
    releaseDate DATE);
INSERT INTO "Games" VALUES(1,'The Witcher III','RPG','2015-05-19');
INSERT INTO "Games" VALUES(2,'Red Dead Redemption II','Action Adventure','2018-10-26');
INSERT INTO "Games" VALUES(3,'Grand Theft Auto V','Action Adventure','2013-09-17');
INSERT INTO "Games" VALUES(4,'Fortnite Battle Royale','Battle Royal','2017-09-26');
INSERT INTO "Games" VALUES(5,'Call of Duty Modern Warfare','First Person Shooter','2019-10-25');
CREATE TABLE Games_Platforms(
    gamesID INTEGER,
    platformsID INTEGER,
    FOREIGN KEY (gamesID) REFERENCES Games (gameID),
    FOREIGN KEY (platformsID) REFERENCES Platforms (platformID));
INSERT INTO "Games_Platforms" VALUES(1,1);
INSERT INTO "Games_Platforms" VALUES(2,1);
INSERT INTO "Games_Platforms" VALUES(3,1);
INSERT INTO "Games_Platforms" VALUES(4,1);
INSERT INTO "Games_Platforms" VALUES(5,1);
INSERT INTO "Games_Platforms" VALUES(5,2);
CREATE TABLE Platforms(
    platformID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Platforms" VALUES(1,'Pc');
INSERT INTO "Platforms" VALUES(2,'PlayStation 4');
INSERT INTO "Platforms" VALUES(3,'Xbox One');
INSERT INTO "Platforms" VALUES(4,'Nintendo Switch');
INSERT INTO "Platforms" VALUES(5,'Mobile');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Platforms',5);
INSERT INTO "sqlite_sequence" VALUES('Games',5);
COMMIT;
