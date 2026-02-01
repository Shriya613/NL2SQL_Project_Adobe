BEGIN TRANSACTION;
CREATE TABLE Actors (
    actorID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT
);
INSERT INTO "Actors" VALUES(1,'Leonardo','DiCaprio');
INSERT INTO "Actors" VALUES(2,'Joseph','Gordon-Levitt');
INSERT INTO "Actors" VALUES(3,'Ellen','Page');
INSERT INTO "Actors" VALUES(4,'Tom','Hardy');
CREATE TABLE Characters (
    characterID INTEGER PRIMARY KEY AUTOINCREMENT,
    roleType TEXT );
INSERT INTO "Characters" VALUES(1,'Protagonist');
INSERT INTO "Characters" VALUES(2,'Antagonist');
INSERT INTO "Characters" VALUES(3,'Supporting Role');
CREATE TABLE Directors (
    directorID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT
);
INSERT INTO "Directors" VALUES(1,'Christopher','Nolan');
INSERT INTO "Directors" VALUES(2,'Quentin','Tarantino');
CREATE TABLE Film_Actors (
    filmID INTEGER,
    actorID INTEGER,
    role TEXT,
    PRIMARY KEY (filmID, actorID),
    FOREIGN KEY (filmID) REFERENCES Films (filmID),
    FOREIGN KEY (actorID) REFERENCES Actors (actorID)
);
INSERT INTO "Film_Actors" VALUES(1,1,'Cobb');
INSERT INTO "Film_Actors" VALUES(1,3,'Arthur');
INSERT INTO "Film_Actors" VALUES(2,1,'Batman');
INSERT INTO "Film_Actors" VALUES(2,2,'Rachel Dawes');
INSERT INTO "Film_Actors" VALUES(3,3,'Cooper');
INSERT INTO "Film_Actors" VALUES(3,2,'Ashley');
INSERT INTO "Film_Actors" VALUES(4,1,'Vincent Vega');
INSERT INTO "Film_Actors" VALUES(4,4,'Jules Winnfield');
CREATE TABLE Films (
    filmID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    releaseYear INTEGER,
    directorID INTEGER,
    FOREIGN KEY (directorID) REFERENCES Directors (directorID)
);
INSERT INTO "Films" VALUES(1,'Inception',2010,1);
INSERT INTO "Films" VALUES(2,'The Dark Knight',2008,1);
INSERT INTO "Films" VALUES(3,'Interstellar',2014,1);
INSERT INTO "Films" VALUES(4,'Pulp Fiction',1994,2);
INSERT INTO "Films" VALUES(5,'Reservoir Dogs',1992,2);
CREATE TABLE Screenplays (
    screenplayID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    director TEXT
);
INSERT INTO "Screenplays" VALUES(1,'Inception','Christopher Nolan');
INSERT INTO "Screenplays" VALUES(2,'The Dark Knight','Christopher Nolan');
INSERT INTO "Screenplays" VALUES(3,'Interstellar','Christopher Nolan');
INSERT INTO "Screenplays" VALUES(4,'Pulp Fiction','Quentin Tarantino');
INSERT INTO "Screenplays" VALUES(5,'Reservoir Dogs','Quentin Tarantino');
CREATE TABLE Screenplays_Characters (
    screenplayID INTEGER,
    characterID INTEGER,
    FOREIGN KEY (screenplayID) REFERENCES Screenplays (screenplayID),
    FOREIGN KEY (characterID) REFERENCES Characters (characterID),
    PRIMARY KEY (screenplayID, characterID)
);
INSERT INTO "Screenplays_Characters" VALUES(1,1);
INSERT INTO "Screenplays_Characters" VALUES(2,1);
INSERT INTO "Screenplays_Characters" VALUES(3,1);
INSERT INTO "Screenplays_Characters" VALUES(4,1);
INSERT INTO "Screenplays_Characters" VALUES(5,1);
INSERT INTO "Screenplays_Characters" VALUES(5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Screenplays',5);
INSERT INTO "sqlite_sequence" VALUES('Characters',6);
INSERT INTO "sqlite_sequence" VALUES('Directors',2);
INSERT INTO "sqlite_sequence" VALUES('Actors',4);
INSERT INTO "sqlite_sequence" VALUES('Films',5);
COMMIT;
