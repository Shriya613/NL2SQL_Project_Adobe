BEGIN TRANSACTION;
CREATE TABLE Actors(
    actorID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT);
INSERT INTO "Actors" VALUES(1,'Leonardo','DiCaprio');
INSERT INTO "Actors" VALUES(2,'Tom','Hanks');
INSERT INTO "Actors" VALUES(3,'Meryl','Streep');
INSERT INTO "Actors" VALUES(4,'Brad','Pitt');
INSERT INTO "Actors" VALUES(5,'Kevin','Spacey');
CREATE TABLE Director_Roles(
    directorID INTEGER,
    movieID INTEGER,
    FOREIGN KEY(directorID) REFERENCES Directors(directorID),
    FOREIGN KEY(movieID) REFERENCES Movies(movieID));
INSERT INTO "Director_Roles" VALUES(1,1);
INSERT INTO "Director_Roles" VALUES(2,2);
INSERT INTO "Director_Roles" VALUES(3,3);
INSERT INTO "Director_Roles" VALUES(4,4);
INSERT INTO "Director_Roles" VALUES(5,5);
CREATE TABLE Directors(
    directorID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT);
INSERT INTO "Directors" VALUES(1,'Christopher','Nolan');
INSERT INTO "Directors" VALUES(2,'Quentin','Tarantino');
INSERT INTO "Directors" VALUES(3,'Steven','Spielberg');
INSERT INTO "Directors" VALUES(4,'Guillermo','del Toro');
INSERT INTO "Directors" VALUES(5,'James','Cameron');
CREATE TABLE Movies(
    movieID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    genre TEXT,
    duration INT,
    productionHouseID INTEGER,
    FOREIGN KEY(productionHouseID) REFERENCES Production_houses(houseID));
INSERT INTO "Movies" VALUES(1,'Inception','Action Movie',120,1);
INSERT INTO "Movies" VALUES(2,'Pulp Fiction','Action Movie',180,2);
INSERT INTO "Movies" VALUES(3,'Notting Hill','Romantic Comedy',120,3);
INSERT INTO "Movies" VALUES(4,'Avatar','Romantic Comedy',190,4);
INSERT INTO "Movies" VALUES(5,'The Dark Knight','Action Movie',160,1);
INSERT INTO "Movies" VALUES(6,'Indiana Jones and the Last Crusade','Action Movie',130,2);
INSERT INTO "Movies" VALUES(7,'Pretty Woman','Romantic Comedy',120,3);
INSERT INTO "Movies" VALUES(8,'Forrest Gump','Romantic Comedy',170,4);
CREATE TABLE Production_houses(
    houseID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Production_houses" VALUES(1,'Warner Bros');
INSERT INTO "Production_houses" VALUES(2,'Disney');
INSERT INTO "Production_houses" VALUES(3,'Universal Pictures');
INSERT INTO "Production_houses" VALUES(4,'Paramount Pictures');
INSERT INTO "Production_houses" VALUES(5,'Fox');
CREATE TABLE Roles(
    roleID INTEGER PRIMARY KEY AUTOINCREMENT,
    actorID INTEGER,
    movieID INTEGER,
    characterType TEXT,
    FOREIGN KEY(actorID) REFERENCES Actors(actorID),
    FOREIGN KEY(movieID) REFERENCES Movies(movieID));
INSERT INTO "Roles" VALUES(1,1,1,'Protagonist');
INSERT INTO "Roles" VALUES(2,2,2,'Antagonist');
INSERT INTO "Roles" VALUES(3,3,3,'Leading Lady');
INSERT INTO "Roles" VALUES(4,4,4,'Supporting Actor');
INSERT INTO "Roles" VALUES(5,5,5,'Villain');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Production_houses',5);
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Movies',8);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('Roles',5);
COMMIT;
