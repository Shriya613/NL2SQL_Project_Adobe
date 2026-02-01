BEGIN TRANSACTION;
CREATE TABLE Actor(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthDate DATE,
    nationality TEXT);
INSERT INTO "Actor" VALUES(1,'John Doe','1980-06-27','American');
INSERT INTO "Actor" VALUES(2,'Jane Smith','1985-11-30','British');
INSERT INTO "Actor" VALUES(3,'Alice Brown','1990-02-14','Canadian');
INSERT INTO "Actor" VALUES(4,'Robert White','1975-04-05','Australian');
INSERT INTO "Actor" VALUES(5,'Emma Green','1992-07-09','New Zealander');
CREATE TABLE Directed(
    movieId INTEGER,
    directorId INTEGER,
    year INTEGER,
    FOREIGN KEY (movieId) REFERENCES Movie(id),
    FOREIGN KEY (directorId) REFERENCES Director(id));
INSERT INTO "Directed" VALUES(1,1,2001);
INSERT INTO "Directed" VALUES(2,2,2010);
INSERT INTO "Directed" VALUES(3,3,2006);
INSERT INTO "Directed" VALUES(4,4,2012);
INSERT INTO "Directed" VALUES(5,5,2018);
CREATE TABLE Director(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthDate DATE,
    nationality TEXT);
INSERT INTO "Director" VALUES(1,'Peter Jackson','1961-10-31','New Zealand');
INSERT INTO "Director" VALUES(2,'Christopher Nolan','1970-07-30','English');
INSERT INTO "Director" VALUES(3,'Guillermo del Toro','1964-09-09','Spanish');
INSERT INTO "Director" VALUES(4,'Quentin Tarantino','1963-03-27','American');
INSERT INTO "Director" VALUES(5,'Ava DuVernay','1978-08-24','American');
CREATE TABLE Movie(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    genre TEXT,
    releaseYear INTEGER,
    director TEXT);
INSERT INTO "Movie" VALUES(1,'The Lord of The Rings: Fellowship of The Ring','Fantasy',2001,'1');
INSERT INTO "Movie" VALUES(2,'Inception','Action/Adventure',2010,'2');
INSERT INTO "Movie" VALUES(3,'Pan''s Labyrinth','Fantasy',2006,'3');
INSERT INTO "Movie" VALUES(4,'Django Unchained','Western',2012,'4');
INSERT INTO "Movie" VALUES(5,'A Wrinkle In Time','Fantasy',2018,'5');
CREATE TABLE Role(
    movieId INTEGER,
    actorId INTEGER,
    roleType TEXT,
    FOREIGN KEY (movieId) REFERENCES Movie(id),
    FOREIGN KEY (actorId) REFERENCES Actor(id));
INSERT INTO "Role" VALUES(1,1,'Lead');
INSERT INTO "Role" VALUES(2,2,'Supporting');
INSERT INTO "Role" VALUES(3,3,'Protagonist');
INSERT INTO "Role" VALUES(4,4,'Antagonist');
INSERT INTO "Role" VALUES(5,5,'Recurring');
CREATE TABLE Tv_Show(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    showName TEXT,
    genre TEXT,
    episodeCount INTEGER,
    network TEXT);
INSERT INTO "Tv_Show" VALUES(1,'Friends','Comedy',236,'NBC');
INSERT INTO "Tv_Show" VALUES(2,'Breaking Bad','Drama',62,'AMC');
INSERT INTO "Tv_Show" VALUES(3,'Game Of Thrones','Fantasy',73,'HBO');
INSERT INTO "Tv_Show" VALUES(4,'Stranger Things','Science Fiction',47,'Netflix');
INSERT INTO "Tv_Show" VALUES(5,'Black Mirror','Anthology',22,'Channel 4');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Actor',5);
INSERT INTO "sqlite_sequence" VALUES('Director',5);
INSERT INTO "sqlite_sequence" VALUES('Movie',5);
INSERT INTO "sqlite_sequence" VALUES('Tv_Show',5);
COMMIT;
