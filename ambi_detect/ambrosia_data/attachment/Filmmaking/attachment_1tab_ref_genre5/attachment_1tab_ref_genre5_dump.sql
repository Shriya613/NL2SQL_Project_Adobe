BEGIN TRANSACTION;
CREATE TABLE Actors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Actors" VALUES(1,'Actor X');
INSERT INTO "Actors" VALUES(2,'Actor Y');
INSERT INTO "Actors" VALUES(3,'Actor Z');
INSERT INTO "Actors" VALUES(4,'Actor W');
INSERT INTO "Actors" VALUES(5,'Actor V');
CREATE TABLE Directors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Directors" VALUES(1,'John Doe');
INSERT INTO "Directors" VALUES(2,'Jane Smith');
INSERT INTO "Directors" VALUES(3,'Michael Johnson');
INSERT INTO "Directors" VALUES(4,'Emily Davis');
INSERT INTO "Directors" VALUES(5,'David Brown');
CREATE TABLE FilmGenres(
    film_id INTEGER,
    genre TEXT ,
    PRIMARY KEY (film_id, genre),
    FOREIGN KEY (film_id) REFERENCES Films(id));
INSERT INTO "FilmGenres" VALUES(1,'Animation');
INSERT INTO "FilmGenres" VALUES(2,'Animation');
INSERT INTO "FilmGenres" VALUES(3,'Comedy');
INSERT INTO "FilmGenres" VALUES(4,'Comedy');
CREATE TABLE Films(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER,
    director_id INTEGER,
    producer_id INTEGER,
    actor_id INTEGER,
    FOREIGN KEY (director_id) REFERENCES Directors(id),
    FOREIGN KEY (producer_id) REFERENCES Producers(id),
    FOREIGN KEY (actor_id) REFERENCES Actors(id));
INSERT INTO "Films" VALUES(1,'Film 1',2021,1,1,1);
INSERT INTO "Films" VALUES(2,'Film 2',2022,2,2,2);
INSERT INTO "Films" VALUES(3,'Film 3',2021,3,3,3);
INSERT INTO "Films" VALUES(4,'Film 4',2023,4,4,4);
INSERT INTO "Films" VALUES(5,'Film 5',2021,1,1,5);
CREATE TABLE Producers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Producers" VALUES(1,'Producer A');
INSERT INTO "Producers" VALUES(2,'Producer B');
INSERT INTO "Producers" VALUES(3,'Producer C');
INSERT INTO "Producers" VALUES(4,'Producer D');
INSERT INTO "Producers" VALUES(5,'Producer E');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Producers',5);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('Films',5);
COMMIT;
