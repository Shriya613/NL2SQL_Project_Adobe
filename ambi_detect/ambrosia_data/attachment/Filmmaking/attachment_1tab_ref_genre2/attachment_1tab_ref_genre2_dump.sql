BEGIN TRANSACTION;
CREATE TABLE Film_Genres(
    film_genre_id INTEGER,
    film_id INTEGER,
    FOREIGN KEY (film_genre_id) REFERENCES Genres(id),
    FOREIGN KEY (film_id) REFERENCES Films(film_id));
INSERT INTO "Film_Genres" VALUES(1,1);
INSERT INTO "Film_Genres" VALUES(1,2);
INSERT INTO "Film_Genres" VALUES(2,3);
INSERT INTO "Film_Genres" VALUES(2,4);
CREATE TABLE Film_Stars(
    film_star_id INTEGER,
    film_id INTEGER,
    role TEXT,
    FOREIGN KEY (film_star_id) REFERENCES Stars(id),
    FOREIGN KEY (film_id) REFERENCES Films(film_id));
INSERT INTO "Film_Stars" VALUES(1,1,'Ethan Hunt');
INSERT INTO "Film_Stars" VALUES(2,2,'John Wick');
INSERT INTO "Film_Stars" VALUES(1,3,'Jack Harper');
INSERT INTO "Film_Stars" VALUES(3,4,'Ray Gaines');
CREATE TABLE Films(
    film_id INTEGER PRIMARY KEY,
    title TEXT,
    release_year INTEGER,
    director_name TEXT,
    country TEXT,
    runtime INTEGER);
INSERT INTO "Films" VALUES(1,'Mission Impossible',2018,'Christopher McQuarrie','USA',147);
INSERT INTO "Films" VALUES(2,'John Wick',2014,'Chad Stahelski & David Leitch','USA',106);
INSERT INTO "Films" VALUES(3,'Oblivion',2013,'Joseph Kosinski','USA',124);
INSERT INTO "Films" VALUES(4,'San Andreas',2015,'Brad Peyton','USA',119);
CREATE TABLE Genres(
    id INTEGER PRIMARY KEY,
    genre TEXT);
INSERT INTO "Genres" VALUES(1,'Action');
INSERT INTO "Genres" VALUES(2,'Adventure');
CREATE TABLE Stars(
    id INTEGER PRIMARY KEY,
    star_name TEXT);
INSERT INTO "Stars" VALUES(1,'Tom Cruise');
INSERT INTO "Stars" VALUES(2,'Keanu Reeves');
INSERT INTO "Stars" VALUES(3,'Dwayne Johnson');
COMMIT;
