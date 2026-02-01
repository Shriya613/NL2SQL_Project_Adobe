BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birth_date DATE,
    nationality TEXT
);
INSERT INTO "Actor" VALUES(1,'Tim Robbins','1958-10-16','American');
INSERT INTO "Actor" VALUES(2,'Morgan Freeman','1937-06-01','American');
INSERT INTO "Actor" VALUES(3,'John Travolta','1954-02-18','American');
INSERT INTO "Actor" VALUES(4,'Liam Neeson','1952-06-07','Irish');
INSERT INTO "Actor" VALUES(5,'Leonardo DiCaprio','1974-11-11','American');
CREATE TABLE Director (
    director_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birth_date DATE,
    nationality TEXT
);
INSERT INTO "Director" VALUES(1,'Frank Darabont','1959-01-28','American');
INSERT INTO "Director" VALUES(2,'Robert Zemeckis','1951-05-14','American');
INSERT INTO "Director" VALUES(3,'Quentin Tarantino','1963-03-27','American');
INSERT INTO "Director" VALUES(4,'Steven Spielberg','1946-12-18','American');
INSERT INTO "Director" VALUES(5,'Chris Nolan','1970-07-30','English');
CREATE TABLE Film (
    film_id INTEGER PRIMARY KEY AUTOINCREMENT,
    director_id INTEGER,
    movie_id INTEGER,
    FOREIGN KEY(director_id) REFERENCES Director(director_id),
    FOREIGN KEY(movie_id) REFERENCES Movie(id)
);
INSERT INTO "Film" VALUES(1,1,1);
INSERT INTO "Film" VALUES(2,2,2);
INSERT INTO "Film" VALUES(3,3,3);
INSERT INTO "Film" VALUES(4,4,4);
INSERT INTO "Film" VALUES(5,5,5);
CREATE TABLE Genre (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    genre_name TEXT UNIQUE
);
INSERT INTO "Genre" VALUES(1,'Drama');
INSERT INTO "Genre" VALUES(2,'Thriller');
INSERT INTO "Genre" VALUES(3,'Comedy');
INSERT INTO "Genre" VALUES(4,'Action');
INSERT INTO "Genre" VALUES(5,'Horror');
CREATE TABLE Movie (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE,
    Total_length INTEGER,
    Number_of_acts INTEGER,
    Theater_play BOOL
);
INSERT INTO "Movie" VALUES(1,'The Shawshank Redemption','1994-09-23',142,2,1);
INSERT INTO "Movie" VALUES(2,'Forrest Gump','1994-06-23',142,1,1);
INSERT INTO "Movie" VALUES(3,'Pulp Fiction','1994-10-14',154,7,0);
INSERT INTO "Movie" VALUES(4,'Schindler''s List','1993-12-15',195,3,1);
INSERT INTO "Movie" VALUES(5,'Inception','2010-07-16',148,3,0);
CREATE TABLE Role (
    movie_id INTEGER,
    actor_id INTEGER,
    role_name TEXT,
    FOREIGN KEY(movie_id) REFERENCES Movie(id),
    FOREIGN KEY(actor_id) REFERENCES Actor(id)
);
INSERT INTO "Role" VALUES(1,1,'Andy DuFresne');
INSERT INTO "Role" VALUES(1,2,'Red');
INSERT INTO "Role" VALUES(2,3,'Benjamin Buford Blue');
INSERT INTO "Role" VALUES(3,4,'Jules Winnfield');
INSERT INTO "Role" VALUES(4,5,'Christian Bodhi');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Movie',5);
INSERT INTO "sqlite_sequence" VALUES('Actor',5);
INSERT INTO "sqlite_sequence" VALUES('Genre',5);
INSERT INTO "sqlite_sequence" VALUES('Director',5);
INSERT INTO "sqlite_sequence" VALUES('Film',5);
COMMIT;
