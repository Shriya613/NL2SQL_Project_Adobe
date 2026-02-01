BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    date_of_birth DATE
);
INSERT INTO "Actor" VALUES(1,'Leonardo','DiCaprio','1974-11-11');
INSERT INTO "Actor" VALUES(2,'Tom','Cruise','1962-07-03');
INSERT INTO "Actor" VALUES(3,'Brad','Pitt','1963-12-18');
INSERT INTO "Actor" VALUES(4,'Joaquin','Phoenix','1974-10-28');
CREATE TABLE Director (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birth_year INT
);
INSERT INTO "Director" VALUES(1,'Christopher Nolan',1970);
INSERT INTO "Director" VALUES(2,'Quentin Tarantino',1963);
INSERT INTO "Director" VALUES(3,'Steven Spielberg',1946);
INSERT INTO "Director" VALUES(4,'James Cameron',1954);
INSERT INTO "Director" VALUES(5,'Martin Scorsese',1942);
CREATE TABLE Genre (
    genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "Genre" VALUES(1,'Action');
INSERT INTO "Genre" VALUES(2,'Drama');
INSERT INTO "Genre" VALUES(3,'Romance');
INSERT INTO "Genre" VALUES(4,'Thriller');
INSERT INTO "Genre" VALUES(5,'Crime');
CREATE TABLE Movie (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE,
    studio_id INTEGER,
    director_id INTEGER,
    FOREIGN KEY(studio_id) REFERENCES Studio(id),
    FOREIGN KEY(director_id) REFERENCES Director(id)
);
INSERT INTO "Movie" VALUES(1,'Inception','2010-07-16',1,1);
INSERT INTO "Movie" VALUES(2,'Pulp Fiction','1994-10-14',2,2);
INSERT INTO "Movie" VALUES(3,'Schindler''s List','1993-12-15',3,3);
INSERT INTO "Movie" VALUES(4,'Titanic','1997-12-18',4,4);
INSERT INTO "Movie" VALUES(5,'Goodfellas','1990-09-22',2,5);
CREATE TABLE MovieGenre (
    movie_id INTEGER,
    genre_id INTEGER,
    FOREIGN KEY(movie_id) REFERENCES Movie(id),
    FOREIGN KEY(genre_id) REFERENCES Genre(genre_id)
);
INSERT INTO "MovieGenre" VALUES(1,1);
INSERT INTO "MovieGenre" VALUES(1,3);
INSERT INTO "MovieGenre" VALUES(2,2);
INSERT INTO "MovieGenre" VALUES(3,4);
INSERT INTO "MovieGenre" VALUES(4,5);
CREATE TABLE Role (
    movie_id INTEGER,
    actor_id INTEGER,
    character TEXT,
    FOREIGN KEY(movie_id) REFERENCES Movie(id),
    FOREIGN KEY(actor_id) REFERENCES Actor(id)
);
INSERT INTO "Role" VALUES(1,1,'Dom Cobb');
INSERT INTO "Role" VALUES(2,2,'Vincent Vega');
INSERT INTO "Role" VALUES(3,3,'Oskar Schindler');
INSERT INTO "Role" VALUES(4,4,'Jack Dawson');
INSERT INTO "Role" VALUES(2,5,'Jules Winnfield');
CREATE TABLE Studio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Studio" VALUES(1,'Warner Bros.','Burbank, CA');
INSERT INTO "Studio" VALUES(2,'Disney','Burbank, CA');
INSERT INTO "Studio" VALUES(3,'Universal Pictures','Universal City, CA');
INSERT INTO "Studio" VALUES(4,'Paramount Pictures','Hollywood, CA');
INSERT INTO "Studio" VALUES(5,'20th Century Fox','Century City, CA');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Studio',5);
INSERT INTO "sqlite_sequence" VALUES('Director',5);
INSERT INTO "sqlite_sequence" VALUES('Movie',5);
INSERT INTO "sqlite_sequence" VALUES('Actor',4);
INSERT INTO "sqlite_sequence" VALUES('Genre',5);
COMMIT;
