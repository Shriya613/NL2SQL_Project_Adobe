BEGIN TRANSACTION;
CREATE TABLE Actor (
    actor_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    date_of_birth DATE,
    gender CHAR(1) 
);
INSERT INTO "Actor" VALUES(1,'Leonardo','DiCaprio','1974-11-11','M');
INSERT INTO "Actor" VALUES(2,'Emma','Watson','1990-04-15','F');
CREATE TABLE Director (
    director_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    birthdate DATE
);
INSERT INTO "Director" VALUES(1,'Christopher','Nolan','1970-06-20');
INSERT INTO "Director" VALUES(2,'Pierre','Perifel','1962-01-20');
CREATE TABLE MovieActor (
    movie_actor_id INTEGER PRIMARY KEY,
    movie_id INTEGER,
    actor_id INTEGER,
    role TEXT,
    FOREIGN KEY(movie_id) REFERENCES Movies(movie_id),
    FOREIGN KEY(actor_id) REFERENCES Actor(actor_id)
);
INSERT INTO "MovieActor" VALUES(1,1,1,'Cobb');
INSERT INTO "MovieActor" VALUES(2,1,2,'Arthur');
CREATE TABLE MovieDirector (
    movie_director_id INTEGER PRIMARY KEY,
    movie_id INTEGER,
    director_id INTEGER,
    FOREIGN KEY(movie_id) REFERENCES Movies(movie_id),
    FOREIGN KEY(director_id) REFERENCES Director(director_id)
);
INSERT INTO "MovieDirector" VALUES(1,1,1);
INSERT INTO "MovieDirector" VALUES(2,2,2);
CREATE TABLE "Movies" (
    movie_id INTEGER PRIMARY KEY,
    title TEXT,
    release_year INTEGER,
    genre TEXT,
    studio_id INTEGER,
    FOREIGN KEY(studio_id) REFERENCES "Production"(studio_id));
INSERT INTO "Movies" VALUES(1,'Inception',2010,'Action/Adventure',1);
INSERT INTO "Movies" VALUES(2,'Minions',2015,'Animation',2);
CREATE TABLE "Production" (
    studio_id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT
);
INSERT INTO "Production" VALUES(1,'Warner Bros','Burbank, California');
INSERT INTO "Production" VALUES(2,'Universal Studios','Universal City, California');
INSERT INTO "Production" VALUES(3,'Disney Studios','Glendale, California');
COMMIT;
