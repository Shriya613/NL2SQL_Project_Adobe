BEGIN TRANSACTION;
CREATE TABLE Actors (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    date_of_birth DATE,
    gender TEXT 
);
INSERT INTO "Actors" VALUES(1,'John','Travolta','1954-02-18','Male');
INSERT INTO "Actors" VALUES(2,'Uma','Thurman','1970-04-29','Female');
INSERT INTO "Actors" VALUES(3,'Leonardo','DiCaprio','1974-11-11','Male');
INSERT INTO "Actors" VALUES(4,'Ellen','Page','1987-02-21','Female');
INSERT INTO "Actors" VALUES(5,'Jamie','Foxx','1967-12-13','Male');
CREATE TABLE Directors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    birth_date DATE,
    nationality TEXT
);
INSERT INTO "Directors" VALUES(1,'Quentin Tarantino','1963-03-27','American');
INSERT INTO "Directors" VALUES(2,'Christopher Nolan','1970-07-30','British');
INSERT INTO "Directors" VALUES(3,'Martin Scorsese','1942-11-22','American');
INSERT INTO "Directors" VALUES(4,'Ava DuVernay','1982-08-24','American');
INSERT INTO "Directors" VALUES(5,'Guillermo del Toro','1964-09-09','Mexican');
CREATE TABLE FilmCrew (
    id INTEGER PRIMARY KEY,
    member_type TEXT, -- e.g., cinematographer, producer, etc.
    full_name TEXT,
    birth_date DATE
);
CREATE TABLE MovieCast (
    movie_id INTEGER,
    actor_id INTEGER,
    role TEXT,
    FOREIGN KEY(movie_id) REFERENCES Movies(id),
    FOREIGN KEY(actor_id) REFERENCES Actors(id),
    PRIMARY KEY(movie_id, actor_id)
);
INSERT INTO "MovieCast" VALUES(1,1,'Hitman');
INSERT INTO "MovieCast" VALUES(1,2,'Mia Wallace');
INSERT INTO "MovieCast" VALUES(2,3,'Dom Cobb');
INSERT INTO "MovieCast" VALUES(2,4,'Mal');
INSERT INTO "MovieCast" VALUES(3,5,'Ralph Fiennes');
CREATE TABLE MovieCrew (
    movie_id INTEGER,
    crew_member_id INTEGER,
    position_held TEXT,
    FOREIGN KEY(movie_id) REFERENCES Movies(id),
    FOREIGN KEY(crew_member_id) REFERENCES FilmCrew(id),
    PRIMARY KEY(movie_id, crew_member_id)
);
CREATE TABLE "Movies" (
    id INTEGER PRIMARY KEY,
    title TEXT,
    release_year INTEGER,
    genre TEXT,
    Release_Date DATE,
    director_id INTEGER,
    FOREIGN KEY(director_id) REFERENCES Directors(id)
);
INSERT INTO "Movies" VALUES(1,'Pulp Fiction',1994,'Crime/Drama','1994-10-14',1);
INSERT INTO "Movies" VALUES(2,'Inception',2010,'Science Fiction','2010-07-16',2);
INSERT INTO "Movies" VALUES(3,'The Departed',2006,'Crime/Thriller','2006-10-05',3);
INSERT INTO "Movies" VALUES(4,'Selma',2014,'Biographical Drama','2014-12-25',4);
INSERT INTO "Movies" VALUES(5,'Pan''s Labyrinth',2006,'Fantasy','2007-01-19',5);
COMMIT;
