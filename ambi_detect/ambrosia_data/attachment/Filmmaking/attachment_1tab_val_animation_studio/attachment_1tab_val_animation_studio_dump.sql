BEGIN TRANSACTION;
CREATE TABLE Actor(
    id INTEGER PRIMARY KEY,
    name TEXT,
    birthYear INTEGER
);
INSERT INTO "Actor" VALUES(1,'Tom Hanks',1954);
INSERT INTO "Actor" VALUES(2,'Tim Allen',1953);
INSERT INTO "Actor" VALUES(3,'David Hyde Pierce',1951);
INSERT INTO "Actor" VALUES(4,'Ian Holm',1931);
INSERT INTO "Actor" VALUES(5,'Billy Crystal',1941);
CREATE TABLE Director(
    id INTEGER PRIMARY KEY,
    name TEXT,
    birthYear INTEGER
);
INSERT INTO "Director" VALUES(1,'John Lasseter',1957);
INSERT INTO "Director" VALUES(2,'Andrew Stanton',1961);
INSERT INTO "Director" VALUES(3,'Adam Foucher',1970);
INSERT INTO "Director" VALUES(4,'Chris Buck',1960);
INSERT INTO "Director" VALUES(5,'Erik de la Espriella',1970);
CREATE TABLE Film(
    id INTEGER PRIMARY KEY,
    title TEXT,
    releaseDate DATE,
    duration INTEGER,
    animationStudio TEXT,
    awards TEXT,
    FOREIGN KEY(animationStudio) REFERENCES Studio(name)
);
INSERT INTO "Film" VALUES(1,'Toy Story','1995-03-26',82,'Pixar','Golden Globe');
INSERT INTO "Film" VALUES(2,'Finding Nemo','2003-07-06',100,'Pixar','Oscar');
INSERT INTO "Film" VALUES(3,'Shrek','2001-04-18',88,'DreamWorks','Golden Globe');
INSERT INTO "Film" VALUES(4,'Madagascar','2005-05-04',89,'DreamWorks','Nickelodeon Kids Choice Award');
CREATE TABLE FilmActors(
    filmID INTEGER,
    actorID INTEGER,
    role TEXT,
    FOREIGN KEY(filmID) REFERENCES Film(id),
    FOREIGN KEY(actorID) REFERENCES Actor(id),
    PRIMARY KEY(filmID, actorID)
);
INSERT INTO "FilmActors" VALUES(1,1,'Woody');
INSERT INTO "FilmActors" VALUES(1,2,'Buzz Lightyear');
INSERT INTO "FilmActors" VALUES(2,3,'Marlin');
INSERT INTO "FilmActors" VALUES(2,4,'Dory');
INSERT INTO "FilmActors" VALUES(3,5,'Donkey');
CREATE TABLE FilmDirectors(
    filmID INTEGER,
    directorID INTEGER,
    FOREIGN KEY(filmID) REFERENCES Film(id),
    FOREIGN KEY(directorID) REFERENCES Director(id),
    PRIMARY KEY(filmID, directorID)
);
CREATE TABLE FilmGenres(
    filmID INTEGER,
    genreID INTEGER,
    FOREIGN KEY(filmID) REFERENCES Film(id),
    FOREIGN KEY(genreID) REFERENCES Genre(id),
    PRIMARY KEY(filmID, genreID)
);
CREATE TABLE Genre(
    id INTEGER PRIMARY KEY,
    genreName TEXT
);
INSERT INTO "Genre" VALUES(1,'Animated Comedy');
INSERT INTO "Genre" VALUES(2,'Family Adventure');
INSERT INTO "Genre" VALUES(3,'Action/Adventure');
INSERT INTO "Genre" VALUES(4,'Drama');
INSERT INTO "Genre" VALUES(5,'Mystery');
COMMIT;
