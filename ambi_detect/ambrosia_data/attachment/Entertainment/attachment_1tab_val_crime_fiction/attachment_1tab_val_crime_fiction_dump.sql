BEGIN TRANSACTION;
CREATE TABLE Actor(
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    dateOfBirth DATE
);
INSERT INTO "Actor" VALUES(1,'Tom','Hanks','1956-07-09');
INSERT INTO "Actor" VALUES(2,'Denzel','Washington','1954-12-28');
INSERT INTO "Actor" VALUES(3,'Scarlett','Johansson','1984-11-22');
INSERT INTO "Actor" VALUES(4,'Leonardo','DiCaprio','1974-11-11');
INSERT INTO "Actor" VALUES(5,'Mark','Ruffalo','1967-11-22');
CREATE TABLE Cast(
    movieId INTEGER REFERENCES Movie(id),
    actorId INTEGER REFERENCES Actor(id),
    role TEXT,
    FOREIGN KEY(movieId, actorId) REFERENCES Movie(id, id)
);
INSERT INTO "Cast" VALUES(1,1,'Lead Role');
INSERT INTO "Cast" VALUES(1,2,'Supporting Role');
INSERT INTO "Cast" VALUES(2,3,'Lead Role');
INSERT INTO "Cast" VALUES(2,4,'Supporting Role');
INSERT INTO "Cast" VALUES(3,5,'Lead Role');
INSERT INTO "Cast" VALUES(3,6,'Supporting Role');
CREATE TABLE Director(
    id INTEGER PRIMARY KEY,
    name TEXT,
    birthYear INT
);
INSERT INTO "Director" VALUES(1,'John Doe',1960);
INSERT INTO "Director" VALUES(2,'Jane Smith',1972);
INSERT INTO "Director" VALUES(3,'Alice Brown',1983);
INSERT INTO "Director" VALUES(4,'Robert Johnson',1990);
INSERT INTO "Director" VALUES(5,'Emma Wilson',1995);
CREATE TABLE Genre(
    id INTEGER PRIMARY KEY,
    genreText TEXT UNIQUE
);
INSERT INTO "Genre" VALUES(1,'Action');
INSERT INTO "Genre" VALUES(2,'Adventure');
INSERT INTO "Genre" VALUES(3,'Drama');
INSERT INTO "Genre" VALUES(4,'Family');
INSERT INTO "Genre" VALUES(5,'Fantasy');
CREATE TABLE Movie(
    id INTEGER PRIMARY KEY,
    title TEXT,
    releaseDate DATE,
    Crime_Fiction TEXT ,
    Studio TEXT,
    directorId INTEGER REFERENCES Director(id)
);
INSERT INTO "Movie" VALUES(1,'MovieA','2000-01-01','Mystery','Warner Bros',1);
INSERT INTO "Movie" VALUES(2,'MovieB','2000-01-01','Mystery','Universal Studios',2);
INSERT INTO "Movie" VALUES(3,'MovieC','2000-01-01','Thriller','Warner Bros',3);
INSERT INTO "Movie" VALUES(4,'MovieD','2000-01-01','Thriller','Disney',4);
INSERT INTO "Movie" VALUES(5,'MovieE','2000-01-01','Comedy','Columbia Pictures',5);
CREATE TABLE MovieGenre(
    movieId INTEGER REFERENCES Movie(id),
    genreId INTEGER REFERENCES Genre(id),
    PRIMARY KEY(movieId, genreId)
);
INSERT INTO "MovieGenre" VALUES(1,1);
INSERT INTO "MovieGenre" VALUES(1,2);
INSERT INTO "MovieGenre" VALUES(2,3);
INSERT INTO "MovieGenre" VALUES(3,4);
INSERT INTO "MovieGenre" VALUES(4,5);
COMMIT;
