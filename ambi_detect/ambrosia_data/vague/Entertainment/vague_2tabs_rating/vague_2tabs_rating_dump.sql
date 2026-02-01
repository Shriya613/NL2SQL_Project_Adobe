BEGIN TRANSACTION;
CREATE TABLE Actor(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT
);
INSERT INTO "Actor" VALUES(1,'Christopher','Nolan');
INSERT INTO "Actor" VALUES(2,'Matthew','McConaughey');
CREATE TABLE Age_Rating(
    ratingId INTEGER PRIMARY KEY AUTOINCREMENT,
    ageMinimum INTEGER 
);
INSERT INTO "Age_Rating" VALUES(1,18);
INSERT INTO "Age_Rating" VALUES(2,15);
CREATE TABLE Cast(
    movieId INTEGER,
    actorId INTEGER,
    role TEXT,
    FOREIGN KEY (movieId) REFERENCES Movie(id),
    FOREIGN KEY (actorId) REFERENCES Actor(id)
);
INSERT INTO "Cast" VALUES(1,1,'Director');
INSERT INTO "Cast" VALUES(1,2,'Cobb');
CREATE TABLE Critical_Rating(
    ratingId INTEGER PRIMARY KEY AUTOINCREMENT,
    score INTEGER 
);
INSERT INTO "Critical_Rating" VALUES(1,92);
INSERT INTO "Critical_Rating" VALUES(2,89);
CREATE TABLE Genre(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    genreText TEXT UNIQUE
);
INSERT INTO "Genre" VALUES(1,'Action');
INSERT INTO "Genre" VALUES(2,'Adventure');
CREATE TABLE Movie(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    releaseYear INTEGER,
    directorId INTEGER,
    FOREIGN KEY (directorId) REFERENCES Director(id) ON DELETE SET NULL
);
INSERT INTO "Movie" VALUES(1,'Inception',2010,1);
INSERT INTO "Movie" VALUES(2,'Interstellar',2014,2);
CREATE TABLE Movie_Genre(
    movieId INTEGER,
    genreId INTEGER,
    FOREIGN KEY (movieId) REFERENCES Movie(id),
    FOREIGN KEY (genreId) REFERENCES Genre(id)
);
INSERT INTO "Movie_Genre" VALUES(1,1);
INSERT INTO "Movie_Genre" VALUES(2,2);
CREATE TABLE Tv_Show(
    showId INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    network TEXT,
    premiereDate DATE,
    ageRatingId INTEGER,
    criticalScoreId INTEGER,
    FOREIGN KEY (ageRatingId) REFERENCES Age_Rating(ratingId),
    FOREIGN KEY (criticalScoreId) REFERENCES Critical_Rating(ratingId)
);
INSERT INTO "Tv_Show" VALUES(1,'Breaking Bad','AMC','2008-01-20',1,1);
INSERT INTO "Tv_Show" VALUES(2,'Game of Thrones','HBO','2011-04-17',2,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Movie',2);
INSERT INTO "sqlite_sequence" VALUES('Actor',2);
INSERT INTO "sqlite_sequence" VALUES('Genre',2);
INSERT INTO "sqlite_sequence" VALUES('Age_Rating',2);
INSERT INTO "sqlite_sequence" VALUES('Critical_Rating',2);
INSERT INTO "sqlite_sequence" VALUES('Tv_Show',2);
COMMIT;
