BEGIN TRANSACTION;
CREATE TABLE AnimationStudios(
    id INTEGER PRIMARY KEY,
    studioName TEXT UNIQUE);
INSERT INTO "AnimationStudios" VALUES(1,'Pixar');
INSERT INTO "AnimationStudios" VALUES(2,'DreamWorks');
CREATE TABLE Awards(
    id INTEGER PRIMARY KEY,
    awardName TEXT UNIQUE);
INSERT INTO "Awards" VALUES(1,'Golden Globe');
INSERT INTO "Awards" VALUES(2,'Oscar');
INSERT INTO "Awards" VALUES(3,'BAFTA');
INSERT INTO "Awards" VALUES(4,'Grammy');
CREATE TABLE Genres(
    id INTEGER PRIMARY KEY,
    genreName TEXT UNIQUE);
INSERT INTO "Genres" VALUES(1,'Adventure');
INSERT INTO "Genres" VALUES(2,'Comedy');
INSERT INTO "Genres" VALUES(3,'Family');
INSERT INTO "Genres" VALUES(4,'Animated');
CREATE TABLE MovieAwards(
    id INTEGER PRIMARY KEY,
    movieId INTEGER,
    awardId INTEGER,
    FOREIGN KEY (movieId) REFERENCES MovieDetails(id),
    FOREIGN KEY (awardId) REFERENCES Awards(id));
INSERT INTO "MovieAwards" VALUES(1,1,1);
INSERT INTO "MovieAwards" VALUES(2,1,3);
INSERT INTO "MovieAwards" VALUES(3,2,1);
INSERT INTO "MovieAwards" VALUES(4,2,2);
INSERT INTO "MovieAwards" VALUES(5,3,2);
INSERT INTO "MovieAwards" VALUES(6,4,3);
INSERT INTO "MovieAwards" VALUES(7,4,4);
CREATE TABLE MovieDetails(
    id INTEGER PRIMARY KEY,
    movieTitle TEXT,
    releaseDate DATE,
    animationStudioId INTEGER,
    FOREIGN KEY (animationStudioId) REFERENCES AnimationStudios(id));
INSERT INTO "MovieDetails" VALUES(1,'Toy Story','2019-06-21',1);
INSERT INTO "MovieDetails" VALUES(2,'Finding Nemo','2003-07-18',1);
INSERT INTO "MovieDetails" VALUES(3,'Shrek','2001-04-18',2);
INSERT INTO "MovieDetails" VALUES(4,'Madagascar','2005-05-27',2);
CREATE TABLE MoviesGenre(
    id INTEGER PRIMARY KEY,
    movieId INTEGER,
    genreId INTEGER,
    FOREIGN KEY (movieId) REFERENCES MovieDetails(id),
    FOREIGN KEY (genreId) REFERENCES Genres(id));
INSERT INTO "MoviesGenre" VALUES(1,1,1);
INSERT INTO "MoviesGenre" VALUES(2,1,2);
INSERT INTO "MoviesGenre" VALUES(3,2,2);
INSERT INTO "MoviesGenre" VALUES(4,2,3);
INSERT INTO "MoviesGenre" VALUES(5,3,1);
INSERT INTO "MoviesGenre" VALUES(6,3,4);
INSERT INTO "MoviesGenre" VALUES(7,4,3);
INSERT INTO "MoviesGenre" VALUES(8,4,4);
COMMIT;
