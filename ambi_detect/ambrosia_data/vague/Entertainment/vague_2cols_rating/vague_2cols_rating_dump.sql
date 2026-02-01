BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT
);
INSERT INTO "Actor" VALUES(1,'Tom','Hanks');
INSERT INTO "Actor" VALUES(2,'Scarlett','Johansson');
INSERT INTO "Actor" VALUES(3,'Leonardo','DiCaprio');
INSERT INTO "Actor" VALUES(4,'Brad','Pitt');
INSERT INTO "Actor" VALUES(5,'Angelina','Jolie');
CREATE TABLE Director (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT
);
INSERT INTO "Director" VALUES(1,'John','Doe');
INSERT INTO "Director" VALUES(2,'Jane','Smith');
INSERT INTO "Director" VALUES(3,'Michael','Jackson');
INSERT INTO "Director" VALUES(4,'Emma','Watson');
INSERT INTO "Director" VALUES(5,'Chris','Evans');
CREATE TABLE Genre (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "Genre" VALUES(1,'Action');
INSERT INTO "Genre" VALUES(2,'Comedy');
INSERT INTO "Genre" VALUES(3,'Drama');
INSERT INTO "Genre" VALUES(4,'Horror');
INSERT INTO "Genre" VALUES(5,'Romance');
CREATE TABLE Movie (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    releaseDate DATE,
    genreId INTEGER,
    directorId INTEGER,
    FOREIGN KEY(genreId) REFERENCES Genre(id),
    FOREIGN KEY(directorId) REFERENCES Director(id)
);
INSERT INTO "Movie" VALUES(1,'Inception','2010-07-16',1,1);
INSERT INTO "Movie" VALUES(2,'Interstellar','2014-11-07',3,2);
INSERT INTO "Movie" VALUES(3,'The Dark Knight','2008-07-18',1,3);
INSERT INTO "Movie" VALUES(4,'Avengers: Endgame','2019-04-26',2,4);
INSERT INTO "Movie" VALUES(5,'Titanic','1997-12-19',4,5);
CREATE TABLE Rating (
    movieId INTEGER,
    ageRating INTEGER,
    criticalRating FLOAT,
    tvShow BOOLEAN,
    FOREIGN KEY(movieId) REFERENCES Movie(id)
);
INSERT INTO "Rating" VALUES(1,18,9.0,0);
INSERT INTO "Rating" VALUES(2,12,8.5,0);
INSERT INTO "Rating" VALUES(3,16,7.5,0);
INSERT INTO "Rating" VALUES(4,12,8.0,0);
INSERT INTO "Rating" VALUES(5,15,7.0,0);
CREATE TABLE Review (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reviewText TEXT,
    rating INTEGER ,
    actorId INTEGER,
    movieId INTEGER,
    FOREIGN KEY(actorId) REFERENCES Actor(id),
    FOREIGN KEY(movieId) REFERENCES Movie(id)
);
INSERT INTO "Review" VALUES(1,'Great acting and a thrilling plot.',5,1,1);
INSERT INTO "Review" VALUES(2,'A masterpiece of cinema.',5,2,2);
INSERT INTO "Review" VALUES(3,'Amazing visuals but weak storyline.',3,3,3);
INSERT INTO "Review" VALUES(4,'An emotional rollercoaster.',5,4,4);
INSERT INTO "Review" VALUES(5,'Too long and not as good as expected.',2,5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Genre',5);
INSERT INTO "sqlite_sequence" VALUES('Director',5);
INSERT INTO "sqlite_sequence" VALUES('Actor',5);
INSERT INTO "sqlite_sequence" VALUES('Movie',5);
INSERT INTO "sqlite_sequence" VALUES('Review',5);
COMMIT;
