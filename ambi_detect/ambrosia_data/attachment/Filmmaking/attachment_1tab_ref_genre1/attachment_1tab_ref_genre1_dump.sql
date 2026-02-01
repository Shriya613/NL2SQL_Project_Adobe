BEGIN TRANSACTION;
CREATE TABLE Directors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT
);
INSERT INTO "Directors" VALUES(1,'John','Doe');
INSERT INTO "Directors" VALUES(2,'Jane','Smith');
CREATE TABLE Durations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    minutes INTEGER UNIQUE
);
INSERT INTO "Durations" VALUES(1,120);
INSERT INTO "Durations" VALUES(2,90);
CREATE TABLE Genres(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "Genres" VALUES(1,'Action Movie');
INSERT INTO "Genres" VALUES(2,'Romantic Comedy');
CREATE TABLE MovieDurations(
    movie_id INTEGER,
    duration_id INTEGER,
    FOREIGN KEY (movie_id) REFERENCES Movies(id),
    FOREIGN KEY (duration_id) REFERENCES Durations(id),
    PRIMARY KEY(movie_id, duration_id)
);
INSERT INTO "MovieDurations" VALUES(1,1);
INSERT INTO "MovieDurations" VALUES(2,2);
INSERT INTO "MovieDurations" VALUES(3,1);
INSERT INTO "MovieDurations" VALUES(4,2);
CREATE TABLE Movies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER,
    genre_id INTEGER,
    director_id INTEGER,
    FOREIGN KEY (genre_id) REFERENCES Genres(id),
    FOREIGN KEY (director_id) REFERENCES Directors(id)
);
INSERT INTO "Movies" VALUES(1,'The Action Movie',2021,1,1);
INSERT INTO "Movies" VALUES(2,'Another Action Movie',2021,1,2);
INSERT INTO "Movies" VALUES(3,'A Romantic Comedy',2021,2,1);
INSERT INTO "Movies" VALUES(4,'Yet Another Romantic Comedy',2021,2,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Genres',2);
INSERT INTO "sqlite_sequence" VALUES('Directors',2);
INSERT INTO "sqlite_sequence" VALUES('Movies',4);
INSERT INTO "sqlite_sequence" VALUES('Durations',2);
COMMIT;
