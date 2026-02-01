BEGIN TRANSACTION;
CREATE TABLE Actors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    birthdate DATE
);
INSERT INTO "Actors" VALUES(1,'Morgan Freeman','1937-06-01');
INSERT INTO "Actors" VALUES(2,'John Travolta','1954-02-18');
INSERT INTO "Actors" VALUES(3,'Tom Hanks','1956-07-09');
INSERT INTO "Actors" VALUES(4,'Jack Nicholson','1937-04-22');
INSERT INTO "Actors" VALUES(5,'Kevin Spacey','1959-07-26');
CREATE TABLE Directors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    birthdate DATE
);
INSERT INTO "Directors" VALUES(1,'Frank Darabont','1959-01-28');
INSERT INTO "Directors" VALUES(2,'Quentin Tarantino','1963-03-27');
INSERT INTO "Directors" VALUES(3,'Robert Zemeckis','1951-05-14');
INSERT INTO "Directors" VALUES(4,'Steven Spielberg','1946-12-18');
CREATE TABLE "Duration" (
    "theater_play_id" INTEGER REFERENCES Theater_Plays(theater_play_id),
    total_length INT 
);
INSERT INTO "Duration" VALUES(1,142);
INSERT INTO "Duration" VALUES(2,154);
INSERT INTO "Duration" VALUES(3,140);
INSERT INTO "Duration" VALUES(4,195);
INSERT INTO "Duration" VALUES(5,140);
CREATE TABLE Genres (
    id INTEGER PRIMARY KEY,
    genre_name TEXT UNIQUE
);
INSERT INTO "Genres" VALUES(1,'Drama');
INSERT INTO "Genres" VALUES(2,'Action');
INSERT INTO "Genres" VALUES(3,'Romance');
INSERT INTO "Genres" VALUES(4,'Thriller');
INSERT INTO "Genres" VALUES(5,'Comedy');
CREATE TABLE Movies (
    id INTEGER PRIMARY KEY,
    title TEXT,
    release_year INTEGER
);
INSERT INTO "Movies" VALUES(1,'The Shawshank Redemption',1994);
INSERT INTO "Movies" VALUES(2,'Pulp Fiction',1994);
INSERT INTO "Movies" VALUES(3,'Forrest Gump',1994);
INSERT INTO "Movies" VALUES(4,'The Godfather',1972);
INSERT INTO "Movies" VALUES(5,'Schindler''s List',1993);
CREATE TABLE Ratings (
    movie_id INTEGER REFERENCES Movies(id),
    rating FLOAT 
);
INSERT INTO "Ratings" VALUES(1,8.7);
INSERT INTO "Ratings" VALUES(2,8.9);
INSERT INTO "Ratings" VALUES(3,8.8);
INSERT INTO "Ratings" VALUES(4,8.5);
INSERT INTO "Ratings" VALUES(5,8.7);
CREATE TABLE "Theater_Plays" (
    theater_play_id INTEGER PRIMARY KEY,
    play_title TEXT,
    total_length_id INTEGER REFERENCES "Duration"("theater_play_id"));
INSERT INTO "Theater_Plays" VALUES(1,'Hamlet',1);
INSERT INTO "Theater_Plays" VALUES(2,'Macbeth',2);
INSERT INTO "Theater_Plays" VALUES(3,'Othello',3);
INSERT INTO "Theater_Plays" VALUES(4,'King Lear',4);
INSERT INTO "Theater_Plays" VALUES(5,'Antony and Cleopatra',5);
COMMIT;
