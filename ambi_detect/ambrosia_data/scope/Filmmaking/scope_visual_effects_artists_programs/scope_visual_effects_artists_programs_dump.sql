BEGIN TRANSACTION;
CREATE TABLE Film_Movies(
    film_id INTEGER,
    movie_id INTEGER,
    FOREIGN KEY (film_id) REFERENCES Films(film_id),
    FOREIGN KEY (movie_id) REFERENCES Movies(movie_id),
    PRIMARY KEY (film_id, movie_id)
);
INSERT INTO "Film_Movies" VALUES(1,1);
INSERT INTO "Film_Movies" VALUES(2,2);
INSERT INTO "Film_Movies" VALUES(3,3);
INSERT INTO "Film_Movies" VALUES(4,4);
INSERT INTO "Film_Movies" VALUES(5,5);
CREATE TABLE Movies(
    movie_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE
);
INSERT INTO "Movies" VALUES(1,'The Matrix','1999-03-31');
INSERT INTO "Movies" VALUES(2,'Inception','2010-07-16');
INSERT INTO "Movies" VALUES(3,'Interstellar','2014-11-07');
INSERT INTO "Movies" VALUES(4,'Gravity','2013-10-04');
INSERT INTO "Movies" VALUES(5,'Avatar','2009-12-18');
CREATE TABLE Programs(
    program_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    version INT
);
INSERT INTO "Programs" VALUES(1,'After_Effects',2021);
INSERT INTO "Programs" VALUES(2,'Maya',2021);
INSERT INTO "Programs" VALUES(3,'Cinema 4D',2021);
INSERT INTO "Programs" VALUES(4,'Nuke',2021);
INSERT INTO "Programs" VALUES(5,'Premiere Pro',2021);
CREATE TABLE Visual_Effects_Artists(
    artist_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Visual_Effects_Artists" VALUES(1,'John Doe','johndoe@example.com');
INSERT INTO "Visual_Effects_Artists" VALUES(2,'Jane Smith','janesmith@example.com');
INSERT INTO "Visual_Effects_Artists" VALUES(3,'Alice Johnson','alicejohnson@example.com');
INSERT INTO "Visual_Effects_Artists" VALUES(4,'Bob Brown','bobbrown@example.com');
INSERT INTO "Visual_Effects_Artists" VALUES(5,'Charlie Green','charliegreen@example.com');
CREATE TABLE Visual_Effects_Artists_Programs(
    visual_effects_artist_id INTEGER,
    program_id INTEGER,
    FOREIGN KEY (visual_effects_artist_id) REFERENCES Visual_Effects_Artists(artist_id),
    FOREIGN KEY (program_id) REFERENCES Programs(program_id),
    PRIMARY KEY (visual_effects_artist_id, program_id)
);
INSERT INTO "Visual_Effects_Artists_Programs" VALUES(1,1);
INSERT INTO "Visual_Effects_Artists_Programs" VALUES(2,1);
INSERT INTO "Visual_Effects_Artists_Programs" VALUES(3,2);
INSERT INTO "Visual_Effects_Artists_Programs" VALUES(4,3);
INSERT INTO "Visual_Effects_Artists_Programs" VALUES(5,4);
INSERT INTO "Visual_Effects_Artists_Programs" VALUES(3,1);
INSERT INTO "Visual_Effects_Artists_Programs" VALUES(4,1);
INSERT INTO "Visual_Effects_Artists_Programs" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Visual_Effects_Artists',5);
INSERT INTO "sqlite_sequence" VALUES('Programs',5);
INSERT INTO "sqlite_sequence" VALUES('Movies',5);
COMMIT;
