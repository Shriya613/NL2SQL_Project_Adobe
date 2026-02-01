BEGIN TRANSACTION;
CREATE TABLE Artist_Techniques(
    artist_id INTEGER,
    technique_id INTEGER,
    FOREIGN KEY(artist_id) REFERENCES Makeup_Artists(artist_id),
    FOREIGN KEY(technique_id) REFERENCES Techniques(technique_id));
INSERT INTO "Artist_Techniques" VALUES(1,1);
INSERT INTO "Artist_Techniques" VALUES(2,2);
INSERT INTO "Artist_Techniques" VALUES(3,3);
INSERT INTO "Artist_Techniques" VALUES(4,4);
INSERT INTO "Artist_Techniques" VALUES(5,5);
INSERT INTO "Artist_Techniques" VALUES(2,1);
INSERT INTO "Artist_Techniques" VALUES(3,1);
INSERT INTO "Artist_Techniques" VALUES(4,1);
INSERT INTO "Artist_Techniques" VALUES(5,1);
CREATE TABLE Makeup_Artists(
    artist_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT);
INSERT INTO "Makeup_Artists" VALUES(1,'John Doe');
INSERT INTO "Makeup_Artists" VALUES(2,'Jane Smith');
INSERT INTO "Makeup_Artists" VALUES(3,'Michael Brown');
INSERT INTO "Makeup_Artists" VALUES(4,'Emily White');
INSERT INTO "Makeup_Artists" VALUES(5,'David Green');
CREATE TABLE Movie_Makeup_Artists(
    movie_id INTEGER,
    artist_id INTEGER,
    FOREIGN KEY(movie_id) REFERENCES Movies(movie_id),
    FOREIGN KEY(artist_id) REFERENCES Makeup_Artists(artist_id));
INSERT INTO "Movie_Makeup_Artists" VALUES(1,1);
INSERT INTO "Movie_Makeup_Artists" VALUES(1,3);
INSERT INTO "Movie_Makeup_Artists" VALUES(2,2);
INSERT INTO "Movie_Makeup_Artists" VALUES(2,4);
INSERT INTO "Movie_Makeup_Artists" VALUES(3,1);
CREATE TABLE Movie_Technique(
    movie_id INTEGER,
    technique_id INTEGER,
    FOREIGN KEY(movie_id) REFERENCES Movies(movie_id),
    FOREIGN KEY(technique_id) REFERENCES Techniques(technique_id));
INSERT INTO "Movie_Technique" VALUES(1,1);
INSERT INTO "Movie_Technique" VALUES(1,2);
INSERT INTO "Movie_Technique" VALUES(2,3);
INSERT INTO "Movie_Technique" VALUES(3,4);
INSERT INTO "Movie_Technique" VALUES(4,5);
CREATE TABLE Movies(
    movie_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE);
INSERT INTO "Movies" VALUES(1,'The_Greatest_Showman','2017-12-08');
INSERT INTO "Movies" VALUES(2,'Black_Panther','2018-02-16');
INSERT INTO "Movies" VALUES(3,'Inception','2010-07-16');
INSERT INTO "Movies" VALUES(4,'Avatar','2009-12-18');
INSERT INTO "Movies" VALUES(5,'Interstellar','2014-11-07');
CREATE TABLE Techniques(
    technique_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Techniques" VALUES(1,'Age_Progression');
INSERT INTO "Techniques" VALUES(2,'Prosthetics');
INSERT INTO "Techniques" VALUES(3,'Special_Effects_Makeup');
INSERT INTO "Techniques" VALUES(4,'Corrective_Makeup');
INSERT INTO "Techniques" VALUES(5,'Period_Makeup');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Makeup_Artists',5);
INSERT INTO "sqlite_sequence" VALUES('Techniques',5);
INSERT INTO "sqlite_sequence" VALUES('Movies',5);
COMMIT;
