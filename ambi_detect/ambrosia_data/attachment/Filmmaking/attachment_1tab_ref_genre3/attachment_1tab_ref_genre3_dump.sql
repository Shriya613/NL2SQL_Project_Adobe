BEGIN TRANSACTION;
CREATE TABLE Directors(
    id INTEGER PRIMARY KEY,
    name TEXT,
    birth_year INTEGER,
    nationality TEXT
);
INSERT INTO "Directors" VALUES(1,'Stanley Kubrick',1928,'American');
INSERT INTO "Directors" VALUES(2,'David Robert Mitchell',1974,'American');
INSERT INTO "Directors" VALUES(3,'Christopher Nolan',1970,'English');
INSERT INTO "Directors" VALUES(4,'David Fincher',1962,'American');
CREATE TABLE Film_Director(
    film_id INTEGER,
    director_id INTEGER,
    FOREIGN KEY(film_id) REFERENCES Films(id),
    FOREIGN KEY(director_id) REFERENCES Directors(id),
    PRIMARY KEY(film_id, director_id)
);
INSERT INTO "Film_Director" VALUES(1,1);
INSERT INTO "Film_Director" VALUES(2,2);
INSERT INTO "Film_Director" VALUES(3,3);
INSERT INTO "Film_Director" VALUES(4,4);
CREATE TABLE Film_Genre(
    film_id INTEGER,
    genre_id INTEGER,
    FOREIGN KEY(film_id) REFERENCES Films(id),
    FOREIGN KEY(genre_id) REFERENCES Genres(id),
    PRIMARY KEY(film_id, genre_id)
);
INSERT INTO "Film_Genre" VALUES(1,1);
INSERT INTO "Film_Genre" VALUES(2,1);
INSERT INTO "Film_Genre" VALUES(3,2);
INSERT INTO "Film_Genre" VALUES(4,2);
CREATE TABLE Films(
    id INTEGER PRIMARY KEY,
    title TEXT,
    release_date DATE,
    director TEXT,
    budget REAL
);
INSERT INTO "Films" VALUES(1,'The Shining','1980-06-22','Stanley Kubrick',20000000.0);
INSERT INTO "Films" VALUES(2,'It Follows','2014-03-14','David Robert Mitchell',5000000.0);
INSERT INTO "Films" VALUES(3,'Inception','2010-07-16','Christopher Nolan',20000000.0);
INSERT INTO "Films" VALUES(4,'Gone Girl','2014-10-03','David Fincher',65000000.0);
CREATE TABLE Genres(
    id INTEGER PRIMARY KEY,
    genre TEXT
);
INSERT INTO "Genres" VALUES(1,'Horror');
INSERT INTO "Genres" VALUES(2,'Thriller');
COMMIT;
