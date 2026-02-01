BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY,
    name TEXT,
    date_of_birth DATE,
    gender CHAR(1) 
);
INSERT INTO "Actor" VALUES(1,'Leonardo DiCaprio','November 11, 1974','Male');
INSERT INTO "Actor" VALUES(2,'Brad Pitt','December 18, 1963','Male');
INSERT INTO "Actor" VALUES(3,'Marion Cotillard','September 30, 1975','Female');
INSERT INTO "Actor" VALUES(4,'Ken Watanabe','October 21, 1959','Male');
INSERT INTO "Actor" VALUES(5,'Ellen Page','February 21, 1987','Female');
CREATE TABLE Award (
    award_name TEXT,
    category TEXT,
    PRIMARY KEY(award_name)
);
INSERT INTO "Award" VALUES('Academy Award','Film');
INSERT INTO "Award" VALUES('Golden Globe Award','Film');
INSERT INTO "Award" VALUES('BAFTA Award','Film');
INSERT INTO "Award" VALUES('Palme d''Or','Film');
INSERT INTO "Award" VALUES('Cannes Film Festival Best Director','Direction');
CREATE TABLE Director (
    id INTEGER PRIMARY KEY,
    name TEXT,
    birth_date DATE,
    nationality TEXT
);
INSERT INTO "Director" VALUES(1,'Christopher Nolan','July 30, 1970','American');
INSERT INTO "Director" VALUES(2,'Quentin Tarantino','March 27, 1963','American');
INSERT INTO "Director" VALUES(3,'Guillermo del Toro','October 9, 1964','Mexican');
INSERT INTO "Director" VALUES(4,'Steven Spielberg','December 18, 1946','American');
INSERT INTO "Director" VALUES(5,'Martin Scorsese','November 17, 1942','American');
CREATE TABLE "Movie" (
    id INTEGER PRIMARY KEY,
    title TEXT,
    genre TEXT,
    release_year INTEGER,
    Film_Creator INTEGER,
    FOREIGN KEY(Film_Creator) REFERENCES Studio(id),
    FOREIGN KEY(Film_Creator) REFERENCES Director(id)
);
INSERT INTO "Movie" VALUES(1,'Inception','Action/Sci-Fi',2010,1);
INSERT INTO "Movie" VALUES(2,'Pulp Fiction','Crime/Drama',1994,3);
INSERT INTO "Movie" VALUES(3,'Pan''s Labyrinth','Fantasy/Drama',2006,4);
INSERT INTO "Movie" VALUES(4,'Schindler''s List','Historical/Drama',1993,2);
INSERT INTO "Movie" VALUES(5,'The Departed','Crime/Thriller',2006,3);
CREATE TABLE Role (
    movie_id INTEGER,
    actor_id INTEGER,
    character TEXT,
    PRIMARY KEY(movie_id, actor_id),
    FOREIGN KEY(movie_id) REFERENCES Movie(id),
    FOREIGN KEY(actor_id) REFERENCES Actor(id)
);
INSERT INTO "Role" VALUES(1,1,'Dom Cobb');
INSERT INTO "Role" VALUES(1,2,'Arthur');
INSERT INTO "Role" VALUES(2,3,'Vincent Vega');
INSERT INTO "Role" VALUES(2,4,'Mia Wallace');
INSERT INTO "Role" VALUES(4,5,'Ofelia');
CREATE TABLE Studio (
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT,
    year_founded INTEGER
);
INSERT INTO "Studio" VALUES(1,'Warner Bros','Los Angeles, CA',1923);
INSERT INTO "Studio" VALUES(2,'Disney Studios','Burbank, CA',1923);
INSERT INTO "Studio" VALUES(3,'Paramount Pictures','Hollywood, CA',1914);
INSERT INTO "Studio" VALUES(4,'Universal Studios','Universal City, CA',1912);
INSERT INTO "Studio" VALUES(5,'Columbia Pictures','Culver City, CA',1924);
COMMIT;
