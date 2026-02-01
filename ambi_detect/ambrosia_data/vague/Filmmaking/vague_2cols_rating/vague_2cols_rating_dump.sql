BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY,
    name TEXT,
    birthdate DATE,
    nationality TEXT
);
INSERT INTO "Actor" VALUES(1,'Tom Hanks','1956-07-09','American');
INSERT INTO "Actor" VALUES(2,'Brad Pitt','1963-12-18','American');
INSERT INTO "Actor" VALUES(3,'Leonardo DiCaprio','1974-11-11','American');
INSERT INTO "Actor" VALUES(4,'Morgan Freeman','1937-06-01','American');
INSERT INTO "Actor" VALUES(5,'Kevin Spacey','1959-07-26','American');
CREATE TABLE Camera (
    id INTEGER PRIMARY KEY,
    camera_type TEXT,
    resolution TEXT
);
INSERT INTO "Camera" VALUES(1,'ARRI Alexa','4K');
INSERT INTO "Camera" VALUES(2,'Red Epic Dragon','5K');
INSERT INTO "Camera" VALUES(3,'Canon C300 Mark II','4K');
INSERT INTO "Camera" VALUES(4,'Sony F55','4K');
INSERT INTO "Camera" VALUES(5,'Blackmagic Ursa Mini 4.6K','4.6K');
CREATE TABLE Cinematographer (
    id INTEGER PRIMARY KEY,
    name TEXT,
    birthdate DATE,
    nationality TEXT
);
INSERT INTO "Cinematographer" VALUES(1,'Roger Deakins','1949-05-24','English');
INSERT INTO "Cinematographer" VALUES(2,'Emmanuel Lubezki','1964-12-30','Mexican');
INSERT INTO "Cinematographer" VALUES(3,'Wally Pfister','1961-02-21','American');
INSERT INTO "Cinematographer" VALUES(4,'Janusz Kaminski','1959-08-05','Polish');
INSERT INTO "Cinematographer" VALUES(5,'Hoyte Van Hoytema','1961-11-24','Swedish');
CREATE TABLE Film (
    id INTEGER PRIMARY KEY,
    title TEXT,
    release_year INT,
    director TEXT,
    genre TEXT
);
INSERT INTO "Film" VALUES(1,'The Shawshank Redemption',1994,'Frank Darabont','Drama');
INSERT INTO "Film" VALUES(2,'Pulp Fiction',1994,'Quentin Tarantino','Crime');
INSERT INTO "Film" VALUES(3,'Schindler''s List',1993,'Steven Spielberg','Biographical Drama');
INSERT INTO "Film" VALUES(4,'Forrest Gump',1994,'Robert Zemeckis','Romantic Comedy-drama');
INSERT INTO "Film" VALUES(5,'Inception',2010,'Christopher Nolan','Action Sci-Fi');
CREATE TABLE FilmReviews (
    id INTEGER PRIMARY KEY,
    critic_rating REAL ,
    audience_rating REAL ,
    film_id INTEGER,
    FOREIGN KEY(film_id) REFERENCES Film(id)
);
INSERT INTO "FilmReviews" VALUES(1,9.0,8.5,1);
INSERT INTO "FilmReviews" VALUES(2,8.5,8.0,2);
INSERT INTO "FilmReviews" VALUES(3,8.0,7.5,3);
INSERT INTO "FilmReviews" VALUES(4,7.5,7.0,4);
INSERT INTO "FilmReviews" VALUES(5,8.5,8.0,5);
CREATE TABLE Role (
    id INTEGER PRIMARY KEY,
    actor_id INTEGER,
    role_name TEXT,
    film_id INTEGER,
    FOREIGN KEY(actor_id) REFERENCES Actor(id),
    FOREIGN KEY(film_id) REFERENCES Film(id)
);
INSERT INTO "Role" VALUES(1,1,'Andrew Beckett',1);
INSERT INTO "Role" VALUES(2,2,'Jared Vennett',1);
INSERT INTO "Role" VALUES(3,3,'Chris Cotton',1);
INSERT INTO "Role" VALUES(4,4,'Daniel Hillard',1);
INSERT INTO "Role" VALUES(5,5,'Philippe Petit',1);
CREATE TABLE ShotList (
    id INTEGER PRIMARY KEY,
    cinematographer_id INTEGER,
    shot_description TEXT,
    camera_id INTEGER,
    FOREIGN KEY(cinematographer_id) REFERENCES Cinematographer(id),
    FOREIGN KEY(camera_id) REFERENCES Camera(id)
);
INSERT INTO "ShotList" VALUES(1,1,'Establishing shot of New York City skyline',1);
INSERT INTO "ShotList" VALUES(2,1,'Medium close-up of main character walking down street',2);
INSERT INTO "ShotList" VALUES(3,2,'Long shot of characters on rooftop at night',3);
INSERT INTO "ShotList" VALUES(4,2,'Close-up of characters discussing plans',4);
INSERT INTO "ShotList" VALUES(5,3,'Wide shot of car chase through city streets',5);
COMMIT;
