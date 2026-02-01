BEGIN TRANSACTION;
CREATE TABLE Actor (
    actor_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    birthdate DATE
);
INSERT INTO "Actor" VALUES(1,'John','Doe','2000-01-01');
INSERT INTO "Actor" VALUES(2,'Jane','Smith','1998-07-30');
INSERT INTO "Actor" VALUES(3,'Michael','Brown','1986-12-10');
INSERT INTO "Actor" VALUES(4,'Emily','White','1995-03-25');
INSERT INTO "Actor" VALUES(5,'Robert','Black','1991-09-09');
CREATE TABLE Actress (
    actress_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    birthdate DATE
);
INSERT INTO "Actress" VALUES(1,'Alice','Green','1975-06-02');
INSERT INTO "Actress" VALUES(2,'Betty','Blue','1981-11-13');
INSERT INTO "Actress" VALUES(3,'Carol','Yellow','1992-04-10');
INSERT INTO "Actress" VALUES(4,'Denise','Red','1990-05-15');
INSERT INTO "Actress" VALUES(5,'Eva','Purple','1985-02-03');
CREATE TABLE Director (
    director_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    birthdate DATE
);
INSERT INTO "Director" VALUES(1,'David','Gray','1964-03-25');
INSERT INTO "Director" VALUES(2,'Steven','Wilson','1968-06-10');
INSERT INTO "Director" VALUES(3,'Quentin','Tarantino','1963-03-27');
INSERT INTO "Director" VALUES(4,'Christopher','Nolan','1970-07-30');
INSERT INTO "Director" VALUES(5,'Martin','Scorsese','1942-11-17');
CREATE TABLE Genre (
    genre_id INTEGER PRIMARY KEY,
    genre_name TEXT UNIQUE
);
INSERT INTO "Genre" VALUES(1,'Action');
INSERT INTO "Genre" VALUES(2,'Adventure');
INSERT INTO "Genre" VALUES(3,'Animation');
INSERT INTO "Genre" VALUES(4,'Biography');
INSERT INTO "Genre" VALUES(5,'Drama');
CREATE TABLE Producer (
    producer_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    birthdate DATE
);
INSERT INTO "Producer" VALUES(1,'Barbara','Lee','1957-05-02');
INSERT INTO "Producer" VALUES(2,'Kevin','Feige','1973-06-19');
INSERT INTO "Producer" VALUES(3,'James','Weaver','1961-08-03');
INSERT INTO "Producer" VALUES(4,'Bryan','Cranston','1956-03-07');
INSERT INTO "Producer" VALUES(5,'Frank','Marshall','1950-01-15');
CREATE TABLE Show (
    show_id INTEGER PRIMARY KEY,
    title TEXT,
    genre TEXT,
    Number_of_seasons INTEGER,
    Number_of_episodes INTEGER,
    release_year INTEGER,
    FOREIGN KEY(genre) REFERENCES Genre(genre_name)
);
INSERT INTO "Show" VALUES(1,'The Big Bang Theory','Comedy',12,279,2007);
INSERT INTO "Show" VALUES(2,'Friends','Romantic Comedy',10,236,1994);
INSERT INTO "Show" VALUES(3,'Game of Thrones','Fantasy Drama',8,60,2011);
INSERT INTO "Show" VALUES(4,'Breaking Bad','Crime Drama',5,62,2008);
INSERT INTO "Show" VALUES(5,'Stranger Things','Horror Sci-Fi',4,39,2016);
COMMIT;
